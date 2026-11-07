"""T1 driver: generate -> extract -> label -> split, one jsonl per (backbone, dataset)."""
from __future__ import annotations

import random
import json
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from ..backbones.registry import build_backbone
from ..config import Config
from ..data.datasets import Sample, load_dataset
from ..data.images import load_image
from ..utils import get_logger, mark_done, read_json, read_jsonl, write_jsonl, write_json
from .claim_extractor import claim_from_question, extract_claims_from_caption
from .label_matcher import SynonymTable, label_claim, label_object_claim, sanity_report

LOG = get_logger("hmqr.build_claims")
SPLIT_SCHEMA_VERSION = 2


def _split_by_image(
    rows: List[Dict[str, Any]],
    all_image_ids: List[str],
    seed: int,
    calib_cp_ratio: float,
    calib_revision_ratio: float,
) -> None:
    """Shared three-way split by image identity.

    ``all_image_ids`` comes from the dataset loader rather than generated claims,
    so every backbone receives the same split even when claim extraction yields a
    different number of rows.
    """
    image_ids = sorted({str(v) for v in all_image_ids})
    rng = random.Random(seed)
    rng.shuffle(image_ids)
    n_cp = int(round(len(image_ids) * calib_cp_ratio))
    n_revision = int(round(len(image_ids) * calib_revision_ratio))
    calib_cp = set(image_ids[:n_cp])
    calib_revision = set(image_ids[n_cp:n_cp + n_revision])
    for r in rows:
        image_id = str(r["image_id"])
        if image_id in calib_cp:
            r["split"] = "calib_cp"
        elif image_id in calib_revision:
            r["split"] = "calib_revision"
        else:
            r["split"] = "test"
        r["split_schema_version"] = SPLIT_SCHEMA_VERSION


def _probe_split(rows: List[Dict[str, Any]], seed: int, train_ratio: float = 0.6) -> None:
    """Disjoint-by-image probe_train / probe_test assignment (in place)."""
    image_ids = sorted({r["image_id"] for r in rows})
    rng = random.Random(seed + 1)
    rng.shuffle(image_ids)
    n_train = int(round(len(image_ids) * train_ratio))
    train = set(image_ids[:n_train])
    for r in rows:
        r["probe_split"] = "probe_train" if r["image_id"] in train else "probe_test"


def build_for(
    cfg: Config,
    backbone_name: str,
    dataset_name: str,
    overwrite: bool = False,
) -> Optional[Path]:
    out_path = cfg.path("processed") / f"claims_{backbone_name}_{dataset_name}.jsonl"
    partial_path = out_path.with_name(out_path.name + ".partial")
    progress_path = out_path.with_name(out_path.name + ".progress.json")
    complete_path = out_path.with_name(out_path.name + ".complete.json")
    generations_path = out_path.with_name(f'generations_{backbone_name}_{dataset_name}.jsonl')
    has_checkpoint = partial_path.exists() and progress_path.exists()
    if out_path.exists() and complete_path.exists() and not overwrite and not has_checkpoint:
        complete = read_json(complete_path)
        existing_rows = read_jsonl(out_path)
        if (
            int(complete.get("claim_count", -1)) == len(existing_rows)
            and all(
                row.get("split_schema_version") == SPLIT_SCHEMA_VERSION
                for row in existing_rows
            )
        ):
            LOG.info("[skip] %s has a verified completion record", out_path.name)
            return out_path
        if int(complete.get("claim_count", -1)) == len(existing_rows):
            raise RuntimeError(
                f"{out_path.name} has stale image/split metadata; run "
                "scripts/migrate_claim_splits.py --apply before step 01"
            )
    elif out_path.exists() and not overwrite and not has_checkpoint:
        LOG.warning("[rebuild] %s has no verified completion record", out_path.name)

    samples = load_dataset(cfg, dataset_name)
    full_sample_count=len(samples)
    audit_only=bool(getattr(cfg,'_audit_batch_only',False))
    if audit_only:
        if dataset_name!='chair500':raise ValueError('Audit-priority generation is defined only for CHAIR-500')
        seed=int(cfg.get('seed',42))
        samples=sorted(samples,key=lambda s:hashlib.sha256(f'{seed}|human_audit|{s.image_id}'.encode()).hexdigest())[:int(cfg['reviewer_revision']['annotation_images_per_backbone'])]
    amber_root = cfg.abspath(cfg["datasets"]["amber"]["root"])
    syn = SynonymTable.from_amber(amber_root) if amber_root.exists() else SynonymTable()
    need_generation = any(s.mode == "generative" for s in samples)
    backbone = build_backbone(cfg, backbone_name, load=False)

    rows: List[Dict[str, Any]] = []
    completed_ids = set()
    if need_generation and not overwrite and partial_path.exists() and progress_path.exists():
        rows = read_jsonl(partial_path)
        progress = read_json(progress_path)
        completed_ids = {str(v) for v in progress.get("completed_image_ids", [])}
        LOG.info(
            "[resume] %s: %d images and %d claims checkpointed",
            out_path.name, len(completed_ids), len(rows),
        )
    max_new = int(cfg["generation"]["max_new_tokens"])
    caption_prompt = cfg["generation"]["caption_prompt"]
    recorded_images = set()
    if need_generation:
        recorded_images = {str(r['image_id']) for r in read_jsonl(generations_path)} if generations_path.exists() else set()
        # Recover available caption text without inventing unavailable raw responses.
        with generations_path.open('a',encoding='utf-8') as stream:
            for row in rows:
                identifier = str(row['image_id'])
                if identifier not in recorded_images:
                    record = dict(image_id=identifier,backbone=backbone_name,dataset=dataset_name,
                                  caption=row['caption'],image_path=row['image_path'],
                                  source='recovered_from_partial_claim_cache',raw_extraction_response=None,
                                  extraction_route='not_recorded_in_historical_partial',schema_version=1)
                    stream.write(json.dumps(record,ensure_ascii=False)+'\n');recorded_images.add(identifier)
        # A completed zero-claim image with no saved caption is incomplete evidence.
        completed_ids &= recorded_images

    if need_generation and any(str(s.image_id) not in completed_ids for s in samples):
        backbone.load()

    for sample_i, s in enumerate(
        tqdm(samples, desc=f"{backbone_name}/{dataset_name}", ncols=88), start=1
    ):
        if str(s.image_id) in completed_ids:
            continue
        if s.mode == "discriminative":
            claim = claim_from_question(s)
            if claim is None:
                continue
            claims = [claim]
            caption = s.question or ""
        else:
            image = load_image(s.image_path)
            if image is None:
                continue
            gen = backbone.generate(image, caption_prompt, max_new_tokens=max_new)
            caption = str(gen["text"]).strip()
            if not caption:raise RuntimeError(f'Empty formal generated caption: {backbone_name}/{dataset_name}/{s.image_id}')
            claims = extract_claims_from_caption(backbone, image, caption)
            audit = getattr(backbone,'last_extraction_audit',{})
            record = dict(image_id=str(s.image_id),image_path=s.image_path,backbone=backbone_name,
                          dataset=dataset_name,caption=caption,caption_prompt=caption_prompt,
                          max_new_tokens=max_new,raw_extraction_response=audit.get('raw_response'),
                          extraction_route=audit.get('route'),extraction_error=audit.get('error'),
                          n_extracted_claims=len(claims),canonical_hf_id=cfg.backbone_cfg(backbone_name)['hf_id'],
                          extracted_claims=claims,
                          generation_prompt_format=getattr(backbone,'generation_prompt_format','backbone_default'),
                          generation_adapter_version=getattr(backbone,'generation_adapter_version',1),
                          dtype=cfg.get('dtype'),seed=cfg.get('seed',42),schema_version=1,
                          config_sha256=hashlib.sha256(json.dumps(cfg.raw).encode()).hexdigest(),
                          source='formal_local_model_generation')
            with generations_path.open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(record,ensure_ascii=False)+'\n')

        # Some discriminative benchmarks reuse question IDs. The loader can
        # provide a row-stable sample ID while ``image_id`` remains the true
        # image identity used by leakage-free calibration/test splits.
        sample_key = str(s.meta.get("sample_id", s.image_id))
        for j, claim in enumerate(claims):
            label = label_claim(claim, s.mode, s.gt_answer, s.gt, syn)
            rows.append(
                {
                    "claim_id": f"{backbone_name}|{dataset_name}|{sample_key}|{j}",
                    "image_id": s.image_id,
                    "image_path": s.image_path,
                    "backbone": backbone_name,
                    "dataset": dataset_name,
                    "mode": s.mode,
                    "caption": caption,
                    "claim_type": claim["claim_type"],
                    "object": claim.get("object"),
                    "attribute": claim.get("attribute"),
                    "subject": claim.get("subject"),
                    "relation": claim.get("relation"),
                    "sentence_idx": claim.get("sentence_idx", 0),
                    "sentence": claim.get("sentence", caption),
                    "label": label,
                }
            )

        # object-level claims for the layer-wise probe (generative samples only)
        if s.mode == "generative" and s.gt:
            seen = set()
            for claim in claims:
                for obj in (claim.get("object"), claim.get("subject")):
                    if not obj or obj in seen:
                        continue
                    seen.add(obj)
                    rows.append(
                        {
                            "claim_id": f"{backbone_name}|{dataset_name}|{sample_key}|obj{len(seen)}",
                            "image_id": s.image_id,
                            "image_path": s.image_path,
                            "backbone": backbone_name,
                            "dataset": dataset_name,
                            "mode": s.mode,
                            "caption": caption,
                            "claim_type": "obj",
                            "object": obj,
                            "attribute": None,
                            "subject": None,
                            "relation": None,
                            "sentence_idx": claim.get("sentence_idx", 0),
                            "sentence": claim.get("sentence", caption),
                            "label": label_object_claim(obj, s.gt, syn),
                        }
                    )

        if need_generation:
            completed_ids.add(str(s.image_id))
            if sample_i % 10 == 0 or sample_i == len(samples):
                write_jsonl(partial_path, rows)
                write_json(
                    progress_path,
                    {"completed_image_ids": sorted(completed_ids), "claim_count": len(rows)},
                )

    if need_generation:
        backbone.unload()

    if audit_only:
        write_jsonl(partial_path,rows)
        write_json(progress_path,{'completed_image_ids':sorted(completed_ids),'claim_count':len(rows)})
        expected={str(s.image_id) for s in samples}
        missing=sorted(expected-completed_ids)
        write_json(cfg.path('logs')/f'audit_generation_{backbone_name}.json',
                   {'status':'complete' if not missing else 'incomplete','backbone':backbone_name,
                    'n_required_audit_images':len(expected),'n_completed_audit_images':len(expected & completed_ids),
                    'n_full_dataset_images':full_sample_count,'n_completed_full_dataset_images':len(completed_ids),
                    'missing_audit_image_ids':missing,'full_dataset_completion_written':False})
        if missing:raise RuntimeError(f'Formal audit captions missing for {backbone_name}: {missing}')
        LOG.info('audit captions complete for %s; full %d-image job remains checkpointed',backbone_name,full_sample_count)
        return partial_path

    if need_generation:
        canonical_order={str(sample.image_id):i for i,sample in enumerate(samples)}
        rows.sort(key=lambda row:canonical_order[str(row['image_id'])])

    cp_cfg = cfg["cp"]
    _split_by_image(
        rows,
        [str(s.image_id) for s in samples],
        cfg.get("seed", 42),
        float(cp_cfg.get("calib_cp_ratio", 0.25)),
        float(cp_cfg.get("calib_revision_ratio", 0.25)),
    )
    _probe_split(rows, cfg.get("seed", 42))
    write_jsonl(out_path, rows)
    write_json(
        complete_path,
        {
            "backbone": backbone_name,
            "dataset": dataset_name,
            "sample_count": len(samples),
            "claim_count": len(rows),
        },
    )
    if partial_path.exists():
        partial_path.unlink()
    if progress_path.exists():
        progress_path.unlink()

    rep = sanity_report(rows, f"{backbone_name}/{dataset_name}")
    write_json(cfg.path("logs") / f"claims_report_{backbone_name}_{dataset_name}.json", rep)
    LOG.info("wrote %s (%d claims)", out_path.name, len(rows))
    return out_path


def build_all(cfg: Config, overwrite: bool = False) -> int:
    """Build claim files for every (backbone, dataset). Returns the failure count.

    The step is marked done only when every requested backbone/dataset pair was
    produced. Paper results may never be assembled from a partial matrix.
    """
    all_datasets = list(cfg["datasets"]["detect"]) + list(cfg["datasets"]["quality"])
    datasets = getattr(cfg, "_selected_datasets", None) or all_datasets

    n_ok = 0
    failures: List[str] = []
    backbone_names = getattr(cfg, "_selected_backbones", None) or cfg.backbone_names
    for backbone_name in backbone_names:
        for dataset_name in datasets:
            try:
                build_for(cfg, backbone_name, dataset_name, overwrite=overwrite)
                n_ok += 1
            except Exception as exc:  # noqa: BLE001
                LOG.exception("[%s/%s] failed: %s", backbone_name, dataset_name, exc)
                failures.append(f"{backbone_name}/{dataset_name}: {exc}")

    total = n_ok + len(failures)
    LOG.info("claim building: %d/%d succeeded, %d failed", n_ok, total, len(failures))
    if failures:
        LOG.error(
            "At least one claim job failed. Not marking this step as done. "
            "Fix the errors above (missing dataset files or unreachable model hub) "
            "and re-run: python run_all.py --from 01"
        )
        return len(failures)

    if (set(backbone_names) == set(cfg.backbone_names)
            and set(datasets) == set(all_datasets)):
        mark_done(cfg.path("processed"), "claims_done")
    return len(failures)
