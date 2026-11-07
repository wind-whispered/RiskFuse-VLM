"""T4 driver: compute ALD / RLD / global-CLIP / internal confidence for every claim.

Order note
----------
`l_p` (internal confidence) is the backbone's own one-shot yes/no verification
belief. Layer-wise probes are a separate mechanism analysis and never overwrite
the mitigation pipeline's confidence signal.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from tqdm import tqdm

from ..backbones.registry import build_backbone
from ..config import Config
from ..data.images import load_image
from ..tools.clip_scorer import ClipScorer
from ..tools.grounding_dino import GroundingDino
from ..utils import get_logger, mark_done, read_jsonl, write_json, write_jsonl
from .global_clip import g_global
from .grounding_attr import g_attr
from .grounding_count import count_consistency
from .grounding_rel import g_rel, score_relation_boxes
from .internal_conf import (
    VERIFY_PROMPT,
    sample_answers,
    semantic_entropy,
    verify_confidence_signals,
)
from .lp import get_lp, has_valid_lp  # re-exported for backwards compatibility
from .support import attribute_kind, matched_signal_group, relation_signal_kind

LOG = get_logger("hmqr.signals")
SIGNALS_SCHEMA_VERSION = 4

__all__ = ["get_lp", "compute_all", "compute_for_file", "normalise_g"]


def _entropy_required(
    row: Dict[str, Any], backbone=None, with_semantic_entropy: bool = True
) -> bool:
    return bool(
        with_semantic_entropy
        and backbone is not None
        and row.get("mode") == "discriminative"
        and row.get("claim_type") in ("attr", "rel")
        and row.get("split") in ("calib_cp", "test")
        and row.get("label") in (0, 1)
    )


def _core_signals_complete(row: Dict[str, Any], backbone=None) -> bool:
    if row.get("signals_schema_version") != SIGNALS_SCHEMA_VERSION:
        return False
    if not bool(row.get("signal_computed")):
        return False
    if "signal_available" not in row or row.get("g_global") is None:
        return False
    return backbone is None or has_valid_lp(row)


def _signals_complete(
    row: Dict[str, Any],
    backbone=None,
    with_semantic_entropy: bool = True,
) -> bool:
    """Return whether a cached row satisfies the current invocation contract.

    Diagnostic runs may intentionally skip the backbone confidence or semantic
    entropy.  A current schema stamp alone therefore cannot make a later paper
    run skip those still-missing outputs.
    """
    if not _core_signals_complete(row, backbone):
        return False
    entropy_required = _entropy_required(row, backbone, with_semantic_entropy)
    return not entropy_required or bool(row.get("semantic_entropy_computed"))


def _minmax(values: List[float]) -> Dict[str, float]:
    arr = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if arr.size == 0:
        return {"min": 0.0, "max": 1.0}
    lo, hi = float(np.min(arr)), float(np.max(arr))
    if hi - lo < 1e-8:
        hi = lo + 1.0
    return {"min": lo, "max": hi}


def _normalisation_group(row: Dict[str, Any]) -> tuple:
    subtype = matched_signal_group(row).split("|", 1)[1]
    return (str(row["backbone"]), str(row["dataset"]), str(row["claim_type"]), str(subtype))


def _route_cached_core(row: Dict[str, Any], cfg: Config) -> bool:
    """Upgrade a cached visual signal without re-running frozen tools."""
    claim_type = str(row.get("claim_type"))
    if row.get("g_global") is None:
        return False
    if claim_type == "attr":
        kind = attribute_kind(row)
        if kind == "state":
            if row.get("g_attr") is None or row.get("box_object") is None:
                return False
            row["g_raw"] = float(row["g_attr"])
            row["evidence_source"] = "clip_object_crop"
        elif kind == "number" and row.get("g_count") is not None:
            row["g_raw"] = float(row["g_count"])
            row["evidence_source"] = "dino_count_consistency"
        elif kind != "number":
            row["g_raw"] = float(row["g_global"])
            row["evidence_source"] = "global_clip_fallback"
        else:
            return False
        row["rel_kind"] = None
        row["signal_available"] = True
        row["signal_computed"] = True
        return True
    if claim_type == "rel":
        kind = relation_signal_kind(row)
        row["rel_kind"] = kind
        if kind == "global":
            row["g_rel"] = None
            row["g_raw"] = float(row["g_global"])
            row["evidence_source"] = "global_clip_fallback"
            row["signal_available"] = True
            row["signal_computed"] = True
            return True
        subject_box, object_box = row.get("box_subject"), row.get("box_object")
        if not subject_box or not object_box:
            return False
        score, scored_kind = score_relation_boxes(
            subject_box,
            object_box,
            str(row.get("relation") or ""),
            cfg["relation_contact_keywords"],
            cfg["relation_direction_vectors"],
        )
        if score is None:
            return False
        row["g_rel"] = float(score)
        row["g_raw"] = float(score)
        row["rel_kind"] = scored_kind
        row["evidence_source"] = f"dino_box_{scored_kind}"
        row["box_found"] = True
        row["signal_available"] = True
        row["signal_computed"] = True
        return True
    if row.get("g_raw") is not None:
        row["signal_available"] = True
        row["signal_computed"] = True
        return True
    return False


def normalise_g(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Fit matched-signal scaling on ``calib_cp`` and apply it unchanged.

    Attribute, contact-relation and direction-relation scores have different raw
    domains and are therefore calibrated separately. Unavailable signals remain
    missing; they are never replaced by a neutral score.
    """
    groups: Dict[tuple, List[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        if r.get("claim_type") in ("attr", "rel"):
            groups[_normalisation_group(r)].append(i)

    params: Dict[str, Dict[str, float]] = {}
    for key, idxs in groups.items():
        fit_idxs = [i for i in idxs if rows[i].get("split") == "calib_cp"]
        raw = [rows[i].get("g_raw") for i in fit_idxs]
        if not any(v is not None for v in raw):
            group_name = "|".join(key)
            LOG.warning(
                "no usable calib_cp grounding scores for %s; disabling the calibrated "
                "grounding signal for this group to avoid test leakage",
                group_name,
            )
            params[group_name] = {
                "min": None,
                "max": None,
                "status": "unavailable_no_calib",
            }
            for i in idxs:
                rows[i]["g"] = None
                rows[i]["signal_available"] = False
                rows[i]["g_normalisation_group"] = group_name
                rows[i]["g_normalisation_fit_split"] = "calib_cp"
                rows[i]["g_normalisation_status"] = "unavailable_no_calib"
            continue
        p = _minmax(raw)
        p["status"] = "fitted"
        params["|".join(key)] = p
        span = p["max"] - p["min"]
        for i in idxs:
            v = rows[i].get("g_raw")
            rows[i]["g"] = (
                float(np.clip((v - p["min"]) / span, 0.0, 1.0))
                if v is not None else None
            )
            rows[i]["g_normalisation_group"] = "|".join(key)
            rows[i]["g_normalisation_fit_split"] = "calib_cp"
            rows[i]["g_normalisation_status"] = "fitted"
    return params


def compute_for_file(
    cfg: Config,
    path: Path,
    clip: ClipScorer,
    dino: GroundingDino,
    backbone=None,
    overwrite: bool = False,
    with_semantic_entropy: bool = True,
) -> None:
    rows = read_jsonl(path)
    if not rows:
        return
    if all(
        _signals_complete(r, backbone, with_semantic_entropy) for r in rows
    ) and not overwrite:
        LOG.info("[skip] signals already present in %s", path.name)
        return

    partial_path = path.with_name(path.name + ".signals.partial")
    if partial_path.exists() and not overwrite:
        partial_rows = read_jsonl(partial_path)
        if (
            len(partial_rows) == len(rows)
            and [r.get("claim_id") for r in partial_rows]
            == [r.get("claim_id") for r in rows]
        ):
            rows = partial_rows
            done = sum(bool(r.get("signal_computed")) for r in rows)
            LOG.info("[resume] %s: %d/%d signal rows checkpointed", path.name, done, len(rows))

    contact_kw = cfg["relation_contact_keywords"]
    dir_vecs = cfg["relation_direction_vectors"]

    for row_i, row in enumerate(
        tqdm(rows, desc=f"signals {path.stem}", ncols=88), start=1
    ):
        core_done = _core_signals_complete(row, backbone) and not overwrite
        lp_done = has_valid_lp(row)
        entropy_required = _entropy_required(row, backbone, with_semantic_entropy)
        entropy_done = bool(row.get("semantic_entropy_computed")) or not entropy_required
        if core_done and (backbone is None or lp_done) and entropy_done:
            continue
        if not core_done:
            routed_from_cache = _route_cached_core(row, cfg)
            if not routed_from_cache:
                image = load_image(row["image_path"])
                if image is None:
                    row["g_raw"], row["g_global"], row["l_p"] = None, None, None
                    row["box_found"] = False
                    row["signal_available"] = False
                    row["signal_computed"] = True
                    continue
                ck = str(row["image_path"])
                if row.get("g_global") is None:
                    row["g_global"] = g_global(clip, image, row)
                if row["claim_type"] == "attr":
                    kind = attribute_kind(row)
                    if kind == "state":
                        res = g_attr(
                            clip, dino, image, row.get("object") or "",
                            row.get("attribute") or "", ck,
                        )
                        row["g_attr"] = res["g_attr"]
                        row["g_raw"] = res["g_attr"]
                        row["box_object"] = res["box_object"]
                        row["box_found"] = bool(res["box_found"])
                        row["signal_available"] = bool(res["signal_available"])
                        row["evidence_source"] = "clip_object_crop"
                    elif kind == "number":
                        res = count_consistency(
                            dino, image, row.get("object") or "",
                            row.get("attribute"), cache_key=ck,
                        )
                        row.update({
                            "g_count": res["g_count"],
                            "expected_count": res["expected_count"],
                            "detected_count": res["detected_count"],
                            "count_detections": res["count_detections"],
                        })
                        row["g_raw"] = res["g_count"]
                        row["box_object"] = (
                            res["count_detections"][0]["box"]
                            if res["count_detections"] else None
                        )
                        row["box_found"] = bool(res["box_found"])
                        row["signal_available"] = bool(res["signal_available"])
                        row["evidence_source"] = "dino_count_consistency"
                    else:
                        row["g_raw"] = row["g_global"]
                        row["box_found"] = False
                        row["signal_available"] = row["g_global"] is not None
                        row["evidence_source"] = "global_clip_fallback"
                    row["box_subject"] = None
                    row["rel_kind"] = None
                elif row["claim_type"] == "rel":
                    kind = relation_signal_kind(row)
                    if kind == "global":
                        row["g_rel"] = None
                        row["g_raw"] = row["g_global"]
                        row["box_subject"] = None
                        row["box_object"] = None
                        row["rel_kind"] = kind
                        row["box_found"] = False
                        row["signal_available"] = row["g_global"] is not None
                        row["evidence_source"] = "global_clip_fallback"
                    else:
                        res = g_rel(
                            dino, image, row.get("subject") or "",
                            row.get("relation") or "", row.get("object") or "",
                            contact_kw, dir_vecs, ck,
                        )
                        row["g_rel"] = res["g_rel"]
                        row["g_raw"] = res["g_rel"]
                        row["box_subject"] = res["box_subject"]
                        row["box_object"] = res["box_object"]
                        row["rel_kind"] = res["rel_kind"]
                        row["box_found"] = bool(res["box_found"])
                        row["signal_available"] = bool(res["signal_available"])
                        row["evidence_source"] = f"dino_box_{res['rel_kind']}"
                else:  # object-level claims (probe only)
                    box = dino.best_box(image, row.get("object") or "", cache_key=ck)
                    row["box_object"] = box
                    row["box_subject"] = None
                    row["rel_kind"] = None
                    row["box_found"] = box is not None
                    row["signal_available"] = box is not None
                    row["g_raw"] = float(
                        clip.cos(
                            clip.encode_image(dino.crop(image, box) if box else image),
                            clip.encode_text(f"a photo of a {row.get('object')}"),
                        )
                    )
                    row["evidence_source"] = "dino_object_clip"
                row["signal_computed"] = True
        image = None
        if backbone is not None and (not has_valid_lp(row) or (
            entropy_required and not row.get("semantic_entropy_computed")
        )):
            image = load_image(row["image_path"])
            if image is None:
                row["l_p"] = None
                continue
        if backbone is not None and not has_valid_lp(row):
            conf = verify_confidence_signals(backbone, image, row.get("sentence") or "")
            row.update(conf)
        elif row.get("l_p") is None:
            # ``--no-lp`` means genuinely missing, never neutral confidence.
            row["lp_computed"] = False
        if entropy_required and not row.get("semantic_entropy_computed"):
            entropy_cfg = cfg.get("semantic_entropy", {})
            answers = sample_answers(
                backbone,
                image,
                VERIFY_PROMPT.format(claim=row.get("sentence") or ""),
                n_samples=int(entropy_cfg.get("n_samples", 10)),
                temperature=float(entropy_cfg.get("temperature", 0.7)),
                top_p=float(entropy_cfg.get("top_p", 0.95)),
            )
            row["semantic_entropy"] = semantic_entropy(answers) if answers else None
            row["semantic_entropy_n"] = len(answers)
            row["semantic_entropy_computed"] = True
        row["signal_computed"] = True

        if row_i % 100 == 0 or row_i == len(rows):
            write_jsonl(partial_path, rows)

    missing_images = [
        row.get("claim_id", "unknown") for row in rows
        if row.get("g_global") is None
    ]
    if missing_images:
        raise RuntimeError(
            f"{path.name}: {len(missing_images)} claims have no readable image; "
            f"first={missing_images[0]}. Full results cannot silently drop images."
        )

    params = normalise_g(rows)
    for row in rows:
        row["signals_schema_version"] = SIGNALS_SCHEMA_VERSION
    write_jsonl(path, rows)
    if partial_path.exists():
        partial_path.unlink()
    write_json(cfg.path("logs") / f"gnorm_{path.stem}.json", params)
    LOG.info("signals written for %s", path.name)


def compute_all(
    cfg: Config,
    overwrite: bool = False,
    with_lp: bool = True,
    with_semantic_entropy: bool = True,
) -> None:
    clip = ClipScorer(
        cfg.resource_id(cfg["tools"]["clip_hf_id"], cfg['tools'].get('clip_local_path')),
        cfg.get("device", "cuda"),
        cfg.get("dtype", "float16"),
        cache_dir=cfg.path("tool_cache"),
    )
    dino = GroundingDino(
        cfg.resource_id(cfg["tools"]["grounding_dino_hf_id"], cfg['tools'].get('grounding_dino_local_path')),
        cfg.get("device", "cuda"),
        cfg.get("dtype", "float16"),
        float(cfg["tools"]["grounding_dino_box_threshold"]),
        float(cfg["tools"]["grounding_dino_text_threshold"]),
        cache_dir=cfg.path("tool_cache"),
    )

    processed = cfg.path("processed")
    full_datasets = list(cfg["datasets"]["detect"]) + list(cfg["datasets"]["quality"])
    datasets = getattr(cfg, "_selected_datasets", None) or full_datasets
    backbone_names = getattr(cfg, "_selected_backbones", None) or cfg.backbone_names
    missing_files = [
        f"claims_{backbone}_{dataset}.jsonl"
        for backbone in backbone_names
        for dataset in datasets
        if not (processed / f"claims_{backbone}_{dataset}.jsonl").exists()
    ]
    if missing_files:
        raise FileNotFoundError(
            "step 02 requires the full backbone/dataset claim matrix; missing: "
            + ", ".join(missing_files)
        )
    for backbone_name in backbone_names:
        files = [
            processed / f"claims_{backbone_name}_{dataset}.jsonl"
            for dataset in datasets
        ]
        if not files:
            continue
        backbone = build_backbone(cfg, backbone_name) if with_lp else None
        for path in files:
            compute_for_file(
                cfg,
                path,
                clip,
                dino,
                backbone,
                overwrite=overwrite,
                with_semantic_entropy=with_semantic_entropy,
            )
        if backbone is not None:
            backbone.unload()

    full_run = (
        set(backbone_names) == set(cfg.backbone_names)
        and set(datasets) == set(full_datasets)
    )
    if full_run:
        log_name = "signal_tool_calls.json"
    elif set(datasets) == set(full_datasets):
        log_name = "signal_tool_calls_" + "_".join(backbone_names) + ".json"
    else:
        log_suffix = "_".join(backbone_names) + "_" + "_".join(datasets)
        log_name = f"signal_tool_calls_{log_suffix}.json"
    write_json(
        cfg.path("logs") / log_name,
        {"clip_image_calls": clip.n_image_calls,
         "clip_text_calls": clip.n_text_calls,
         "dino_forward": dino.n_forward},
    )
    if full_run:
        mark_done(processed, "signals_done")
