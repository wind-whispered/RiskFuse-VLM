"""Namespaced identities plus explicitly scoped VG pixel-fingerprint screening."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from ..utils import read_jsonl, write_json


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _int(value):
    try:
        return str(int(value)) if value is not None and str(value).strip() else None
    except (TypeError, ValueError):
        return None


def vg_identities(
    row: dict[str, Any], coco_to_vg: dict[str, set[str]]
) -> tuple[bool, set[str], str]:
    """A numeric filename has no implied shared namespace across datasets."""
    for field in ("vg_image_id", "visual_genome_id"):
        if (identifier := _int(row.get(field))) is not None:
            return True, {identifier}, field
    coco = _int(row.get("coco_id"))
    if row.get("dataset") == "chair500":
        coco = _int(row.get("image_id"))
    path = str(row.get("image_path", "")).replace("\\", "/")
    if match := re.search(r"COCO_(?:train|val|test)\d*_(\d+)", path, re.I):
        coco = _int(match.group(1))
    if coco is not None:
        return True, set(coco_to_vg.get(coco, set())), "coco_id"
    if re.search(r"/(?:visual_genome|vg_100k(?:_2)?|vg)/", path, re.I):
        identifier = _int(Path(path.split("?", 1)[0]).stem)
        if identifier is not None:
            return True, {identifier}, "explicit_vg_path_namespace"
    return False, set(), "unresolved_identity_namespace"


def collect_images(cfg):
    """Prefer actual frozen claim inventories; otherwise inspect native loaders."""
    from ..data.datasets import load_dataset

    result = {}
    failures = {}
    for dataset in [*cfg["datasets"]["detect"], *cfg["datasets"]["quality"]]:
        images = {}
        paths = sorted(cfg.path("processed").glob(f"claims_*_{dataset}.jsonl"))
        rows = [r for p in paths for r in read_jsonl(p)]
        if not rows or dataset in cfg['datasets']['quality']:
            try:
                samples = load_dataset(cfg, dataset, limit=None)
                rows = [
                    dict(
                        s.meta,
                        dataset=dataset,
                        image_id=s.image_id,
                        image_path=s.image_path,
                    )
                    for s in samples
                ]
            except Exception as exc:
                failures[dataset] = f"{type(exc).__name__}: {exc}"
                # Image identity inventory must not depend on having the pixels.
                if dataset in ("amber_attr", "amber_rel"):
                    root = cfg.abspath(cfg["datasets"]["amber"]["root"])
                    suffix = "attribute" if dataset == "amber_attr" else "relation"
                    native = root / "data/query" / f"query_discriminative-{suffix}.json"
                    if native.exists():
                        questions = json.loads(native.read_text(encoding="utf-8"))
                        rows = [
                            dict(
                                dataset=dataset,
                                image_id=Path(r["image"]).stem,
                                image_path=str(root / "image" / r["image"]),
                            )
                            for r in questions
                        ]
        for row in rows:
            identifier = str(row.get("image_id", ""))
            if not identifier:
                failures[dataset] = "records_missing_image_id"
                continue
            previous = images.get(identifier)
            if previous is not None:
                # Cross-backbone copies must not disagree about explicit IDs.
                for key in ("coco_id", "vg_image_id", "visual_genome_id"):
                    if (
                        previous.get(key) is not None
                        and row.get(key) is not None
                        and str(previous[key]) != str(row[key])
                    ):
                        raise ValueError(
                            f"conflicting identity for {dataset}/{identifier}: {key}"
                        )
            images.setdefault(identifier, dict(row, dataset=dataset))
        result[dataset] = images
    return result, failures


def audit(cfg, output_dir: Path | None = None) -> dict[str, Any]:
    output_dir = output_dir or cfg.path("review") / "reviewer_revision"
    output_dir.mkdir(parents=True, exist_ok=True)
    vg_root = cfg.abspath(cfg["datasets"]["visual_genome"]["root"])
    metadata = vg_root / "image_data.json"
    if not metadata.exists():
        raise FileNotFoundError(f"VG image identity metadata missing: {metadata}")
    data = json.loads(metadata.read_text(encoding="utf-8"))
    coco_to_vg = {}
    vg_ids = set()
    for row in data:
        identifier = _int(row.get("image_id", row.get("id")))
        if identifier is None:
            continue
        vg_ids.add(identifier)
        if (coco := _int(row.get("coco_id"))) is not None:
            coco_to_vg.setdefault(coco, set()).add(identifier)
    external = {}
    mapping_path = cfg.get("reviewer_revision", {}).get("identity_map")
    if mapping_path:
        with cfg.abspath(mapping_path).open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                if not row.get("source") or not (
                    row.get("vg_image_id") or row.get("coco_id")
                ):
                    raise ValueError(
                        "each supplied identity map needs source and an explicit VG/COCO ID"
                    )
                key = (row["dataset"], row["image_id"])
                if key in external:
                    raise ValueError(f"duplicate supplied image mapping: {key}")
                external[key] = row
    images, failures = collect_images(cfg)
    pixel_search = None
    pixel_manifest = None
    if cfg.get('reviewer_revision', {}).get('pixel_identity_audit', False):
        from .pixels import build_search
        pixel_search, pixel_manifest = build_search(cfg)
    pixel_cache = {}
    detail = []
    excluded = set()
    summary = []
    for dataset, entries in images.items():
        resolved_count = overlap_count = missing_pixels = 0
        for image_id, row in entries.items():
            mapped = external.get((dataset, image_id), {})
            resolved, hits, method = vg_identities({**row, **mapped}, coco_to_vg)
            identity_row = {**row, **mapped}
            canonical_keys = {f'vg:{identifier}' for identifier in hits}
            coco_identity = _int(identity_row.get('coco_id'))
            if dataset == 'chair500':
                coco_identity = _int(image_id)
            match = re.search(r'COCO_(?:train|val|test)\d*_(\d+)', str(row.get('image_path','')), re.I)
            if match:
                coco_identity = _int(match.group(1))
            if coco_identity is not None:
                canonical_keys.add(f'coco:{coco_identity}')
            hits &= vg_ids
            resolved_count += int(resolved)
            overlap_count += int(bool(hits))
            excluded.update(hits)
            image_path = str(row.get("image_path", ""))
            pixels_present = bool(
                image_path
                and not image_path.startswith("http")
                and Path(image_path).is_file()
            )
            pixel = {}
            pixel_error = ''
            if pixel_search is not None and pixels_present:
                try:
                    if image_path not in pixel_cache:
                        pixel_cache[image_path] = pixel_search.query(image_path)
                    pixel = pixel_cache[image_path]
                    additional = set(pixel['candidate_vg_ids']) & vg_ids
                    if additional and not hits:
                        overlap_count += 1
                    hits.update(additional)
                    excluded.update(additional)
                    canonical_keys.update(f'vg:{identifier}' for identifier in additional)
                    canonical_keys.add('rgb:' + pixel['rgb_sha256'])
                    if not resolved and pixel_manifest['complete']:
                        resolved = True
                        resolved_count += 1
                        method = 'full_VG_RGB_and_phash_screen'
                except Exception as exc:
                    pixel_error = str(exc)
            missing_pixels += int(not pixels_present)
            detail.append(
                dict(
                    dataset=dataset,
                    image_id=image_id,
                    image_path=image_path,
                    identity_resolved=resolved,
                    matched_vg_ids=";".join(sorted(hits)),
                    identity_method=method,
                    source=mapped.get("source", method),
                    canonical_image_keys=';'.join(sorted(canonical_keys)),
                    pixels_present=pixels_present,
                    rgb_sha256=pixel.get('rgb_sha256', ''),
                    exact_pixel_vg_ids=';'.join(pixel.get('exact_vg_ids', [])),
                    phash_minimum_matched_distance=pixel.get('minimum_matched_phash_distance'),
                    pixel_search_complete=bool(pixel and pixel_manifest['complete']),
                    pixel_error=pixel_error,
                )
            )
        n = len(entries)
        summary.append(
            dict(
                dataset=dataset,
                n_images=n,
                n_identity_resolved=resolved_count,
                n_known_overlap=overlap_count,
                n_unresolved=n - resolved_count,
                overlap_scope="VG_metadata_corpus_not_proof_of_old_index_leakage",
                n_missing_pixels=missing_pixels,
                loader_error=failures.get(dataset, ""),
                status=(
                    "complete_identity_or_full_fingerprint_search"
                    if n and resolved_count == n and dataset not in failures
                    else "incomplete"
                ),
            )
        )
    signature_payload = dict(
        summary=summary,
        images=[{k: v for k, v in r.items() if k != "pixels_present"} for r in detail],
        metadata_sha256=sha256(metadata),
        pixel_manifest=pixel_manifest,
    )
    signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True).encode()
    ).hexdigest()
    report = dict(
        schema_version=2,
        signature=signature,
        datasets=summary,
        excluded_vg_image_ids=sorted(excluded),
        n_vg_images=len(vg_ids),
        identity_audit_complete=all(
            r["status"] == "complete_identity_or_full_fingerprint_search" for r in summary
        ),
        pixel_corpus=pixel_manifest,
        near_duplicate_status=(
            'full_corpus_RGB_SHA256_and_phash_radius4_plus_flip'
            if pixel_manifest and pixel_manifest['complete'] and all(r['pixel_search_complete'] for r in detail)
            else 'partial_pixel_screen' if pixel_manifest else 'not_assessed_no_vg_pixel_audit'
        ),
        validity_note='Identity mapping and fingerprints have stated coverage. A no-match does not certify absence under arbitrary transformations.',
    )
    write_json(output_dir / "vg_overlap_audit.json", report)
    for name, rows in [
        ("vg_overlap_by_dataset.csv", summary),
        ("vg_overlap_images.csv", detail),
    ]:
        if rows:
            with (output_dir / name).open(
                "w", encoding="utf-8-sig", newline=""
            ) as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    return report
