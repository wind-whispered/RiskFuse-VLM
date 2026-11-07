"""Budgeted one-shot internal-confidence pass for residual/gate experiments."""
from __future__ import annotations

import zlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

import numpy as np
from tqdm import tqdm

from ..backbones.registry import build_backbone, verification_adapter_version
from ..data.images import load_image
from ..revise.pipeline import mark_revision_evaluation_sample
from ..sampling import one_index_per_image, stable_claim_key
from ..utils import claim_to_sentence, mark_done, read_jsonl, write_json, write_jsonl
from .internal_conf import LP_CONFIDENCE_DEFINITION, verify_confidence_signals
from .lp import has_current_lp, has_valid_lp, verification_adapter_matches
from .support import matched_signal_group


INTERNAL_CONFIDENCE_SAMPLING_SCHEMA_VERSION = 3


def _stable_claim_key(row: Dict[str, Any]) -> str:
    return stable_claim_key(row)


def _group_key(row: Dict[str, Any]) -> str:
    return matched_signal_group(row)


def _confidence_claim(row: Dict[str, Any], cfg) -> str:
    quality_datasets = set(cfg.get("datasets", {}).get("quality", []))
    return (
        claim_to_sentence(row)
        if row.get("dataset") in quality_datasets
        else str(row.get("sentence") or "")
    )


def _confidence_key(row: Dict[str, Any], cfg) -> Tuple[str, str]:
    text = " ".join(_confidence_claim(row, cfg).lower().split())
    return str(row.get("image_id")), text


def _quality_transfer_eligible(row: Dict[str, Any], cfg) -> bool:
    if str(cfg.get("quality_transfer", {}).get("policy")) == "audit_only_keep":
        return False
    supported_rel_kinds = set(
        cfg.get("quality_transfer", {}).get("supported_relation_kinds", ["contact"])
    )
    return bool(
        row.get("claim_type") == "attr"
        or (
            row.get("claim_type") == "rel"
            and str(row.get("rel_kind") or "all") in supported_rel_kinds
        )
    )


def mark_internal_confidence_selection(rows: List[Dict[str, Any]], cfg) -> Set[int]:
    """Mark detector samples plus quality-transfer rows required by the gate."""
    cap = int(cfg.get("internal_confidence", {}).get("max_images_per_group_split", 250))
    seed = int(cfg.get("seed", 42))
    by_stratum: Dict[Tuple[str, str], Dict[str, List[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for index, row in enumerate(rows):
        eligible = (
            row.get("mode") == "discriminative"
            and row.get("claim_type") in ("attr", "rel")
            and row.get("split") in ("calib_cp", "test")
            and row.get("label") in (0, 1)
            and bool(row.get("signal_available", row.get("g") is not None))
        )
        if eligible:
            by_stratum[(_group_key(row), str(row.get("split")))][
                str(row.get("image_id"))
            ].append(index)

    selected: Set[int] = set()
    for stratum in sorted(by_stratum):
        candidates = one_index_per_image(
            rows,
            by_stratum[stratum],
            seed,
            "internal_confidence|" + "|".join(stratum),
        )
        stratum_seed = seed + zlib.crc32(
            ("internal_confidence|" + "|".join(stratum)).encode("utf-8")
        )
        rng = np.random.default_rng(stratum_seed)
        order = rng.permutation(len(candidates))[:cap]
        selected.update(candidates[int(position)] for position in order)

    # The correction sample is fixed independently of scores, labels, flags,
    # and decisions. Its calib_revision/test rows also require l_p for the gate.
    mark_revision_evaluation_sample(rows, cfg)
    quality_datasets = set(cfg.get("datasets", {}).get("quality", []))
    selected.update(
        index for index, row in enumerate(rows)
        if row.get("revision_evaluation_selected")
        and row.get("claim_type") in ("attr", "rel")
        and (
            row.get("dataset") not in quality_datasets
            or _quality_transfer_eligible(row, cfg)
        )
        and bool(row.get("signal_available", row.get("g") is not None))
    )
    selected.update(
        index for index, row in enumerate(rows)
        if row.get("dataset") in quality_datasets
        and _quality_transfer_eligible(row, cfg)
        and bool(row.get("signal_available", row.get("g") is not None))
    )
    for index, row in enumerate(rows):
        row["internal_confidence_selected"] = index in selected
        row["internal_confidence_selection_cap"] = cap
        row["internal_confidence_sampling_unit"] = "image_id"
        row["internal_confidence_sampling_schema_version"] = (
            INTERNAL_CONFIDENCE_SAMPLING_SCHEMA_VERSION
        )
    return selected


def needs_internal_confidence(
    row: Dict[str, Any],
    overwrite: bool = False,
    expected_adapter_version: int = 1,
    retry_failed: bool = False,
) -> bool:
    return bool(
        row.get("internal_confidence_selected")
        and (
            overwrite
            or not row.get("lp_attempted")
            or not verification_adapter_matches(row, expected_adapter_version)
            or (retry_failed and not has_current_lp(row, expected_adapter_version))
        )
    )


def selected_paths(cfg, backbone_names: Iterable[str], datasets: Iterable[str]) -> List[Path]:
    return [
        cfg.path("processed") / f"claims_{backbone}_{dataset}.jsonl"
        for backbone in backbone_names
        for dataset in datasets
    ]


def _rows_with_partial(path: Path, overwrite: bool = False) -> List[Dict[str, Any]]:
    """Use an aligned l_p checkpoint for exact resume budgeting."""
    rows = read_jsonl(path)
    partial_path = path.with_name(path.name + ".lp.partial")
    if partial_path.exists() and not overwrite:
        partial = read_jsonl(partial_path)
        if (
            len(partial) == len(rows)
            and [row.get("claim_id") for row in partial]
            == [row.get("claim_id") for row in rows]
        ):
            rows = partial
    return rows


def estimate_calls(
    cfg,
    backbone_names: Iterable[str],
    datasets: Iterable[str],
    overwrite: bool = False,
    max_claims_per_file: int | None = None,
    retry_failed: bool = False,
) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for path in selected_paths(cfg, backbone_names, datasets):
        if not path.exists():
            raise FileNotFoundError(path)
        rows = _rows_with_partial(path, overwrite=overwrite)
        mark_internal_confidence_selection(rows, cfg)
        backbone_name = str(rows[0].get("backbone")) if rows else path.name.split("_")[1]
        expected_adapter = verification_adapter_version(cfg, backbone_name)
        pending_indices = [
            index for index, row in enumerate(rows)
            if needs_internal_confidence(
                row, overwrite, expected_adapter, retry_failed=retry_failed
            )
        ]
        if max_claims_per_file is not None:
            pending_indices = pending_indices[: int(max_claims_per_file)]
        reusable = set() if overwrite else {
            _confidence_key(row, cfg) for row in rows
            if has_current_lp(row, expected_adapter)
        }
        pending_keys = {
            _confidence_key(rows[index], cfg) for index in pending_indices
        }
        counts[path.name] = int(len(pending_keys - reusable))
    return counts


def audit_existing(
    cfg,
    backbone_names: Iterable[str],
    datasets: Iterable[str],
    stamp_schema: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Validate the selected one-shot confidence matrix without loading a model."""
    report: Dict[str, Dict[str, Any]] = {}
    for path in selected_paths(cfg, backbone_names, datasets):
        if not path.exists():
            raise FileNotFoundError(path)
        rows = read_jsonl(path)
        selected = mark_internal_confidence_selection(rows, cfg)
        backbone_name = str(rows[0].get("backbone")) if rows else path.name.split("_")[1]
        expected_adapter = verification_adapter_version(cfg, backbone_name)
        attempted = [
            index for index in selected
            if rows[index].get("lp_attempted")
            and verification_adapter_matches(rows[index], expected_adapter)
        ]
        valid = [
            index for index in selected
            if has_current_lp(rows[index], expected_adapter)
        ]
        # A dataset excluded by policy (for example CHAIR500 under
        # ``audit_only_keep``) legitimately has an empty selection matrix.
        # Treat that vacuous audit as complete while still failing when a
        # non-empty selection has no attempted calls.
        parse_rate = (
            len(valid) / len(attempted)
            if attempted
            else (1.0 if not selected else 0.0)
        )
        if len(attempted) != len(selected) or parse_rate < 0.95:
            raise RuntimeError(
                f"{path.name}: selected={len(selected)}, attempted={len(attempted)}, "
                f"valid={len(valid)}, parse_rate={parse_rate:.3f}"
            )
        if stamp_schema:
            for index in attempted:
                rows[index]["verification_adapter_version"] = expected_adapter
                rows[index]["lp_confidence_definition"] = LP_CONFIDENCE_DEFINITION
            write_jsonl(path, rows)
        values = [float(rows[index]["l_p"]) for index in valid]
        report[path.name] = {
            "selected": len(selected),
            "attempted": len(attempted),
            "valid": len(valid),
            "parse_rate": parse_rate,
            "lp_min": min(values) if values else None,
            "lp_max": max(values) if values else None,
            "sampling_schema_stamped": bool(stamp_schema),
            "lp_confidence_definition": LP_CONFIDENCE_DEFINITION,
        }
    return report


def run(
    cfg,
    backbone_names: List[str],
    datasets: List[str],
    max_generation_calls: int,
    overwrite: bool = False,
    max_claims_per_file: int | None = None,
    retry_failed: bool = False,
) -> None:
    estimate = estimate_calls(
        cfg, backbone_names, datasets, overwrite, max_claims_per_file,
        retry_failed=retry_failed,
    )
    total_calls = sum(estimate.values())
    if total_calls > int(max_generation_calls):
        raise RuntimeError(
            f"internal-confidence budget exceeded: pending={total_calls} > "
            f"max_generation_calls={max_generation_calls}"
        )

    for backbone_name in backbone_names:
        backbone = build_backbone(cfg, backbone_name)
        expected_adapter = verification_adapter_version(cfg, backbone_name)
        attempts = parsed = unparsed = reused_rows = 0
        for dataset in datasets:
            path = cfg.path("processed") / f"claims_{backbone_name}_{dataset}.jsonl"
            rows = _rows_with_partial(path, overwrite=overwrite)
            mark_internal_confidence_selection(rows, cfg)
            partial_path = path.with_name(path.name + ".lp.partial")

            # Remove the legacy --no-lp neutral placeholder. Downstream code
            # recognises only an explicitly parsed lp_computed value.
            for row in rows:
                if not has_valid_lp(row):
                    row["l_p"] = None
                    row["lp_computed"] = False

            pending = [
                i for i, row in enumerate(rows)
                if needs_internal_confidence(
                    row, overwrite, expected_adapter, retry_failed=retry_failed
                )
            ]
            if max_claims_per_file is not None:
                pending = pending[: int(max_claims_per_file)]
            confidence_cache: Dict[Tuple[str, str], Tuple[Dict[str, Any], str]] = {}
            if not overwrite:
                for cached_row in rows:
                    if has_current_lp(cached_row, expected_adapter):
                        confidence_cache[_confidence_key(cached_row, cfg)] = (
                            {
                                key: cached_row.get(key)
                                for key in (
                                    "l_p", "lp_computed", "lp_attempted",
                                    "lp_parse_status", "lp_answer",
                                    "lp_predictive_entropy",
                                    "verification_adapter_version",
                                    "lp_confidence_definition",
                                )
                            },
                            str(cached_row.get("claim_id", "")),
                        )
            for done_i, row_i in enumerate(
                tqdm(pending, desc=f"internal {backbone_name}/{dataset}", ncols=88),
                start=1,
            ):
                row = rows[row_i]
                confidence_key = _confidence_key(row, cfg)
                cached = confidence_cache.get(confidence_key)
                if cached is not None:
                    signals = dict(cached[0])
                    signals.update(
                        lp_attempted=True,
                        lp_reused=True,
                        lp_reused_from_claim_id=cached[1],
                    )
                    reused_rows += 1
                else:
                    image = load_image(row["image_path"])
                    if image is None:
                        signals = {
                            "l_p": None,
                            "lp_computed": False,
                            "lp_attempted": True,
                            "lp_parse_status": "unreadable_image",
                            "lp_answer": "",
                            "lp_predictive_entropy": None,
                        }
                    else:
                        signals = verify_confidence_signals(
                            backbone, image, _confidence_claim(row, cfg)
                        )
                    signals.update(lp_reused=False, lp_reused_from_claim_id=None)
                    confidence_cache[confidence_key] = (
                        dict(signals), str(row.get("claim_id", ""))
                    )
                    attempts += 1
                    if bool(signals.get("lp_computed")):
                        parsed += 1
                    else:
                        unparsed += 1
                row.update(signals)
                if done_i % 25 == 0 or done_i == len(pending):
                    write_jsonl(partial_path, rows)
            write_jsonl(path, rows)
            if partial_path.exists():
                partial_path.unlink()

        required_datasets = set(cfg["datasets"]["detect"]) | set(
            cfg.get("datasets", {}).get("quality", [])
        )
        dataset_set = set(datasets)
        dataset_suffix = (
            "" if dataset_set == required_datasets
            else "_" + "_".join(str(dataset) for dataset in datasets)
        )
        write_json(
            cfg.path("logs")
            / f"internal_confidence_calls_{backbone_name}{dataset_suffix}.json",
            {
                "datasets": list(datasets),
                "generation_attempts": attempts,
                "parsed_yes_no": parsed,
                "unparsed_or_failed": unparsed,
                "reused_duplicate_rows": reused_rows,
                "image_level_sample": True,
                "max_images_per_group_split": int(
                    cfg.get("internal_confidence", {}).get(
                        "max_images_per_group_split", 250
                    )
                ),
                "pilot_max_claims_per_file": max_claims_per_file,
                "verification_adapter_version": expected_adapter,
                "lp_confidence_definition": LP_CONFIDENCE_DEFINITION,
            },
        )
        backbone.unload()

    required_datasets = set(cfg["datasets"]["detect"]) | set(
        cfg.get("datasets", {}).get("quality", [])
    )
    full = set(backbone_names) == set(cfg.backbone_names) and set(datasets) == required_datasets
    if full:
        mark_done(cfg.path("processed"), "internal_confidence_done")
