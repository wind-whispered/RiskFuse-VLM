"""Budgeted matched-subset uncertainty baselines, isolated from core grounding."""
from __future__ import annotations

import zlib
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

import numpy as np

from tqdm import tqdm

from ..backbones.registry import build_backbone, verification_adapter_version
from ..data.images import load_image
from ..utils import mark_done, read_jsonl, write_json, write_jsonl
from .internal_conf import (
    VERIFY_PROMPT,
    sample_answer_signals,
    semantic_answer_class,
    semantic_entropy,
)


SEMANTIC_ENTROPY_SCHEMA_VERSION = 2


def repair_answer_metadata(row: Dict[str, Any]) -> bool:
    """Deterministically reconstruct audit metadata from saved raw answers."""
    answers = row.get("semantic_answers")
    if not isinstance(answers, list):
        return False
    classes = [semantic_answer_class(answer) for answer in answers]
    unparsed = int(sum(value not in ("yes", "no") for value in classes))
    row["semantic_answer_classes"] = classes
    row["semantic_answer_empty_count"] = int(
        sum(not str(answer).strip() for answer in answers)
    )
    row["semantic_answer_unparsed_count"] = unparsed
    row["semantic_answer_binary_parse_rate"] = (
        float(1.0 - unparsed / len(answers)) if answers else 0.0
    )
    return True


def _stable_claim_key(row: Dict[str, Any]) -> str:
    claim_id = str(row.get("claim_id", ""))
    return claim_id.split("|", 1)[1] if "|" in claim_id else claim_id


def _selection_stratum(row: Dict[str, Any]) -> Tuple[str, str, int, str]:
    subtype = str(row.get("rel_kind") or "all") if row.get("claim_type") == "rel" else "attr"
    return (
        str(row.get("claim_type")), subtype, int(row.get("label")), str(row.get("split"))
    )


def mark_matched_selection(rows: List[Dict[str, Any]], cfg) -> Set[int]:
    """Mark one claim/image in each label/split stratum with a deterministic cap."""
    cap = int(cfg.get("rq1_uncertainty", {}).get("max_images_per_label_split", 50))
    seed = int(cfg.get("seed", 42))
    by_stratum: Dict[Tuple[str, str, int, str], Dict[str, int]] = defaultdict(dict)
    for index, row in enumerate(rows):
        eligible = (
            row.get("mode") == "discriminative"
            and row.get("claim_type") in ("attr", "rel")
            and row.get("split") in ("calib_cp", "test")
            and row.get("label") in (0, 1)
        )
        if eligible:
            by_stratum[_selection_stratum(row)].setdefault(str(row.get("image_id")), index)

    selected: Set[int] = set()
    for stratum in sorted(by_stratum):
        candidates = sorted(
            by_stratum[stratum].values(), key=lambda i: _stable_claim_key(rows[i])
        )
        stratum_seed = seed + zlib.crc32("|".join(map(str, stratum)).encode("utf-8"))
        rng = np.random.default_rng(stratum_seed)
        order = rng.permutation(len(candidates))[:cap]
        selected.update(candidates[int(position)] for position in order)
    for index, row in enumerate(rows):
        row["rq1_uncertainty_selected"] = index in selected
        row["rq1_uncertainty_selection_cap"] = cap
    return selected


def needs_semantic_entropy(
    row: Dict[str, Any],
    overwrite: bool = False,
    expected_adapter_version: int = 1,
) -> bool:
    eligible = (
        row.get("mode") == "discriminative"
        and row.get("claim_type") in ("attr", "rel")
        and row.get("split") in ("calib_cp", "test")
        and row.get("label") in (0, 1)
    )
    complete = bool(
        row.get("semantic_entropy_computed")
        and row.get("reefknot_entropy_computed")
        and row.get("semantic_entropy_schema_version")
        == SEMANTIC_ENTROPY_SCHEMA_VERSION
        and (
            row.get("verification_adapter_version") == expected_adapter_version
            or (
                expected_adapter_version == 1
                and row.get("verification_adapter_version") is None
            )
        )
        and (
            expected_adapter_version == 1
            or (
                isinstance(row.get("semantic_answers"), list)
                and len(row["semantic_answers"])
                == int(row.get("semantic_entropy_n", 0))
                and (
                    isinstance(row.get("semantic_answer_classes"), list)
                    or repair_answer_metadata(row)
                )
                and int(row.get("semantic_answer_empty_count", -1)) == 0
                and int(row.get("semantic_answer_unparsed_count", -1)) >= 0
            )
        )
    )
    return bool(
        eligible
        and row.get("rq1_uncertainty_selected")
        and (overwrite or not complete)
    )


def selected_paths(cfg, backbone_names: Iterable[str], datasets: Iterable[str]) -> List[Path]:
    return [
        cfg.path("processed") / f"claims_{backbone}_{dataset}.jsonl"
        for backbone in backbone_names
        for dataset in datasets
    ]


def _rows_with_partial(path: Path, overwrite: bool = False) -> List[Dict[str, Any]]:
    """Use an aligned checkpoint for both budgeting and execution."""
    rows = read_jsonl(path)
    partial_path = path.with_name(path.name + ".entropy.partial")
    if partial_path.exists() and not overwrite:
        partial = read_jsonl(partial_path)
        if (
            len(partial) == len(rows)
            and [row.get("claim_id") for row in partial]
            == [row.get("claim_id") for row in rows]
        ):
            rows = partial
    return rows


def audit_existing(
    cfg,
    backbone_names: Iterable[str],
    datasets: Iterable[str],
    stamp_schema: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Validate a completed matched matrix and optionally stamp decoding schema 2."""
    report: Dict[str, Dict[str, Any]] = {}
    expected_n = int(cfg.get("semantic_entropy", {}).get("n_samples", 10))
    for path in selected_paths(cfg, backbone_names, datasets):
        rows = read_jsonl(path)
        selected = mark_matched_selection(rows, cfg)
        backbone_name = str(rows[0].get("backbone")) if rows else path.name.split("_")[1]
        expected_adapter = verification_adapter_version(cfg, backbone_name)
        invalid: List[str] = []
        semantic_values: List[float] = []
        reefknot_values: List[float] = []
        answer_count = 0
        unparsed_answer_count = 0
        for index in sorted(selected):
            row = rows[index]
            if expected_adapter != 1 and stamp_schema:
                repair_answer_metadata(row)
            if stamp_schema and row.get("predictive_entropy") is None:
                row["predictive_entropy"] = row.get("reefknot_entropy")
                row["predictive_entropy_computed"] = bool(
                    row.get("reefknot_entropy_computed")
                )
            try:
                semantic_value = float(row.get("semantic_entropy"))
                reefknot_value = float(
                    row.get("predictive_entropy", row.get("reefknot_entropy"))
                )
            except (TypeError, ValueError):
                invalid.append(str(row.get("claim_id")))
                continue
            valid = bool(
                row.get("semantic_entropy_computed")
                and row.get("reefknot_entropy_computed")
                and int(row.get("semantic_entropy_n", 0)) == expected_n
                and math.isfinite(semantic_value)
                and 0.0 <= semantic_value <= 1.0
                and math.isfinite(reefknot_value)
                and (
                    row.get("verification_adapter_version") == expected_adapter
                    or (
                        expected_adapter == 1
                        and row.get("verification_adapter_version") is None
                    )
                )
                and (
                    expected_adapter == 1
                    or (
                        isinstance(row.get("semantic_answers"), list)
                        and len(row["semantic_answers"]) == expected_n
                        and isinstance(row.get("semantic_answer_classes"), list)
                        and len(row["semantic_answer_classes"]) == expected_n
                        and int(row.get("semantic_answer_empty_count", -1)) == 0
                        and int(row.get("semantic_answer_unparsed_count", -1)) >= 0
                    )
                )
            )
            if not valid:
                invalid.append(str(row.get("claim_id")))
                continue
            semantic_values.append(semantic_value)
            reefknot_values.append(reefknot_value)
            if isinstance(row.get("semantic_answers"), list):
                answer_count += len(row["semantic_answers"])
                unparsed_answer_count += int(
                    row.get("semantic_answer_unparsed_count", 0)
                )
            if stamp_schema:
                row["semantic_entropy_schema_version"] = (
                    SEMANTIC_ENTROPY_SCHEMA_VERSION
                )
                row["verification_adapter_version"] = expected_adapter
        if invalid or len(semantic_values) != len(selected):
            raise RuntimeError(
                f"{path.name}: invalid matched uncertainty rows="
                f"{len(invalid)}/{len(selected)}; first={invalid[:1]}"
            )
        if stamp_schema:
            write_jsonl(path, rows)
        report[path.name] = {
            "selected": len(selected),
            "semantic_unique": len(set(semantic_values)),
            "semantic_min": min(semantic_values) if semantic_values else None,
            "semantic_max": max(semantic_values) if semantic_values else None,
            "predictive_entropy_min": min(reefknot_values) if reefknot_values else None,
            "predictive_entropy_max": max(reefknot_values) if reefknot_values else None,
            "schema_stamped": bool(stamp_schema),
            "semantic_binary_parse_rate": (
                1.0 - unparsed_answer_count / answer_count
                if answer_count else None
            ),
            "semantic_answer_count": answer_count,
        }
    return report


def estimate_calls(
    cfg,
    backbone_names: Iterable[str],
    datasets: Iterable[str],
    n_samples: int,
    overwrite: bool = False,
    max_claims_per_file: int | None = None,
) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for path in selected_paths(cfg, backbone_names, datasets):
        if not path.exists():
            raise FileNotFoundError(path)
        rows = _rows_with_partial(path, overwrite=overwrite)
        mark_matched_selection(rows, cfg)
        backbone_name = str(rows[0].get("backbone")) if rows else path.name.split("_")[1]
        expected_adapter = verification_adapter_version(cfg, backbone_name)
        pending = sum(
            needs_semantic_entropy(row, overwrite, expected_adapter) for row in rows
        )
        if max_claims_per_file is not None:
            pending = min(pending, int(max_claims_per_file))
        counts[path.name] = int(pending * n_samples)
    return counts


def run(
    cfg,
    backbone_names: List[str],
    datasets: List[str],
    n_samples: int,
    max_generation_calls: int,
    overwrite: bool = False,
    max_claims_per_file: int | None = None,
) -> None:
    estimate = estimate_calls(
        cfg, backbone_names, datasets, n_samples, overwrite, max_claims_per_file
    )
    total_calls = sum(estimate.values())
    if total_calls > int(max_generation_calls):
        raise RuntimeError(
            f"semantic-entropy budget exceeded: pending={total_calls} > "
            f"max_generation_calls={max_generation_calls}"
        )

    entropy_cfg = cfg.get("semantic_entropy", {})
    for backbone_name in backbone_names:
        backbone = build_backbone(cfg, backbone_name)
        expected_adapter = verification_adapter_version(cfg, backbone_name)
        completed_calls = 0
        model_invocations = 0
        for dataset in datasets:
            path = cfg.path("processed") / f"claims_{backbone_name}_{dataset}.jsonl"
            rows = _rows_with_partial(path, overwrite=overwrite)
            mark_matched_selection(rows, cfg)
            partial_path = path.with_name(path.name + ".entropy.partial")

            pending = [
                i for i, row in enumerate(rows)
                if needs_semantic_entropy(row, overwrite, expected_adapter)
            ]
            if max_claims_per_file is not None:
                pending = pending[: int(max_claims_per_file)]
            for done_i, row_i in enumerate(
                tqdm(pending, desc=f"semantic {backbone_name}/{dataset}", ncols=88), start=1
            ):
                row = rows[row_i]
                image = load_image(row["image_path"])
                if image is None:
                    raise RuntimeError(f"unreadable image for {row.get('claim_id')}")
                signals = sample_answer_signals(
                    backbone,
                    image,
                    VERIFY_PROMPT.format(claim=row.get("sentence") or ""),
                    n_samples=n_samples,
                    max_return_sequences_per_generate=int(
                        entropy_cfg.get("max_return_sequences_per_generate", {}).get(
                            backbone_name, n_samples
                        )
                    ),
                    temperature=float(entropy_cfg.get("temperature", 0.7)),
                    top_p=float(entropy_cfg.get("top_p", 0.95)),
                )
                answers = signals["answers"]
                if len(answers) != n_samples:
                    write_jsonl(partial_path, rows)
                    raise RuntimeError(
                        f"{row.get('claim_id')}: semantic entropy returned "
                        f"{len(answers)}/{n_samples} samples"
                    )
                if int(signals["empty_answer_count"]) > 0:
                    write_jsonl(partial_path, rows)
                    raise RuntimeError(
                        f"{row.get('claim_id')}: semantic verifier returned "
                        f"empty={signals['empty_answer_count']}, "
                        f"answers={answers!r}"
                    )
                row["semantic_entropy"] = semantic_entropy(answers)
                row["reefknot_entropy"] = signals["reefknot_entropy"]
                row["predictive_entropy"] = signals["predictive_entropy"]
                row["semantic_entropy_n"] = n_samples
                row["semantic_entropy_computed"] = True
                row["reefknot_entropy_computed"] = True
                row["predictive_entropy_computed"] = True
                row["semantic_entropy_schema_version"] = (
                    SEMANTIC_ENTROPY_SCHEMA_VERSION
                )
                row["verification_adapter_version"] = int(
                    signals["verification_adapter_version"]
                )
                row["semantic_answers"] = answers
                row["semantic_parsed_answers"] = signals["parsed_answers"]
                row["semantic_answer_classes"] = signals[
                    "semantic_answer_classes"
                ]
                row["semantic_answer_empty_count"] = int(
                    signals["empty_answer_count"]
                )
                row["semantic_answer_unparsed_count"] = int(
                    signals["unparsed_answer_count"]
                )
                row["semantic_answer_binary_parse_rate"] = float(
                    1.0 - signals["unparsed_answer_count"] / max(len(answers), 1)
                )
                completed_calls += n_samples
                model_invocations += int(signals["model_generate_invocations"])
                if done_i % 25 == 0 or done_i == len(pending):
                    write_jsonl(partial_path, rows)
            write_jsonl(path, rows)
            if partial_path.exists():
                partial_path.unlink()

        write_json(
            cfg.path("logs") / f"semantic_entropy_calls_{backbone_name}.json",
            {
                "sampled_sequences": completed_calls,
                "model_generate_invocations": model_invocations,
                "n_samples_per_claim": n_samples,
                "matched_subset": True,
                "max_images_per_label_split": int(
                    cfg.get("rq1_uncertainty", {}).get("max_images_per_label_split", 50)
                ),
                "pilot_max_claims_per_file": max_claims_per_file,
                "semantic_entropy_schema_version": SEMANTIC_ENTROPY_SCHEMA_VERSION,
                "verification_adapter_version": expected_adapter,
            },
        )
        backbone.unload()

    full = set(backbone_names) == set(cfg.backbone_names) and set(datasets) == set(
        cfg["datasets"]["detect"]
    )
    if full:
        mark_done(cfg.path("processed"), "semantic_entropy_done")
