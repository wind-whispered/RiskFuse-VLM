#!/usr/bin/env python
"""Audit cached detection/intervention artifacts without loading any model.

The audit is deliberately separated from Steps 00--12.  It never rewrites
claim, revision, metric, table, or figure artifacts.  Its outputs live under
``results/review`` and are intended to diagnose support coverage, calibration
transfer, and whether purported intervention components are actually active.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hmqr.config import load_config  # noqa: E402
from hmqr.revise import pipeline  # noqa: E402
from hmqr.revise.residual_gate import select_theta  # noqa: E402
from hmqr.signals.support import (  # noqa: E402
    matched_signal_group,
    matched_signal_supported,
    normalise_relation,
)
from hmqr.utils import read_jsonl  # noqa: E402


CORE_DATASETS = ("amber_attr", "amber_rel", "rbench", "mmrel")
RISK_BUDGETS = (0.0, 0.01, 0.025, 0.05, 0.10, 0.20)


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Tie-aware binary AUROC without requiring scikit-learn."""
    pairs = sorted(zip(scores, labels), key=lambda pair: pair[0])
    n_pos = sum(label == 1 for _, label in pairs)
    n_neg = sum(label == 0 for _, label in pairs)
    if not n_pos or not n_neg:
        return float("nan")
    rank_sum = 0.0
    index = 0
    while index < len(pairs):
        end = index + 1
        while end < len(pairs) and pairs[end][0] == pairs[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2.0
        rank_sum += average_rank * sum(label == 1 for _, label in pairs[index:end])
        index = end
    return float((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def _fmt(value: float) -> str:
    return "-" if not math.isfinite(value) else f"{value:.6f}"


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _selected_metrics(records: Iterable[dict[str, Any]], theta: float | None) -> dict[str, Any]:
    records = list(records)
    labels = [int(row.get("original_label", row.get("label", -1))) for row in records]
    decisions = [
        pipeline.decide(
            bool(row.get("flagged")),
            bool(row.get("signal_available")),
            row.get("d_conflict"),
            theta,
        )
        for row in records
    ]
    hall = sum(label == 1 for label in labels)
    correct = sum(label == 0 for label in labels)
    removed = sum(decision == "revise" and label == 1 for decision, label in zip(decisions, labels))
    incorrect = sum(decision == "revise" and label == 0 for decision, label in zip(decisions, labels))
    return {
        "n_test": len(records),
        "n_hallucination": hall,
        "n_correct": correct,
        "n_revise": sum(decision == "revise" for decision in decisions),
        "n_abstain": sum(decision == "abstain" for decision in decisions),
        "n_keep": sum(decision == "keep" for decision in decisions),
        "n_removed": removed,
        "n_incorrect": incorrect,
        "removal_rate": _safe_rate(removed, hall),
        "incorrect_intervention_rate": _safe_rate(incorrect, correct),
        "net_hallucination_reduction": _safe_rate(removed - incorrect, hall),
    }


def _audit_group(
    backbone: str,
    dataset: str,
    group: str,
    raw_rows: Sequence[dict[str, Any]],
    calibration: Sequence[dict[str, Any]],
    test: Sequence[dict[str, Any]],
    q_hat: float | None,
    theta: float | None,
    gate_status: dict[str, Any],
) -> dict[str, Any]:
    cp_rows = [
        row for row in raw_rows
        if row.get("split") == "calib_cp"
        and row.get("label") in (0, 1)
        and pipeline._signal_value(row) is not None
    ]
    detection_labels = [int(row["label"]) for row in cp_rows]
    detection_scores = [1.0 - float(pipeline._signal_value(row)) for row in cp_rows]
    conflict_rows = [
        row for row in test
        if row.get("original_label") in (0, 1)
        and _finite(row.get("d_conflict")) is not None
        and bool(row.get("signal_available"))
    ]
    conflict_labels = [int(row["original_label"]) for row in conflict_rows]
    conflict_scores = [float(row["d_conflict"]) for row in conflict_rows]
    flagged = [row for row in test if bool(row.get("flagged")) and row.get("original_label") in (0, 1)]
    test_hall = sum(int(row.get("original_label", -1)) == 1 for row in test)
    test_correct = sum(int(row.get("original_label", -1)) == 0 for row in test)
    metrics = _selected_metrics(test, theta)
    return {
        "backbone": backbone,
        "dataset": dataset,
        "group": group,
        "n_all": len(raw_rows),
        "n_calib_cp_supported": len(cp_rows),
        "n_calib_revision_sample": len(calibration),
        "n_test_sample": len(test),
        "test_signal_available_rate": _safe_rate(
            sum(bool(row.get("signal_available")) for row in test), len(test)
        ),
        "test_hallucinations": test_hall,
        "test_correct": test_correct,
        "grounding_deficit_auc_calib_cp": _auc(detection_labels, detection_scores),
        "conflict_auc_test": _auc(conflict_labels, conflict_scores),
        "cp_hallucination_recall_test": _safe_rate(
            sum(int(row.get("original_label", -1)) == 1 for row in flagged), test_hall
        ),
        "cp_false_positive_rate_test": _safe_rate(
            sum(int(row.get("original_label", -1)) == 0 for row in flagged), test_correct
        ),
        "q_hat": q_hat,
        "theta": theta,
        "gate_status": gate_status.get("status", "missing"),
        "gate_reason": gate_status.get("reason", ""),
        **metrics,
    }


def _audit_core(cfg: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    group_rows: list[dict[str, Any]] = []
    operating_rows: list[dict[str, Any]] = []
    unsupported_rows: list[dict[str, Any]] = []
    for backbone in cfg.backbone_names:
        for dataset in CORE_DATASETS:
            path = cfg.path("processed") / f"claims_{backbone}_{dataset}.jsonl"
            if not path.exists():
                continue
            rows = [row for row in read_jsonl(path) if row.get("claim_type") in ("attr", "rel")]
            pipeline.mark_revision_evaluation_sample(rows, cfg)
            calibration, test, q_by_group = pipeline._prepare_records(
                rows, "full", float(cfg["cp"]["alpha_default"]), cfg
            )
            theta_by_group, status_by_group = pipeline._select_gate_thresholds(
                calibration, q_by_group, cfg["residual_gate"]
            )
            raw_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
            calibration_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
            test_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in rows:
                raw_by_group[matched_signal_group(row)].append(row)
            for row in calibration:
                calibration_by_group[str(row.get("calibration_group"))].append(row)
            for row in test:
                test_by_group[str(row.get("calibration_group"))].append(row)
            for group in sorted(raw_by_group):
                group_rows.append(_audit_group(
                    backbone, dataset, group, raw_by_group[group],
                    calibration_by_group[group], test_by_group[group],
                    q_by_group.get(group), theta_by_group.get(group),
                    status_by_group.get(group, {}),
                ))
                if group in q_by_group:
                    for budget in RISK_BUDGETS:
                        gate_cfg = dict(cfg["residual_gate"])
                        gate_cfg["target_incorrect_intervention_rate"] = budget
                        try:
                            theta = select_theta(
                                calibration_by_group[group],
                                target_incorrect_intervention_rate=budget,
                                min_calibration_claims=int(gate_cfg.get("min_calibration_claims", 10)),
                                min_correct_claims_for_risk=int(gate_cfg.get("min_correct_claims_for_risk", 5)),
                            )
                            status = "calibrated"
                            reason = ""
                        except ValueError as exc:
                            theta = None
                            status = "no_action"
                            reason = str(exc)
                        operating_rows.append({
                            "backbone": backbone,
                            "dataset": dataset,
                            "group": group,
                            "calibration_risk_budget": budget,
                            "theta": theta,
                            "status": status,
                            "reason": reason,
                            **_selected_metrics(test_by_group[group], theta),
                        })
            unsupported: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
            for row in rows:
                if matched_signal_supported(row):
                    continue
                split = str(row.get("split", ""))
                predicate = (
                    normalise_relation(row.get("relation", ""))
                    if row.get("claim_type") == "rel"
                    else str(row.get("attribute") or row.get("question") or "").strip().lower()
                )
                unsupported[(split, predicate)].append(row)
            for (split, predicate), predicate_rows in sorted(
                unsupported.items(), key=lambda item: len(item[1]), reverse=True
            ):
                labelled = [row for row in predicate_rows if row.get("label") in (0, 1)]
                raw = [
                    row for row in labelled if _finite(row.get("g_raw")) is not None
                ]
                unsupported_rows.append({
                    "backbone": backbone,
                    "dataset": dataset,
                    "split": split,
                    "predicate": predicate,
                    "count": len(predicate_rows),
                    "n_labelled": len(labelled),
                    "n_raw_signal": len(raw),
                    "n_box_found": sum(bool(row.get("box_found")) for row in predicate_rows),
                    "raw_grounding_deficit_auc": _auc(
                        [int(row["label"]) for row in raw],
                        [1.0 - float(row["g_raw"]) for row in raw],
                    ),
                })
    return group_rows, operating_rows, unsupported_rows


def _audit_pathways(cfg: Any) -> dict[str, Any]:
    processed = cfg.path("processed")
    payload: dict[str, Any] = {
        "quality_transfer_policy": cfg.get("quality_transfer", {}).get("policy"),
        "discriminative_action": cfg.get("intervention", {}).get("discriminative_action"),
        "core_full_revision": {},
        "chair500": {},
    }
    for backbone in cfg.backbone_names:
        for dataset in CORE_DATASETS:
            path = processed / f"revised_labeled_{backbone}_{dataset}.jsonl"
            if not path.exists():
                continue
            rows = read_jsonl(path)
            revised = [row for row in rows if row.get("decision") == "revise"]
            payload["core_full_revision"][f"{backbone}|{dataset}"] = {
                "n_rows": len(rows),
                "n_revision_decisions": len(revised),
                "n_revision_attempted": sum(bool(row.get("revision_attempted")) for row in revised),
                "n_with_retrieval_ids": sum(bool(row.get("retrieval_ids")) for row in revised),
                "n_deterministic_retractions": sum(
                    row.get("revision_parse_mode") == "deterministic_target_retraction"
                    for row in revised
                ),
            }
        chair = processed / f"revised_labeled_{backbone}_chair500.jsonl"
        if chair.exists():
            rows = read_jsonl(chair)
            payload["chair500"][backbone] = {
                "n_rows": len(rows),
                "n_keep": sum(row.get("decision") == "keep" for row in rows),
                "n_abstain": sum(row.get("decision") == "abstain" for row in rows),
                "n_revise": sum(row.get("decision") == "revise" for row in rows),
                "n_changed_caption": sum(
                    str(row.get("final_caption", row.get("caption", ""))).strip()
                    != str(row.get("caption", "")).strip()
                    for row in rows
                ),
                "n_with_retrieval_ids": sum(bool(row.get("retrieval_ids")) for row in rows),
            }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else cfg.path("review")
    group_rows, operating_rows, unsupported_rows = _audit_core(cfg)
    pathways = _audit_pathways(cfg)
    _write_csv(output_dir / "intervention_audit_groups.csv", group_rows)
    _write_csv(output_dir / "intervention_operating_points.csv", operating_rows)
    _write_csv(output_dir / "unsupported_predicates.csv", unsupported_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "intervention_pathways.json").write_text(
        json.dumps(pathways, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(group_rows)} group rows")
    print(f"wrote {len(operating_rows)} operating-point rows")
    print(f"wrote {len(unsupported_rows)} unsupported-predicate rows")
    print(json.dumps(pathways, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
