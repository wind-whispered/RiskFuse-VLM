"""Native-label intervention metrics for RQ2/RQ3/RQ4."""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator > 0 else float("nan")


def hallucination_rate(labels: Sequence[int]) -> float:
    kept = [int(value) for value in labels if value in (0, 1)]
    return _rate(sum(value == 1 for value in kept), len(kept))


def validate_revision_outcomes(rows: Sequence[Dict[str, Any]]) -> None:
    """Require an audited target-only action outcome for every revision."""
    invalid: List[str] = []
    for row in rows:
        if row.get("decision") != "revise":
            continue
        required = (
            row.get("intervention_evaluable") is True
            and row.get("target_only_intervention", row.get("target_only_edit")) is True
            and "target_hallucination_removed" in row
            and "incorrect_intervention" in row
        )
        if not required:
            invalid.append(str(row.get("claim_id", "unknown")))
    if invalid:
        raise ValueError(
            f"{len(invalid)} revised rows lack native-label intervention audit; "
            f"first={invalid[0]}"
        )


def summarise(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    validate_revision_outcomes(rows)
    rows = [
        row for row in rows
        if row.get("original_label", row.get("label")) in (0, 1)
    ]
    n = len(rows)
    original = [int(row.get("original_label", row.get("label"))) for row in rows]
    decisions = [str(row.get("decision", "keep")) for row in rows]
    revised = [row for row in rows if row.get("decision") == "revise"]
    requested = [row for row in rows if bool(row.get("revision_requested"))]
    attempted = [row for row in rows if bool(row.get("revision_attempted"))]
    n_original_hallu = sum(value == 1 for value in original)
    n_original_correct = sum(value == 0 for value in original)
    removed = sum(bool(row.get("target_hallucination_removed")) for row in revised)
    incorrect = sum(bool(row.get("incorrect_intervention")) for row in revised)
    remaining_target_errors = n_original_hallu - removed + incorrect
    covered = [row for row in rows if str(row.get("decision", "keep")) != "abstain"]
    covered_errors = sum(
        (
            int(row.get("original_label", row.get("label"))) == 1
            and not bool(row.get("target_hallucination_removed"))
        )
        or bool(row.get("incorrect_intervention"))
        for row in covered
    )
    failed_attempts = sum(
        not bool(row.get("parse_ok"))
        or not bool(row.get("schema_ok"))
        or bool(row.get("revision_error"))
        for row in attempted
    )

    return {
        "base_hallu_rate": hallucination_rate(original),
        "remaining_target_error_rate": _rate(remaining_target_errors, n),
        "target_hallucination_removal_rate": _rate(removed, n_original_hallu),
        "incorrect_intervention_rate": _rate(incorrect, n_original_correct),
        "net_hallucination_reduction": _rate(removed - incorrect, n_original_hallu),
        "selection_risk": _rate(covered_errors, len(covered)),
        "output_coverage": _rate(sum(d != "abstain" for d in decisions), n),
        "abstention_rate": _rate(sum(d == "abstain" for d in decisions), n),
        "revision_request_rate": _rate(len(requested), n),
        "revision_attempt_rate": _rate(len(attempted), n),
        "revision_rate": _rate(sum(d == "revise" for d in decisions), n),
        "valid_revision_yield": _rate(len(revised), len(attempted)),
        "failed_revision_attempt_rate": _rate(failed_attempts, len(attempted)),
        "non_target_preservation_rate": _rate(
            sum(bool(row.get("non_target_preserved")) for row in revised), len(revised)
        ),
        "external_label_coverage": _rate(n, n),
        "n_claims": n,
        "n_original_hallucinations": n_original_hallu,
        "n_original_correct": n_original_correct,
        "n_target_hallucinations_removed": removed,
        "n_incorrect_interventions": incorrect,
        "n_keep": sum(d == "keep" for d in decisions),
        "n_abstain": sum(d == "abstain" for d in decisions),
        "n_revise": sum(d == "revise" for d in decisions),
        "n_revision_requested": len(requested),
        "n_revision_attempted": len(attempted),
        "n_failed_revision_attempts": failed_attempts,
    }


def image_bootstrap_ci(
    rows: Sequence[Dict[str, Any]],
    metric: str = "net_hallucination_reduction",
    n_boot: int = 500,
    seed: int = 42,
) -> tuple[float, float]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get("image_id")), []).append(dict(row))
    image_ids = sorted(groups)
    if len(image_ids) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(int(n_boot)):
        sampled = rng.choice(image_ids, size=len(image_ids), replace=True)
        sample_rows: List[Dict[str, Any]] = []
        for draw_index, image_id in enumerate(sampled):
            for row in groups[str(image_id)]:
                copied = dict(row)
                copied["image_id"] = f"{image_id}|bootstrap{draw_index}"
                sample_rows.append(copied)
        value = summarise(sample_rows).get(metric, float("nan"))
        if np.isfinite(value):
            values.append(float(value))
    if len(values) < max(20, int(n_boot) // 10):
        return float("nan"), float("nan")
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))
