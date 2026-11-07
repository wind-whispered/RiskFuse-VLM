"""Claim-level signed residual-conflict gate.

The gate is calibrated on the independent ``calib_revision`` split using the
external benchmark's original binary labels. No generated calibration text or
post-revision pseudo-label is needed.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np

from ..utils import get_logger

LOG = get_logger("hmqr.gate")


def _finite_conflict(row: Dict[str, Any]) -> Optional[float]:
    if not bool(row.get("signal_available", False)):
        return None
    try:
        value = float(row.get("d_conflict"))
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def select_theta(
    calib_rows: Sequence[Dict[str, Any]],
    flagged: Optional[Sequence[bool]] = None,
    target_incorrect_intervention_rate: float = 0.05,
    min_calibration_claims: int = 10,
    min_correct_claims_for_risk: int = 5,
    n_grid: int = 50,
) -> float:
    """Choose theta from native-label intervention utility.

    Among thresholds whose incorrect-intervention rate is at most the target,
    maximise ``target errors selected - correct targets selected``. The risk
    denominator is every evaluable originally-correct target in the calibration
    group.  CP flags restrict which targets can be selected for revision, but
    they must not shrink the population used to estimate intervention risk.
    Ties prefer the larger (more conservative) threshold.
    """
    if flagged is None:
        flagged_arr = np.asarray(
            [bool(r.get("flagged")) for r in calib_rows], dtype=bool
        )
    else:
        flagged_arr = np.asarray(list(flagged), dtype=bool)
    if len(flagged_arr) != len(calib_rows):
        raise ValueError("flagged length must match calibration rows")

    conflicts = np.asarray(
        [
            value if (value := _finite_conflict(row)) is not None else np.nan
            for row in calib_rows
        ]
    )
    original = np.asarray(
        [int(row.get("original_label", row.get("label", -1))) for row in calib_rows]
    )
    evaluable = np.isfinite(conflicts) & np.isin(original, [0, 1])
    usable = flagged_arr & evaluable
    n_usable = int(usable.sum())
    if n_usable < int(min_calibration_claims):
        raise ValueError(
            f"insufficient native-labelled calibration claims: {n_usable} < "
            f"{int(min_calibration_claims)}"
        )
    n_correct = int((evaluable & (original == 0)).sum())
    if n_correct < int(min_correct_claims_for_risk):
        raise ValueError(
            f"insufficient correct calibration claims for risk estimation: {n_correct} < "
            f"{int(min_correct_claims_for_risk)}"
        )

    values = conflicts[usable]
    candidates = np.unique(np.quantile(values, np.linspace(0.0, 1.0, n_grid)))
    candidates = np.unique(
        np.concatenate(
            ([np.nextafter(values.min(), -np.inf)], candidates, [values.max()])
        )
    )
    feasible = []
    for theta in candidates:
        revise = usable & (conflicts > theta)
        removed = int((revise & (original == 1)).sum())
        incorrect = int((revise & (original == 0)).sum())
        risk = float(incorrect / n_correct)
        if risk <= float(target_incorrect_intervention_rate):
            feasible.append(
                (removed - incorrect, removed, float(theta), risk, int(revise.sum()))
            )
    if not feasible:
        raise ValueError(
            "no residual-gate threshold satisfies the calibration risk constraint"
        )
    net, removed, theta, risk, n_revise = max(
        feasible, key=lambda item: (item[0], item[1], item[2])
    )
    LOG.info(
        "residual gate theta=%.6g net=%d removed=%d incorrect_rate=%.4f n=%d",
        theta,
        net,
        removed,
        risk,
        n_revise,
    )
    return theta


def passes_gate(row: Dict[str, Any], theta: float) -> bool:
    value = _finite_conflict(row)
    return value is not None and value > float(theta)


def select_theta_ltt(
    calib_rows: Sequence[Dict[str, Any]],
    candidate_thresholds: Sequence[float],
    target_risk: float = 0.05,
    failure_probability: float = 0.05,
) -> Dict[str, Any]:
    """Simultaneously certify a fixed, independently learned threshold family.

    For each threshold, incorrect interventions on n native-correct independent
    images are Bernoulli observations. A one-sided Clopper-Pearson bound at
    failure_probability / M gives simultaneous per-family validity by a union
    bound. Utility-based selection among certified thresholds is then allowed.
    This is conditional on the independent learning data and targets IIR, not
    the error fraction among interventions. It is not shift or all-group control.

    The caller must form candidates exclusively from a disjoint learning split.
    Missing conflict/evidence cannot trigger an action but stays in the risk
    denominator. A no-action return is structurally safe and never extrapolates
    a finite maximum calibration score as a guaranteed no-action threshold.
    """
    from scipy.stats import beta

    if not 0 < target_risk < 1 or not 0 < failure_probability < 1:
        raise ValueError("risk and failure probability must be in (0, 1)")
    images = [str(r.get("image_id", "")) for r in calib_rows]
    if any(not x for x in images) or len(set(images)) != len(images):
        raise ValueError("LTT calibration requires one record per independent image")
    candidates = np.unique(np.asarray(candidate_thresholds, dtype=float))
    if not candidates.size or not np.isfinite(candidates).all():
        raise ValueError("LTT candidates must be a nonempty fixed finite family")
    labels = np.asarray(
        [int(r.get("original_label", r.get("label", -1))) for r in calib_rows]
    )
    correct = labels == 0
    n = int(correct.sum())
    eta = failure_probability / len(candidates)
    tests = []
    certified = []
    for threshold in candidates:
        selected = np.asarray(
            [bool(r.get("flagged")) and passes_gate(r, threshold) for r in calib_rows]
        )
        errors = int((selected & correct).sum())
        removed = int((selected & (labels == 1)).sum())
        upper = (
            1.0
            if n == 0 or errors == n
            else float(beta.ppf(1.0 - eta, errors + 1, n - errors))
        )
        record = dict(
            theta=float(threshold),
            n_correct=n,
            n_incorrect=errors,
            n_removed=removed,
            risk_upper=upper,
            accepted=upper <= target_risk,
        )
        tests.append(record)
        if record["accepted"]:
            certified.append(record)
    chosen = (
        max(
            certified,
            key=lambda r: (
                r["n_removed"] - r["n_incorrect"],
                r["n_removed"],
                r["theta"],
            ),
        )
        if certified
        else None
    )
    return dict(
        theta=chosen["theta"] if chosen else None,
        status="certified" if chosen else "no_action",
        risk_upper=chosen["risk_upper"] if chosen else 0.0,
        n_correct=n,
        family_size=len(candidates),
        failure_probability=failure_probability,
        target_risk=target_risk,
        scope="per_group_fixed_family_iid_images",
        candidate_source="independent_learning_split_required",
        tests=tests,
    )
