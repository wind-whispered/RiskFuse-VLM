"""Detection metrics with image-level uncertainty and calibration-only thresholds."""
from __future__ import annotations

from typing import Any, Callable, Dict, Sequence, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score


def _as_arrays(scores: Sequence[float], labels: Sequence[int]):
    scores_arr = np.asarray(scores, dtype=np.float64)
    labels_arr = np.asarray(labels, dtype=int)
    finite = np.isfinite(scores_arr) & np.isin(labels_arr, [0, 1])
    return scores_arr[finite], labels_arr[finite], finite


def auroc(scores: Sequence[float], labels: Sequence[int]) -> float:
    scores_arr, labels_arr, _ = _as_arrays(scores, labels)
    if labels_arr.size == 0 or len(np.unique(labels_arr)) < 2:
        return float("nan")
    return float(roc_auc_score(labels_arr, scores_arr))


def average_precision(scores: Sequence[float], labels: Sequence[int]) -> float:
    scores_arr, labels_arr, _ = _as_arrays(scores, labels)
    if labels_arr.size == 0 or len(np.unique(labels_arr)) < 2:
        return float("nan")
    return float(average_precision_score(labels_arr, scores_arr))


def select_f1_threshold(scores_calib: Sequence[float], labels_calib: Sequence[int]) -> float:
    scores_arr, labels_arr, _ = _as_arrays(scores_calib, labels_calib)
    if labels_arr.size == 0 or len(np.unique(labels_arr)) < 2:
        return float("nan")
    precision, recall, thresholds = precision_recall_curve(labels_arr, scores_arr)
    if thresholds.size == 0:
        return float("nan")
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    return float(thresholds[int(np.nanargmax(f1))])


def f1_at_threshold(scores: Sequence[float], labels: Sequence[int], threshold: float) -> float:
    scores_arr, labels_arr, _ = _as_arrays(scores, labels)
    if labels_arr.size == 0 or not np.isfinite(threshold):
        return float("nan")
    return float(f1_score(labels_arr, scores_arr >= float(threshold), zero_division=0))


def best_f1(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Diagnostic only; paper tables must use a calibration-selected threshold."""
    threshold = select_f1_threshold(scores, labels)
    return f1_at_threshold(scores, labels, threshold)


def image_bootstrap_ci(
    scores: Sequence[float],
    labels: Sequence[int],
    image_ids: Sequence[str],
    metric: Callable[[Sequence[float], Sequence[int]], float],
    n_boot: int = 500,
    seed: int = 42,
) -> Tuple[float, float]:
    scores_arr = np.asarray(scores, dtype=np.float64)
    labels_arr = np.asarray(labels, dtype=int)
    groups = np.asarray([str(value) for value in image_ids])
    finite = np.isfinite(scores_arr) & np.isin(labels_arr, [0, 1])
    scores_arr, labels_arr, groups = scores_arr[finite], labels_arr[finite], groups[finite]
    unique = np.unique(groups)
    if unique.size < 2:
        return float("nan"), float("nan")
    indices = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(int(n_boot)):
        sampled = rng.choice(unique, size=unique.size, replace=True)
        idx = np.concatenate([indices[group] for group in sampled])
        value = metric(scores_arr[idx], labels_arr[idx])
        if np.isfinite(value):
            values.append(value)
    if len(values) < max(20, int(n_boot) // 10):
        return float("nan"), float("nan")
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def summarise(
    scores: Sequence[float],
    labels: Sequence[int],
    image_ids: Sequence[str] | None = None,
    f1_threshold: float | None = None,
    n_boot: int = 500,
    seed: int = 42,
    **_legacy: Any,
) -> Dict[str, Any]:
    if image_ids is None:
        image_ids = [str(index) for index in range(len(labels))]
    threshold = (
        float(f1_threshold) if f1_threshold is not None else select_f1_threshold(scores, labels)
    )
    auroc_ci = image_bootstrap_ci(scores, labels, image_ids, auroc, n_boot, seed)
    ap_ci = image_bootstrap_ci(scores, labels, image_ids, average_precision, n_boot, seed + 1)
    return {
        "auroc": auroc(scores, labels),
        "auroc_ci_low": auroc_ci[0],
        "auroc_ci_high": auroc_ci[1],
        "ap": average_precision(scores, labels),
        "ap_ci_low": ap_ci[0],
        "ap_ci_high": ap_ci[1],
        "f1": f1_at_threshold(scores, labels, threshold),
        "f1_threshold": threshold,
        "n_images": len(set(str(value) for value in image_ids)),
        "n_claims": int(len(labels)),
    }
