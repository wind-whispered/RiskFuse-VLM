"""T6: split conformal prediction for claim-level hallucination detection.

Nonconformity score
-------------------
    s(c) = 1 - g(c)          larger s = weaker visual grounding

Guarantee
---------
The paper controls the *hallucination recall*: a true hallucination must be
flagged with probability at least 1 - alpha. Calibration therefore uses the
scores of the hallucinated calibration claims, and the threshold is their
LOWER-tail quantile:

    k     = floor((n + 1) * alpha),  requiring k >= 1
    q_hat = the k-th smallest calibration score
    flag  <=>  s >= q_hat

so that a fraction of about 1 - alpha of the calibration hallucinations lie above
q_hat. By exchangeability the same holds for a new hallucinated claim from the
same distribution when the eligible scores are exchangeable. Independent
images do not by themselves make a pooled collection of dependent claims
exchangeable. The primary protocol selects one claim per image before reading
labels or scores. Unsupported sample sizes fail closed rather than clipping k.

(The upper-tail quantile ceil((n+1)(1-alpha))/n is the right choice for the dual
formulation that instead bounds the false-alarm rate on grounded claims; that is
not the guarantee reported in the paper.)

Attribute and relation claims have different score distributions and are
calibrated as independent conformal instances.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def conformal_threshold(scores_calib: Sequence[float], alpha: float) -> float:
    """Lower-tail conformal quantile of the calibration nonconformity scores."""
    if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be finite and strictly between zero and one")
    s = np.asarray(scores_calib, dtype=np.float64)
    s = s[np.isfinite(s)]
    n = s.size
    if n == 0:
        raise ValueError("cannot calibrate conformal threshold without finite scores")
    k = int(np.floor((n + 1) * alpha))
    if k < 1:
        raise ValueError(
            f"insufficient_calibration: n={n}, alpha={alpha}, lower-tail rank={k}"
        )
    return float(np.sort(s)[k - 1])


def flag_hallucination(g_test: Sequence[float], q_hat: float) -> np.ndarray:
    s = 1.0 - np.asarray(g_test, dtype=np.float64)
    return s >= q_hat


def calibrate(
    g_calib: Sequence[float],
    labels_calib: Sequence[int],
    alpha: float,
    hallucination_only: bool = True,
    min_hallucinations: int = 5,
) -> float:
    """Calibrate the threshold for a 1 - alpha hallucination-recall guarantee."""
    g = np.asarray(g_calib, dtype=np.float64)
    y = np.asarray(labels_calib)
    if hallucination_only:
        if g.shape != y.shape:
            raise ValueError("calibration scores and labels must have the same shape")
        sel = (y == 1) & np.isfinite(g)
        n_hallu = int(sel.sum())
        if n_hallu < int(min_hallucinations):
            raise ValueError(
                f"insufficient hallucination calibration rows: {n_hallu} "
                f"< {int(min_hallucinations)}"
            )
        g = g[sel]
    return conformal_threshold(1.0 - g, alpha)


def empirical_coverage(
    g_test: Sequence[float], labels_test: Sequence[int], q_hat: float
) -> float:
    """Fraction of true hallucinations that are flagged (empirical recall)."""
    y = np.asarray(labels_test)
    flagged = flag_hallucination(g_test, q_hat)
    pos = y == 1
    if pos.sum() == 0:
        return float("nan")
    return float(flagged[pos].mean())


def false_alarm_rate(
    g_test: Sequence[float], labels_test: Sequence[int], q_hat: float
) -> float:
    """Fraction of grounded claims that are flagged."""
    y = np.asarray(labels_test)
    flagged = flag_hallucination(g_test, q_hat)
    neg = y == 0
    if neg.sum() == 0:
        return float("nan")
    return float(flagged[neg].mean())


def fixed_threshold(g_calib: Sequence[float], quantile: float = 0.5) -> float:
    """Baseline: a fixed score quantile with no finite-sample guarantee."""
    g = np.asarray(g_calib, dtype=np.float64)
    return float(np.quantile(1.0 - g, quantile))
