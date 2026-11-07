"""T5: residual non-redundancy and incremental utility.

``Var(g_perp)`` is a group-level property. Individual claims receive a signed
residual and a robustly standardised conflict score; they never receive a
per-claim variance. Projection and robust-scaling parameters are fitted on
``calib_cp`` only and then frozen.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..backbones.registry import verification_adapter_version
from ..config import Config
from ..signals.lp import get_lp, has_current_lp, has_valid_lp
from ..signals.kb_prior import assign_kb_priors
from ..signals.support import matched_signal_group, matched_signal_supported
from ..utils import get_logger, read_jsonl, write_jsonl

LOG = get_logger("hmqr.residual")
RESIDUAL_SCHEMA_VERSION = 11
DEFAULT_MIN_RESIDUAL_IMAGES = 3


def _n_images(rows: Sequence[Dict[str, Any]]) -> int:
    """Count independent image units rather than correlated claim rows."""
    return len({str(row.get("image_id")) for row in rows})


def _min_residual_images(cfg: Config) -> int:
    return int(
        cfg.get("residual_gate", {}).get(
            "min_residual_images", DEFAULT_MIN_RESIDUAL_IMAGES
        )
    )


def validate_internal_confidence_ready(cfg: Config) -> None:
    """Fail closed when Step 05 would otherwise reuse neutral/missing l_p."""
    failures: List[str] = []
    checked_files = 0
    min_images = _min_residual_images(cfg)
    for path in sorted(cfg.path("processed").glob("claims_*.jsonl")):
        rows = read_jsonl(path)
        selected = [
            row
            for row in rows
            if row.get("internal_confidence_selected") and matched_signal_supported(row)
        ]
        if not selected:
            continue
        checked_files += 1
        backbone_name = str(rows[0].get("backbone", "")) if rows else ""
        expected_adapter = verification_adapter_version(cfg, backbone_name)
        attempted = [
            row
            for row in selected
            if row.get("lp_attempted")
            and (
                row.get("verification_adapter_version") == expected_adapter
                or (
                    expected_adapter == 1
                    and row.get("verification_adapter_version") is None
                )
            )
        ]
        valid = [row for row in selected if has_current_lp(row, expected_adapter)]
        parse_rate = len(valid) / len(attempted) if attempted else 0.0
        if len(attempted) != len(selected) or parse_rate < 0.95:
            failures.append(
                f"{path.name}: selected={len(selected)}, attempted={len(attempted)}, "
                f"valid={len(valid)}, parse_rate={parse_rate:.3f}"
            )
            continue
        dataset = str(rows[0].get("dataset", "")) if rows else ""
        if dataset in set(cfg.get("datasets", {}).get("quality", [])):
            continue
        for (ctype, subtype), idxs in _groups(rows).items():
            eligible_cp = [
                rows[i]
                for i in idxs
                if rows[i].get("split") == "calib_cp"
                and rows[i].get("label") in (0, 1)
                and _signal_value(rows[i], "g") is not None
            ]
            if _n_images(eligible_cp) < min_images:
                LOG.warning(
                    "[%s/%s/%s] supported group is underpowered by image count; "
                    "residual disabled (calib_cp=%d)",
                    path.name,
                    ctype,
                    subtype,
                    _n_images(eligible_cp),
                )
                continue
            cp = [
                rows[i]
                for i in idxs
                if rows[i].get("internal_confidence_selected")
                and rows[i].get("split") == "calib_cp"
                and rows[i].get("label") in (0, 1)
                and _signal_value(rows[i], "g") is not None
                and has_current_lp(rows[i], expected_adapter)
            ]
            if len(cp) < min_images:
                failures.append(f"{path.name}/{ctype}/{subtype}: calib_cp={len(cp)}")
                continue
            cp_labels = [int(row["label"]) for row in cp]
            if (
                set(cp_labels) != {0, 1}
                or min(cp_labels.count(0), cp_labels.count(1)) < 5
            ):
                failures.append(
                    f"{path.name}/{ctype}/{subtype}: label support "
                    f"calib_cp=({cp_labels.count(0)},{cp_labels.count(1)})"
                )
    if checked_files == 0:
        failures.append("no internal_confidence_selected rows found")
    if failures:
        raise RuntimeError(
            "budgeted internal confidence is incomplete; run "
            "scripts/02_internal_confidence.py first:\n  " + "\n  ".join(failures[:12])
        )


def fit_residual_model(
    g: Sequence[float],
    l_p: Sequence[float],
    kb_prior: Sequence[float] | None = None,
    kb_available: Sequence[bool] | None = None,
) -> Dict[str, float]:
    g_arr = np.asarray(g, dtype=np.float64)
    lp_arr = np.asarray(l_p, dtype=np.float64)
    if g_arr.shape != lp_arr.shape:
        raise ValueError("grounding and belief shapes must match")
    kb_arr = None if kb_prior is None else np.asarray(kb_prior, dtype=np.float64)
    availability = (
        None if kb_available is None else np.asarray(kb_available, dtype=float)
    )
    if availability is not None and (
        kb_arr is None or availability.shape != g_arr.shape
    ):
        raise ValueError("KB availability requires a matching prior vector")
    finite = np.isfinite(g_arr) & np.isfinite(lp_arr)
    if kb_arr is not None:
        if len(kb_arr) != len(g_arr):
            raise ValueError("kb_prior length must match g")
        if availability is not None:
            finite &= np.isfinite(availability)
            kb_arr = np.where(availability > 0, kb_arr, 0.0)
            availability = availability[finite]
        finite &= np.isfinite(kb_arr)
        if availability is not None:
            # Re-select directly from the original vector after all filters.
            availability = np.asarray(kb_available, dtype=float)[finite]
        kb_arr = kb_arr[finite]
    g_arr, lp_arr = g_arr[finite], lp_arr[finite]
    if g_arr.size < 3:
        raise ValueError("at least three finite calibration rows are required")
    columns = [np.ones_like(lp_arr), lp_arr]
    if kb_arr is not None:
        columns.append(kb_arr)
        if availability is not None:
            columns.append(availability)
    design = np.column_stack(columns)
    coefficients = np.linalg.lstsq(design, g_arr, rcond=None)[0]
    intercept, beta = float(coefficients[0]), float(coefficients[1])
    beta_kb = float(coefficients[2]) if kb_arr is not None else 0.0
    residual = g_arr - design @ coefficients
    center = float(np.median(residual))
    mad = float(np.median(np.abs(residual - center)))
    return {
        "intercept": intercept,
        "beta": beta,
        "beta_kb": beta_kb,
        "beta_kb_available": (
            float(coefficients[3]) if availability is not None else 0.0
        ),
        "uses_kb_availability": availability is not None,
        "n_calibration": int(g_arr.size),
        "design_rank": int(np.linalg.matrix_rank(design)),
        "uses_kb_prior": kb_arr is not None,
        "center": center,
        "mad": mad,
    }


def apply_residual_model(
    g: Sequence[float],
    l_p: Sequence[float],
    model: Dict[str, float],
    kb_prior: Sequence[float] | None = None,
    kb_available: Sequence[bool] | None = None,
) -> np.ndarray:
    g_arr = np.asarray(g, dtype=np.float64)
    lp_arr = np.asarray(l_p, dtype=np.float64)
    prediction = float(model["intercept"]) + float(model["beta"]) * lp_arr
    if bool(model.get("uses_kb_prior")):
        if kb_prior is None:
            raise ValueError("kb_prior is required by this residual model")
        values = np.asarray(kb_prior, dtype=np.float64)
        if model.get("uses_kb_availability"):
            if kb_available is None:
                raise ValueError("KB availability is required by this residual model")
            available = np.asarray(kb_available, dtype=float)
            values = np.where(available > 0, values, 0.0)
            prediction = prediction + float(model["beta_kb_available"]) * available
        prediction = prediction + float(model.get("beta_kb", 0.0)) * values
    return g_arr - prediction


def _kb_prior(row: Dict[str, Any]) -> float:
    try:
        value = float(row.get("kb_prior", 0.0))
    except (TypeError, ValueError):
        return 0.0
    return value if np.isfinite(value) else 0.0


def _kb_available(row: Dict[str, Any]) -> bool:
    return bool(row.get("kb_prior_available", False))


def conflict_scores(residual: Sequence[float], center: float, mad: float) -> np.ndarray:
    """Positive values mean grounding is weaker than the belief projection."""
    residual_arr = np.asarray(residual, dtype=np.float64)
    return -(residual_arr - float(center)) / (float(mad) + 1e-8)


def residual_variance(
    g: Sequence[float], l_p: Sequence[float]
) -> Tuple[float, float, float]:
    """Return ``(rho, Var(g), Var(g_perp))`` for a signal group."""
    g_arr = np.asarray(g, dtype=np.float64)
    lp_arr = np.asarray(l_p, dtype=np.float64)
    finite = np.isfinite(g_arr) & np.isfinite(lp_arr)
    g_arr, lp_arr = g_arr[finite], lp_arr[finite]
    if g_arr.size < 3 or np.var(g_arr) < 1e-12:
        return 0.0, float(np.var(g_arr)), float(np.var(g_arr))
    model = fit_residual_model(g_arr, lp_arr)
    residual = apply_residual_model(g_arr, lp_arr, model)
    rho = 0.0 if np.var(lp_arr) < 1e-12 else float(np.corrcoef(g_arr, lp_arr)[0, 1])
    return rho, float(np.var(g_arr)), float(np.var(residual))


def per_claim_residual(g: Sequence[float], l_p: Sequence[float]) -> np.ndarray:
    """Backward-compatible helper returning signed residuals, not squared values."""
    model = fit_residual_model(g, l_p)
    return apply_residual_model(g, l_p, model)


def _subtype(row: Dict[str, Any]) -> str:
    return matched_signal_group(row).split("|", 1)[1]


def _groups(rows: List[Dict[str, Any]]) -> Dict[Tuple[str, str], List[int]]:
    groups: Dict[Tuple[str, str], List[int]] = {}
    for index, row in enumerate(rows):
        ctype = str(row.get("claim_type"))
        if ctype in ("attr", "rel") and matched_signal_supported(row):
            groups.setdefault((ctype, _subtype(row)), []).append(index)
    return groups


def _signal_value(row: Dict[str, Any], signal: str) -> float | None:
    if signal != "g_global" and not matched_signal_supported(row):
        return None
    if signal != "g_global" and not bool(
        row.get("signal_available", row.get("g") is not None)
    ):
        return None
    value = row.get(signal)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _partial_alignment(
    g: np.ndarray, lp: np.ndarray, y: np.ndarray
) -> Tuple[float, float]:
    """Linear residual-label alignment after controlling for internal belief."""
    if len(y) < 4 or np.var(y) < 1e-12:
        return float("nan"), float("nan")
    g_resid = apply_residual_model(g, lp, fit_residual_model(g, lp))
    y_resid = apply_residual_model(y, lp, fit_residual_model(y, lp))
    if np.var(g_resid) < 1e-12 or np.var(y_resid) < 1e-12:
        return 0.0, 0.0
    corr = float(np.corrcoef(g_resid, y_resid)[0, 1])
    return corr, float(corr**2)


def _partial_alignment_ci(
    g: np.ndarray,
    lp: np.ndarray,
    y: np.ndarray,
    image_ids: Sequence[str],
    n_boot: int = 500,
    seed: int = 42,
) -> Tuple[float, float]:
    groups = np.asarray([str(value) for value in image_ids])
    unique = np.unique(groups)
    if unique.size < 2:
        return float("nan"), float("nan")
    indices = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(int(n_boot)):
        sampled = rng.choice(unique, size=unique.size, replace=True)
        idx = np.concatenate([indices[group] for group in sampled])
        corr, _ = _partial_alignment(g[idx], lp[idx], y[idx])
        if np.isfinite(corr):
            values.append(corr)
    if len(values) < max(20, int(n_boot) // 10):
        return float("nan"), float("nan")
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def _fixed_residual_alignment_ci(
    g_residual: np.ndarray,
    y_residual: np.ndarray,
    image_ids: Sequence[str],
    n_boot: int = 500,
    seed: int = 42,
) -> Tuple[float, float]:
    """Image bootstrap with calibration-fitted projections held fixed."""
    groups = np.asarray([str(value) for value in image_ids])
    unique = np.unique(groups)
    if unique.size < 2:
        return float("nan"), float("nan")
    indices = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(int(n_boot)):
        sampled = rng.choice(unique, size=unique.size, replace=True)
        idx = np.concatenate([indices[group] for group in sampled])
        if np.var(g_residual[idx]) < 1e-12 or np.var(y_residual[idx]) < 1e-12:
            continue
        values.append(float(np.corrcoef(g_residual[idx], y_residual[idx])[0, 1]))
    if len(values) < max(20, int(n_boot) // 10):
        return float("nan"), float("nan")
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def _oof_predictions(
    features: np.ndarray, labels: np.ndarray, image_ids: Sequence[str], seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=int)
    features = np.asarray(features, dtype=np.float64)
    groups = np.asarray([str(v) for v in image_ids])
    finite = np.isfinite(features).all(axis=1) & np.isin(labels, [0, 1])
    features, labels, groups = features[finite], labels[finite], groups[finite]
    if labels.size < 10 or np.unique(labels).size < 2 or np.unique(groups).size < 2:
        return np.asarray([]), np.asarray([]), np.asarray([])
    class_counts = np.bincount(labels, minlength=2)
    n_splits = min(5, int(class_counts.min()), int(np.unique(groups).size))
    if n_splits < 2:
        return np.asarray([]), np.asarray([]), np.asarray([])
    model = make_pipeline(
        StandardScaler(), LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
    )
    try:
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        pred = cross_val_predict(
            model, features, labels, groups=groups, cv=cv, method="predict_proba"
        )[:, 1]
    except ValueError:
        cv = GroupKFold(n_splits=n_splits)
        pred = cross_val_predict(
            model, features, labels, groups=groups, cv=cv, method="predict_proba"
        )[:, 1]
    return pred.astype(np.float64), labels, groups


def _oof_auroc(
    features: np.ndarray, labels: np.ndarray, image_ids: Sequence[str], seed: int
) -> float:
    pred, kept_labels, _ = _oof_predictions(features, labels, image_ids, seed)
    if pred.size == 0:
        return float("nan")
    return float(roc_auc_score(kept_labels, pred))


def _paired_image_bootstrap_gain(
    internal_scores: np.ndarray,
    combined_scores: np.ndarray,
    labels: np.ndarray,
    image_ids: np.ndarray,
    n_boot: int = 500,
    seed: int = 42,
) -> Tuple[float, float]:
    """Paired image bootstrap of AUROC(combined) - AUROC(internal)."""
    if internal_scores.size == 0 or combined_scores.size != internal_scores.size:
        return float("nan"), float("nan")
    unique = np.unique(image_ids)
    if unique.size < 2:
        return float("nan"), float("nan")
    indices = {group: np.flatnonzero(image_ids == group) for group in unique}
    rng = np.random.default_rng(seed)
    gains = []
    for _ in range(int(n_boot)):
        sampled = rng.choice(unique, size=unique.size, replace=True)
        idx = np.concatenate([indices[group] for group in sampled])
        sampled_labels = labels[idx]
        if np.unique(sampled_labels).size < 2:
            continue
        gains.append(
            float(
                roc_auc_score(sampled_labels, combined_scores[idx])
                - roc_auc_score(sampled_labels, internal_scores[idx])
            )
        )
    if len(gains) < max(20, int(n_boot) // 10):
        return float("nan"), float("nan")
    return float(np.quantile(gains, 0.025)), float(np.quantile(gains, 0.975))


def _transfer_residual_models(
    cfg: Config,
) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    """Fit frozen AMBER residual models used by the CHAIR quality transfer."""
    models: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    processed = cfg.path("processed")
    for backbone in cfg.backbone_names:
        for dataset, wanted_type in (("amber_attr", "attr"), ("amber_rel", "rel")):
            path = processed / f"claims_{backbone}_{dataset}.jsonl"
            if not path.exists():
                continue
            rows = read_jsonl(path)
            for (ctype, subtype), idxs in _groups(rows).items():
                if ctype != wanted_type:
                    continue
                fit_idxs = [
                    i
                    for i in idxs
                    if rows[i].get("split") == "calib_cp"
                    and rows[i].get("label") in (0, 1)
                    and _signal_value(rows[i], "g") is not None
                    and has_valid_lp(rows[i])
                ]
                if _n_images([rows[i] for i in fit_idxs]) < _min_residual_images(cfg):
                    continue
                g_values = [_signal_value(rows[i], "g") for i in fit_idxs]
                lp_values = [get_lp(rows[i]) for i in fit_idxs]
                models[(backbone, ctype, subtype)] = {
                    "full": fit_residual_model(
                        g_values,
                        lp_values,
                        [_kb_prior(rows[i]) for i in fit_idxs],
                        [_kb_available(rows[i]) for i in fit_idxs],
                    ),
                    "no_kb": fit_residual_model(g_values, lp_values),
                    "source_dataset": dataset,
                }
    return models


def assign_claim_residuals(cfg: Config, *, prior_cache_signature: str | None = None) -> None:
    """Write signed residual and conflict score to each claim file."""
    processed = cfg.path("processed")
    if prior_cache_signature is None:
        assign_kb_priors(cfg)
    else:
        # A transported cache can reuse the already measured prior from the
        # exact cleaned index. Never relabel a stale or absent prior as current.
        from ..review.overlap import sha256
        manifest=cfg.path('vg_index')/'index_manifest.json'
        if not manifest.is_file() or sha256(manifest)!=prior_cache_signature:
            raise ValueError('Transported prior cache does not match the cleaned index manifest')
        for cached in processed.glob('claims_*.jsonl'):
            for row in read_jsonl(cached):
                if row.get('claim_type') not in ('attr','rel'):continue
                if (row.get('kb_prior_schema_version')!=2
                    or row.get('kb_prior_index_signature')!=prior_cache_signature
                    or not isinstance(row.get('kb_prior_available'),bool)
                    or not np.isfinite(float(row.get('kb_prior',np.nan)))):
                    raise ValueError('A transported prior is missing or stale: '+str(row.get('claim_id')))
    transfer_models = _transfer_residual_models(cfg)
    quality_datasets = set(cfg.get("datasets", {}).get("quality", []))
    for path in sorted(processed.glob("claims_*.jsonl")):
        rows = read_jsonl(path)
        if not rows:
            continue
        backbone = str(rows[0].get("backbone", ""))
        dataset = str(rows[0].get("dataset", ""))
        score_rows: List[Dict[str, Any]] = []
        for (ctype, subtype), idxs in _groups(rows).items():
            if dataset in quality_datasets:
                payload = transfer_models.get((backbone, ctype, subtype))
                model = payload.get("full") if payload is not None else None
                no_kb_model = payload.get("no_kb") if payload is not None else None
                fit_source = (
                    f"{payload['source_dataset']}|calib_cp"
                    if payload is not None
                    else None
                )
            else:
                fit_idxs = [
                    i
                    for i in idxs
                    if rows[i].get("split") == "calib_cp"
                    and rows[i].get("label") in (0, 1)
                    and _signal_value(rows[i], "g") is not None
                    and has_valid_lp(rows[i])
                ]
                if _n_images([rows[i] for i in fit_idxs]) >= _min_residual_images(cfg):
                    g_values = [_signal_value(rows[i], "g") for i in fit_idxs]
                    lp_values = [get_lp(rows[i]) for i in fit_idxs]
                    model = fit_residual_model(
                        g_values,
                        lp_values,
                        [_kb_prior(rows[i]) for i in fit_idxs],
                        [_kb_available(rows[i]) for i in fit_idxs],
                    )
                    no_kb_model = fit_residual_model(g_values, lp_values)
                else:
                    model = no_kb_model = None
                fit_source = "calib_cp"
            if model is None or no_kb_model is None:
                LOG.warning(
                    "[%s/%s/%s] insufficient calib_cp rows for residual fit",
                    path.stem,
                    ctype,
                    subtype,
                )
                for i in idxs:
                    rows[i]["g_perp"] = None
                    rows[i]["d_conflict"] = None
                    rows[i]["g_perp_no_kb"] = None
                    rows[i]["d_conflict_no_kb"] = None
                continue
            for i in idxs:
                value = _signal_value(rows[i], "g")
                if value is None or not has_valid_lp(rows[i]):
                    rows[i]["g_perp"] = None
                    rows[i]["d_conflict"] = None
                    rows[i]["g_perp_no_kb"] = None
                    rows[i]["d_conflict_no_kb"] = None
                    continue
                residual = float(
                    apply_residual_model(
                        [value],
                        [get_lp(rows[i])],
                        model,
                        [_kb_prior(rows[i])],
                        [_kb_available(rows[i])],
                    )[0]
                )
                conflict = float(
                    conflict_scores([residual], model["center"], model["mad"])[0]
                )
                residual_no_kb = float(
                    apply_residual_model([value], [get_lp(rows[i])], no_kb_model)[0]
                )
                conflict_no_kb = float(
                    conflict_scores(
                        [residual_no_kb], no_kb_model["center"], no_kb_model["mad"]
                    )[0]
                )
                rows[i].update(
                    g_perp=residual,
                    residual_center=model["center"],
                    residual_mad=model["mad"],
                    d_conflict=conflict,
                    residual_beta_kb=model.get("beta_kb", 0.0),
                    residual_intercept=model["intercept"],
                    residual_beta=model["beta"],
                    residual_beta_kb_available=model.get("beta_kb_available", 0.0),
                    residual_n_calibration=model["n_calibration"],
                    residual_design_rank=model["design_rank"],
                    g_perp_no_kb=residual_no_kb,
                    residual_center_no_kb=no_kb_model["center"],
                    residual_mad_no_kb=no_kb_model["mad"],
                    d_conflict_no_kb=conflict_no_kb,
                    residual_fit_split=fit_source,
                    residual_group=f"{ctype}|{subtype}",
                    residual_schema_version=RESIDUAL_SCHEMA_VERSION,
                )
                score_rows.append(
                    {
                        "claim_id": rows[i].get("claim_id"),
                        "image_id": rows[i].get("image_id"),
                        "backbone": rows[i].get("backbone"),
                        "dataset": rows[i].get("dataset"),
                        "claim_type": ctype,
                        "rel_kind": subtype,
                        "g_perp": residual,
                        "residual_center": model["center"],
                        "residual_mad": model["mad"],
                        "d_conflict": conflict,
                        "d_conflict_no_kb": conflict_no_kb,
                        "kb_prior": _kb_prior(rows[i]),
                        "kb_prior_available": _kb_available(rows[i]),
                        "intercept": model["intercept"],
                        "beta": model["beta"],
                        "beta_kb": model["beta_kb"],
                        "beta_kb_available": model.get("beta_kb_available", 0.0),
                        "n_calibration": model["n_calibration"],
                        "design_rank": model["design_rank"],
                        "fit_split": fit_source,
                    }
                )
        write_jsonl(path, rows)
        suffix = path.stem.removeprefix("claims_")
        write_jsonl(processed / f"residual_claim_scores_{suffix}.jsonl", score_rows)
    LOG.info("signed residual conflict scores written to claim files")


def _analysis_rows(
    rows: List[Dict[str, Any]],
    ctype: str,
    subtype: str,
    signal: str,
    split: str = "test",
):
    selected = []
    for row in rows:
        if row.get("claim_type") != ctype or _subtype(row) != subtype:
            continue
        if row.get("split") != split or row.get("label") not in (0, 1):
            continue
        if not has_valid_lp(row):
            continue
        value = _signal_value(row, signal)
        if value is not None:
            selected.append((row, value))
    return selected


def _calibration_frozen_residuals(
    rows: List[Dict[str, Any]],
    ctype: str,
    subtype: str,
    signal: str,
) -> Dict[str, Any] | None:
    """Fit nuisance projections on calib_cp and apply them once to test."""
    calibration = _analysis_rows(rows, ctype, subtype, signal, split="calib_cp")
    test = _analysis_rows(rows, ctype, subtype, signal, split="test")
    if len(calibration) < 3 or len(test) < 10:
        return None
    calib_rows = [row for row, _ in calibration]
    test_rows = [row for row, _ in test]
    g_calib = np.asarray([value for _, value in calibration], dtype=np.float64)
    lp_calib = np.asarray([get_lp(row) for row in calib_rows], dtype=np.float64)
    kb_calib = np.asarray([_kb_prior(row) for row in calib_rows], dtype=np.float64)
    y_calib = np.asarray([int(row["label"]) for row in calib_rows], dtype=np.float64)
    g_test = np.asarray([value for _, value in test], dtype=np.float64)
    lp_test = np.asarray([get_lp(row) for row in test_rows], dtype=np.float64)
    kb_test = np.asarray([_kb_prior(row) for row in test_rows], dtype=np.float64)
    y_test = np.asarray([int(row["label"]) for row in test_rows], dtype=np.float64)
    avail_calib = [_kb_available(row) for row in calib_rows]
    avail_test = [_kb_available(row) for row in test_rows]
    g_model = fit_residual_model(g_calib, lp_calib, kb_calib, avail_calib)
    y_model = fit_residual_model(y_calib, lp_calib, kb_calib, avail_calib)
    g_residual = apply_residual_model(g_test, lp_test, g_model, kb_test, avail_test)
    y_residual = apply_residual_model(y_test, lp_test, y_model, kb_test, avail_test)
    rho = 0.0 if np.var(lp_test) < 1e-12 else float(np.corrcoef(g_test, lp_test)[0, 1])
    partial_corr = (
        0.0
        if np.var(g_residual) < 1e-12 or np.var(y_residual) < 1e-12
        else float(np.corrcoef(g_residual, y_residual)[0, 1])
    )
    return {
        "rows": test_rows,
        "g": g_test,
        "lp": lp_test,
        "kb_prior": kb_test,
        "kb_prior_available": np.asarray(avail_test, dtype=float),
        "y": y_test,
        "g_residual": g_residual,
        "y_residual": y_residual,
        "rho": rho,
        "var_g": float(np.var(g_test)),
        "var_g_perp": float(np.var(g_residual)),
        "partial_corr": partial_corr,
        "partial_r2": float(partial_corr**2),
        "fit_split": "calib_cp",
    }


def collect_group_stats(cfg: Config) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for path in sorted(cfg.path("processed").glob("claims_*.jsonl")):
        rows = read_jsonl(path)
        for ctype, subtype in _groups(rows):
            for signal in ("g", "g_global"):
                frozen = _calibration_frozen_residuals(rows, ctype, subtype, signal)
                if frozen is None:
                    continue
                group_rows = frozen["rows"]
                image_ids = [str(row.get("image_id")) for row in group_rows]
                partial_ci_low, partial_ci_high = _fixed_residual_alignment_ci(
                    frozen["g_residual"],
                    frozen["y_residual"],
                    image_ids,
                    seed=int(cfg.get("seed", 42)),
                )
                records.append(
                    {
                        "backbone": group_rows[0].get("backbone"),
                        "dataset": group_rows[0].get("dataset"),
                        "claim_type": ctype,
                        "rel_kind": subtype,
                        "signal": "g_matched" if signal == "g" else signal,
                        "rho": frozen["rho"],
                        "var_g": frozen["var_g"],
                        "var_g_perp": frozen["var_g_perp"],
                        "residual_label_partial_corr": frozen["partial_corr"],
                        "partial_r2": frozen["partial_r2"],
                        "residual_label_ci_low": partial_ci_low,
                        "residual_label_ci_high": partial_ci_high,
                        "n_images": len(
                            {str(row.get("image_id")) for row in group_rows}
                        ),
                        "n_claims": len(group_rows),
                        "residual_fit_split": frozen["fit_split"],
                        "residual_schema_version": RESIDUAL_SCHEMA_VERSION,
                    }
                )
    return pd.DataFrame(records)


def collect_signal_gain(cfg: Config) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    seed = int(cfg.get("seed", 42))
    for path in sorted(cfg.path("processed").glob("claims_*.jsonl")):
        rows = read_jsonl(path)
        for ctype, subtype in _groups(rows):
            for signal in ("g", "g_global"):
                frozen = _calibration_frozen_residuals(rows, ctype, subtype, signal)
                if frozen is None:
                    continue
                group_rows = frozen["rows"]
                g = frozen["g"]
                lp = frozen["lp"]
                y = frozen["y"].astype(int)
                image_ids = [str(row.get("image_id")) for row in group_rows]
                kb_prior = frozen["kb_prior"]
                internal_scores, kept_y, kept_groups = _oof_predictions(
                    np.column_stack(
                        [1.0 - lp, 1.0 - kb_prior, frozen["kb_prior_available"]]
                    ),
                    y,
                    image_ids,
                    seed,
                )
                combined_scores, combined_y, combined_groups = _oof_predictions(
                    np.column_stack(
                        [
                            1.0 - lp,
                            1.0 - kb_prior,
                            frozen["kb_prior_available"],
                            1.0 - g,
                        ]
                    ),
                    y,
                    image_ids,
                    seed,
                )
                aligned = (
                    internal_scores.size > 0
                    and internal_scores.size == combined_scores.size
                    and np.array_equal(kept_y, combined_y)
                    and np.array_equal(kept_groups, combined_groups)
                )
                if aligned:
                    internal = float(roc_auc_score(kept_y, internal_scores))
                    combined = float(roc_auc_score(kept_y, combined_scores))
                    gain_ci_low, gain_ci_high = _paired_image_bootstrap_gain(
                        internal_scores, combined_scores, kept_y, kept_groups, seed=seed
                    )
                else:
                    internal = combined = gain_ci_low = gain_ci_high = float("nan")
                records.append(
                    {
                        "backbone": group_rows[0].get("backbone"),
                        "dataset": group_rows[0].get("dataset"),
                        "claim_type": ctype,
                        "rel_kind": subtype,
                        "signal": "g_matched" if signal == "g" else signal,
                        "rho": frozen["rho"],
                        "var_g": frozen["var_g"],
                        "var_g_perp": frozen["var_g_perp"],
                        "auroc_internal": internal,
                        "auroc_with_signal": combined,
                        "auroc_gain": combined - internal,
                        "gain_ci_low": gain_ci_low,
                        "gain_ci_high": gain_ci_high,
                        "n_images": len(set(image_ids)),
                        "n_claims": len(group_rows),
                        "residual_fit_split": frozen["fit_split"],
                        "residual_schema_version": RESIDUAL_SCHEMA_VERSION,
                    }
                )
    return pd.DataFrame(records)


def aggregate_gain_groups(signal_gain: pd.DataFrame) -> pd.DataFrame:
    if signal_gain.empty:
        return pd.DataFrame()
    df = signal_gain.copy()
    df["signal_group"] = np.where(
        df["signal"] == "g_global", "global", "granularity_matched"
    )
    summary = (
        df.groupby(["signal_group", "claim_type"], sort=True)["auroc_gain"]
        .agg(mean_auroc_gain="mean", n_signals="count")
        .reset_index()
    )
    ci_records = []
    for (signal_group, claim_type), group in df.groupby(
        ["signal_group", "claim_type"], sort=True
    ):
        values = group["auroc_gain"].dropna().to_numpy(float)
        if values.size < 2:
            low = high = float("nan")
        else:
            rng = np.random.default_rng(42)
            means = np.asarray(
                [
                    rng.choice(values, size=values.size, replace=True).mean()
                    for _ in range(2000)
                ]
            )
            low, high = (float(v) for v in np.quantile(means, [0.025, 0.975]))
        ci_records.append(
            {
                "signal_group": signal_group,
                "claim_type": claim_type,
                "gain_ci_low": low,
                "gain_ci_high": high,
            }
        )
    result = summary.merge(pd.DataFrame(ci_records), on=["signal_group", "claim_type"])
    result["residual_schema_version"] = RESIDUAL_SCHEMA_VERSION
    return result


def run(cfg: Config, overwrite: bool = False) -> pd.DataFrame:
    out = cfg.path("tables") / "residual_stats.csv"
    gain_out = cfg.path("tables") / "residual_signal_gain.csv"
    group_out = cfg.path("tables") / "residual_gain_by_group.csv"
    if out.exists() and gain_out.exists() and group_out.exists() and not overwrite:
        existing = pd.read_csv(out)
        if {
            "residual_label_partial_corr",
            "partial_r2",
            "n_images",
            "residual_schema_version",
        } <= set(existing.columns) and existing["residual_schema_version"].eq(
            RESIDUAL_SCHEMA_VERSION
        ).all():
            LOG.info("[skip] residual analysis tables exist with current schema")
            return existing
    validate_internal_confidence_ready(cfg)
    assign_claim_residuals(cfg)
    df = collect_group_stats(cfg)
    gain = collect_signal_gain(cfg)
    if df.empty or gain.empty:
        LOG.error("no residual statistics produced")
        return pd.DataFrame()
    df.to_csv(out, index=False)
    gain.to_csv(gain_out, index=False)
    aggregate_gain_groups(gain).to_csv(group_out, index=False)
    LOG.info("wrote %s, %s and %s", out.name, gain_out.name, group_out.name)
    return df
