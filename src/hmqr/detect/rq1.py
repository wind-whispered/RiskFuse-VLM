"""RQ1: per-backbone detection quality and conformal operating points."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..backbones.registry import verification_adapter_version

from ..config import Config
from ..sampling import image_unit_rows
from ..eval.detection_metrics import select_f1_threshold, summarise
from ..signals.support import matched_signal_group, matched_signal_supported
from ..utils import get_logger, read_jsonl
from .conformal import (
    calibrate,
    empirical_coverage,
    false_alarm_rate,
    fixed_threshold,
    flag_hallucination,
)

LOG = get_logger("hmqr.rq1")
METHODS = ["Ours", "GlobalCLIP-thr", "PredictiveEntropy", "SemanticEntropy"]
RQ1_SCHEMA_VERSION = 14


def _subtype(row: Dict[str, Any]) -> str:
    return matched_signal_group(row).split("|", 1)[1]


def _value(row: Dict[str, Any], key: str) -> Optional[float]:
    if key == "g" and not matched_signal_supported(row):
        return None
    try:
        value = float(row.get(key))
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _method_scores(
    rows: List[Dict[str, Any]],
    method: str,
    params: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    if method == "Ours":
        return np.asarray([1.0 - float(row["g"]) for row in rows], dtype=np.float64)
    if method == "GlobalCLIP-thr":
        values = np.asarray([float(row["g_global"]) for row in rows], dtype=np.float64)
        if params is None:
            lo, hi = float(values.min()), float(values.max())
        else:
            lo, hi = params
        normalised = np.clip((values - lo) / (hi - lo + 1e-8), 0.0, 1.0)
        return 1.0 - normalised
    if method == "PredictiveEntropy":
        return np.asarray(
            [float(row["predictive_entropy"]) for row in rows], dtype=np.float64
        )
    return np.asarray(
        [float(row["semantic_entropy"]) for row in rows], dtype=np.float64
    )


def _required_key(method: str) -> str:
    return {
        "Ours": "g",
        "GlobalCLIP-thr": "g_global",
        "PredictiveEntropy": "predictive_entropy",
        "SemanticEntropy": "semantic_entropy",
    }[method]


def load_dataset_rows(cfg: Config, dataset: str, backbone: str) -> List[Dict[str, Any]]:
    path = cfg.path("processed") / f"claims_{backbone}_{dataset}.jsonl"
    return [
        row
        for row in read_jsonl(path)
        if row.get("claim_type") in ("attr", "rel")
    ]


def _balanced_common_subset(group: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return one paired, label-balanced subset shared by every RQ1 method."""
    required = [_required_key(method) for method in METHODS]
    buckets: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for row in group:
        if row.get('label') not in (0, 1):
            continue
        if not bool(row.get("rq1_uncertainty_selected")):
            continue
        if not all(_value(row, key) is not None for key in required):
            continue
        key = (str(row.get("split")), int(row.get("label")))
        buckets.setdefault(key, []).append(row)
    expected = {(split, label) for split in ("calib_cp", "test") for label in (0, 1)}
    if set(buckets) != expected:
        return []
    per_stratum = min(len(buckets[key]) for key in expected)
    if per_stratum < 1:
        return []
    selected: List[Dict[str, Any]] = []
    for key in sorted(expected):
        selected.extend(sorted(buckets[key], key=_stable_row_key)[:per_stratum])
    return selected


def _stable_row_key(row: Dict[str, Any]) -> str:
    claim_id = str(row.get("claim_id", ""))
    return claim_id.split("|", 1)[1] if "|" in claim_id else claim_id


def _append_macro_average(df: pd.DataFrame, value_columns: List[str]) -> pd.DataFrame:
    if df.empty:
        return df
    base = df[df["backbone"] != "Macro"]
    keys = ["dataset", "claim_type", "rel_kind", "method"]
    macro = base.groupby(keys, dropna=False, as_index=False)[value_columns].mean(
        numeric_only=True
    )
    macro["backbone"] = "Macro"
    # Averaging bootstrap endpoints is not a confidence interval for the macro
    # mean. Keep the point estimate and expose group heterogeneity in the plot.
    for column in ("auroc_ci_low", "auroc_ci_high", "ap_ci_low", "ap_ci_high"):
        if column in macro:
            macro[column] = np.nan
    macro["aggregation"] = "unweighted_macro_backbone_group"
    base = base.copy()
    base["aggregation"] = "image_bootstrap_group"
    macro["n_images"] = base.groupby(keys, dropna=False)["n_images"].sum().values
    macro["n_claims"] = base.groupby(keys, dropna=False)["n_claims"].sum().values
    return pd.concat([base, macro], ignore_index=True, sort=False)


def run(cfg: Config, overwrite: bool = False) -> pd.DataFrame:
    tables = cfg.path("tables")
    det_path = tables / "rq1_detection.csv"
    cov_path = tables / "rq1_coverage.csv"
    if det_path.exists() and cov_path.exists() and not overwrite:
        existing = pd.read_csv(det_path)
        coverage_existing = pd.read_csv(cov_path)
        if (
            {"backbone", "auroc_ci_low", "n_images", "rq1_schema_version"}
            <= set(existing.columns)
            and existing["rq1_schema_version"].eq(RQ1_SCHEMA_VERSION).all()
            and {"method", "rq1_schema_version"} <= set(coverage_existing.columns)
            and coverage_existing["rq1_schema_version"].eq(RQ1_SCHEMA_VERSION).all()
            and "Ours-fixed" in set(coverage_existing["method"].astype(str))
        ):
            LOG.info("[skip] RQ1 tables exist with current schema")
            return existing

    seed = int(cfg.get("seed", 42))
    alphas = [float(value) for value in cfg["cp"]["alpha_grid"]]
    min_hallu = int(cfg["cp"].get("min_hallucinations_per_group", 5))
    det_rows: List[Dict[str, Any]] = []
    cov_rows: List[Dict[str, Any]] = []
    insufficient_rows: List[Dict[str, Any]] = []

    for backbone in cfg.backbone_names:
        for dataset in cfg["datasets"]["detect"]:
            rows = load_dataset_rows(cfg, dataset, backbone)
            grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
            for row in rows:
                if not matched_signal_supported(row):
                    continue
                grouped.setdefault((str(row["claim_type"]), _subtype(row)), []).append(
                    row
                )
            for (claim_type, rel_kind), group in sorted(grouped.items()):
                selected_uncertainty = [
                    row for row in group if bool(row.get("rq1_uncertainty_selected"))
                ]
                comparison_group = _balanced_common_subset(group)
                if not comparison_group:
                    availability = {
                        f"{split}|label={label}": sum(
                            row.get("split") == split
                            and int(row.get("label")) == label
                            and bool(row.get("rq1_uncertainty_selected"))
                            and all(
                                _value(row, _required_key(method)) is not None
                                for method in METHODS
                            )
                            for row in group
                        )
                        for split in ("calib_cp", "test")
                        for label in (0, 1)
                    }
                    LOG.warning(
                        "[%s/%s/%s/%s] no label-balanced four-method common "
                        "intersection; omitting this unsupported comparison group: %s",
                        backbone,
                        dataset,
                        claim_type,
                        rel_kind,
                        availability,
                    )
                    continue
                stale_uncertainty = [
                    row.get("claim_id")
                    for row in selected_uncertainty
                    if (
                        row.get("semantic_entropy_schema_version") != 2
                        or row.get("verification_adapter_version")
                        != verification_adapter_version(cfg, backbone)
                    )
                ]
                if stale_uncertainty:
                    raise RuntimeError(
                        "matched uncertainty rows use a stale generation/decoding "
                        f"schema; first={stale_uncertainty[0]}"
                    )
                for method in METHODS:
                    key = _required_key(method)
                    calib = [
                        row
                        for row in comparison_group
                        if row.get("split") == "calib_cp"
                        and _value(row, key) is not None
                    ]
                    test = [
                        row
                        for row in comparison_group
                        if row.get("split") == "test" and _value(row, key) is not None
                    ]
                    if len(calib) < 10 or len(test) < 10:
                        LOG.warning(
                            "[%s/%s/%s/%s] %s unavailable or too small",
                            backbone,
                            dataset,
                            claim_type,
                            rel_kind,
                            method,
                        )
                        continue
                    params = None
                    if method == "GlobalCLIP-thr":
                        calib_values = np.asarray([float(row[key]) for row in calib])
                        params = (float(calib_values.min()), float(calib_values.max()))
                    calib_scores = _method_scores(calib, method, params)
                    test_scores = _method_scores(test, method, params)
                    threshold = select_f1_threshold(
                        calib_scores, [int(row["label"]) for row in calib]
                    )
                    stats = summarise(
                        test_scores,
                        [int(row["label"]) for row in test],
                        [str(row.get("image_id")) for row in test],
                        f1_threshold=threshold,
                        seed=seed,
                    )
                    det_rows.append(
                        {
                            "backbone": backbone,
                            "dataset": dataset,
                            "claim_type": claim_type,
                            # ``rel_kind`` stores the matched signal subtype for
                            # both claim classes (for example attr/state and
                            # attr/count).  Dropping attribute subtypes here would
                            # collapse distinct rows during macro aggregation.
                            "rel_kind": rel_kind,
                            "method": method,
                            "evaluation_scope": "balanced_common_available_uncertainty_subset",
                            "selection_balanced_by_label": True,
                            "n_selected_before_common_filter": len(
                                selected_uncertainty
                            ),
                            **stats,
                        }
                    )

                cp_candidates = [row for row in group if row.get("split") == "calib_cp"]
                test_candidates = [row for row in group if row.get("split") == "test"]
                sampling_unit = cfg["cp"].get("sampling_unit", "image_one_claim")
                if sampling_unit != "image_one_claim":
                    raise ValueError(
                        "primary conformal evaluation requires image_one_claim"
                    )
                cp_candidates = image_unit_rows(cp_candidates, seed, "conformal")
                test_candidates = image_unit_rows(test_candidates, seed, "conformal")
                cp_calib = [
                    row
                    for row in cp_candidates
                    if row.get("split") == "calib_cp" and _value(row, "g") is not None
                    and row.get('label') in (0, 1)
                ]
                cp_test = [
                    row
                    for row in test_candidates
                    if row.get("split") == "test" and _value(row, "g") is not None
                    and row.get('label') in (0, 1)
                ]
                all_test = [row for row in test_candidates if row.get('label') in (0, 1)]
                if len(cp_calib) < 10 or len(cp_test) < 10:
                    continue
                g_calib = np.asarray([float(row["g"]) for row in cp_calib])
                g_test = np.asarray([float(row["g"]) for row in cp_test])
                y_calib = [int(row["label"]) for row in cp_calib]
                y_test = [int(row["label"]) for row in cp_test]
                for alpha in alphas:
                    try:
                        q_hat = calibrate(
                            g_calib, y_calib, alpha, min_hallucinations=min_hallu
                        )
                    except ValueError as exc:
                        LOG.warning("[%s/%s/%s] %s", backbone, dataset, rel_kind, exc)
                        insufficient_rows.append(
                            dict(
                                backbone=backbone,
                                dataset=dataset,
                                claim_type=claim_type,
                                rel_kind=rel_kind,
                                alpha=alpha,
                                n_calibration_images=len(cp_calib),
                                n_calibration_hallucinations=int(
                                    sum(v == 1 for v in y_calib)
                                ),
                                reason=str(exc),
                                sampling_unit=sampling_unit,
                            )
                        )
                        continue
                    flagged = flag_hallucination(g_test, q_hat)
                    ties = int(
                        np.isclose(1.0 - g_test, q_hat, rtol=0.0, atol=1e-12).sum()
                    )
                    cov_rows.append(
                        {
                            "backbone": backbone,
                            "dataset": dataset,
                            "claim_type": claim_type,
                            "rel_kind": rel_kind,
                            "method": "Ours",
                            "alpha": alpha,
                            "q_hat": q_hat,
                            "sampling_unit": sampling_unit,
                            "n_calibration_images": len(cp_calib),
                            "empirical_recall": empirical_coverage(
                                g_test, y_test, q_hat
                            ),
                            "target": 1.0 - alpha,
                            "fpr": false_alarm_rate(g_test, y_test, q_hat),
                            "flag_rate": float(flagged.mean()),
                            "ties": ties,
                            "n_hallu": int(sum(value == 1 for value in y_test)),
                            "n_grounded": int(sum(value == 0 for value in y_test)),
                            "n_claims": len(cp_test),
                            "n_total_test": len(all_test),
                            "n_images": len(
                                {str(row.get("image_id")) for row in cp_test}
                            ),
                            "image_ids_json": json.dumps(
                                sorted({str(row.get("image_id")) for row in cp_test})
                            ),
                            "signal_unavailable_rate": (
                                1.0 - len(cp_test) / len(all_test)
                                if all_test
                                else float("nan")
                            ),
                        }
                    )

                    # Keep the score fixed and change only the calibration
                    # rule, so RQ4 can isolate fixed-threshold vs CP effects.
                    q_granular_fixed = fixed_threshold(g_calib, quantile=1.0 - alpha)
                    granular_fixed_flags = flag_hallucination(g_test, q_granular_fixed)
                    cov_rows.append(
                        {
                            "backbone": backbone,
                            "dataset": dataset,
                            "claim_type": claim_type,
                            "rel_kind": rel_kind,
                            "method": "Ours-fixed",
                            "alpha": alpha,
                            "q_hat": q_granular_fixed,
                            "empirical_recall": empirical_coverage(
                                g_test, y_test, q_granular_fixed
                            ),
                            "target": float("nan"),
                            "fpr": false_alarm_rate(g_test, y_test, q_granular_fixed),
                            "flag_rate": float(granular_fixed_flags.mean()),
                            "ties": int(
                                np.isclose(
                                    1.0 - g_test,
                                    q_granular_fixed,
                                    rtol=0.0,
                                    atol=1e-12,
                                ).sum()
                            ),
                            "n_hallu": int(sum(value == 1 for value in y_test)),
                            "n_grounded": int(sum(value == 0 for value in y_test)),
                            "n_claims": len(cp_test),
                            "n_total_test": len(all_test),
                            "n_images": len(
                                {str(row.get("image_id")) for row in cp_test}
                            ),
                            "image_ids_json": json.dumps(
                                sorted({str(row.get("image_id")) for row in cp_test})
                            ),
                            "signal_unavailable_rate": (
                                1.0 - len(cp_test) / len(all_test)
                                if all_test
                                else float("nan")
                            ),
                        }
                    )

                    global_calib = [
                        row
                        for row in cp_candidates
                        if row.get("split") == "calib_cp"
                        and _value(row, "g_global") is not None
                        and row.get('label') in (0, 1)
                    ]
                    global_test = [
                        row
                        for row in test_candidates
                        if row.get("split") == "test"
                        and _value(row, "g_global") is not None
                        and row.get('label') in (0, 1)
                    ]
                    if global_calib and global_test:
                        values = np.asarray(
                            [float(row["g_global"]) for row in global_calib]
                        )
                        lo, hi = float(values.min()), float(values.max())
                        gc = np.clip((values - lo) / (hi - lo + 1e-8), 0.0, 1.0)
                        gt = np.clip(
                            (
                                np.asarray(
                                    [float(row["g_global"]) for row in global_test]
                                )
                                - lo
                            )
                            / (hi - lo + 1e-8),
                            0.0,
                            1.0,
                        )
                        q_fixed = fixed_threshold(gc, quantile=1.0 - alpha)
                        y_global = [int(row["label"]) for row in global_test]
                        fixed_flags = flag_hallucination(gt, q_fixed)
                        cov_rows.append(
                            {
                                "backbone": backbone,
                                "dataset": dataset,
                                "claim_type": claim_type,
                                "rel_kind": rel_kind,
                                "method": "GlobalCLIP-thr",
                                "alpha": alpha,
                                "q_hat": q_fixed,
                                "empirical_recall": empirical_coverage(
                                    gt, y_global, q_fixed
                                ),
                                "target": 1.0 - alpha,
                                "fpr": false_alarm_rate(gt, y_global, q_fixed),
                                "flag_rate": float(fixed_flags.mean()),
                                "ties": int(
                                    np.isclose(1.0 - gt, q_fixed, atol=1e-12).sum()
                                ),
                                "n_hallu": int(sum(value == 1 for value in y_global)),
                                "n_grounded": int(
                                    sum(value == 0 for value in y_global)
                                ),
                                "n_claims": len(global_test),
                                "n_total_test": len(global_test),
                                "n_images": len(
                                    {str(row.get("image_id")) for row in global_test}
                                ),
                                "image_ids_json": json.dumps(
                                    sorted(
                                        {
                                            str(row.get("image_id"))
                                            for row in global_test
                                        }
                                    )
                                ),
                                "signal_unavailable_rate": 0.0,
                            }
                        )

    pd.DataFrame(
        insufficient_rows,
        columns=[
            "backbone",
            "dataset",
            "claim_type",
            "rel_kind",
            "alpha",
            "n_calibration_images",
            "n_calibration_hallucinations",
            "reason",
            "sampling_unit",
        ],
    ).to_csv(tables / "rq1_insufficient_calibration.csv", index=False)
    det = pd.DataFrame(det_rows)
    coverage = pd.DataFrame(cov_rows)
    if det.empty:
        LOG.error("no detection results produced")
        return det
    matrix_keys = ["backbone", "dataset", "claim_type", "rel_kind"]
    incomplete = []
    for key, group in det.groupby(matrix_keys, dropna=False, sort=True):
        observed = set(group["method"].astype(str))
        if observed != set(METHODS):
            incomplete.append((key, sorted(set(METHODS) - observed)))
    if incomplete:
        raise RuntimeError(f"incomplete matched RQ1 method matrix: {incomplete[:5]}")
    observed_datasets = set(det["dataset"].astype(str))
    missing_datasets = set(cfg["datasets"]["detect"]) - observed_datasets
    if missing_datasets:
        raise RuntimeError(
            "RQ1 has no valid four-method matched comparison for datasets: "
            f"{sorted(missing_datasets)}"
        )
    det = _append_macro_average(
        det,
        [
            "auroc",
            "auroc_ci_low",
            "auroc_ci_high",
            "ap",
            "ap_ci_low",
            "ap_ci_high",
            "f1",
        ],
    )
    det["rq1_schema_version"] = RQ1_SCHEMA_VERSION
    coverage["rq1_schema_version"] = RQ1_SCHEMA_VERSION
    det.to_csv(det_path, index=False)
    coverage.to_csv(cov_path, index=False)
    LOG.info("wrote %s and %s", det_path.name, cov_path.name)
    return det
