#!/usr/bin/env python
"""Cached analyses requested by the manuscript revision (B2, B4, and B5).

This script is deliberately model-free.  It derives operating-frontier points,
a calibrated grounding/entropy fusion baseline, and an auditable cost summary
from the frozen claim files and canonical run logs.  It never rewrites claim
JSONL files or the primary RQ1--RQ4 tables.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hmqr.config import load_config, seed_everything  # noqa: E402
from hmqr.revise import pipeline  # noqa: E402
from hmqr.revise.pipeline import mark_revision_evaluation_sample  # noqa: E402
from hmqr.signals.support import attribute_kind, relation_signal_kind  # noqa: E402
from hmqr.utils import read_jsonl, write_json  # noqa: E402


ALPHAS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
DELTAS = [0.01, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20]
AMBER_DATASETS = ("amber_attr", "amber_rel")
FUSION_DATASETS = ("rbench", "mmrel")
SCHEMA_VERSION = 1


def _finite(row: Dict[str, Any], key: str) -> float | None:
    try:
        value = float(row.get(key))
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _label(row: Dict[str, Any]) -> int:
    return int(row.get("original_label", row.get("label", -1)))


def _decision_metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, float | int]:
    evaluable = [row for row in rows if _label(row) in (0, 1)]
    n1 = sum(_label(row) == 1 for row in evaluable)
    n0 = sum(_label(row) == 0 for row in evaluable)
    removed = sum(
        _label(row) == 1 and str(row.get("decision")) == "revise"
        for row in evaluable
    )
    incorrect = sum(
        _label(row) == 0 and str(row.get("decision")) == "revise"
        for row in evaluable
    )
    released = [row for row in evaluable if str(row.get("decision")) != "abstain"]
    released_errors = sum(
        (_label(row) == 1 and str(row.get("decision")) == "keep")
        or (_label(row) == 0 and str(row.get("decision")) == "revise")
        for row in released
    )
    total = len(evaluable)
    return {
        "n_claims": total,
        "n_hallucinated": n1,
        "n_grounded": n0,
        "h_removed": removed,
        "c_intervened": incorrect,
        "target_hallucination_removal_rate": removed / n1 if n1 else np.nan,
        "incorrect_intervention_rate": incorrect / n0 if n0 else np.nan,
        "net_hallucination_reduction": (removed - incorrect) / n1 if n1 else np.nan,
        "output_coverage": len(released) / total if total else np.nan,
        "selection_risk": released_errors / len(released) if released else np.nan,
        "abstention_rate": 1.0 - len(released) / total if total else np.nan,
    }


def _apply_policy(
    calibration: Sequence[Dict[str, Any]],
    test: Sequence[Dict[str, Any]],
    q_by_group: Dict[str, float],
    strategy: str,
    delta: float | None,
    cfg,
) -> List[Dict[str, Any]]:
    output = [dict(row) for row in test]
    if strategy == "no_gate":
        for row in output:
            if bool(row.get("flagged")) and bool(row.get("signal_available")):
                row["decision"] = "revise"
            elif bool(row.get("flagged")):
                row["decision"] = "abstain"
            else:
                row["decision"] = "keep"
        return output

    gate_cfg = dict(cfg["residual_gate"])
    gate_cfg["target_incorrect_intervention_rate"] = float(delta)
    theta_by_group, _ = pipeline._select_gate_thresholds(
        calibration, q_by_group, gate_cfg
    )
    for row in output:
        theta = theta_by_group.get(str(row.get("calibration_group")))
        row["theta"] = theta
        row["decision"] = pipeline.decide(
            bool(row.get("flagged")), bool(row.get("signal_available")),
            row.get("d_conflict"), theta,
        )
    return output


def operating_frontiers(cfg) -> pd.DataFrame:
    backbone = "llava15" if "llava15" in cfg.backbone_names else cfg.backbone_names[0]
    records: List[Dict[str, Any]] = []
    for dataset in AMBER_DATASETS:
        path = cfg.path("processed") / f"claims_{backbone}_{dataset}.jsonl"
        rows = [
            row for row in read_jsonl(path)
            if row.get("claim_type") in ("attr", "rel")
        ]
        mark_revision_evaluation_sample(rows, cfg)
        for alpha in ALPHAS:
            for strategy, label in (
                ("full", "Full"),
                ("no_cp", "Fixed threshold"),
                ("no_gate", "No gate"),
            ):
                calibration, test, q_by_group = pipeline._prepare_records(
                    rows, strategy, float(alpha), cfg
                )
                deltas: Iterable[float | None] = (
                    [None] if strategy == "no_gate" else DELTAS
                )
                for delta in deltas:
                    decided = _apply_policy(
                        calibration, test, q_by_group, strategy, delta, cfg
                    )
                    records.append({
                        "schema_version": SCHEMA_VERSION,
                        "backbone": backbone,
                        "dataset": dataset,
                        "strategy": label,
                        "alpha": float(alpha),
                        "delta": np.nan if delta is None else float(delta),
                        **_decision_metrics(decided),
                    })

    detail = pd.DataFrame(records)
    metric_cols = [
        "target_hallucination_removal_rate", "incorrect_intervention_rate",
        "net_hallucination_reduction", "output_coverage", "selection_risk",
        "abstention_rate",
    ]
    group_cols = ["schema_version", "backbone", "strategy", "alpha", "delta"]
    macro = detail.groupby(group_cols, dropna=False, as_index=False)[metric_cols].mean()
    macro["dataset"] = "AMBER macro"
    for col in ("n_claims", "n_hallucinated", "n_grounded", "h_removed", "c_intervened"):
        macro[col] = np.nan
    return pd.concat([detail, macro], ignore_index=True, sort=False)


def plot_frontiers(df: pd.DataFrame, output: Path) -> None:
    macro = df[df["dataset"] == "AMBER macro"].copy()
    colors = {"Full": "#D55E00", "No gate": "#0072B2", "Fixed threshold": "#009E73"}
    markers = {0.05: "o", 0.10: "s", 0.15: "^", 0.20: "D", 0.25: "P", 0.30: "X"}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.45))
    panels = [
        ("output_coverage", "selection_risk", "Output coverage (%)", "Selection risk (%)"),
        ("incorrect_intervention_rate", "net_hallucination_reduction", "Incorrect intervention, IIR (%)", "Net reduction, NHR (pp)"),
    ]
    for ax, (xcol, ycol, xlabel, ylabel) in zip(axes, panels):
        for strategy, group in macro.groupby("strategy", sort=False):
            color = colors[strategy]
            for alpha, alpha_group in group.groupby("alpha", sort=True):
                ordered = alpha_group.sort_values("delta", na_position="last")
                x = 100.0 * ordered[xcol].to_numpy(float)
                y = 100.0 * ordered[ycol].to_numpy(float)
                ax.plot(x, y, color=color, alpha=0.36, linewidth=0.85)
                ax.scatter(
                    x, y, s=24, marker=markers[float(alpha)], color=color,
                    edgecolor="white", linewidth=0.35, zorder=3,
                )
        ax.axhline(0.0, color="#777777", linestyle="--", linewidth=0.7)
        ax.grid(True, color="#E5E5E5", linewidth=0.6)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    strategy_handles = [
        plt.Line2D([0], [0], color=colors[name], marker="o", linewidth=1.2,
                   markersize=5, label=name)
        for name in ("Full", "No gate", "Fixed threshold")
    ]
    alpha_handles = [
        plt.Line2D([0], [0], color="#555555", marker=marker, linestyle="None",
                   markersize=5, label=rf"$\alpha={alpha:.2f}$")
        for alpha, marker in markers.items()
    ]
    fig.legend(handles=strategy_handles, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.995), fontsize=7)
    fig.legend(handles=alpha_handles, loc="upper center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, 0.925), fontsize=7)
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.16, top=0.72, wspace=0.27)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    fig.savefig(output.with_suffix(".png"), dpi=300)
    plt.close(fig)


def _bootstrap_auc(
    labels: np.ndarray,
    scores: np.ndarray,
    image_ids: np.ndarray,
    seed: int,
    n_boot: int = 500,
) -> tuple[float, float]:
    unique = np.unique(image_ids)
    rng = np.random.default_rng(seed)
    values: List[float] = []
    by_image = {image: np.flatnonzero(image_ids == image) for image in unique}
    for _ in range(n_boot):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([by_image[image] for image in sampled])
        if np.unique(labels[idx]).size < 2:
            continue
        values.append(float(roc_auc_score(labels[idx], scores[idx])))
    if not values:
        return np.nan, np.nan
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def fusion_detection(cfg) -> pd.DataFrame:
    seed = int(cfg.get("seed", 42))
    records: List[Dict[str, Any]] = []
    for backbone in cfg.backbone_names:
        for dataset in FUSION_DATASETS:
            path = cfg.path("processed") / f"claims_{backbone}_{dataset}.jsonl"
            rows = [
                row for row in read_jsonl(path)
                if row.get("label") in (0, 1)
                and bool(row.get("rq1_uncertainty_selected"))
                and _finite(row, "g") is not None
                and _finite(row, "predictive_entropy") is not None
            ]
            calib = [row for row in rows if row.get("split") == "calib_cp"]
            test = [row for row in rows if row.get("split") == "test"]
            if len(calib) < 10 or len(test) < 10:
                continue
            x_calib = np.asarray([
                [1.0 - float(row["g"]), float(row["predictive_entropy"])]
                for row in calib
            ])
            x_test = np.asarray([
                [1.0 - float(row["g"]), float(row["predictive_entropy"])]
                for row in test
            ])
            mean = x_calib.mean(axis=0)
            scale = x_calib.std(axis=0)
            scale[scale < 1e-8] = 1.0
            x_calib = (x_calib - mean) / scale
            x_test = (x_test - mean) / scale
            y_calib = np.asarray([int(row["label"]) for row in calib])
            y_test = np.asarray([int(row["label"]) for row in test])
            model = LogisticRegression(
                C=1.0, class_weight="balanced", solver="liblinear",
                random_state=seed,
            )
            model.fit(x_calib, y_calib)
            method_scores = {
                "Grounding": x_test[:, 0],
                "Predictive entropy": x_test[:, 1],
                "Logistic fusion": model.predict_proba(x_test)[:, 1],
            }
            image_ids = np.asarray([str(row.get("image_id")) for row in test])
            for method, scores in method_scores.items():
                ci_low, ci_high = _bootstrap_auc(
                    y_test, np.asarray(scores), image_ids,
                    seed + sum(ord(ch) for ch in backbone + dataset + method),
                )
                records.append({
                    "schema_version": SCHEMA_VERSION,
                    "backbone": backbone,
                    "dataset": dataset,
                    "method": method,
                    "auroc": float(roc_auc_score(y_test, scores)),
                    "auroc_ci_low": ci_low,
                    "auroc_ci_high": ci_high,
                    "average_precision": float(average_precision_score(y_test, scores)),
                    "n_calibration": len(calib),
                    "n_test": len(test),
                    "n_images": len(np.unique(image_ids)),
                    "fusion_weight_grounding": (
                        float(model.coef_[0, 0]) if method == "Logistic fusion" else np.nan
                    ),
                    "fusion_weight_entropy": (
                        float(model.coef_[0, 1]) if method == "Logistic fusion" else np.nan
                    ),
                })
    detail = pd.DataFrame(records)
    if detail.empty:
        return detail
    macro = detail.groupby(["schema_version", "dataset", "method"], as_index=False)[
        ["auroc", "average_precision"]
    ].mean()
    macro["backbone"] = "Macro"
    macro["auroc_ci_low"] = np.nan
    macro["auroc_ci_high"] = np.nan
    macro["n_calibration"] = np.nan
    macro["n_test"] = np.nan
    macro["n_images"] = np.nan
    macro["fusion_weight_grounding"] = np.nan
    macro["fusion_weight_entropy"] = np.nan
    return pd.concat([detail, macro], ignore_index=True, sort=False)


def _dino_keys(row: Dict[str, Any]) -> List[tuple[str, str]]:
    image = str(row.get("image_path") or "")
    if not image or row.get("claim_type") not in ("attr", "rel"):
        return []
    if row.get("claim_type") == "attr":
        kind = attribute_kind(row)
        if kind in ("state", "number"):
            return [(image, str(row.get("object") or "").strip().lower())]
        return []
    if relation_signal_kind(row) == "global":
        return []
    return [
        (image, str(row.get("subject") or "").strip().lower()),
        (image, str(row.get("object") or "").strip().lower()),
    ]


def _parse_duration(value: str) -> float:
    parts = [float(part) for part in value.split(":")]
    if len(parts) == 2:
        return 60.0 * parts[0] + parts[1]
    if len(parts) == 3:
        return 3600.0 * parts[0] + 60.0 * parts[1] + parts[2]
    return parts[0]


def _tqdm_seconds(path: Path) -> float:
    if not path.exists():
        return 0.0
    text = path.read_text(errors="replace")
    completed: Dict[str, float] = {}
    pattern = re.compile(
        r"(?P<label>[^\r\n:]{2,80}):\s*100%\|[^\[]*\[(?P<elapsed>\d+(?::\d+){1,2})<"
    )
    for match in pattern.finditer(text):
        label = re.sub(r"\s+", " ", match.group("label")).strip()
        completed[label] = max(
            completed.get(label, 0.0), _parse_duration(match.group("elapsed"))
        )
    return float(sum(completed.values()))


def overhead_summary(cfg) -> Dict[str, Any]:
    processed = cfg.path("processed")
    all_rows: List[Dict[str, Any]] = []
    by_backbone: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for backbone in cfg.backbone_names:
        for dataset in list(cfg["datasets"]["detect"]) + list(cfg["datasets"]["quality"]):
            path = processed / f"claims_{backbone}_{dataset}.jsonl"
            if not path.exists():
                continue
            rows = [row for row in read_jsonl(path) if row.get("claim_type") in ("attr", "rel")]
            all_rows.extend(rows)
            by_backbone[backbone].extend(rows)

    dino_forwards = 0
    for rows in by_backbone.values():
        dino_forwards += len({key for row in rows for key in _dino_keys(row) if key[1]})
    clip_image_calls = len(all_rows) + sum(
        row.get("claim_type") == "attr" and attribute_kind(row) == "state"
        and row.get("box_object") is not None
        for row in all_rows
    )
    clip_text_calls = len(all_rows) + 2 * sum(
        row.get("claim_type") == "attr" and attribute_kind(row) == "state"
        and row.get("box_object") is not None
        for row in all_rows
    )
    self_verify_calls = sum(
        _finite(row, "l_p") is not None and bool(row.get("lp_computed", True))
        for row in all_rows
    )
    images = {
        (str(row.get("backbone")), str(row.get("dataset")), str(row.get("image_id")))
        for row in all_rows if row.get("image_id") is not None
    }

    logs = cfg.path("logs")
    canonical = {
        "visual_signals": [logs / "multigpu" / f"phase02_{name}.log" for name in cfg.backbone_names],
        "self_verification": [
            logs / "internal_llava15_orderfix.log",
            logs / "internal_instructblip_orderfix.log",
            logs / "llava16_internal_full.log",
            logs / "qwen_internal_attr_rel.log",
            logs / "qwen_internal_rbench.log",
            logs / "qwen_internal_mmrel.log",
            logs / "qwen_internal_chair500.log",
        ],
        "semantic_entropy_baseline": [
            logs / "uncertainty_llava15.log",
            logs / "uncertainty_llava16.log",
            logs / "uncertainty_instructblip_decodefix.log",
            logs / "uncertainty_qwenvl.log",
        ],
        "mechanism_diagnostics": [
            *[logs / "multigpu" / f"phase03_{name}.log" for name in cfg.backbone_names],
            *[logs / "multigpu" / f"phase04_{name}.log" for name in cfg.backbone_names],
        ],
    }
    seconds_by_stage = {
        stage: float(sum(_tqdm_seconds(path) for path in paths))
        for stage, paths in canonical.items()
    }
    total_seconds = sum(seconds_by_stage.values())
    n_claims = len(all_rows)
    n_images = len(images)
    return {
        "schema_version": SCHEMA_VERSION,
        "scope": "canonical successful model loops; failed/restarted jobs and model-load overhead excluded",
        "n_claim_rows": n_claims,
        "n_backbone_dataset_images": n_images,
        "dino_forward_calls": int(dino_forwards),
        "clip_image_forward_calls": int(clip_image_calls),
        "clip_text_encoding_calls": int(clip_text_calls),
        "vlm_self_verification_calls": int(self_verify_calls),
        "dino_forwards_per_image": dino_forwards / n_images if n_images else np.nan,
        "clip_image_forwards_per_image": clip_image_calls / n_images if n_images else np.nan,
        "clip_text_encodings_per_image": clip_text_calls / n_images if n_images else np.nan,
        "vlm_self_verifications_per_image": self_verify_calls / n_images if n_images else np.nan,
        "retained_self_verification_seconds": seconds_by_stage["self_verification"],
        "retained_self_verification_seconds_per_call": (
            seconds_by_stage["self_verification"] / self_verify_calls
            if self_verify_calls
            else np.nan
        ),
        "retained_model_loop_gpu_hours_lower_bound": total_seconds / 3600.0,
        "gpu_seconds_by_stage": seconds_by_stage,
        "canonical_logs": {
            stage: [str(path) for path in paths if path.exists()]
            for stage, paths in canonical.items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_everything(int(cfg.get("seed", 42)))

    tables = cfg.path("tables")
    figures = cfg.path("figures")
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    frontiers = operating_frontiers(cfg)
    frontiers.to_csv(tables / "requested_tradeoffs.csv", index=False)
    plot_frontiers(frontiers, figures / "figure9.pdf")

    fusion = fusion_detection(cfg)
    fusion.to_csv(tables / "requested_fusion.csv", index=False)

    overhead = overhead_summary(cfg)
    write_json(cfg.path("logs") / "requested_overhead.json", overhead)
    print("wrote", tables / "requested_tradeoffs.csv")
    print("wrote", figures / "figure9.pdf")
    print("wrote", tables / "requested_fusion.csv")
    print("wrote", cfg.path("logs") / "requested_overhead.json")
    if not fusion.empty:
        print(
            "\nFusion macro rows:\n",
            fusion[fusion["backbone"] == "Macro"].to_string(index=False),
        )
    else:
        print("\nFusion macro rows: unavailable (no current matched rows)")
    print("\nOverhead:\n", json.dumps(overhead, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
