#!/usr/bin/env python
"""Render every submission-ready figure and table from the result csv files.

Produces:
    results/figures/figure1.pdf/.png    layer-wise probing        (Section 3.2)
    results/figures/figure2.pdf/.png    attention analysis        (Section 3.3)
    results/figures/figure3.pdf/.png    residual principle        (Section 3.4)
    results/figures/figure5.pdf/.png    detection + coverage      (RQ1)
    results/figures/figure6.pdf/.png    residual-gate trade-off   (RQ2)
    results/figures/figure7.pdf/.png    cross-backbone reduction  (RQ3)
    results/figures/figure8.pdf/.png    component ablation        (RQ4)
    results/tables/table1.pdf/.tex/.csv quality preservation      (RQ3)
    results/tables/table2.pdf/.tex/.csv ablation                  (RQ4)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hmqr.config import load_config, seed_everything  # noqa: E402
from hmqr.detect.rq1 import RQ1_SCHEMA_VERSION  # noqa: E402
from hmqr.eval.rq_studies import (  # noqa: E402
    RQ2_BACKBONES,
    RQ2_STRATEGIES,
    RQ4_REVISION_ABLATIONS,
    RQ_SCHEMA_VERSION,
)
from hmqr.figures import (  # noqa: E402
    fig1_layer_auroc,
    fig2_attention_box,
    fig3_residual,
    fig5_detection,
    fig6_gate,
    fig7_crossbackbone,
    fig8_ablation,
    table1_quality,
    table2_ablation,
)
from hmqr.mechanism.attention_analysis import (  # noqa: E402
    ATTENTION_SCHEMA_VERSION,
    descriptive_summary,
)
from hmqr.mechanism.layer_probe import PROBE_SCHEMA_VERSION  # noqa: E402
from hmqr.residual.decomposition import RESIDUAL_SCHEMA_VERSION  # noqa: E402
from hmqr.revise.pipeline import PIPELINE_SCHEMA_VERSION  # noqa: E402


PLOT_INPUTS = [
    "layer_probe_auroc.csv",
    "probe_peak_layers.csv",
    "attention_vattn.csv",
    "attention_summary.csv",
    "residual_stats.csv",
    "residual_signal_gain.csv",
    "residual_gain_by_group.csv",
    "rq1_detection.csv",
    "rq1_coverage.csv",
    "rq2_gate.csv",
    "rq3_reduction.csv",
    "rq3_quality.csv",
    "rq4_ablation.csv",
]

ARTIFACT_INPUTS = {
    "figure1": ["layer_probe_auroc.csv", "probe_peak_layers.csv"],
    "figure2": ["attention_vattn.csv", "attention_summary.csv"],
    "figure3": ["residual_stats.csv", "residual_signal_gain.csv", "residual_gain_by_group.csv"],
    "figure5": ["rq1_detection.csv", "rq1_coverage.csv"],
    "figure6": ["rq2_gate.csv"],
    "figure7": ["rq3_reduction.csv", "rq3_quality.csv"],
    "figure8": ["rq4_ablation.csv"],
    "table1": ["rq3_reduction.csv", "rq3_quality.csv"],
    "table2": ["rq4_ablation.csv"],
}

SCHEMA_CONTRACTS = {
    "layer_probe_auroc.csv": ("probe_schema_version", PROBE_SCHEMA_VERSION),
    "probe_peak_layers.csv": ("probe_schema_version", PROBE_SCHEMA_VERSION),
    "attention_vattn.csv": ("attention_schema_version", ATTENTION_SCHEMA_VERSION),
    "attention_summary.csv": ("attention_schema_version", ATTENTION_SCHEMA_VERSION),
    "residual_stats.csv": ("residual_schema_version", RESIDUAL_SCHEMA_VERSION),
    "residual_signal_gain.csv": ("residual_schema_version", RESIDUAL_SCHEMA_VERSION),
    "residual_gain_by_group.csv": ("residual_schema_version", RESIDUAL_SCHEMA_VERSION),
    "rq1_detection.csv": ("rq1_schema_version", RQ1_SCHEMA_VERSION),
    "rq1_coverage.csv": ("rq1_schema_version", RQ1_SCHEMA_VERSION),
    "rq2_gate.csv": ("rq_schema_version", RQ_SCHEMA_VERSION),
    "rq3_reduction.csv": ("rq_schema_version", RQ_SCHEMA_VERSION),
    "rq3_quality.csv": ("rq_schema_version", RQ_SCHEMA_VERSION),
    "rq4_ablation.csv": ("rq_schema_version", RQ_SCHEMA_VERSION),
}


def _key_set(frame: pd.DataFrame, columns) -> set[tuple[str, ...]]:
    return {
        tuple("" if pd.isna(value) else str(value) for value in values)
        for values in frame[list(columns)].itertuples(index=False, name=None)
    }


def _coverage_failures(cfg, name: str, frame: pd.DataFrame) -> list[str]:
    """Validate the experiment matrix, not merely a non-empty current schema."""
    failures: list[str] = []
    backbones = set(map(str, cfg.backbone_names))
    if "backbone" in frame and name not in {
        "residual_gain_by_group.csv", "rq2_gate.csv", "rq4_ablation.csv"
    }:
        observed = set(frame["backbone"].dropna().astype(str)) - {"Macro"}
        if observed != backbones:
            failures.append(
                f"backbones={sorted(observed)} (expected {sorted(backbones)})"
            )

    if name in {"layer_probe_auroc.csv", "probe_peak_layers.csv"}:
        expected = {
            (backbone, claim_class)
            for backbone in backbones
            for claim_class in ("object", "attribute", "relation")
        }
        observed = _key_set(frame, ("backbone", "class"))
        if not expected <= observed:
            failures.append(f"missing backbone/class groups={sorted(expected - observed)}")

    elif name in {"attention_vattn.csv", "attention_summary.csv"}:
        expected = {
            (backbone, claim_type, str(label))
            for backbone in backbones
            for claim_type in ("attr", "rel")
            for label in (0, 1)
        }
        observed = _key_set(frame, ("backbone", "claim_type", "label"))
        if expected != observed:
            failures.append(f"attention groups mismatch; missing={sorted(expected - observed)}")
        if name == "attention_vattn.csv":
            counts = frame.groupby(["backbone", "claim_type", "label"]).size()
            small = [tuple(map(str, key)) for key, value in counts.items() if int(value) < 10]
            if small:
                failures.append(f"attention groups with <10 claims={small}")

    elif name in {"residual_stats.csv", "residual_signal_gain.csv"}:
        expected_signals = {"g_global", "g_matched"}
        identity_cols = ["dataset", "claim_type", "rel_kind"]
        reference_groups = None
        for backbone in backbones:
            subset = frame[frame["backbone"].astype(str) == backbone]
            observed_groups = _key_set(subset, identity_cols)
            if reference_groups is None:
                reference_groups = observed_groups
            elif observed_groups != reference_groups:
                failures.append(
                    f"{backbone} residual groups mismatch; "
                    f"missing={sorted(reference_groups - observed_groups)}"
                )
            for key, group in subset.groupby(identity_cols, dropna=False):
                signals = set(group["signal"].astype(str))
                if signals != expected_signals or len(group) != len(expected_signals):
                    failures.append(
                        f"{backbone} residual signal pair incomplete for {key}: "
                        f"rows/signals={len(group)}/{sorted(signals)}"
                    )

    elif name == "residual_gain_by_group.csv":
        expected = {
            (signal_group, claim_type)
            for signal_group in ("global", "granularity_matched")
            for claim_type in ("attr", "rel")
        }
        observed = _key_set(frame, ("signal_group", "claim_type"))
        if observed != expected:
            failures.append(f"residual aggregate groups mismatch={sorted(observed)}")

    elif name == "rq1_detection.csv":
        required_methods = {
            "Ours", "GlobalCLIP-thr", "PredictiveEntropy", "SemanticEntropy"
        }
        detail = frame[frame["backbone"].astype(str) != "Macro"]
        group_cols = ["backbone", "dataset", "claim_type", "rel_kind"]
        for key, group in detail.groupby(group_cols, dropna=False):
            methods = set(group["method"].astype(str))
            if methods != required_methods or len(group) != len(required_methods):
                failures.append(f"RQ1 detection methods incomplete for {key}")
        identity_cols = ["dataset", "claim_type", "rel_kind"]
        groups_by_backbone = {
            backbone: _key_set(
                detail[detail["backbone"].astype(str) == backbone], identity_cols
            )
            for backbone in backbones
        }
        reference_groups = next(iter(groups_by_backbone.values()), set())
        for backbone, observed_groups in groups_by_backbone.items():
            if observed_groups != reference_groups:
                failures.append(
                    f"RQ1 detection groups mismatch for {backbone}; "
                    f"missing={sorted(reference_groups - observed_groups)}"
                )
        expected_macro = {
            (*group_key, method)
            for group_key in reference_groups
            for method in required_methods
        }
        macro = frame[frame["backbone"].astype(str) == "Macro"]
        observed_macro = _key_set(
            macro, ["dataset", "claim_type", "rel_kind", "method"]
        )
        if observed_macro != expected_macro or len(macro) != len(expected_macro):
            failures.append(
                "RQ1 detection Macro matrix does not match the backbone groups"
            )

    elif name == "rq1_coverage.csv":
        required_methods = {"Ours", "GlobalCLIP-thr", "Ours-fixed"}
        required_alpha = {0.05, 0.10, 0.20}
        group_cols = ["backbone", "dataset", "claim_type", "rel_kind"]
        for key, group in frame.groupby(group_cols, dropna=False):
            combinations = {
                (str(method), round(float(alpha), 8))
                for method, alpha in group[["method", "alpha"]].itertuples(
                    index=False, name=None
                )
            }
            expected = {
                (method, round(alpha, 8))
                for method in required_methods for alpha in required_alpha
            }
            if combinations != expected or len(group) != len(expected):
                failures.append(f"RQ1 coverage combinations incomplete for {key}")
        identity_cols = ["dataset", "claim_type", "rel_kind"]
        groups_by_backbone = {
            backbone: _key_set(
                frame[frame["backbone"].astype(str) == backbone], identity_cols
            )
            for backbone in backbones
        }
        reference_groups = next(iter(groups_by_backbone.values()), set())
        for backbone, observed_groups in groups_by_backbone.items():
            if observed_groups != reference_groups:
                failures.append(
                    f"RQ1 coverage groups mismatch for {backbone}; "
                    f"missing={sorted(reference_groups - observed_groups)}"
                )

    elif name == "rq2_gate.csv":
        expected = {
            (backbone, label, dataset, scope)
            for backbone in set(RQ2_BACKBONES) & backbones
            for _, label in RQ2_STRATEGIES
            for dataset, scope in (
                ("amber_attr", "dataset"),
                ("amber_rel", "dataset"),
                ("pooled", "pooled"),
            )
        }
        observed = _key_set(frame, ("backbone", "strategy", "dataset", "plot_scope"))
        if observed != expected:
            failures.append(f"RQ2 matrix incomplete; missing={sorted(expected - observed)}")

    elif name == "rq3_reduction.csv":
        expected = {
            (backbone, dataset)
            for backbone in backbones
            for dataset in (
                "amber_attr", "amber_rel", "rbench", "mmrel", "chair500"
            )
        }
        observed = _key_set(frame, ("backbone", "dataset"))
        if observed != expected:
            failures.append(f"RQ3 reduction matrix incomplete; missing={sorted(expected - observed)}")

    elif name == "rq3_quality.csv":
        metrics = {"CHAIR-i", "BLEU-4", "ROUGE-L", "obj_recall"}
        expected = {(backbone, metric) for backbone in backbones for metric in metrics}
        observed = _key_set(frame, ("backbone", "metric"))
        if observed != expected:
            failures.append(f"RQ3 quality matrix incomplete; missing={sorted(expected - observed)}")
        expected_images = int(cfg["datasets"]["coco"].get("chair_num_images", 500))
        if "n_images" not in frame or not frame["n_images"].eq(expected_images).all():
            failures.append(f"RQ3 quality must cover exactly {expected_images} images")

    elif name == "rq4_ablation.csv":
        datasets = ("amber_attr", "amber_rel")
        expected_a = {
            ("A", dataset, variant)
            for dataset in datasets
            for variant in (
                "Granularity-matched signal", "Global CLIP", "CP", "Fixed threshold"
            )
        }
        expected_b = {
            ("B", dataset, label)
            for dataset in datasets
            for _, label in RQ4_REVISION_ABLATIONS
        }
        observed = _key_set(frame, ("panel", "dataset", "variant"))
        expected = expected_a | expected_b
        if observed != expected:
            failures.append(f"RQ4 matrix incomplete; missing={sorted(expected - observed)}")

    if name in {"rq2_gate.csv", "rq3_reduction.csv", "rq4_ablation.csv"}:
        revision_rows = frame
        if name == "rq3_reduction.csv":
            revision_rows = frame[
                frame["dataset"].isin(
                    ["amber_attr", "amber_rel", "rbench", "mmrel"]
                )
            ]
        elif name == "rq4_ablation.csv":
            revision_rows = frame[frame["panel"].astype(str) == "B"]
        required = {
            "n_revise", "external_label_coverage",
            "target_hallucination_removal_rate", "incorrect_intervention_rate",
        }
        if required <= set(revision_rows):
            incomplete = revision_rows[
                ~np.isclose(
                    pd.to_numeric(
                        revision_rows["external_label_coverage"], errors="coerce"
                    ),
                    1.0, equal_nan=False,
                )
            ]
            if not incomplete.empty:
                failures.append(
                    f"{len(incomplete)} revision rows lack complete external labels"
                )
        else:
            failures.append(
                "missing native-label intervention completion columns: "
                + ", ".join(sorted(required - set(revision_rows)))
            )
    return failures


def _validate_plot_inputs(cfg, input_names=None) -> None:
    """Fail closed instead of rendering current-looking plots from stale CSVs."""
    failures = []
    for name in (PLOT_INPUTS if input_names is None else input_names):
        path = cfg.path("tables") / name
        if not path.exists():
            failures.append(f"{name}: missing")
            continue
        try:
            frame = pd.read_csv(path)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{name}: unreadable ({exc})")
            continue
        if frame.empty:
            failures.append(f"{name}: empty")
            continue
        contract = SCHEMA_CONTRACTS.get(name)
        if contract is None:
            continue
        column, expected = contract
        if column not in frame.columns:
            failures.append(f"{name}: missing {column} (expected v{expected})")
            continue
        observed = set(pd.to_numeric(frame[column], errors="coerce").dropna().astype(int))
        if observed != {int(expected)}:
            failures.append(
                f"{name}: stale {column}={sorted(observed)} (expected v{expected})"
            )
            continue
        failures.extend(
            f"{name}: {failure}" for failure in _coverage_failures(cfg, name, frame)
        )
    if failures:
        raise RuntimeError(
            "Step 12 refused stale/incomplete inputs:\n  " + "\n  ".join(failures)
        )


def _refresh_derived_inputs(cfg, input_names) -> None:
    """Refresh CPU-only derivatives without invoking models or changing raw scores."""
    if "attention_summary.csv" not in set(input_names):
        return
    raw_path = cfg.path("tables") / "attention_vattn.csv"
    if not raw_path.exists():
        return
    raw = pd.read_csv(raw_path)
    if (
        raw.empty
        or "attention_schema_version" not in raw
        or set(pd.to_numeric(
            raw["attention_schema_version"], errors="coerce"
        ).dropna().astype(int)) != {ATTENTION_SCHEMA_VERSION}
    ):
        return
    descriptive_summary(raw).to_csv(
        cfg.path("tables") / "attention_summary.csv", index=False
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_manifest(cfg, config_path: str) -> Path:
    overhead = cfg.path("logs") / "overhead.txt"
    if (
        not overhead.exists()
        or "# computational overhead (Section 5.5)"
        not in overhead.read_text(encoding="utf-8")
        or f"pipeline schema version   : {PIPELINE_SCHEMA_VERSION}"
        not in overhead.read_text(encoding="utf-8")
    ):
        raise RuntimeError(
            "Step 12 refused a missing/stale results/logs/overhead.txt; "
            "complete Step 08 with the current pipeline first"
        )
    files = []
    candidates = [cfg.path("tables") / name for name in PLOT_INPUTS]
    candidates += sorted(cfg.path("figures").glob("figure*.*"))
    candidates += sorted(cfg.path("tables").glob("table[12].*"))
    candidates += [overhead]
    for path in candidates:
        if path.is_file():
            files.append(
                {
                    "path": path.relative_to(cfg.root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    config = Path(config_path)
    if not config.is_absolute():
        config = cfg.root / config
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": config.relative_to(cfg.root).as_posix(),
        "config_sha256": _sha256(config),
        "files": files,
    }
    path = cfg.path("logs") / "artifact_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--only", nargs="+", choices=sorted(ARTIFACT_INPUTS), default=None,
        help="render only the named artifacts after validating their direct CSV inputs",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.get("seed", 42))
    makers = {
        "figure1": ("Figure 1", fig1_layer_auroc.make),
        "figure2": ("Figure 2", fig2_attention_box.make),
        "figure3": ("Figure 3", fig3_residual.make),
        "figure5": ("Figure 5", fig5_detection.make),
        "figure6": ("Figure 6", fig6_gate.make),
        "figure7": ("Figure 7", fig7_crossbackbone.make),
        "figure8": ("Figure 8", fig8_ablation.make),
        "table1": ("Table 1", table1_quality.make),
        "table2": ("Table 2", table2_ablation.make),
    }
    selected = args.only or list(makers)
    required_inputs = sorted({
        name for artifact in selected for name in ARTIFACT_INPUTS[artifact]
    })
    _refresh_derived_inputs(cfg, required_inputs)
    _validate_plot_inputs(cfg, required_inputs)
    results = {
        makers[key][0]: makers[key][1](cfg)
        for key in selected
    }

    print("\n================ figure summary ================")
    for name, ok in results.items():
        print(f"  {name:<10} {'OK' if ok else 'MISSING INPUT'}")
    print("===============================================")
    print("Figure 4 is the method schematic; it is drawn separately and is not "
          "produced by this code base.")
    if all(results.values()):
        if args.only is None:
            print(f"Artifact manifest: {_write_manifest(cfg, args.config)}")
        else:
            print("Partial render complete; full artifact manifest was not overwritten.")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
