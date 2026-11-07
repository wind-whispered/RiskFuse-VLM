"""Figure 5: detection ranking and empirical conformal risk control."""
from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from ..config import Config
from ..utils import get_logger
from .style import (
    ALPHA,
    DATASET_LABELS,
    LINE_WIDTHS,
    MARKER_SIZES,
    METHOD_COLORS,
    METHOD_MARKERS,
    NEUTRAL_COLORS,
    OKABE_ITO,
    apply_style,
    clean_axis,
    figure_size,
    panel_label,
    save_figure,
)

LOG = get_logger("hmqr.fig5")
METHOD_ORDER = ("Ours", "GlobalCLIP-thr", "PredictiveEntropy", "SemanticEntropy")
METHOD_LABELS = {
    "Ours": "Granularity-matched",
    "GlobalCLIP-thr": "Global CLIP",
    "PredictiveEntropy": "Predictive entropy",
    "SemanticEntropy": "Semantic entropy",
}


def make(cfg: Config) -> bool:
    detection_path = cfg.path("tables") / "rq1_detection.csv"
    coverage_path = cfg.path("tables") / "rq1_coverage.csv"
    if not detection_path.exists() or not coverage_path.exists():
        LOG.error("missing Figure 5 inputs -- run scripts/06_conformal_detect.py first")
        return False

    detection = pd.read_csv(detection_path)
    coverage = pd.read_csv(coverage_path)
    if detection.empty or coverage.empty:
        LOG.error("RQ1 tables are empty")
        return False

    datasets = [name for name in cfg["datasets"]["detect"] if name in set(detection["dataset"])]
    methods = [name for name in METHOD_ORDER if name in set(detection["method"])]
    if not datasets or not methods:
        return False

    apply_style()
    fig, (ax_rank, ax_control) = plt.subplots(
        1,
        2,
        figsize=figure_size("figure5"),
        gridspec_kw={"width_ratios": [1.14, 0.86]},
    )

    macro = detection[detection["backbone"] == "Macro"]
    if macro.empty:
        macro = detection
    individual = detection[detection["backbone"] != "Macro"]
    y_base = np.arange(len(datasets))[::-1]
    offsets = np.linspace(0.24, -0.24, len(methods))

    for method, offset in zip(methods, offsets):
        colour = METHOD_COLORS[method]
        marker = METHOD_MARKERS[method]
        for y, dataset in zip(y_base, datasets):
            row = macro[(macro["dataset"] == dataset) & (macro["method"] == method)]
            value = float(row["auroc"].mean()) if not row.empty else np.nan
            points = (
                individual[
                    (individual["dataset"] == dataset)
                    & (individual["method"] == method)
                ]
                .groupby("backbone")["auroc"]
                .mean()
                .dropna()
                .to_numpy(float)
            )
            if points.size:
                jitter = np.linspace(-0.028, 0.028, points.size)
                ax_rank.scatter(
                    points,
                    np.full(points.size, y + offset) + jitter,
                    s=MARKER_SIZES["small_scatter"],
                    marker=marker,
                    facecolor=NEUTRAL_COLORS["white"],
                    edgecolor=colour,
                    linewidth=LINE_WIDTHS["edge"],
                    alpha=ALPHA["points"],
                    zorder=2,
                )
            if np.isfinite(value):
                ax_rank.scatter(
                    value,
                    y + offset,
                    s=MARKER_SIZES["scatter"],
                    marker=marker,
                    facecolor=colour,
                    edgecolor=NEUTRAL_COLORS["white"],
                    linewidth=LINE_WIDTHS["edge"],
                    zorder=4,
                )

    finite_scores = pd.to_numeric(detection["auroc"], errors="coerce").dropna()
    lower = max(
        0.0,
        min(0.5, math.floor((float(finite_scores.min()) - 0.04) * 10) / 10),
    )
    ax_rank.axvline(
        0.5,
        color=NEUTRAL_COLORS["mid"],
        linewidth=LINE_WIDTHS["reference"],
        linestyle=(0, (2.0, 2.5)),
    )
    ax_rank.set_xlim(lower, 1.01)
    ax_rank.set_yticks(y_base)
    ax_rank.set_yticklabels([DATASET_LABELS.get(name, name) for name in datasets])
    ax_rank.set_xlabel("Detection AUROC")
    ax_rank.set_title("Cross-dataset detection", loc="left", pad=5)
    clean_axis(ax_rank, grid="x")
    panel_label(ax_rank, "(a)", dx=-0.18, dy=1.10)

    ours = coverage[coverage["method"] == "Ours"]
    alphas = sorted(float(value) for value in ours["alpha"].dropna().unique())
    curve_meta = (
        ("empirical_recall", "Empirical recall", "o", "-", METHOD_COLORS["Ours"]),
        ("fpr", "False-positive rate", "s", (0, (4.0, 2.0)), OKABE_ITO["blue"]),
        ("flag_rate", "Flag rate", "^", (0, (1.2, 1.8)), OKABE_ITO["green"]),
    )
    for column, label, marker, linestyle, colour in curve_meta:
        means = [float(ours[ours["alpha"] == alpha][column].mean()) for alpha in alphas]
        ax_control.plot(
            alphas,
            means,
            marker=marker,
            linestyle=linestyle,
            color=colour,
            markerfacecolor=NEUTRAL_COLORS["white"],
            markeredgecolor=colour,
            markeredgewidth=0.7,
            label=label,
            zorder=3,
        )
    ax_control.plot(
        alphas,
        [1.0 - alpha for alpha in alphas],
        color=NEUTRAL_COLORS["black"],
        linestyle=(0, (2.0, 2.5)),
        linewidth=LINE_WIDTHS["reference"],
        marker="_",
        markersize=5,
        label=r"Target $1-\alpha$",
        zorder=2,
    )
    ax_control.set_xlim(min(alphas) - 0.015, max(alphas) + 0.015)
    ax_control.set_ylim(0.0, 1.02)
    ax_control.set_xticks(alphas)
    ax_control.set_xlabel(r"Miscoverage level, $\alpha$")
    ax_control.set_ylabel("Rate")
    ax_control.set_title("Conformal screening", loc="left", pad=5)
    clean_axis(ax_control, grid="y")
    ax_control.legend(loc="lower left", bbox_to_anchor=(0.02, 0.05))
    panel_label(ax_control, "(b)", dx=-0.18, dy=1.10)

    method_handles = [
        Line2D(
            [0],
            [0],
            marker=METHOD_MARKERS[method],
            color="none",
            markerfacecolor=METHOD_COLORS[method],
            markeredgecolor=NEUTRAL_COLORS["white"],
            markeredgewidth=LINE_WIDTHS["edge"],
            markersize=5.2,
            label=METHOD_LABELS[method],
        )
        for method in methods
    ]
    fig.legend(
        handles=method_handles,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.34, 0.995),
    )
    fig.subplots_adjust(left=0.205, right=0.985, bottom=0.18, top=0.80, wspace=0.34)
    save_figure(fig, "figure5", cfg.path("figures"))
    return True
