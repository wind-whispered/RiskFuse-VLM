"""Figure 8: stage-specific ablation comparisons without structural missing cells."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from ..config import Config
from ..utils import get_logger
from .style import (
    LINE_WIDTHS,
    MARKER_SIZES,
    NEUTRAL_COLORS,
    OKABE_ITO,
    apply_style,
    clean_axis,
    figure_size,
    panel_label,
    save_figure,
)

LOG = get_logger("hmqr.fig8")
INTERVENTION_VARIANTS = (
    "Full",
    "w/o residual gate",
    "fixed threshold (w/o CP)",
    "w/o residualization",
    "w/o VG prior",
    "abstain-all",
)
INTERVENTION_LABELS = {
    "Full": "Full",
    "w/o residual gate": "No residual gate",
    "fixed threshold (w/o CP)": "Fixed threshold (no CP)",
    "w/o residualization": "No residualisation",
    "w/o VG prior": "No VG prior",
    "abstain-all": "Abstain all",
}


def _aggregate(frame: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    return frame.groupby("variant", sort=False, as_index=False)[metrics].mean(numeric_only=True)


def _value(frame: pd.DataFrame, variant: str, metric: str) -> float:
    hit = frame.loc[frame["variant"].astype(str).eq(variant), metric]
    return float(hit.iloc[0]) if not hit.empty else np.nan


def _paired_panel(ax, frame: pd.DataFrame, rows, variants, colours, markers, *, percent: bool, xlabel: str) -> None:
    y_positions = np.arange(len(rows))[::-1]
    scale = 100.0 if percent else 1.0
    for y, (metric, label) in zip(y_positions, rows):
        values = [scale * _value(frame, variant, metric) for variant in variants]
        ax.plot(values, [y, y], color=NEUTRAL_COLORS["light"], linewidth=LINE_WIDTHS["data"], zorder=1)
        for value, colour, marker in zip(values, colours, markers):
            ax.scatter(
                [value], [y], s=MARKER_SIZES["scatter"], marker=marker,
                facecolor=colour, edgecolor=NEUTRAL_COLORS["white"],
                linewidth=LINE_WIDTHS["edge"], zorder=3,
            )
    ax.set_yticks(y_positions)
    ax.set_yticklabels([label for _, label in rows])
    ax.set_xlabel(xlabel)
    clean_axis(ax, grid="x")


def _draw_intervention(ax, source: pd.DataFrame, revision: pd.DataFrame) -> None:
    y_positions = np.arange(len(INTERVENTION_VARIANTS))[::-1]
    for stripe, y in enumerate(y_positions):
        if stripe % 2 == 0:
            ax.axhspan(y - 0.46, y + 0.46, color=NEUTRAL_COLORS["near_white"], zorder=0)
    for y, variant in zip(y_positions, INTERVENTION_VARIANTS):
        net = 100.0 * _value(revision, variant, "net_hallucination_reduction")
        risk = -100.0 * _value(revision, variant, "incorrect_intervention_rate")
        ax.plot([risk, net], [y, y], color=NEUTRAL_COLORS["light"], linewidth=LINE_WIDTHS["data"], zorder=1)
        raw = source[(source["panel"].astype(str).eq("B")) & source["variant"].astype(str).eq(variant)]
        jitter = np.linspace(-0.07, 0.07, max(len(raw), 1))
        for offset, (_, row) in zip(jitter, raw.iterrows()):
            ax.scatter(
                [-100.0 * float(row["incorrect_intervention_rate"])], [y + offset],
                s=MARKER_SIZES["small_scatter"], marker="s", facecolor=NEUTRAL_COLORS["white"],
                edgecolor=OKABE_ITO["vermillion"], linewidth=LINE_WIDTHS["edge"], alpha=0.58, zorder=2,
            )
            ax.scatter(
                [100.0 * float(row["net_hallucination_reduction"])], [y + offset],
                s=MARKER_SIZES["small_scatter"], marker="o", facecolor=NEUTRAL_COLORS["white"],
                edgecolor=OKABE_ITO["green"], linewidth=LINE_WIDTHS["edge"], alpha=0.58, zorder=2,
            )
        ax.scatter([risk], [y], s=MARKER_SIZES["scatter"], marker="s", facecolor=OKABE_ITO["vermillion"], edgecolor=NEUTRAL_COLORS["white"], linewidth=LINE_WIDTHS["edge"], zorder=4)
        ax.scatter([net], [y], s=MARKER_SIZES["scatter"], marker="o", facecolor=OKABE_ITO["green"], edgecolor=NEUTRAL_COLORS["white"], linewidth=LINE_WIDTHS["edge"], zorder=4)
    ax.axvline(0, color=NEUTRAL_COLORS["mid"], linewidth=LINE_WIDTHS["reference"], linestyle=(0, (2, 2.5)), zorder=1)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([INTERVENTION_LABELS[name] for name in INTERVENTION_VARIANTS])
    ax.set_xlabel("Signed outcome (percentage points)")
    clean_axis(ax, grid="x")


def make(cfg: Config) -> bool:
    path = cfg.path("tables") / "rq4_ablation.csv"
    if not path.exists():
        LOG.error("missing %s -- run scripts/11_rq4_ablation.py first", path)
        return False
    source = pd.read_csv(path)
    if source.empty or "panel" not in source:
        return False
    panel_a = source[source["panel"].astype(str).eq("A")]
    panel_b = source[source["panel"].astype(str).eq("B")]
    ranking = _aggregate(panel_a, ["auroc", "ap"])
    risk = _aggregate(panel_a, ["empirical_recall", "fpr", "flag_rate"])
    revision = _aggregate(panel_b, ["net_hallucination_reduction", "incorrect_intervention_rate"])
    if ranking.empty or risk.empty or revision.empty:
        return False
    if not set(INTERVENTION_VARIANTS).issubset(set(revision["variant"].astype(str))):
        LOG.error("Figure 8 is missing one or more intervention ablations")
        return False

    apply_style()
    fig = plt.figure(figsize=figure_size("figure8"))
    grid = fig.add_gridspec(1, 3, width_ratios=(0.82, 1.0, 1.52), wspace=0.48)
    rank_ax = fig.add_subplot(grid[0, 0])
    risk_ax = fig.add_subplot(grid[0, 1])
    intervention_ax = fig.add_subplot(grid[0, 2])

    _paired_panel(
        rank_ax, ranking, (("auroc", "AUROC"), ("ap", "AP")),
        ("Granularity-matched signal", "Global CLIP"),
        (OKABE_ITO["vermillion"], OKABE_ITO["blue"]), ("o", "s"),
        percent=False, xlabel="Score",
    )
    rank_ax.set_title("Score ranking", loc="left", pad=5)
    rank_values = pd.to_numeric(ranking[["auroc", "ap"]].stack(), errors="coerce").dropna()
    rank_ax.set_xlim(float(rank_values.min()) - 0.025, float(rank_values.max()) + 0.025)
    rank_ax.legend(handles=[
        Line2D([0], [0], marker="o", linestyle="none", color=OKABE_ITO["vermillion"], label="Matched"),
        Line2D([0], [0], marker="s", linestyle="none", color=OKABE_ITO["blue"], label="Global CLIP"),
    ], loc="center right")
    panel_label(rank_ax, "(a)", dx=-0.22, dy=1.11)

    _paired_panel(
        risk_ax, risk, (("empirical_recall", "Recall"), ("fpr", "FPR"), ("flag_rate", "Flag rate")),
        ("CP", "Fixed threshold"),
        (OKABE_ITO["vermillion"], OKABE_ITO["blue"]), ("o", "s"),
        percent=True, xlabel="Rate (%)",
    )
    risk_ax.set_xlim(-3, 103)
    risk_ax.set_xticks([0, 50, 100])
    risk_ax.set_title("Risk control", loc="left", pad=5)
    risk_ax.legend(handles=[
        Line2D([0], [0], marker="o", linestyle="none", color=OKABE_ITO["vermillion"], label="CP"),
        Line2D([0], [0], marker="s", linestyle="none", color=OKABE_ITO["blue"], label="Fixed"),
    ], loc="lower right")
    panel_label(risk_ax, "(b)", dx=-0.20, dy=1.11)

    _draw_intervention(intervention_ax, source, revision)
    intervention_ax.set_title("Selective intervention", loc="left", pad=5)
    intervention_ax.legend(handles=[
        Line2D([0], [0], marker="o", linestyle="none", color=OKABE_ITO["green"], label="Net reduction"),
        Line2D([0], [0], marker="s", linestyle="none", color=OKABE_ITO["vermillion"], label="Incorrect intervention (negative)"),
    ], loc="lower left")
    panel_label(intervention_ax, "(c)", dx=-0.25, dy=1.11)

    fig.subplots_adjust(left=0.08, right=0.985, top=0.84, bottom=0.20)
    save_figure(fig, "figure8", cfg.path("figures"))
    return True
