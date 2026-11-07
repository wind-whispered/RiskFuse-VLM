"""Figure 3: residual non-redundancy, label alignment, and task utility."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from ..config import Config
from ..utils import get_logger
from .style import (
    ALPHA,
    CAPSIZE,
    CLASS_COLORS,
    LINE_WIDTHS,
    MARKER_SIZES,
    NEUTRAL_COLORS,
    apply_style,
    clean_axis,
    figure_size,
    panel_label,
    save_figure,
)

LOG = get_logger("hmqr.fig3")
CLAIM_META = {
    "attr": ("Attribute", CLASS_COLORS["attribute"]),
    "rel": ("Relation", CLASS_COLORS["relation"]),
}
SIGNAL_META = {
    "matched": ("Granularity matched", "o"),
    "global": ("Global CLIP", "s"),
}


def _signal_group(series: pd.Series) -> pd.Series:
    return np.where(series.astype(str).eq("g_global"), "global", "matched")


def _reference_lines(ax, *, vertical: float = 0.0) -> None:
    ax.axvline(vertical, color=NEUTRAL_COLORS["mid"], linewidth=LINE_WIDTHS["reference"], linestyle=(0, (2, 2.5)), zorder=0)
    ax.axhline(0, color=NEUTRAL_COLORS["mid"], linewidth=LINE_WIDTHS["reference"], linestyle=(0, (2, 2.5)), zorder=0)


def make(cfg: Config) -> bool:
    gain_path = cfg.path("tables") / "residual_signal_gain.csv"
    stats_path = cfg.path("tables") / "residual_stats.csv"
    if not gain_path.exists() or not stats_path.exists():
        LOG.error("missing Figure 3 inputs -- run scripts/05_residual.py first")
        return False
    gain = pd.read_csv(gain_path)
    stats = pd.read_csv(stats_path)
    if gain.empty or stats.empty:
        LOG.error("residual tables are empty")
        return False
    gain["signal_group"] = _signal_group(gain["signal"])
    stats["signal_group"] = _signal_group(stats["signal"])
    stats["residual_variance_fraction"] = (
        pd.to_numeric(stats["var_g_perp"], errors="coerce")
        / pd.to_numeric(stats["var_g"], errors="coerce").clip(lower=1e-12)
    )

    apply_style()
    fig, (ax_gain, ax_alignment) = plt.subplots(
        1, 2, figsize=figure_size("figure3"), gridspec_kw={"width_ratios": [1.05, 0.95]},
    )

    for claim_type, (_, colour) in CLAIM_META.items():
        for signal_group, (_, marker) in SIGNAL_META.items():
            rows = gain[(gain["claim_type"] == claim_type) & (gain["signal_group"] == signal_group)]
            if rows.empty:
                continue
            y = rows["auroc_gain"].to_numpy(float) * 100.0
            yerr = np.vstack([
                np.clip(rows["auroc_gain"].to_numpy(float) - rows["gain_ci_low"].to_numpy(float), 0, None) * 100.0,
                np.clip(rows["gain_ci_high"].to_numpy(float) - rows["auroc_gain"].to_numpy(float), 0, None) * 100.0,
            ])
            ax_gain.errorbar(
                rows["rho"], y, yerr=yerr, fmt=marker,
                markersize=MARKER_SIZES["summary"],
                markerfacecolor=colour if signal_group == "matched" else NEUTRAL_COLORS["white"],
                markeredgecolor=colour, markeredgewidth=LINE_WIDTHS["edge"],
                ecolor=colour, elinewidth=LINE_WIDTHS["error"], capsize=CAPSIZE,
                alpha=0.78, zorder=2,
            )
    _reference_lines(ax_gain)
    ax_gain.set_xlabel(r"Signal-belief correlation, $\rho$")
    ax_gain.set_ylabel("Cross-fitted AUROC gain (pp)")
    ax_gain.set_title("Incremental detection utility", loc="left", pad=5)
    ax_gain.margins(x=0.08, y=0.12)
    clean_axis(ax_gain, grid="y")
    panel_label(ax_gain, "(a)", dx=-0.13, dy=1.10)

    for claim_type, (_, colour) in CLAIM_META.items():
        for signal_group, (_, marker) in SIGNAL_META.items():
            rows = stats[(stats["claim_type"] == claim_type) & (stats["signal_group"] == signal_group)].dropna(
                subset=["residual_variance_fraction", "residual_label_partial_corr"]
            )
            if rows.empty:
                continue
            face = colour if signal_group == "matched" else NEUTRAL_COLORS["white"]
            ax_alignment.scatter(
                rows["residual_variance_fraction"], rows["residual_label_partial_corr"],
                s=MARKER_SIZES["small_scatter"], marker=marker,
                facecolor=face, edgecolor=colour, linewidth=LINE_WIDTHS["edge"],
                alpha=ALPHA["points"], zorder=2,
            )
            ax_alignment.scatter(
                [rows["residual_variance_fraction"].mean()],
                [rows["residual_label_partial_corr"].mean()],
                s=MARKER_SIZES["scatter"] * 1.15, marker=marker,
                facecolor=face, edgecolor=colour, linewidth=LINE_WIDTHS["data"],
                zorder=4,
            )
    _reference_lines(ax_alignment, vertical=1.0)
    ax_alignment.set_xlabel(r"Residual variance fraction, $\mathrm{Var}(g_\perp)/\mathrm{Var}(g)$")
    ax_alignment.set_ylabel("Residual-label partial correlation")
    ax_alignment.set_title("Non-redundancy versus label alignment", loc="left", pad=5)
    clean_axis(ax_alignment, grid="y")
    panel_label(ax_alignment, "(b)", dx=-0.14, dy=1.10)

    handles = [
        Line2D(
            [0], [0], marker=marker, linestyle="none",
            markerfacecolor=colour if signal_group == "matched" else NEUTRAL_COLORS["white"],
            markeredgecolor=colour, markeredgewidth=LINE_WIDTHS["edge"],
            color=colour, label=f"{claim_label}, {signal_label}",
        )
        for claim_type, (claim_label, colour) in CLAIM_META.items()
        for signal_group, (signal_label, marker) in SIGNAL_META.items()
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.52, 0.995), columnspacing=1.0)
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.20, top=0.78, wspace=0.32)
    save_figure(fig, "figure3", cfg.path("figures"))
    return True
