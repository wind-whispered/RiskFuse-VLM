"""Figure 1: layer-wise probing AUROC for object, attribute, and relation claims."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator

from ..config import Config
from ..utils import get_logger
from .style import (
    BACKBONE_LABELS,
    CLASS_COLORS,
    CLASS_MARKERS,
    CLASS_STYLES,
    LINE_WIDTHS,
    NEUTRAL_COLORS,
    apply_style,
    clean_axis,
    figure_size,
    panel_label,
    save_figure,
)

LOG = get_logger("hmqr.fig1")
CLASS_LABELS = {"object": "Object", "attribute": "Attribute", "relation": "Relation"}


def make(cfg: Config) -> bool:
    curve_path = cfg.path("tables") / "layer_probe_auroc.csv"
    peak_path = cfg.path("tables") / "probe_peak_layers.csv"
    if not curve_path.exists() or not peak_path.exists():
        LOG.error("missing Figure 1 inputs -- run scripts/03_layer_probe.py first")
        return False

    curves = pd.read_csv(curve_path)
    peaks = pd.read_csv(peak_path)
    backbones = [name for name in cfg.backbone_names if name in set(curves["backbone"])]
    if not backbones:
        LOG.error("no backbones present in %s", curve_path.name)
        return False

    apply_style()
    fig, axes = plt.subplots(
        2,
        2,
        figsize=figure_size("figure1"),
        sharex=False,
        sharey=True,
    )
    axes = axes.ravel()

    for index, (ax, backbone) in enumerate(zip(axes, backbones)):
        subset = curves[curves["backbone"] == backbone]
        for claim_class in ("object", "attribute", "relation"):
            rows = subset[subset["class"] == claim_class].sort_values("layer")
            if rows.empty:
                continue
            ax.plot(
                rows["layer"],
                rows["auroc"],
                color=CLASS_COLORS[claim_class],
                linestyle=CLASS_STYLES[claim_class],
                marker=CLASS_MARKERS[claim_class],
                markerfacecolor=NEUTRAL_COLORS["white"],
                markeredgecolor=CLASS_COLORS[claim_class],
                markeredgewidth=0.65,
                markevery=max(1, len(rows) // 7),
                linewidth=(
                    LINE_WIDTHS["emphasis"]
                    if claim_class == "attribute"
                    else LINE_WIDTHS["data"]
                ),
                label=CLASS_LABELS[claim_class],
                zorder=3,
            )

        peak_rows = peaks[peaks["backbone"] == backbone]
        for claim_class, text_y, prefix in (
            ("attribute", 0.515, "A"),
            ("relation", 0.548, "R"),
        ):
            row = peak_rows[peak_rows["class"] == claim_class]
            if row.empty:
                continue
            layer = int(row["peak_layer"].iloc[0])
            ax.axvline(
                layer,
                color=CLASS_COLORS[claim_class],
                linestyle=(0, (2.0, 2.2)),
                linewidth=LINE_WIDTHS["reference"],
                alpha=0.82,
                zorder=1,
            )
            ax.text(
                layer,
                text_y,
                f"{prefix} L{layer}",
                color=CLASS_COLORS[claim_class],
                fontsize=6.2,
                ha="center",
                va="bottom",
                bbox={
                    "facecolor": NEUTRAL_COLORS["white"],
                    "edgecolor": "none",
                    "pad": 0.6,
                    "alpha": 0.88,
                },
                zorder=4,
            )

        ax.axhline(
            0.5,
            color=NEUTRAL_COLORS["mid"],
            linewidth=LINE_WIDTHS["reference"],
            linestyle=(0, (2.0, 2.5)),
            zorder=1,
        )
        ax.set_ylim(0.5, 1.0)
        ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        ax.set_title(BACKBONE_LABELS.get(backbone, backbone), loc="left", pad=5)
        clean_axis(ax, grid="y")
        panel_label(ax, f"({chr(ord('a') + index)})", dx=-0.11, dy=1.10)

    for ax in axes[len(backbones):]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.995),
    )
    fig.supxlabel("Transformer layer index", y=0.025)
    fig.supylabel("Probe AUROC", x=0.018)
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.12, top=0.90, wspace=0.18, hspace=0.29)
    save_figure(fig, "figure1", cfg.path("figures"))
    return True
