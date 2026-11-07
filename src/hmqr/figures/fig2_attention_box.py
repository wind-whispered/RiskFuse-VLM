"""Figure 2: violin distributions of visual-attention enrichment."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from ..config import Config
from ..utils import get_logger
from .style import (
    ANNOTATION_SIZE,
    BACKBONE_LABELS,
    CLASS_COLORS,
    LINE_WIDTHS,
    NEUTRAL_COLORS,
    apply_style,
    clean_axis,
    figure_size,
    panel_label,
    save_figure,
)

LOG = get_logger("hmqr.fig2")

GROUPS = (
    ("attr", 1, 0.0, "Attr.\nhalluc."),
    ("attr", 0, 1.0, "Attr.\ngrounded"),
    ("rel", 1, 2.7, "Rel.\nhalluc."),
    ("rel", 0, 3.7, "Rel.\ngrounded"),
)


def _effect_text(summary: pd.DataFrame, backbone: str) -> str:
    required = {"effect_size", "effect_ci_low", "effect_ci_high"}
    if not required.issubset(summary.columns):
        return ""
    fragments = []
    for claim_type, label in (("attr", "Attr."), ("rel", "Rel.")):
        row = summary[
            (summary["backbone"] == backbone)
            & (summary["claim_type"] == claim_type)
        ]
        if row.empty:
            continue
        first = row.iloc[0]
        fragments.append(
            f"{label} d={float(first['effect_size']):+.2f} "
            f"[{float(first['effect_ci_low']):+.2f}, {float(first['effect_ci_high']):+.2f}]"
        )
    return "Effect size (95% CI)\n" + "   ".join(fragments) if fragments else ""


def make(cfg: Config) -> bool:
    values_path = cfg.path("tables") / "attention_vattn.csv"
    summary_path = cfg.path("tables") / "attention_summary.csv"
    if not values_path.exists() or not summary_path.exists():
        LOG.error("missing Figure 2 inputs -- run scripts/04_attention.py first")
        return False

    values = pd.read_csv(values_path)
    summary = pd.read_csv(summary_path)
    backbones = [name for name in cfg.backbone_names if name in set(values["backbone"])]
    if not backbones:
        LOG.error("no backbones in %s", values_path.name)
        return False

    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=figure_size("figure2"), sharex=True)
    axes = axes.ravel()

    for index, (ax, backbone) in enumerate(zip(axes, backbones)):
        panel_arrays = []
        for claim_type, label_value, position, _ in GROUPS:
            data = values[
                (values["backbone"] == backbone)
                & (values["claim_type"] == claim_type)
                & (values["label"] == label_value)
            ]["vattn_enrichment"].dropna().to_numpy(float)
            if not data.size:
                continue
            panel_arrays.append(data)
            colour = CLASS_COLORS["attribute" if claim_type == "attr" else "relation"]
            grounded = label_value == 0
            artists = ax.violinplot(
                [data], positions=[position], widths=0.78,
                showmeans=False, showmedians=False, showextrema=False,
                points=120, bw_method=0.25,
            )
            body = artists["bodies"][0]
            body.set_facecolor(NEUTRAL_COLORS["white"] if grounded else colour)
            body.set_edgecolor(colour)
            body.set_linewidth(LINE_WIDTHS["data"])
            body.set_alpha(1.0 if grounded else 0.62)
            body.set_hatch("////" if grounded else "")
            q1, median, q3 = np.percentile(data, [25, 50, 75])
            ax.plot(
                [position, position], [q1, q3],
                color=NEUTRAL_COLORS["black"],
                linewidth=LINE_WIDTHS["emphasis"], zorder=4,
            )
            ax.scatter(
                [position], [median], s=8.0,
                facecolor=NEUTRAL_COLORS["white"],
                edgecolor=NEUTRAL_COLORS["black"],
                linewidth=LINE_WIDTHS["edge"], zorder=5,
            )

        if panel_arrays:
            pooled = np.concatenate(panel_arrays)
            low, high = np.percentile(pooled, [0.2, 99.8])
            span = max(high - low, 1e-3)
            ax.set_ylim(low - 0.12 * span, high + 0.26 * span)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

        effect = _effect_text(summary, backbone)
        if effect:
            ax.text(
                0.02,
                0.985,
                effect,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=ANNOTATION_SIZE,
                color=NEUTRAL_COLORS["dark"],
                linespacing=1.25,
            )

        ax.set_xlim(-0.65, 4.35)
        ax.set_xticks([group[2] for group in GROUPS])
        ax.set_xticklabels([group[3] for group in GROUPS])
        ax.set_title(BACKBONE_LABELS.get(backbone, backbone), loc="left", pad=5)
        clean_axis(ax, grid="y")
        panel_label(ax, f"({chr(ord('a') + index)})", dx=-0.11, dy=1.10)

    for ax in axes[len(backbones):]:
        ax.axis("off")

    legend_handles = [
        Patch(
            facecolor=CLASS_COLORS["attribute"],
            edgecolor=CLASS_COLORS["attribute"],
            label="Attribute hallucinated",
        ),
        Patch(
            facecolor=NEUTRAL_COLORS["white"],
            edgecolor=CLASS_COLORS["attribute"],
            hatch="////",
            label="Attribute grounded",
        ),
        Patch(
            facecolor=CLASS_COLORS["relation"],
            edgecolor=CLASS_COLORS["relation"],
            label="Relation hallucinated",
        ),
        Patch(
            facecolor=NEUTRAL_COLORS["white"],
            edgecolor=CLASS_COLORS["relation"],
            hatch="////",
            label="Relation grounded",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.995),
    )
    fig.supylabel("Visual-attention enrichment", x=0.018)
    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.12, top=0.90, wspace=0.20, hspace=0.30)
    save_figure(fig, "figure2", cfg.path("figures"))
    return True
