"""Figure 6: unified residual-gate decision and outcome profile."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory

from ..config import Config
from ..utils import get_logger
from .style import (
    ANNOTATION_SIZE,
    BACKBONE_LABELS,
    CAPSIZE,
    LINE_WIDTHS,
    MARKER_SIZES,
    NEUTRAL_COLORS,
    OKABE_ITO,
    apply_style,
    clean_axis,
    figure_size,
    save_figure,
)

LOG = get_logger("hmqr.fig6")
BACKBONE_ORDER = ("llava15", "instructblip")
STRATEGY_ORDER = ("full", "revise-all (no gate)", "abstain-all")
STRATEGY_STYLE = {
    "full": ("Full", OKABE_ITO["vermillion"], "o"),
    "revise-all (no gate)": ("No gate", OKABE_ITO["blue"], "s"),
    "abstain-all": ("Abstain all", NEUTRAL_COLORS["mid"], "^"),
}
ACTION_STYLE = {
    "Keep": OKABE_ITO["skyblue"],
    "Revise": OKABE_ITO["orange"],
    "Abstain": NEUTRAL_COLORS["light"],
}
ACTION_MARKERS = {"Keep": "o", "Revise": "s", "Abstain": "^"}
ROW_POSITIONS = np.asarray([6.0, 5.0, 4.0, 2.6, 1.6, 0.6])


def _pooled_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Use joint image-level bootstrap rows; never average dataset intervals."""
    if "plot_scope" in frame:
        pooled = frame[frame["plot_scope"].astype(str).eq("pooled")].copy()
        if not pooled.empty:
            return pooled
    raise ValueError("Figure 6 requires plot_scope='pooled' rows from Step 09")


def _rate(row: pd.Series, name: str) -> float:
    value = float(row.get(name, np.nan))
    return value * 100.0 if np.isfinite(value) else np.nan


def _entry(row: pd.Series, backbone: str) -> dict:
    label, colour, marker = STRATEGY_STYLE[str(row["strategy"])]
    net = _rate(row, "net_hallucination_reduction")
    incorrect = _rate(row, "incorrect_intervention_rate")
    net_low, net_high = _rate(row, "ci_low"), _rate(row, "ci_high")
    incorrect_low = _rate(row, "incorrect_intervention_ci_low")
    incorrect_high = _rate(row, "incorrect_intervention_ci_high")
    return {
        "strategy": str(row["strategy"]),
        "backbone": backbone,
        "label": label,
        "colour": colour,
        "marker": marker,
        "target": _rate(row, "target_hallucination_removal_rate"),
        "incorrect": incorrect,
        "incorrect_low": incorrect if not np.isfinite(incorrect_low) else incorrect_low,
        "incorrect_high": incorrect if not np.isfinite(incorrect_high) else incorrect_high,
        "net": net,
        "net_low": net if not np.isfinite(net_low) else net_low,
        "net_high": net if not np.isfinite(net_high) else net_high,
        "keep": 100.0 * float(row["n_keep"]) / max(float(row["n_claims"]), 1.0),
        "revise": _rate(row, "revision_rate"),
        "abstain": _rate(row, "abstention_rate"),
    }


def _ordered(entries: list[dict]) -> list[dict]:
    by_key = {(item["backbone"], item["strategy"]): item for item in entries}
    return [by_key[(backbone, strategy)] for backbone in BACKBONE_ORDER for strategy in STRATEGY_ORDER]


def _format_rows(ax, ordered: list[dict], *, labels: bool) -> None:
    ax.set_ylim(0.0, 6.65)
    ax.set_yticks(ROW_POSITIONS)
    if labels:
        ax.set_yticklabels([
            f"{BACKBONE_LABELS.get(item['backbone'], item['backbone'])} · {item['label']}"
            for item in ordered
        ])
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)
    ax.axhline(3.3, color=NEUTRAL_COLORS["light"], linewidth=0.8, zorder=0)
    clean_axis(ax, grid="x")


def _draw_redistribution(ax, entries: list[dict]) -> None:
    """Show what the residual gate changes, not two hard-to-compare stacks."""
    lookup = {(item["backbone"], item["strategy"]): item for item in entries}
    action_rows = (("keep", "Keep"), ("revise", "Revise"), ("abstain", "Abstain"))
    positions = {
        "llava15": (5.7, 4.7, 3.7),
        "instructblip": (1.9, 0.9, -0.1),
    }
    transform = blended_transform_factory(ax.transAxes, ax.transData)
    for backbone in BACKBONE_ORDER:
        no_gate = lookup[(backbone, "revise-all (no gate)")]
        full = lookup[(backbone, "full")]
        for (key, label), y in zip(action_rows, positions[backbone]):
            start, end = float(no_gate[key]), float(full[key])
            colour = ACTION_STYLE[label]
            ax.annotate(
                "", xy=(end, y), xytext=(start, y),
                arrowprops={
                    "arrowstyle": "-|>", "color": colour,
                    "linewidth": LINE_WIDTHS["emphasis"],
                    "mutation_scale": 8.0, "shrinkA": 4.5, "shrinkB": 4.5,
                },
                zorder=2,
            )
            ax.scatter(
                start, y, s=MARKER_SIZES["scatter"], marker=ACTION_MARKERS[label],
                facecolor=NEUTRAL_COLORS["white"], edgecolor=colour,
                linewidth=LINE_WIDTHS["emphasis"], zorder=3,
            )
            ax.scatter(
                end, y, s=MARKER_SIZES["scatter"], marker=ACTION_MARKERS[label],
                facecolor=colour, edgecolor=NEUTRAL_COLORS["white"],
                linewidth=LINE_WIDTHS["edge"], zorder=4,
            )
            delta = end - start
            midpoint = (start + end) / 2.0
            ax.text(
                midpoint, y + 0.27, f"{delta:+.0f} pp",
                ha="center", va="bottom", fontsize=ANNOTATION_SIZE,
                color=NEUTRAL_COLORS["dark"],
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.4},
                zorder=5,
            )
        group_y = positions[backbone][0] + 0.55
        ax.text(
            -0.02, group_y, BACKBONE_LABELS.get(backbone, backbone),
            transform=transform, ha="right", va="bottom", fontweight="bold",
            clip_on=False,
        )
    ax.axhline(2.8, color=NEUTRAL_COLORS["light"], linewidth=0.8, zorder=0)
    ax.set_ylim(-0.7, 6.55)
    ax.set_yticks([*positions["llava15"], *positions["instructblip"]])
    ax.set_yticklabels(["Keep", "Revise", "Abstain"] * 2)
    ax.set_xlim(-2.0, 102.0)
    ax.set_xticks([0, 25, 50, 75, 100])
    clean_axis(ax, grid="x")
    ax.set_title("Gate-induced decision redistribution", loc="left", pad=6)
    ax.set_xlabel("Share of claims (%)")
    state_handles = [
        Line2D(
            [0], [0], marker="o", linestyle="none", color=NEUTRAL_COLORS["dark"],
            markerfacecolor=NEUTRAL_COLORS["white"],
            markeredgecolor=NEUTRAL_COLORS["dark"], label="No gate",
        ),
        Line2D(
            [0], [0], marker="o", linestyle="none", color=NEUTRAL_COLORS["dark"],
            markerfacecolor=NEUTRAL_COLORS["dark"],
            markeredgecolor=NEUTRAL_COLORS["white"], label="Full",
        ),
    ]
    ax.legend(
        handles=state_handles, loc="lower right", ncol=2,
        bbox_to_anchor=(1.0, 1.005), borderaxespad=0.0,
        handletextpad=0.3, columnspacing=0.8,
    )


def _draw_rate(ax, ordered: list[dict], metric: str, title: str, *, interval: bool = False) -> None:
    _format_rows(ax, ordered, labels=False)
    values = [float(item[metric]) for item in ordered]
    for y, item, value in zip(ROW_POSITIONS, ordered, values):
        kwargs = {}
        if interval:
            low, high = float(item[f"{metric}_low"]), float(item[f"{metric}_high"])
            kwargs["xerr"] = [[max(0.0, value - low)], [max(0.0, high - value)]]
        ax.errorbar(
            value, y,
            fmt=item["marker"], markersize=MARKER_SIZES["summary"],
            markerfacecolor=item["colour"], markeredgecolor=NEUTRAL_COLORS["white"],
            markeredgewidth=LINE_WIDTHS["edge"], ecolor=item["colour"],
            elinewidth=LINE_WIDTHS["error"], capsize=CAPSIZE if interval else 0,
            zorder=3, **kwargs,
        )
    finite = np.asarray(values)[np.isfinite(values)]
    if metric in {"target", "incorrect"}:
        upper_candidates = list(finite)
        if interval:
            upper_candidates.extend(float(item[f"{metric}_high"]) for item in ordered)
        upper = max(10.0, float(np.nanmax(upper_candidates)))
        right = min(100.0, np.ceil((upper + 6.0) / 10.0) * 10.0)
        ax.set_xlim(-2.5, right)
        ticks = np.linspace(0, right, 3)
        ax.set_xticks(ticks)
    else:
        lows = [float(item["net_low"]) for item in ordered]
        highs = [float(item["net_high"]) for item in ordered]
        minimum, maximum = min(lows + [0.0]), max(highs + [0.0])
        span = max(maximum - minimum, 1.0)
        ax.set_xlim(minimum - 0.06 * span, maximum + 0.06 * span)
        ax.axvline(0, color=NEUTRAL_COLORS["mid"], linewidth=LINE_WIDTHS["reference"], linestyle=(0, (2, 2.5)), zorder=1)
    ax.set_title(title, loc="center", pad=6)
    ax.set_xlabel("Rate (%)" if metric != "net" else "Net change (pp)")


def make(cfg: Config) -> bool:
    path = cfg.path("tables") / "rq2_gate.csv"
    if not path.exists():
        LOG.error("missing %s -- run scripts/09_rq2_gate.py first", path)
        return False
    pooled = _pooled_rows(pd.read_csv(path))
    expected = {(backbone, strategy) for backbone in BACKBONE_ORDER for strategy in STRATEGY_ORDER}
    observed = set(pooled[["backbone", "strategy"]].astype(str).itertuples(index=False, name=None))
    if observed != expected:
        LOG.error("Figure 6 requires the complete two-backbone by three-strategy matrix")
        return False
    entries = [
        _entry(
            pooled[
                pooled["backbone"].astype(str).eq(backbone)
                & pooled["strategy"].astype(str).eq(strategy)
            ].iloc[0],
            backbone,
        )
        for backbone in BACKBONE_ORDER for strategy in STRATEGY_ORDER
    ]
    ordered = _ordered(entries)

    apply_style()
    fig = plt.figure(figsize=figure_size("figure6"))
    grid = fig.add_gridspec(1, 4, width_ratios=(2.35, 0.88, 0.98, 1.15), wspace=0.16)
    composition_ax = fig.add_subplot(grid[0, 0])
    target_ax = fig.add_subplot(grid[0, 1])
    incorrect_ax = fig.add_subplot(grid[0, 2])
    net_ax = fig.add_subplot(grid[0, 3])
    _draw_redistribution(composition_ax, entries)
    _draw_rate(target_ax, ordered, "target", "Target removal")
    _draw_rate(incorrect_ax, ordered, "incorrect", "Incorrect interv.", interval=True)
    _draw_rate(net_ax, ordered, "net", "Net reduction", interval=True)

    strategy_handles = [Line2D(
        [0], [0], marker=STRATEGY_STYLE[name][2], linestyle="none",
        markerfacecolor=STRATEGY_STYLE[name][1], markeredgecolor=NEUTRAL_COLORS["white"],
        color=STRATEGY_STYLE[name][1], markersize=MARKER_SIZES["summary"],
        label=STRATEGY_STYLE[name][0],
    ) for name in STRATEGY_ORDER]
    fig.legend(
        handles=strategy_handles,
        loc="upper center", ncol=3, bbox_to_anchor=(0.77, 0.985),
        handletextpad=0.35, columnspacing=1.1,
    )
    fig.subplots_adjust(left=0.125, right=0.985, top=0.79, bottom=0.19)
    save_figure(fig, "figure6", cfg.path("figures"))
    return True
