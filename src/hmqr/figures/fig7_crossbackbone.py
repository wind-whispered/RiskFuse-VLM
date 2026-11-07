"""Figure 7: unified cross-backbone forest-grid for outcomes and operating rates."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from ..config import Config
from ..utils import get_logger
from .style import (
    BACKBONE_COLORS,
    BACKBONE_LABELS,
    CAPSIZE,
    DATASET_LABELS,
    LINE_WIDTHS,
    MARKER_SIZES,
    NEUTRAL_COLORS,
    apply_style,
    clean_axis,
    figure_size,
    save_figure,
)

LOG = get_logger("hmqr.fig7")
DATASET_ORDER = ("amber_attr", "amber_rel", "rbench", "mmrel", "chair500")
DISCRIMINATIVE_DATASETS = set(DATASET_ORDER[:-1])
BACKBONE_MARKERS = {"llava15": "o", "llava16": "s", "instructblip": "^", "qwenvl": "D"}
BACKBONE_OFFSETS = (0.27, 0.09, -0.09, -0.27)
RATE_SPECS = (
    ("selection_risk", "Selection risk", "Risk (%)"),
    ("abstention_rate", "Abstention", "Abstention (%)"),
    ("signal_unavailable_rate", "Unavailable", "Unavailable (%)"),
)


def _finite_or(value: float, fallback: float) -> float:
    return float(value) if np.isfinite(value) else float(fallback)


def _outcome_interval(row: pd.Series) -> tuple[float, float, float]:
    """Return an improvement-oriented outcome and interval in percentage points."""
    if str(row["dataset"]) == "chair500":
        value = -float(row["object_hallucination_change"])
        low = -float(row.get("ci_high", np.nan))
        high = -float(row.get("ci_low", np.nan))
    else:
        value = float(row["net_hallucination_reduction"])
        low = float(row.get("ci_low", np.nan))
        high = float(row.get("ci_high", np.nan))
    low = _finite_or(low, value)
    high = _finite_or(high, value)
    return value * 100.0, low * 100.0, high * 100.0


def _row(frame: pd.DataFrame, backbone: str, dataset: str) -> pd.Series:
    hit = frame[
        frame["backbone"].astype(str).eq(backbone)
        & frame["dataset"].astype(str).eq(dataset)
    ]
    if len(hit) != 1:
        raise ValueError(f"Figure 7 expected one {backbone}/{dataset} row, found {len(hit)}")
    return hit.iloc[0]


def _format_common_axis(ax, positions: np.ndarray, *, show_labels: bool) -> None:
    for stripe, y in enumerate(positions):
        if stripe % 2 == 0:
            ax.axhspan(y - 0.48, y + 0.48, color=NEUTRAL_COLORS["near_white"], zorder=0)
    ax.set_ylim(-0.5, len(DATASET_ORDER) - 0.5)
    ax.set_yticks(positions)
    ax.set_yticklabels([DATASET_LABELS[name] for name in DATASET_ORDER] if show_labels else [])
    ax.tick_params(axis="y", length=0 if not show_labels else 3, pad=5)
    clean_axis(ax, grid="x")


def _draw_outcomes(ax, frame: pd.DataFrame, backbones: list[str], limits) -> None:
    positions = np.arange(len(DATASET_ORDER))[::-1]
    _format_common_axis(ax, positions, show_labels=True)
    for index, backbone in enumerate(backbones):
        colour = BACKBONE_COLORS[backbone]
        marker = BACKBONE_MARKERS[backbone]
        for base_y, dataset in zip(positions, DATASET_ORDER):
            value, low, high = _outcome_interval(_row(frame, backbone, dataset))
            ax.errorbar(
                value,
                base_y + BACKBONE_OFFSETS[index],
                xerr=[[max(0.0, value - low)], [max(0.0, high - value)]],
                fmt=marker,
                markersize=MARKER_SIZES["summary"],
                markerfacecolor=colour,
                markeredgecolor=NEUTRAL_COLORS["white"],
                markeredgewidth=LINE_WIDTHS["edge"],
                ecolor=colour,
                elinewidth=LINE_WIDTHS["error"],
                capsize=CAPSIZE,
                zorder=3,
            )
    ax.axvline(0, color=NEUTRAL_COLORS["mid"], linewidth=LINE_WIDTHS["reference"], linestyle=(0, (2, 2.5)), zorder=1)
    ax.set_xlim(*limits)
    ax.set_title("Outcome improvement", loc="left", pad=6)
    ax.set_xlabel("Net change (percentage points)")


def _draw_rate_axis(ax, frame: pd.DataFrame, backbones: list[str], metric: str, title: str, xlabel: str) -> None:
    positions = np.arange(len(DATASET_ORDER))[::-1]
    _format_common_axis(ax, positions, show_labels=False)
    for index, backbone in enumerate(backbones):
        for base_y, dataset in zip(positions, DATASET_ORDER):
            if dataset not in DISCRIMINATIVE_DATASETS:
                continue
            value = float(_row(frame, backbone, dataset).get(metric, np.nan))
            if not np.isfinite(value):
                continue
            ax.plot(
                100.0 * value,
                base_y + BACKBONE_OFFSETS[index],
                marker=BACKBONE_MARKERS[backbone],
                linestyle="none",
                markersize=MARKER_SIZES["summary"],
                markerfacecolor=BACKBONE_COLORS[backbone],
                markeredgecolor=NEUTRAL_COLORS["white"],
                markeredgewidth=LINE_WIDTHS["edge"],
                zorder=3,
            )
    ax.text(50, 0, "-", ha="center", va="center", fontsize=7.0, color=NEUTRAL_COLORS["mid"])
    ax.set_xlim(-4, 104)
    ax.set_xticks([0, 50, 100])
    ax.set_title(title, loc="center", pad=6)
    ax.set_xlabel(xlabel)


def make(cfg: Config) -> bool:
    path = cfg.path("tables") / "rq3_reduction.csv"
    if not path.exists():
        LOG.error("missing %s -- run scripts/10_rq3_crossbackbone.py first", path)
        return False
    frame = pd.read_csv(path)
    backbones = [name for name in cfg.backbone_names if name in set(frame["backbone"])]
    expected = {(backbone, dataset) for backbone in backbones for dataset in DATASET_ORDER}
    observed = set(frame[["backbone", "dataset"]].astype(str).itertuples(index=False, name=None))
    if len(backbones) != 4 or observed != expected:
        LOG.error("Figure 7 requires a complete four-backbone by five-dataset matrix")
        return False

    bounds = [number for _, row in frame.iterrows() for number in _outcome_interval(row)]
    finite = np.asarray(bounds)[np.isfinite(bounds)]
    x_min, x_max = min(0.0, float(finite.min())), max(0.0, float(finite.max()))
    span = max(x_max - x_min, 1.0)
    limits = (x_min - max(1.3, 0.07 * span), x_max + max(1.3, 0.07 * span))

    apply_style()
    fig = plt.figure(figsize=figure_size("figure7"))
    grid = fig.add_gridspec(1, 4, width_ratios=(2.8, 1.0, 1.0, 1.0), wspace=0.12)
    outcome_ax = fig.add_subplot(grid[0, 0])
    rate_axes = [fig.add_subplot(grid[0, index]) for index in range(1, 4)]
    _draw_outcomes(outcome_ax, frame, backbones, limits)
    for ax, (metric, title, xlabel) in zip(rate_axes, RATE_SPECS):
        _draw_rate_axis(ax, frame, backbones, metric, title, xlabel)

    handles = [Line2D(
        [0], [0], marker=BACKBONE_MARKERS[name], linestyle="none",
        markerfacecolor=BACKBONE_COLORS[name], markeredgecolor=NEUTRAL_COLORS["white"],
        color=BACKBONE_COLORS[name], markersize=MARKER_SIZES["summary"],
        label=BACKBONE_LABELS.get(name, name),
    ) for name in backbones]
    handles.append(Line2D([0], [0], color=NEUTRAL_COLORS["dark"], linewidth=LINE_WIDTHS["error"], marker="|", markersize=6, label="Image-level 95% CI"))
    fig.legend(handles=handles, loc="upper center", ncol=5, bbox_to_anchor=(0.55, 0.995), handletextpad=0.35, columnspacing=1.15)
    fig.subplots_adjust(left=0.16, right=0.985, top=0.78, bottom=0.18)
    save_figure(fig, "figure7", cfg.path("figures"))
    return True
