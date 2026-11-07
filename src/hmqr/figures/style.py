"""Central publication style for every HMQR figure and standalone table.

This is the only file that should be edited for global appearance changes.
Canvas dimensions, typography, colours, strokes, markers, spacing, and export
settings are intentionally collected here so that all seven figures remain
visually coherent.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib as mpl
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# User-facing controls: canvas and export
# ---------------------------------------------------------------------------
MM_PER_INCH = 25.4
FIGURE_WIDTHS_MM = {
    "large": 183.0,
    "medium": 140.0,
    "small": 89.0,
}

# Figure 1/2 are mechanism grids; Figures 6/7 are unified forest grids and
# Figure 8 is a consolidated ablation matrix. These use the large width.
FIGURE_LAYOUT_MM = {
    "figure1": ("large", 104.0),
    "figure2": ("large", 108.0),
    "figure3": ("large", 72.0),
    "figure5": ("large", 70.0),
    "figure6": ("large", 78.0),
    "figure7": ("large", 78.0),
    "figure8": ("large", 72.0),
}

OUTPUT_DPI = 600
OUTPUT_FORMATS = ("pdf", "png")
OUTPUT_PAD_INCHES = 0.0
PNG_COMPRESS_LEVEL = 6

# ---------------------------------------------------------------------------
# User-facing controls: typography
# ---------------------------------------------------------------------------
FONT_FAMILY = "Arial"
FONT_FALLBACKS = ("Helvetica", "Liberation Sans", "Nimbus Sans", "DejaVu Sans")
FONT_SIZE = 6.5
AXIS_LABEL_SIZE = 7.0
TITLE_SIZE = 7.0
LEGEND_SIZE = 6.2
ANNOTATION_SIZE = 5.8
PANEL_LABEL_SIZE = 8.0

# ---------------------------------------------------------------------------
# User-facing controls: strokes, markers, spacing, and opacity
# ---------------------------------------------------------------------------
LINE_WIDTHS = {
    "axis": 0.50,
    "data": 1.00,
    "emphasis": 1.25,
    "reference": 0.55,
    "error": 0.65,
    "grid": 0.35,
    "edge": 0.45,
}
MARKER_SIZES = {
    "line": 3.0,
    "scatter": 20.0,
    "small_scatter": 8.0,
    "tradeoff": 6.2,
    "summary": 4.6,
}
ALPHA = {
    "solid": 0.92,
    "soft": 0.24,
    "points": 0.42,
    "connectors": 0.20,
    "grid": 0.90,
}
CAPSIZE = 2.0

# ---------------------------------------------------------------------------
# User-facing controls: one coherent colour system
# ---------------------------------------------------------------------------
# Okabe-Ito categorical colours. Large surfaces use low opacity; saturated
# colours are reserved for data marks and direct labels.
OKABE_ITO: Dict[str, str] = {
    "black": "#1F1F1F",
    "orange": "#E69F00",
    "skyblue": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
}

NEUTRAL_COLORS = {
    "black": "#202124",
    "dark": "#4B4F54",
    "mid": "#7A7F85",
    "light": "#D6D9DD",
    "grid": "#E7E9EC",
    "near_white": "#F7F8F9",
    "white": "#FFFFFF",
}

CLASS_COLORS: Dict[str, str] = {
    "object": OKABE_ITO["blue"],
    "attribute": OKABE_ITO["orange"],
    "relation": OKABE_ITO["green"],
}
CLASS_STYLES: Dict[str, object] = {
    "object": "-",
    "attribute": (0, (4.0, 2.0)),
    "relation": (0, (1.2, 1.8)),
}
CLASS_MARKERS: Dict[str, str] = {"object": "o", "attribute": "s", "relation": "^"}

METHOD_COLORS: Dict[str, str] = {
    "Ours": OKABE_ITO["vermillion"],
    "GlobalCLIP-thr": OKABE_ITO["blue"],
    "PredictiveEntropy": OKABE_ITO["green"],
    "SemanticEntropy": OKABE_ITO["purple"],
}
METHOD_MARKERS: Dict[str, str] = {
    "Ours": "o",
    "GlobalCLIP-thr": "s",
    "PredictiveEntropy": "^",
    "SemanticEntropy": "D",
}
METHOD_HATCH: Dict[str, str] = {
    "Ours": "",
    "GlobalCLIP-thr": "//",
    "PredictiveEntropy": "\\\\",
    "SemanticEntropy": "xx",
}

BACKBONE_COLORS: Dict[str, str] = {
    "llava15": OKABE_ITO["blue"],
    "llava16": OKABE_ITO["orange"],
    "instructblip": OKABE_ITO["green"],
    "qwenvl": OKABE_ITO["purple"],
}
BACKBONE_LABELS: Dict[str, str] = {
    "llava15": "LLaVA-1.5-7B",
    "llava16": "LLaVA-1.6-7B",
    "instructblip": "InstructBLIP-7B",
    "qwenvl": "Qwen-VL-Chat",
}
DATASET_LABELS: Dict[str, str] = {
    "amber_attr": "AMBER attribute",
    "amber_rel": "AMBER relation",
    "rbench": "R-Bench relation",
    "mmrel": "MMRel",
    "chair500": "CHAIR-500",
}

# Muted diverging endpoints for Figure 8. White is the exact stage reference.
HEATMAP_COLORS = {
    "negative": "#C95B51",
    "neutral": "#F7F7F7",
    "positive": "#3B6EA8",
}


def mm_to_inches(value_mm: float) -> float:
    return float(value_mm) / MM_PER_INCH


def figure_size(name: str) -> tuple[float, float]:
    """Return the exact configured canvas size in inches."""
    if name not in FIGURE_LAYOUT_MM:
        raise KeyError(f"No central figure size configured for {name!r}")
    size_class, height_mm = FIGURE_LAYOUT_MM[name]
    return mm_to_inches(FIGURE_WIDTHS_MM[size_class]), mm_to_inches(height_mm)


# Compatibility aliases used by the standalone table modules.
WIDTH_SINGLE = mm_to_inches(FIGURE_WIDTHS_MM["small"])
WIDTH_ONEHALF = mm_to_inches(FIGURE_WIDTHS_MM["medium"])
WIDTH_DOUBLE = mm_to_inches(FIGURE_WIDTHS_MM["large"])


def check_fonts() -> str:
    """Return Arial when installed, otherwise the closest available sans font."""
    from matplotlib import font_manager

    available = {font.name for font in font_manager.fontManager.ttflist}
    for preferred in (FONT_FAMILY, *FONT_FALLBACKS):
        if preferred in available:
            if preferred != FONT_FAMILY:
                print(f"[style] WARNING: {FONT_FAMILY} unavailable; using {preferred}.")
            return preferred
    return "DejaVu Sans"


def apply_style() -> None:
    """Apply the shared Nature-style plotting grammar."""
    font = check_fonts()
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [font, FONT_FAMILY, *FONT_FALLBACKS],
            "mathtext.fontset": "custom",
            "mathtext.rm": font,
            "mathtext.it": f"{font}:italic",
            "mathtext.bf": f"{font}:bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "pdf.compression": 9,
            "font.size": FONT_SIZE,
            "axes.titlesize": TITLE_SIZE,
            "axes.titleweight": "normal",
            "axes.labelsize": AXIS_LABEL_SIZE,
            "axes.linewidth": LINE_WIDTHS["axis"],
            "axes.edgecolor": NEUTRAL_COLORS["black"],
            "axes.labelcolor": NEUTRAL_COLORS["black"],
            "axes.titlecolor": NEUTRAL_COLORS["black"],
            "axes.axisbelow": True,
            "xtick.labelsize": FONT_SIZE,
            "ytick.labelsize": FONT_SIZE,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": LINE_WIDTHS["axis"],
            "ytick.major.width": LINE_WIDTHS["axis"],
            "xtick.major.size": 2.8,
            "ytick.major.size": 2.8,
            "xtick.color": NEUTRAL_COLORS["black"],
            "ytick.color": NEUTRAL_COLORS["black"],
            "lines.linewidth": LINE_WIDTHS["data"],
            "lines.markersize": MARKER_SIZES["line"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "grid.color": NEUTRAL_COLORS["grid"],
            "grid.linewidth": LINE_WIDTHS["grid"],
            "grid.linestyle": (0, (1.5, 2.2)),
            "legend.fontsize": LEGEND_SIZE,
            "legend.frameon": False,
            "legend.handlelength": 1.8,
            "legend.handletextpad": 0.45,
            "legend.columnspacing": 1.15,
            "text.color": NEUTRAL_COLORS["black"],
            "figure.facecolor": NEUTRAL_COLORS["white"],
            "axes.facecolor": NEUTRAL_COLORS["white"],
            "figure.dpi": OUTPUT_DPI,
            "savefig.dpi": OUTPUT_DPI,
            "savefig.facecolor": NEUTRAL_COLORS["white"],
            "savefig.transparent": False,
            "savefig.bbox": None,
            "savefig.pad_inches": OUTPUT_PAD_INCHES,
            "axes.unicode_minus": True,
        }
    )


def clean_axis(ax, *, grid: str | None = None) -> None:
    """Apply a consistent light-touch axis treatment."""
    ax.spines["left"].set_linewidth(LINE_WIDTHS["axis"])
    ax.spines["bottom"].set_linewidth(LINE_WIDTHS["axis"])
    if grid:
        ax.grid(axis=grid, alpha=ALPHA["grid"])


def panel_label(ax, text: str, dx: float = -0.13, dy: float = 1.08) -> None:
    """Place a bold panel letter just outside the plotting area."""
    ax.text(
        dx,
        dy,
        text.strip("()"),
        transform=ax.transAxes,
        fontsize=PANEL_LABEL_SIZE,
        fontweight="bold",
        va="top",
        ha="left",
        clip_on=False,
    )


def underline_cells(fig, table, cells) -> None:
    """Draw true PDF underlines beneath selected matplotlib table cells."""
    from matplotlib.lines import Line2D

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    for cell in cells:
        text = cell.get_text()
        if not text.get_text().strip():
            continue
        bbox = text.get_window_extent(renderer)
        (x0, y0), (x1, _) = inv.transform([(bbox.x0, bbox.y0), (bbox.x1, bbox.y1)])
        fig.add_artist(
            Line2D(
                [x0, x1],
                [y0 - 0.004, y0 - 0.004],
                transform=fig.transFigure,
                color=NEUTRAL_COLORS["black"],
                linewidth=LINE_WIDTHS["axis"],
            )
        )


def save_figure(fig, name: str, figures_dir: Path) -> None:
    """Write an exact-size vector PDF and a validated 600 dpi PNG."""
    from PIL import Image

    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = figures_dir / f"{name}.pdf"
    png_path = figures_dir / f"{name}.png"
    png_tmp = figures_dir / f".{name}.tmp.png"
    expected_size = tuple(
        int(round(value * OUTPUT_DPI)) for value in fig.get_size_inches()
    )

    fig.savefig(pdf_path, format="pdf")
    try:
        fig.savefig(
            png_tmp,
            format="png",
            dpi=OUTPUT_DPI,
            pil_kwargs={"compress_level": PNG_COMPRESS_LEVEL},
        )
        with Image.open(png_tmp) as image:
            dpi = tuple(float(value) for value in image.info.get("dpi", (0.0, 0.0)))
            pixel_size = tuple(int(value) for value in image.size)
        if len(dpi) != 2 or any(abs(value - OUTPUT_DPI) > 1.0 for value in dpi):
            raise RuntimeError(f"{name}: PNG dpi={dpi}, expected {OUTPUT_DPI}x{OUTPUT_DPI}")
        if any(
            abs(actual - expected) > 1
            for actual, expected in zip(pixel_size, expected_size)
        ):
            raise RuntimeError(
                f"{name}: PNG pixels={pixel_size}, expected approximately {expected_size}"
            )
        png_tmp.replace(png_path)
    except Exception:
        if png_tmp.exists():
            png_tmp.unlink()
        raise
    finally:
        plt.close(fig)
    print(f"[figure] {pdf_path.name} + {png_path.name} ({OUTPUT_DPI} dpi)")


def save_table_pdf(fig, name: str, tables_dir: Path) -> Path:
    """Write one standalone vector PDF for a manuscript table."""
    tables_dir = Path(tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = tables_dir / f"{name}.pdf"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"[table] {pdf_path.name}")
    return pdf_path
