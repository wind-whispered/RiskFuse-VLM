"""Table 2: compact check-mark table for comparable intervention ablations."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..config import Config
from ..utils import get_logger
from .style import WIDTH_DOUBLE, apply_style, save_table_pdf

LOG = get_logger("hmqr.table2")
COMPONENTS = (
    ("cp", "CP"), ("residual", "Resid."), ("kb", "VG"),
    ("gate", "Gate"), ("intervene", "Act"),
)
METRICS = (
    ("target_hallucination_removal_rate", "Removal ↑", True),
    ("incorrect_intervention_rate", "Incorrect ↓", False),
    ("net_hallucination_reduction", "Net ↑", True),
    ("output_coverage", "Coverage ↑", True),
)
ROW_SPEC = (
    ("Full", "Full", True, True, True, True, True),
    ("w/o residual gate", "No residual gate", True, True, True, False, True),
    ("fixed threshold (w/o CP)", "Fixed threshold (no CP)", False, True, True, True, True),
    ("w/o residualization", "No residualisation", True, False, False, True, True),
    ("w/o VG prior", "No VG prior", True, True, False, True, True),
    ("abstain-all", "Abstain all", True, False, False, False, False),
)


def _build_display(source: pd.DataFrame) -> pd.DataFrame:
    numeric = [key for key, *_ in METRICS]
    panel = source[source["panel"].astype(str).eq("B")]
    grouped = panel.groupby("variant", sort=False, as_index=False)[numeric].mean(numeric_only=True)
    records = []
    for source_name, label, *flags in ROW_SPEC:
        hit = grouped[grouped["variant"].astype(str).eq(source_name)]
        if len(hit) != 1:
            raise ValueError(f"Table 2 expected one aggregated intervention row for {source_name!r}")
        row = hit.iloc[0]
        record = {"variant": label}
        record.update({key: bool(flag) for (key, _), flag in zip(COMPONENTS, flags)})
        record.update({key: float(row[key]) for key in numeric})
        records.append(record)
    return pd.DataFrame.from_records(records)


def _fmt(value: float) -> str:
    return "-" if not np.isfinite(value) else f"{100.0 * value:.1f}"


def _rank(values: np.ndarray, higher: bool) -> List[str]:
    rounded = {i: round(100.0 * float(v), 1) for i, v in enumerate(values) if np.isfinite(v)}
    marks = [""] * len(values)
    if not rounded:
        return marks
    levels = sorted(set(rounded.values()), reverse=higher)
    for index, value in rounded.items():
        if value == levels[0]:
            marks[index] = "best"
        elif len(levels) > 1 and value == levels[1]:
            marks[index] = "second"
    return marks


def _marks(display: pd.DataFrame) -> Dict[str, List[str]]:
    return {key: _rank(display[key].to_numpy(float), higher) for key, _, higher in METRICS}


def _write_tex(display: pd.DataFrame, path: Path) -> None:
    marks = _marks(display)
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Component ablation of selective intervention on the pooled AMBER attribute and relation evaluation.}",
        r"\label{tab:ablation}", r"\small", r"\setlength{\tabcolsep}{3.4pt}",
        r"\begin{tabular}{@{}lccccccccc@{}}", r"\toprule",
        r" & \multicolumn{5}{c}{Enabled component} & \multicolumn{4}{c}{Outcome (\%)} \\",
        r"\cmidrule(lr){2-6}\cmidrule(l){7-10}",
        "Variant & " + " & ".join(label for _, label in COMPONENTS) + " & "
        + " & ".join(label.replace("↑", r"$\uparrow$").replace("↓", r"$\downarrow$") for _, label, _ in METRICS)
        + r" \\", r"\midrule",
    ]
    for index, row in display.iterrows():
        cells = [str(row["variant"])]
        cells.extend(r"$\checkmark$" if bool(row[key]) else "-" for key, _ in COMPONENTS)
        for key, _, _ in METRICS:
            text = _fmt(float(row[key]))
            if marks[key][index] == "best":
                text = r"\textbf{" + text + "}"
            elif marks[key][index] == "second":
                text = r"\underline{" + text + "}"
            cells.append(text)
        lines.append(" & ".join(cells) + r" \\")
    lines += [
        r"\bottomrule", r"\end{tabular}", r"\par\smallskip",
        r"\begin{minipage}{0.96\linewidth}A check mark denotes an enabled component; - denotes a removed component. Best values are bold and second-best values are underlined.\end{minipage}",
        r"\end{table}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make(cfg: Config) -> bool:
    source_path = cfg.path("tables") / "rq4_ablation.csv"
    if not source_path.exists():
        LOG.error("missing %s -- run scripts/11_rq4_ablation.py first", source_path)
        return False
    display = _build_display(pd.read_csv(source_path))
    tables = cfg.path("tables")
    display.to_csv(tables / "table2.csv", index=False)
    _write_tex(display, tables / "table2.tex")

    apply_style()
    fig, ax = plt.subplots(1, 1, figsize=(WIDTH_DOUBLE, 2.25))
    ax.axis("off")
    marks = _marks(display)
    cells = []
    for _, row in display.iterrows():
        cells.append(
            [str(row["variant"])]
            + ["✓" if bool(row[key]) else "-" for key, _ in COMPONENTS]
            + [_fmt(float(row[key])) for key, _, _ in METRICS]
        )
    labels = ["Variant"] + [label for _, label in COMPONENTS] + [label for _, label, _ in METRICS]
    widths = [0.25] + [0.07] * len(COMPONENTS) + [0.115] * len(METRICS)
    table = ax.table(cellText=cells, colLabels=labels, cellLoc="center", loc="center", colWidths=widths)
    table.auto_set_font_size(False)
    table.set_fontsize(7.4)
    table.scale(1.0, 1.38)
    for (row_index, column_index), cell in table.get_celld().items():
        cell.set_facecolor("white")
        cell.set_linewidth(0.5)
        cell.visible_edges = "BT" if row_index == 0 else ("B" if row_index == len(cells) else "")
        if column_index == 0:
            cell.set_text_props(ha="left")
        if row_index == 0:
            cell.set_text_props(fontweight="bold")
        elif cell.get_text().get_text() == "✓":
            cell.set_text_props(fontfamily="DejaVu Sans")
        elif column_index >= 1 + len(COMPONENTS):
            key = METRICS[column_index - 1 - len(COMPONENTS)][0]
            if marks[key][row_index - 1] == "best":
                cell.set_text_props(fontweight="bold")
    fig.tight_layout(pad=0.2)
    save_table_pdf(fig, "table2", tables)
    LOG.info("wrote compact table2.pdf, table2.tex and table2.csv")
    return True
