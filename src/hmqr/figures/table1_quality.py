"""Table 1: equal-status cross-backbone and cross-dataset results."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..config import Config
from ..utils import get_logger
from .style import BACKBONE_LABELS, WIDTH_DOUBLE, apply_style, save_table_pdf

LOG = get_logger("hmqr.table1")

DATASETS = [
    ("amber_attr", "AMBER-Attribute"),
    ("amber_rel", "AMBER-Relation"),
    ("rbench", "R-Bench"),
    ("mmrel", "MMRel"),
    ("chair500", "CHAIR-500"),
]
QUALITY_METRICS = [
    ("CHAIR-i", "CHAIR-i"),
    ("BLEU-4", "BLEU-4"),
    ("ROUGE-L", "ROUGE-L"),
    ("obj_recall", "Object recall"),
]
ROW_METRICS = ("NHR [95% CI]", "Risk / coverage", "Abstention", "Unavailable")
TABLE_BACKBONE_LABELS = {
    "llava15": "LLaVA-1.5",
    "llava16": "LLaVA-1.6",
    "instructblip": "InstructBLIP",
    "qwenvl": "Qwen-VL",
}


def _pct(value: float) -> float:
    return 100.0 * float(value)


def _fmt(value: float, signed: bool = False) -> str:
    if not np.isfinite(value):
        return "-"
    if signed:
        return "0.0" if abs(value) < 0.05 else f"{value:+.1f}"
    return f"{value:.1f}"


def _build_table_data(
    reduction: pd.DataFrame,
    quality: pd.DataFrame,
    backbones: List[str],
) -> tuple[pd.DataFrame, Dict[tuple[str, str], List[str]]]:
    """Return a tidy 4x5 audit CSV and four display values per dataset."""
    records: List[Dict[str, float | str]] = []
    cells: Dict[tuple[str, str], List[str]] = {}

    for backbone in backbones:
        for dataset, dataset_label in DATASETS:
            hit = reduction[
                (reduction["backbone"] == backbone)
                & (reduction["dataset"] == dataset)
            ]
            if hit.empty:
                raise ValueError(f"missing RQ3 row for {backbone}/{dataset}")
            row = hit.iloc[0]
            record: Dict[str, float | str] = {
                "backbone": TABLE_BACKBONE_LABELS.get(
                    backbone, BACKBONE_LABELS.get(backbone, backbone)
                ),
                "dataset": dataset_label,
                "net_hallucination_reduction": _pct(row["net_hallucination_reduction"]),
                "ci_low": _pct(row["ci_low"]),
                "ci_high": _pct(row["ci_high"]),
                "selection_risk": _pct(row["selection_risk"]),
                "output_coverage": _pct(row["output_coverage"]),
                "abstention_rate": _pct(row["abstention_rate"]),
                "signal_unavailable_rate": _pct(row["signal_unavailable_rate"]),
            }

            if dataset != "chair500":
                cell = [
                    f"{_fmt(record['net_hallucination_reduction'])} "
                    f"[{_fmt(record['ci_low'])},{_fmt(record['ci_high'])}]",
                    f"{_fmt(record['selection_risk'])}/{_fmt(record['output_coverage'])}",
                    f"{_fmt(record['abstention_rate'])}",
                    f"{_fmt(record['signal_unavailable_rate'])}",
                ]
            else:
                cell = []
                for metric, _ in QUALITY_METRICS:
                    qhit = quality[
                        (quality["backbone"] == backbone)
                        & (quality["metric"] == metric)
                    ]
                    if qhit.empty:
                        raise ValueError(f"missing CHAIR quality row for {backbone}/{metric}")
                    qrow = qhit.iloc[0]
                    method = _pct(qrow["ours"])
                    record[metric] = method
                    cell.append(_fmt(method))
            records.append(record)
            cells[(backbone, dataset)] = cell

    return pd.DataFrame(records), cells


def _write_tex(
    cells: Dict[tuple[str, str], List[str]],
    backbones: List[str],
    path: Path,
) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Cross-backbone results with equal status for all five evaluation datasets.}",
        r"\label{tab:transfer-values}",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{0.105\textwidth}|>{\raggedright\arraybackslash}p{0.12\textwidth}|*{4}{>{\centering\arraybackslash}p{0.126\textwidth}}|>{\raggedright\arraybackslash}p{0.095\textwidth}|>{\centering\arraybackslash}p{0.065\textwidth}@{}}",
        r"\toprule",
        r"\multicolumn{2}{c}{} & \multicolumn{6}{c}{Evaluation dataset} \\",
        r"\hline",
        r"Backbone & Metric & \makecell{AMBER-\\Attribute} & \makecell{AMBER-\\Relation} & R-Bench & MMRel & Metric & CHAIR-500 \\",
        r"\hline",
    ]
    for index, backbone in enumerate(backbones):
        for metric_index, metric_label in enumerate(ROW_METRICS):
            backbone_cell = (
                rf"\multirow{{4}}{{*}}{{{TABLE_BACKBONE_LABELS.get(backbone, backbone)}}}"
                if metric_index == 0 else ""
            )
            row = [backbone_cell, metric_label]
            row.extend(
                cells[(backbone, dataset)][metric_index]
                for dataset, _ in DATASETS[:-1]
            )
            row.extend(
                [
                    QUALITY_METRICS[metric_index][1],
                    cells[(backbone, "chair500")][metric_index],
                ]
            )
            lines.append(" & ".join(row) + r" \\")
        if index < len(backbones) - 1:
            lines.append(r"\hline")
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"\par\smallskip",
            r"\begin{minipage}{0.98\linewidth}",
            r"All values are percentages. For AMBER-Attribute, AMBER-Relation, R-Bench, and MMRel, rows report NHR [image-level 95\% CI], selection risk/output coverage, abstention, and signal-unavailability. For CHAIR-500, the right-hand Metric and CHAIR-500 columns report the post-intervention Method value for CHAIR-i, BLEU-4, ROUGE-L, and object recall. Dataset-native metrics are not pooled across incompatible tasks.",
            r"\end{minipage}",
            r"\end{table*}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make(cfg: Config) -> bool:
    tables = cfg.path("tables")
    reduction_path = tables / "rq3_reduction.csv"
    quality_path = tables / "rq3_quality.csv"
    if not reduction_path.exists() or not quality_path.exists():
        LOG.error("missing RQ3 inputs -- run scripts/10_rq3_crossbackbone.py first")
        return False

    reduction = pd.read_csv(reduction_path)
    quality = pd.read_csv(quality_path)
    available = set(reduction["backbone"]) & set(quality["backbone"])
    backbones = [b for b in cfg.backbone_names if b in available]
    if len(backbones) != 4:
        LOG.error("Table 1 requires four complete backbones; found %d", len(backbones))
        return False

    display, cells = _build_table_data(reduction, quality, backbones)
    display.to_csv(tables / "table1.csv", index=False)
    _write_tex(cells, backbones, tables / "table1.tex")

    apply_style()
    fig, ax = plt.subplots(figsize=(WIDTH_DOUBLE, 4.1))
    ax.axis("off")
    column_labels = [
        "Backbone", "Metric", "AMBER-\nAttribute", "AMBER-\nRelation",
        "R-Bench", "MMRel", "Metric", "CHAIR-500",
    ]
    cell_text = []
    for backbone in backbones:
        for metric_index, metric_label in enumerate(ROW_METRICS):
            row = [
                TABLE_BACKBONE_LABELS.get(backbone, backbone) if metric_index == 0 else "",
                "NHR\n[95% CI]" if metric_index == 0 else metric_label,
            ]
            dataset_values = [
                cells[(backbone, dataset)][metric_index]
                for dataset, _ in DATASETS[:-1]
            ]
            if metric_index == 0:
                dataset_values = [value.replace(" [", "\n[", 1) for value in dataset_values]
            row.extend(dataset_values)
            row.extend(
                [
                    QUALITY_METRICS[metric_index][1],
                    cells[(backbone, "chair500")][metric_index],
                ]
            )
            cell_text.append(row)

    table = ax.table(
        cellText=cell_text,
        colLabels=column_labels,
        cellLoc="left",
        colLoc="left",
        loc="center",
        colWidths=[0.115, 0.12, 0.135, 0.135, 0.125, 0.125, 0.14, 0.085],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6.3)
    table.scale(1.0, 1.45)
    for (row, col), cell in table.get_celld().items():
        cell.set_facecolor("white")
        cell.set_linewidth(0.7)
        group_end = row > 0 and row % 4 == 0
        horizontal = "TB" if row == 0 else ("B" if group_end else "")
        cell.visible_edges = horizontal + ("LR" if col in {1, 6} else "")
        cell.PAD = 0.05
        cell.set_text_props(va="center", fontweight="bold" if row == 0 else "normal")
    save_table_pdf(fig, "table1", tables)
    LOG.info("wrote Table 1 with separate CHAIR-500 Metric and Value columns")
    return True
