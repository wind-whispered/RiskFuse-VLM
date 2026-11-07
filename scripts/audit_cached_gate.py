#!/usr/bin/env python
"""Audit frozen CP plus residual-gate decisions from cached claim files only."""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hmqr.config import load_config  # noqa: E402
from hmqr.revise import pipeline  # noqa: E402
from hmqr.utils import read_jsonl  # noqa: E402


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--backbone", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--by-relation", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    backbones = [args.backbone] if args.backbone else cfg.backbone_names
    datasets = [args.dataset] if args.dataset else [
        "amber_attr", "amber_rel", "rbench", "mmrel"
    ]
    header = (
        "backbone\tdataset\tgroup\trelation\tn\tn_hallu\tn_correct\t"
        "selected_hallu\tselected_correct\tremoval\tiir\tnet"
    )
    print(header)
    for backbone in backbones:
        for dataset in datasets:
            path = cfg.path("processed") / f"claims_{backbone}_{dataset}.jsonl"
            rows = [
                row for row in read_jsonl(path)
                if row.get("claim_type") in ("attr", "rel")
            ]
            pipeline.mark_revision_evaluation_sample(rows, cfg)
            calibration, test, q_by_group = pipeline._prepare_records(
                rows, "full", float(cfg["cp"]["alpha_default"]), cfg
            )
            theta_by_group, _ = pipeline._select_gate_thresholds(
                calibration, q_by_group, cfg["residual_gate"]
            )
            grouped = defaultdict(list)
            for row in test:
                group = str(row.get("calibration_group"))
                theta = theta_by_group.get(group)
                selected = pipeline.decide(
                    bool(row.get("flagged")), bool(row.get("signal_available")),
                    row.get("d_conflict"), theta,
                ) == "revise"
                copied = dict(row)
                copied["selected"] = selected
                relation = (
                    str(row.get("relation") or row.get("attribute") or "all")
                    if args.by_relation else "all"
                )
                grouped[(group, relation)].append(copied)
            for (group, relation), group_rows in sorted(grouped.items()):
                labels = [int(row.get("original_label", -1)) for row in group_rows]
                n_hallu = sum(label == 1 for label in labels)
                n_correct = sum(label == 0 for label in labels)
                selected_hallu = sum(
                    row["selected"] and int(row.get("original_label", -1)) == 1
                    for row in group_rows
                )
                selected_correct = sum(
                    row["selected"] and int(row.get("original_label", -1)) == 0
                    for row in group_rows
                )
                print(
                    f"{backbone}\t{dataset}\t{group}\t{relation}\t{len(group_rows)}\t"
                    f"{n_hallu}\t{n_correct}\t{selected_hallu}\t{selected_correct}\t"
                    f"{_rate(selected_hallu, n_hallu):.6f}\t"
                    f"{_rate(selected_correct, n_correct):.6f}\t"
                    f"{_rate(selected_hallu - selected_correct, n_hallu):.6f}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
