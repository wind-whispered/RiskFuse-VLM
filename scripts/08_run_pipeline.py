#!/usr/bin/env python
"""T7: run native-label selective correction and frozen transfer evaluation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hmqr.config import load_config, seed_everything  # noqa: E402
from hmqr.revise import pipeline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--backbone", default=None, help="run a single backbone")
    parser.add_argument("--dataset", default=None, help="run a single dataset")
    parser.add_argument("--strategy", default="full", choices=pipeline.STRATEGIES)
    parser.add_argument(
        "--preflight", action="store_true",
        help="report exact cached generation budgets without loading a model or writing outputs",
    )
    parser.add_argument(
        "--overhead-only", action="store_true",
        help="rebuild the CPU-only Section 5.5 overhead report and exit",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.get("seed", 42))
    if args.overhead_only:
        if args.preflight or args.backbone or args.dataset or args.overwrite:
            parser.error(
                "--overhead-only cannot be combined with preflight/backbone/dataset/overwrite"
            )
        pipeline.write_overhead(cfg)
        return 0

    backbones = [args.backbone] if args.backbone else cfg.backbone_names
    if args.dataset:
        datasets = [args.dataset]
    else:
        # AMBER, R-Bench and MMRel use dataset-native calibration; CHAIR-500 is
        # the subsequent frozen-threshold caption-quality transfer.
        datasets = ["amber_attr", "amber_rel", "rbench", "mmrel", "chair500"]
    missing = [
        cfg.path("processed") / f"claims_{backbone}_{dataset}.jsonl"
        for backbone in backbones
        for dataset in datasets
        if not (cfg.path("processed") / f"claims_{backbone}_{dataset}.jsonl").is_file()
    ]
    if missing:
        print("missing required claim inputs:")
        for path in missing:
            print(f"  {path.name}")
        return 1

    if args.preflight:
        reports = []
        for backbone in backbones:
            for dataset in datasets:
                reports.append(
                    pipeline.preflight_one(
                        cfg, backbone, dataset, strategy=args.strategy,
                        overwrite=args.overwrite,
                    )
                )
        print(json.dumps(reports, indent=2, ensure_ascii=False))
        return 0

    if args.backbone or args.dataset:
        for b in backbones:
            for d in datasets:
                pipeline.run_one(
                    cfg, b, d, args.strategy, overwrite=args.overwrite
                )
            # Four-GPU shards must not race while writing one shared report.
            # Rebuild it once with --overhead-only after every Full Step 08 job
            # (including CHAIR-500) has completed.
        if not args.backbone:
            pipeline.write_overhead(cfg)
    else:
        pipeline.run_all(cfg, overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
