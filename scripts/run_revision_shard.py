#!/usr/bin/env python
"""Run a resumable list of revision jobs on the single GPU visible to this process."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hmqr.config import load_config, seed_everything  # noqa: E402
from hmqr.revise import pipeline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--strategy", required=True, choices=pipeline.STRATEGIES)
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.backbone not in cfg.backbone_names:
        parser.error(f"unknown backbone: {args.backbone}")
    seed_everything(cfg.get("seed", 42))
    for dataset in args.dataset:
        pipeline.run_one(
            cfg, args.backbone, dataset, args.strategy, overwrite=args.recompute
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
