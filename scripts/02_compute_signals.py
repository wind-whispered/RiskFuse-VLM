#!/usr/bin/env python
"""T4: granularity-matched grounding signals (ALD / RLD), global CLIP baseline,
and the bootstrap internal confidence l_p."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hmqr.config import load_config, seed_everything  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--backbone", action="append", default=None,
                        help="run only this backbone (repeatable; for multi-GPU sharding)")
    parser.add_argument("--dataset", action="append", default=None,
                        help="run only this dataset (repeatable; for targeted pilots)")
    parser.add_argument(
        "--no-lp",
        action="store_true",
        help="skip one-shot l_p and leave it missing for the budgeted confidence pass",
    )
    parser.add_argument(
        "--with-semantic-entropy",
        action="store_true",
        help=(
            "legacy combined mode; normally run scripts/02_semantic_entropy.py "
            "separately so its generation budget is explicit"
        ),
    )
    args = parser.parse_args()

    # imported after argument parsing: this step needs the GPU stack, but --help
    # should work in any environment
    from hmqr.signals.compute import compute_all

    cfg = load_config(args.config)
    if args.backbone:
        unknown = sorted(set(args.backbone) - set(cfg.backbone_names))
        if unknown:
            parser.error(f"unknown backbone(s): {unknown}")
        cfg._selected_backbones = args.backbone
    if args.dataset:
        known = set(cfg["datasets"]["detect"]) | set(cfg["datasets"]["quality"])
        unknown = sorted(set(args.dataset) - known)
        if unknown:
            parser.error(f"unknown dataset(s): {unknown}")
        cfg._selected_datasets = args.dataset
    seed_everything(cfg.get("seed", 42))
    compute_all(
        cfg,
        overwrite=args.overwrite,
        with_lp=not args.no_lp,
        with_semantic_entropy=args.with_semantic_entropy,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
