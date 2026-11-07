#!/usr/bin/env python
"""T1: VLM generation + structured claim extraction + ground-truth labelling."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hmqr.config import load_config, seed_everything  # noqa: E402
from hmqr.extract.build_claims import build_all  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument('--audit-batch-only',action='store_true',
                        help='generate the fixed human-audit images first; retain incomplete full-data checkpoints')
    parser.add_argument("--backbone", action="append", default=None,
                        help="run only this backbone (repeatable; for multi-GPU sharding)")
    parser.add_argument("--dataset", action="append", default=None,
                        help="run only this dataset (repeatable; for targeted rebuilds)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg._audit_batch_only=args.audit_batch_only
    if args.audit_batch_only and (not args.dataset or set(args.dataset)!={'chair500'}):
        parser.error('--audit-batch-only requires exactly --dataset chair500')
    if args.backbone:
        unknown = sorted(set(args.backbone) - set(cfg.backbone_names))
        if unknown:
            parser.error(f"unknown backbone(s): {unknown}")
        cfg._selected_backbones = args.backbone
    if args.dataset:
        known_datasets = set(cfg["datasets"]["detect"]) | set(cfg["datasets"]["quality"])
        unknown = sorted(set(args.dataset) - known_datasets)
        if unknown:
            parser.error(f"unknown dataset(s): {unknown}")
        cfg._selected_datasets = args.dataset
    seed_everything(cfg.get("seed", 42))
    n_failed = build_all(cfg, overwrite=args.overwrite)
    # A paper run is complete only when every requested backbone/dataset job
    # succeeded. Partial claim files must never cascade into apparently valid
    # downstream figures.
    return 1 if n_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
