#!/usr/bin/env python
"""T3: visual attention weight analysis (Figure 2)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hmqr.config import load_config, seed_everything  # noqa: E402
from hmqr.mechanism import attention_analysis  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--backbone", action="append", default=None,
                        help="run only this backbone; repeat for more than one")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.backbone:
        unknown = sorted(set(args.backbone) - set(cfg.backbone_names))
        if unknown:
            parser.error(f"unknown backbone(s): {unknown}")
        cfg._selected_backbones = args.backbone
    seed_everything(cfg.get("seed", 42))
    result = attention_analysis.run(cfg, overwrite=args.overwrite, max_rows=args.max_rows)
    return 1 if result.empty else 0


if __name__ == "__main__":
    raise SystemExit(main())
