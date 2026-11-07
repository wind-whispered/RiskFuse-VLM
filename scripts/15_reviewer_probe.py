#!/usr/bin/env python
"""Estimate or explicitly run the development-selected linear-probe baseline."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hmqr.config import load_config, seed_everything
from hmqr.review.probe import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--max-forward-calls", type=int)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.backbone not in cfg.backbone_names:
        parser.error("backbone must be configured")
    if args.apply:
        seed_everything(int(cfg.get("seed", 42)))
    report = run(cfg, args.backbone, args.apply, args.max_forward_calls)
    print(json.dumps(report, indent=2))
    return 2 if args.apply and report["status"] not in ('complete','complete_with_unsupported_groups') else 0


if __name__ == "__main__":
    raise SystemExit(main())
