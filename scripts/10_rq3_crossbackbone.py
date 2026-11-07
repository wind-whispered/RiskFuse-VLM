#!/usr/bin/env python
"""RQ3: cross-backbone generalisation and quality preservation (Figure 7/Table 1)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hmqr.config import load_config, seed_everything  # noqa: E402
from hmqr.eval import rq_studies  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--recompute-revisions", action="store_true",
                        help="also discard verified revision checkpoints")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.get("seed", 42))
    result = rq_studies.run_rq3(
        cfg, overwrite=args.overwrite, recompute_revisions=args.recompute_revisions
    )
    return 1 if any(table.empty for table in result.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
