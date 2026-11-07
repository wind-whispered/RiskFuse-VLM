#!/usr/bin/env python
"""T5: residual decomposition and the corrective-value principle (Section 3.4)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hmqr.config import load_config, seed_everything  # noqa: E402
from hmqr.residual import decomposition  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.get("seed", 42))
    result = decomposition.run(cfg, overwrite=args.overwrite)
    return 1 if result.empty else 0


if __name__ == "__main__":
    raise SystemExit(main())
