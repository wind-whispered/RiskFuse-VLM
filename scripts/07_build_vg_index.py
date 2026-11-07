#!/usr/bin/env python
"""T7a: build the Visual Genome region-level evidence index for RAG."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hmqr.config import load_config, seed_everything  # noqa: E402
from hmqr.revise import rag_index  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit-images", type=int, default=None,
                        help="cap the number of VG images indexed (smoke test)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.get("seed", 42))
    rag_index.run(cfg, overwrite=args.overwrite, limit_images=args.limit_images)
    out_dir = cfg.path("vg_index")
    required = [
        out_dir / rag_index.ATTR_INDEX,
        out_dir / rag_index.REL_INDEX,
        out_dir / "region_texts.json",
    ]
    return 0 if all(path.is_file() and path.stat().st_size > 2 for path in required) else 1


if __name__ == "__main__":
    raise SystemExit(main())
