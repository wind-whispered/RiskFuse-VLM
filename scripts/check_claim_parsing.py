#!/usr/bin/env python
"""Fail-fast audit of every discriminative benchmark question -> structured claim."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hmqr.config import load_config  # noqa: E402
from hmqr.data.datasets import load_dataset  # noqa: E402

EXPECTED = {
    "amber_attr": 7628,
    "amber_rel": 1664,
    "rbench": 7756,
    "mmrel": 14927,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if cfg["datasets"].get("subset_limit") is not None:
        raise RuntimeError("claim parsing audit requires datasets.subset_limit=null")

    for dataset, expected in EXPECTED.items():
        samples = load_dataset(cfg, dataset)
        malformed = []
        for sample in samples:
            if dataset == "amber_attr":
                ok = sample.gt.get("object") and sample.gt.get("attribute")
            else:
                ok = (
                    sample.gt.get("subject")
                    and sample.gt.get("relation")
                    and sample.gt.get("object")
                )
            if not ok:
                malformed.append(sample.image_id)
        if len(samples) != expected or malformed:
            raise RuntimeError(
                f"{dataset}: expected={expected}, loaded={len(samples)}, "
                f"malformed={len(malformed)}, first={malformed[:1]}"
            )
        print(f"[PASS] {dataset}: {len(samples)} structured claims")
        print(f"       first={samples[0].gt}")
    print("ALL DISCRIMINATIVE CLAIM PARSERS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
