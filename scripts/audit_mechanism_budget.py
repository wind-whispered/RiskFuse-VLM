#!/usr/bin/env python
"""Estimate Step 03/04 forward-pass counts without loading any model."""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hmqr.config import load_config  # noqa: E402
from hmqr.mechanism.attention_analysis import _balanced_rows  # noqa: E402
from hmqr.mechanism.layer_probe import _balanced_indices  # noqa: E402
from hmqr.utils import read_jsonl, write_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    mechanism = cfg.get("mechanism", {})
    probe_cap = int(mechanism["probe_max_images_per_stratum"])
    attention_cap = int(mechanism["attention_max_images_per_stratum"])
    report = {"backbones": {}, "total_forward_passes": 0}

    for backbone in cfg.backbone_names:
        rows = []
        for path in sorted(cfg.path("processed").glob(f"claims_{backbone}_*.jsonl")):
            if any(name in path.stem for name in ("amber", "rbench")):
                rows.extend(read_jsonl(path))

        probe_idx = _balanced_indices(
            rows,
            None,
            max_images_per_stratum=probe_cap,
            seed=int(cfg.get("seed", 42)),
        )
        probe_rows = [rows[i] for i in probe_idx]
        attention_source = [
            row for row in rows
            if row.get("claim_type") in ("attr", "rel") and row.get("label") in (0, 1)
        ]
        attention_rows = _balanced_rows(
            attention_source,
            None,
            max_images_per_stratum=attention_cap,
            seed=int(cfg.get("seed", 42)),
        )
        probe_strata = Counter(
            f"{row.get('claim_type')}|{row.get('label')}|{row.get('probe_split')}"
            for row in probe_rows
        )
        attention_strata = Counter(
            f"{row.get('claim_type')}|{row.get('label')}" for row in attention_rows
        )
        n_forward = len(probe_rows) + len(attention_rows)
        report["backbones"][backbone] = {
            "probe_forward_passes": len(probe_rows),
            "attention_forward_passes": len(attention_rows),
            "total_forward_passes": n_forward,
            "probe_strata": dict(sorted(probe_strata.items())),
            "attention_strata": dict(sorted(attention_strata.items())),
        }
        report["total_forward_passes"] += n_forward
        print(
            f"{backbone}: probe={len(probe_rows)} attention={len(attention_rows)} "
            f"total={n_forward}"
        )

    write_json(cfg.path("logs") / "mechanism_budget.json", report)
    print(f"total_forward_passes={report['total_forward_passes']}")
    print("CPU-only estimate; no model loaded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
