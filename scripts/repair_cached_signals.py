#!/usr/bin/env python
"""Repair cached signals without loading CLIP, Grounding DINO, or a VLM.

The default mode is a dry run. Pass ``--apply`` only after reviewing the per-file
availability counts. Raw attribute/CLIP/confidence values are reused; relation
scores are deterministically recomputed from the cached subject/object boxes.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hmqr.config import load_config  # noqa: E402
from hmqr.signals.cache_repair import repair_rows  # noqa: E402
from hmqr.utils import read_jsonl, write_json, write_jsonl  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    files = sorted(cfg.path("processed").glob("claims_*.jsonl"))
    if not files:
        raise FileNotFoundError("no claims_*.jsonl files found")
    repaired = []
    backbone_counts = defaultdict(lambda: {"rows": 0, "available": 0, "unavailable": 0})
    for path in files:
        rows = read_jsonl(path)
        counts, params = repair_rows(rows, cfg)
        if counts["invalid_core"]:
            raise RuntimeError(
                f"{path.name}: {counts['invalid_core']} rows lack cached g_global/l_p; "
                "targeted Step 02 recomputation is required"
            )
        print(f"{path.name}: {counts}")
        repaired.append((path, rows, params))
        for backbone in cfg.backbone_names:
            if path.stem.startswith(f"claims_{backbone}_"):
                for key in backbone_counts[backbone]:
                    backbone_counts[backbone][key] += int(counts[key])
                break

    if args.apply:
        for path, rows, params in repaired:
            write_jsonl(path, rows)
            write_json(cfg.path("logs") / f"gnorm_{path.stem}.json", params)
        for backbone in cfg.backbone_names:
            write_json(
                cfg.path("logs") / f"signal_tool_calls_{backbone}.json",
                {
                    "clip_image_calls": 0,
                    "clip_text_calls": 0,
                    "dino_forward": 0,
                    "repair_mode": "cpu_cached",
                    **backbone_counts[backbone],
                },
            )
    action = "updated" if args.apply else "would update"
    print(f"{action} {len(repaired)} files; CPU only, no model inference")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
