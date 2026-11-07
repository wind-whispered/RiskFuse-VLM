#!/usr/bin/env python
"""Run a tiny isolated Step 03/04 timing pilot without touching paper outputs."""
from __future__ import annotations

import argparse
import copy
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hmqr.config import Config, load_config, seed_everything  # noqa: E402
from hmqr.mechanism import attention_analysis, layer_probe  # noqa: E402
from hmqr.utils import read_jsonl, write_json  # noqa: E402


def _isolated_config(cfg: Config, backbone: str) -> Config:
    raw = copy.deepcopy(cfg.raw)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = cfg.root / "results" / "pilot" / f"mechanism_{backbone}_{stamp}"
    raw["paths"]["tables"] = str(root / "tables")
    raw["paths"]["logs"] = str(root / "logs")
    pilot = Config(raw=raw, root=cfg.root)
    pilot.path("tables").mkdir(parents=True, exist_ok=True)
    pilot.path("logs").mkdir(parents=True, exist_ok=True)
    pilot._selected_backbones = [backbone]
    return pilot


def _pooled_rows(cfg: Config, backbone: str):
    rows = []
    for path in sorted(cfg.path("processed").glob(f"claims_{backbone}_*.jsonl")):
        if any(name in path.stem for name in ("amber", "rbench")):
            rows.extend(read_jsonl(path))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--probe-rows", type=int, default=24)
    parser.add_argument("--attention-rows", type=int, default=8)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.backbone not in cfg.backbone_names:
        parser.error(f"unknown backbone: {args.backbone}")
    seed_everything(int(cfg.get("seed", 42)))
    pilot = _isolated_config(cfg, args.backbone)
    rows = _pooled_rows(cfg, args.backbone)

    started = time.perf_counter()
    features, _, _, _, _ = layer_probe.extract_features(
        pilot, args.backbone, rows, max_rows=args.probe_rows
    )
    probe_seconds = time.perf_counter() - started
    probe_completed = int(features.shape[0]) if features.ndim == 3 else 0
    print(
        f"probe: requested={args.probe_rows} completed={probe_completed} "
        f"seconds={probe_seconds:.1f}"
    )

    started = time.perf_counter()
    attention = attention_analysis.collect(pilot, max_rows=args.attention_rows)
    attention_seconds = time.perf_counter() - started
    attention_completed = int(len(attention))
    print(
        f"attention: requested={args.attention_rows} completed={attention_completed} "
        f"seconds={attention_seconds:.1f}"
    )
    record = {
        "backbone": args.backbone,
        "probe_requested": args.probe_rows,
        "probe_completed": probe_completed,
        "probe_seconds_including_load": probe_seconds,
        "attention_requested": args.attention_rows,
        "attention_completed": attention_completed,
        "attention_seconds_including_load": attention_seconds,
        "production_outputs_modified": False,
    }
    write_json(pilot.path("logs") / "timing.json", record)
    print(f"timing_log={pilot.path('logs') / 'timing.json'}")
    return 0 if probe_completed and attention_completed else 2


if __name__ == "__main__":
    raise SystemExit(main())
