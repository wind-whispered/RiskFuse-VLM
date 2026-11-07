#!/usr/bin/env python
"""Estimate or run the budgeted one-shot internal-confidence pass."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hmqr.config import load_config, seed_everything  # noqa: E402
from hmqr.signals.lp_runner import audit_existing, estimate_calls, run  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--backbone", action="append", default=None)
    parser.add_argument("--dataset", action="append", default=None)
    parser.add_argument("--max-generation-calls", type=int, default=None)
    parser.add_argument("--max-claims-per-file", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--retry-failed", action="store_true",
        help="retry only selected rows that were attempted but did not yield a valid yes/no parse",
    )
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--audit-existing", action="store_true")
    parser.add_argument("--stamp-schema", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    backbones = args.backbone or cfg.backbone_names
    allowed_datasets = list(cfg["datasets"]["detect"]) + list(
        cfg.get("datasets", {}).get("quality", [])
    )
    datasets = args.dataset or allowed_datasets
    unknown_backbones = sorted(set(backbones) - set(cfg.backbone_names))
    unknown_datasets = sorted(set(datasets) - set(allowed_datasets))
    if unknown_backbones:
        parser.error(f"unknown backbone(s): {unknown_backbones}")
    if unknown_datasets:
        parser.error(f"invalid dataset(s): {unknown_datasets}")
    if args.max_claims_per_file is not None and args.max_claims_per_file < 1:
        parser.error("--max-claims-per-file must be positive")
    if args.stamp_schema and not args.audit_existing:
        parser.error("--stamp-schema requires --audit-existing")

    if args.audit_existing:
        report = audit_existing(cfg, backbones, datasets, args.stamp_schema)
        for name, row in report.items():
            lp_range = (
                f"[{row['lp_min']:.6f},{row['lp_max']:.6f}]"
                if row["lp_min"] is not None and row["lp_max"] is not None
                else "n/a"
            )
            print(
                f"{name}: selected={row['selected']} attempted={row['attempted']} "
                f"valid={row['valid']} parse_rate={row['parse_rate']:.3f} "
                f"l_p={lp_range} "
                f"schema_stamped={row['sampling_schema_stamped']}"
            )
        return 0

    estimate = estimate_calls(
        cfg, backbones, datasets, args.overwrite, args.max_claims_per_file,
        retry_failed=args.retry_failed,
    )
    for name, calls in estimate.items():
        print(f"{name}: pending_generation_calls={calls}")
    total = sum(estimate.values())
    print(f"total_pending_generation_calls={total}")
    if not args.apply:
        print("estimate only; no model loaded")
        return 0
    if args.max_generation_calls is None:
        parser.error("--apply requires --max-generation-calls")

    seed_everything(cfg.get("seed", 42))
    run(
        cfg,
        backbones,
        datasets,
        max_generation_calls=args.max_generation_calls,
        overwrite=args.overwrite,
        max_claims_per_file=args.max_claims_per_file,
        retry_failed=args.retry_failed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
