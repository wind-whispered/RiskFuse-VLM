#!/usr/bin/env python
"""Estimate or run matched SemanticEntropy/Reefknot baselines under a budget.

Without ``--apply`` this command only prints pending generation calls and loads
no model. A real run requires ``--max-generation-calls`` so an accidental
million-sequence launch is impossible. Each selected claim preserves all
semantic samples; long-context backbones may split only the return-sequence
dimension into fixed micro-batches.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hmqr.config import load_config, seed_everything  # noqa: E402
from hmqr.signals.semantic_runner import audit_existing, estimate_calls, run  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--backbone", action="append", default=None)
    parser.add_argument("--dataset", action="append", default=None)
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--max-generation-calls", type=int, default=None)
    parser.add_argument(
        "--max-claims-per-file", type=int, default=None,
        help="pilot/resume cap; completed rows remain valid for the later full run",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--audit-existing", action="store_true")
    parser.add_argument("--stamp-schema", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    backbones = args.backbone or cfg.backbone_names
    datasets = args.dataset or list(cfg["datasets"]["detect"])
    unknown_backbones = sorted(set(backbones) - set(cfg.backbone_names))
    unknown_datasets = sorted(set(datasets) - set(cfg["datasets"]["detect"]))
    if unknown_backbones:
        parser.error(f"unknown backbone(s): {unknown_backbones}")
    if unknown_datasets:
        parser.error(f"semantic entropy is RQ1-only; invalid dataset(s): {unknown_datasets}")
    n_samples = int(args.n_samples or cfg["semantic_entropy"].get("n_samples", 10))
    if n_samples < 2:
        parser.error("--n-samples must be at least 2")
    if args.max_claims_per_file is not None and args.max_claims_per_file < 1:
        parser.error("--max-claims-per-file must be positive")
    if args.stamp_schema and not args.audit_existing:
        parser.error("--stamp-schema requires --audit-existing")
    if args.audit_existing:
        report = audit_existing(cfg, backbones, datasets, args.stamp_schema)
        for name, values in report.items():
            print(name, values)
        return 0

    estimate = estimate_calls(
        cfg, backbones, datasets, n_samples, args.overwrite, args.max_claims_per_file
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
        n_samples=n_samples,
        max_generation_calls=args.max_generation_calls,
        overwrite=args.overwrite,
        max_claims_per_file=args.max_claims_per_file,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
