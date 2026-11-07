#!/usr/bin/env python
"""Model-free B0-B4 audit and analysis. Never launches GPU jobs implicitly."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hmqr.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--stage",
        choices=[
            "baseline",
            "overlap",
            "preflight",
            "analyse",
            "annotation-export",
            "annotation-score",
        ],
        required=True,
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.stage == "baseline":
        from hmqr.review.audit import baseline

        report = baseline(cfg)
        print(
            json.dumps(
                dict(
                    status=report["status"],
                    missing_claim_files=report["missing_claim_files"],
                ),
                indent=2,
            )
        )
        return 0
    if args.stage == "overlap":
        from hmqr.review.overlap import audit

        report = audit(cfg)
        print(
            json.dumps(
                {k: v for k, v in report.items() if k != "excluded_vg_image_ids"},
                indent=2,
            )
        )
        return 0 if report["identity_audit_complete"] else 2
    if args.stage == "preflight":
        from hmqr.review.audit import preflight

        report = preflight(cfg)
        print(json.dumps(report, indent=2))
        return 0 if report["status"] == "ready_for_budget_review" else 2
    if args.stage == "analyse":
        from hmqr.review.studies import run

        report = run(cfg)
        print(json.dumps(report, indent=2))
        return 0 if report["status"] in ('analysis_complete','analysis_complete_with_unsupported_groups') else 2
    from hmqr.review.human import export_tasks, score_annotations

    report = (
        export_tasks(cfg)
        if args.stage == "annotation-export"
        else score_annotations(cfg)
    )
    print(json.dumps(report, indent=2))
    if args.stage == 'annotation-export':
        # Exporting a partial, explicitly pending workbook is a successful
        # authoring operation; it does not certify annotations or metrics.
        return 0
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
