#!/usr/bin/env python
"""Run exactly one revision generation and print the raw auditable output."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hmqr.backbones.registry import build_backbone  # noqa: E402
from hmqr.config import load_config, seed_everything  # noqa: E402
from hmqr.data.images import load_image  # noqa: E402
from hmqr.revise.pipeline import (  # noqa: E402
    _current_image_summary,
    _prepare_records,
    _shared_resources,
    mark_revision_evaluation_sample,
)
from hmqr.revise.reviser import revise_claim  # noqa: E402
from hmqr.utils import claim_to_sentence, read_jsonl  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--dataset", default="amber_attr")
    parser.add_argument("--claim-id", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.get("seed", 42))
    path = cfg.path("processed") / f"claims_{args.backbone}_{args.dataset}.jsonl"
    rows = [
        row for row in read_jsonl(path)
        if row.get("claim_type") in ("attr", "rel")
    ]
    mark_revision_evaluation_sample(rows, cfg)
    calibration, _, _ = _prepare_records(
        rows, "full", float(cfg["cp"]["alpha_default"]), cfg
    )
    candidates = [
        row for row in calibration
        if row.get("flagged") and row.get("signal_available")
    ]
    if args.claim_id:
        candidates = [
            row for row in candidates if str(row.get("claim_id")) == args.claim_id
        ]
    if not candidates:
        raise RuntimeError("no matching flagged calibration claim")
    claim = candidates[0]
    # Mirror the formal pipeline exactly. Discriminative benchmarks store the
    # source yes/no question in ``sentence``, but revision operates on the
    # canonical declarative claim constructed from its parsed fields.
    claim["original_sentence"] = (
        claim_to_sentence(claim)
        if claim.get("mode") == "discriminative"
        else str(claim.get("sentence", ""))
    )

    backbone = build_backbone(cfg, args.backbone)
    _, retriever = _shared_resources(cfg)
    image = load_image(claim["image_path"])
    if image is None:
        raise RuntimeError(f"unreadable image for {claim.get('claim_id')}")
    hits = retriever.retrieve(claim)
    result = revise_claim(
        backbone,
        image,
        claim,
        retriever.format_evidence(hits),
        _current_image_summary(claim),
    )
    print(json.dumps({
        "claim_id": claim.get("claim_id"),
        "mode": claim.get("mode"),
        "original_sentence": claim.get("original_sentence"),
        "retrieval_ids": [hit.get("image_id") for hit in hits],
        **result,
    }, ensure_ascii=False, indent=2))
    backbone.unload()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
