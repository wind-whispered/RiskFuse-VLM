#!/usr/bin/env python
"""Run a tiny engineering pilot of the revision adapter without paper outputs."""
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
from hmqr.revise.post_revision import _target_only_edit  # noqa: E402
from hmqr.revise.reviser import revise_claim  # noqa: E402
from hmqr.utils import claim_to_sentence, read_jsonl, write_json  # noqa: E402


def _candidates(cfg, backbone: str, datasets: list[str]) -> list[dict]:
    pools: list[list[dict]] = []
    for dataset in datasets:
        path = cfg.path("processed") / f"claims_{backbone}_{dataset}.jsonl"
        rows = [
            row for row in read_jsonl(path)
            if row.get("claim_type") in ("attr", "rel")
        ]
        mark_revision_evaluation_sample(rows, cfg)
        _, test, _ = _prepare_records(
            rows, "no_gate", float(cfg["cp"]["alpha_default"]), cfg
        )
        selected = [
            dict(row, pilot_dataset=dataset)
            for row in test
            if row.get("flagged")
            and row.get("signal_available")
            and row.get("original_label") == 1
        ]
        selected.sort(
            key=lambda row: (
                -float(row.get("d_conflict", float("-inf"))),
                str(row.get("claim_id")),
            )
        )
        pools.append(selected)

    # Round-robin keeps attribute and relation cases represented in a tiny run.
    output: list[dict] = []
    while any(pools):
        for pool in pools:
            if pool:
                output.append(pool.pop(0))
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--max-claims", type=int, default=3)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.get("seed", 42))
    datasets = args.dataset or ["amber_attr", "amber_rel"]
    candidates = _candidates(cfg, args.backbone, datasets)[: max(0, args.max_claims)]
    if not candidates:
        raise RuntimeError("no labelled, flagged pilot candidates")

    backbone = build_backbone(cfg, args.backbone)
    _, retriever = _shared_resources(cfg)
    records = []
    try:
        for claim in candidates:
            image = load_image(claim["image_path"])
            if image is None:
                raise RuntimeError(f"unreadable image: {claim.get('image_path')}")
            claim["original_sentence"] = (
                claim_to_sentence(claim)
                if claim.get("mode") == "discriminative"
                else str(claim.get("sentence", ""))
            )
            hits = retriever.retrieve(claim)
            result = revise_claim(
                backbone,
                image,
                claim,
                retriever.format_evidence(hits),
                _current_image_summary(claim),
            )
            audited = dict(claim)
            audited.update(result)
            records.append({
                "backbone": args.backbone,
                "dataset": claim["pilot_dataset"],
                "claim_id": claim.get("claim_id"),
                "original_sentence": claim["original_sentence"],
                "action": result.get("action"),
                "replacement": result.get("replacement"),
                "revised_sentence": result.get("revised_sentence"),
                "parse_ok": result.get("parse_ok"),
                "schema_ok": result.get("schema_ok"),
                "target_only_edit": (
                    result.get("action") != "revise" or _target_only_edit(audited)
                ),
                "revision_error": result.get("revision_error"),
                "revision_parse_mode": result.get("revision_parse_mode"),
                "raw_revision_output": result.get("raw_revision_output"),
            })
    finally:
        backbone.unload()

    summary = {
        "backbone": args.backbone,
        "n": len(records),
        "n_contract_valid": sum(
            bool(row["parse_ok"] and row["schema_ok"] and row["target_only_edit"])
            for row in records
        ),
        "n_revise": sum(row["action"] == "revise" for row in records),
        "n_abstain": sum(row["action"] == "abstain" for row in records),
        "records": records,
    }
    out = cfg.path("logs") / f"revision_adapter_pilot_{args.backbone}.json"
    write_json(out, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["n_contract_valid"] == summary["n"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
