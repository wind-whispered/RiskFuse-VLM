#!/usr/bin/env python
"""Audit a Visual Genome exact-key plausibility prior on cached claims."""
from __future__ import annotations

import argparse
import csv
import difflib
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hmqr.config import load_config  # noqa: E402
from hmqr.revise.rag_retrieve import EvidenceRetriever  # noqa: E402
from hmqr.revise.rag_index import _norm  # noqa: E402
from hmqr.utils import read_jsonl  # noqa: E402


OUTPUT_FIELDS = [
    "dataset",
    "split",
    "claim_type",
    "n",
    "exact_key_coverage",
    "nonzero_prior_coverage",
    "kb_deficit_auc",
    "mean_support_hallucination",
    "mean_support_correct",
]


def _auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    pairs = sorted(zip(scores, labels), key=lambda pair: pair[0])
    n_pos = sum(label == 1 for _, label in pairs)
    n_neg = sum(label == 0 for _, label in pairs)
    if not n_pos or not n_neg:
        return float("nan")
    rank_sum = 0.0
    index = 0
    while index < len(pairs):
        end = index + 1
        while end < len(pairs) and pairs[end][0] == pairs[index][0]:
            end += 1
        rank = ((index + 1) + end) / 2.0
        rank_sum += rank * sum(label == 1 for _, label in pairs[index:end])
        index = end
    return float((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _similarity(left: Any, right: Any) -> float:
    a, b = _norm(left), _norm(right)
    if not a or not b:
        return 0.0
    a_tokens, b_tokens = set(a.split()), set(b.split())
    jaccard = len(a_tokens & b_tokens) / max(len(a_tokens | b_tokens), 1)
    sequence = difflib.SequenceMatcher(None, a, b).ratio()
    return float(max(jaccard, sequence))


def _support(retriever: EvidenceRetriever, row: dict[str, Any]) -> tuple[float, int]:
    hits = retriever._exact_key_hits(row)
    target = row.get("attribute") if row.get("claim_type") == "attr" else row.get("relation")
    field = "attribute" if row.get("claim_type") == "attr" else "relation"
    return (
        max((_similarity(target, hit.get(field, hit.get("text", ""))) for hit in hits), default=0.0),
        len(hits),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--backbone", default="llava15")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    retriever = EvidenceRetriever(cfg, clip=None)
    buckets: dict[tuple[str, str, str], list[tuple[int, float, int]]] = defaultdict(list)
    for dataset in ("amber_attr", "amber_rel", "rbench", "mmrel"):
        path = cfg.path("processed") / f"claims_{args.backbone}_{dataset}.jsonl"
        for row in read_jsonl(path):
            if row.get("label") not in (0, 1) or row.get("claim_type") not in ("attr", "rel"):
                continue
            support, n_hits = _support(retriever, row)
            buckets[(dataset, str(row.get("split")), str(row.get("claim_type")))].append(
                (int(row["label"]), support, n_hits)
            )
    output = []
    for (dataset, split, claim_type), values in sorted(buckets.items()):
        labels = [label for label, _, _ in values]
        support = [score for _, score, _ in values]
        output.append({
            "dataset": dataset,
            "split": split,
            "claim_type": claim_type,
            "n": len(values),
            "exact_key_coverage": sum(n_hits > 0 for _, _, n_hits in values) / len(values),
            "nonzero_prior_coverage": sum(score > 0 for _, score, _ in values) / len(values),
            "kb_deficit_auc": _auc(labels, [1.0 - score for score in support]),
            "mean_support_hallucination": sum(
                score for label, score, _ in values if label == 1
            ) / max(sum(label == 1 for label in labels), 1),
            "mean_support_correct": sum(
                score for label, score, _ in values if label == 0
            ) / max(sum(label == 0 for label in labels), 1),
        })
    out = Path(args.output) if args.output else cfg.path("review") / "kb_prior_audit.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(output)
    if not output:
        print(
            "no evaluable cached claims were found; wrote a header-only audit file "
            f"to {out}"
        )
        return 0
    for row in output:
        if row["split"] in ("calib_cp", "test"):
            print(
                f"{row['dataset']}\t{row['split']}\tn={row['n']}\t"
                f"coverage={row['exact_key_coverage']:.3f}\tauc={row['kb_deficit_auc']:.3f}"
            )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
