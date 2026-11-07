#!/usr/bin/env python
"""Audit DINO object-presence support on R-Bench calib_cp only."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hmqr.config import load_config  # noqa: E402
from hmqr.data.images import load_image  # noqa: E402
from hmqr.signals.grounding_object import object_presence  # noqa: E402
from hmqr.tools.grounding_dino import GroundingDino  # noqa: E402
from hmqr.utils import read_jsonl, write_jsonl  # noqa: E402


def _auc(labels, scores) -> float:
    pairs = sorted(zip(scores, labels), key=lambda pair: pair[0])
    n_pos = sum(label == 1 for _, label in pairs)
    n_neg = sum(label == 0 for _, label in pairs)
    if not n_pos or not n_neg:
        return float("nan")
    rank_sum = 0.0
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        rank_sum += ((i + 1 + j) / 2.0) * sum(label == 1 for _, label in pairs[i:j])
        i = j
    return float((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--backbone", default="llava15")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    rows = [
        row for row in read_jsonl(
            cfg.path("processed") / f"claims_{args.backbone}_rbench.jsonl"
        )
        if row.get("split") == "calib_cp"
        and row.get("label") in (0, 1)
        and str(row.get("relation")) == "present"
    ]
    if args.n > 0:
        rng = np.random.default_rng(args.seed)
        selected = []
        for label in (0, 1):
            candidates = [row for row in rows if int(row["label"]) == label]
            take = min(len(candidates), args.n // 2)
            selected.extend(candidates[int(i)] for i in rng.permutation(len(candidates))[:take])
    else:
        selected = rows
    output_path = Path(args.output) if args.output else None
    cached = {
        str(row["claim_id"]): row
        for row in (read_jsonl(output_path) if output_path and output_path.exists() else [])
    }
    dino = GroundingDino(
        cfg["tools"]["grounding_dino_hf_id"], cfg.get("device", "cuda"),
        cfg.get("dtype", "float16"),
        float(cfg["tools"]["grounding_dino_box_threshold"]),
        float(cfg["tools"]["grounding_dino_text_threshold"]),
        cache_dir=cfg.path("tool_cache"),
    )
    output = []
    for index, row in enumerate(selected, start=1):
        cached_row = cached.get(str(row["claim_id"]))
        if cached_row:
            output.append(cached_row)
            continue
        result = object_presence(
            dino, load_image(row["image_path"]), str(row.get("object") or row.get("subject") or ""),
            cache_key=str(row["image_path"]),
        )
        output.append({
            "claim_id": row["claim_id"], "label": int(row["label"]),
            "object": row.get("object") or row.get("subject"),
            "g_object": result["g_object"], "box_found": result["box_found"],
        })
        if output_path and (index % 25 == 0 or index == len(selected)):
            write_jsonl(output_path, output)
    labels = [int(row["label"]) for row in output]
    deficits = [1.0 - float(row["g_object"]) for row in output]
    if args.n > 0:
        print(json.dumps(output, indent=2, ensure_ascii=False))
    print(json.dumps({
        "n": len(output),
        "grounding_deficit_auc": _auc(labels, deficits),
        "mean_deficit_hallucination": float(np.mean([s for s, y in zip(deficits, labels) if y == 1])),
        "mean_deficit_correct": float(np.mean([s for s, y in zip(deficits, labels) if y == 0])),
        "dino_forwards": int(dino.n_forward),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
