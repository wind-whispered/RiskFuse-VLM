#!/usr/bin/env python
"""Run a small calibration-only AMBER count-signal pilot."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hmqr.config import load_config  # noqa: E402
from hmqr.data.images import load_image  # noqa: E402
from hmqr.signals.grounding_count import count_consistency, expected_count  # noqa: E402
from hmqr.tools.grounding_dino import GroundingDino  # noqa: E402
from hmqr.utils import read_jsonl, write_jsonl  # noqa: E402


def _auc(labels, scores) -> float:
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
            cfg.path("processed") / f"claims_{args.backbone}_amber_attr.jsonl"
        )
        if row.get("split") == "calib_cp"
        and row.get("label") in (0, 1)
        and expected_count(row.get("attribute")) is not None
    ]
    rng = np.random.default_rng(args.seed)
    if args.n > 0:
        selected = []
        for label in (0, 1):
            candidates = [row for row in rows if int(row["label"]) == label]
            take = min(len(candidates), (args.n + 1) // 2 if label == 1 else args.n // 2)
            selected.extend(candidates[int(index)] for index in rng.permutation(len(candidates))[:take])
    else:
        selected = rows
    output_path = Path(args.output) if args.output else None
    cached = {
        str(row.get("claim_id")): row
        for row in (read_jsonl(output_path) if output_path and output_path.exists() else [])
    }
    dino = GroundingDino(
        cfg["tools"]["grounding_dino_hf_id"],
        cfg.get("device", "cuda"),
        cfg.get("dtype", "float16"),
        float(cfg["tools"]["grounding_dino_box_threshold"]),
        float(cfg["tools"]["grounding_dino_text_threshold"]),
        cache_dir=cfg.path("tool_cache"),
    )
    output = []
    for index, row in enumerate(selected, start=1):
        if str(row.get("claim_id")) in cached:
            output.append(cached[str(row.get("claim_id"))])
            continue
        image = load_image(row["image_path"])
        result = count_consistency(
            dino, image, str(row.get("object") or ""), row.get("attribute"),
            cache_key=str(row.get("image_path")),
        )
        output.append({
            "claim_id": row.get("claim_id"),
            "label": int(row["label"]),
            "object": row.get("object"),
            "expected_count": result["expected_count"],
            "detected_count": result["detected_count"],
            "g_count": result["g_count"],
            "n_detections": len(result["count_detections"]),
        })
        if output_path and (index % 25 == 0 or index == len(selected)):
            write_jsonl(output_path, output)
    labels = np.asarray([row["label"] for row in output], dtype=int)
    deficits = np.asarray([1.0 - row["g_count"] for row in output], dtype=float)
    if args.n > 0:
        print(json.dumps(output, indent=2, ensure_ascii=False))
    auc = _auc(labels.tolist(), deficits.tolist())
    print(
        "summary",
        json.dumps({
            "n": len(output),
            "mean_deficit_hallucination": float(deficits[labels == 1].mean()),
            "mean_deficit_correct": float(deficits[labels == 0].mean()),
            "grounding_deficit_auc": auc,
            "dino_forwards": int(dino.n_forward),
            "output": str(output_path) if output_path else None,
        }, ensure_ascii=False),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
