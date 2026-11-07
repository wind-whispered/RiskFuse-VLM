#!/usr/bin/env python
"""Evaluate expanded relation geometry from cached Grounding-DINO boxes only.

This is an exploratory, split-aware audit.  It does not modify claim caches or
formal results.  Candidate families are declared from relation semantics before
their calibration/test AUROC is reported.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hmqr.config import load_config  # noqa: E402
from hmqr.signals.support import matched_signal_supported, normalise_relation  # noqa: E402
from hmqr.tools.geometry import box_area, box_intersection_area, centroid  # noqa: E402
from hmqr.utils import read_jsonl  # noqa: E402


SIZE_AREA_POS = {"bigger_than", "larger_than"}
SIZE_AREA_NEG = {"smaller_than"}
SIZE_HEIGHT_POS = {"taller_than"}
SIZE_HEIGHT_NEG = {"shorter_than"}
CONTAINED_BY = {"in", "inside", "within"}
CONTAINS = {"contains", "filled_with", "surrounded_by", "cover", "covering"}
PROXIMITY = {"next_to", "near", "beside", "with"}
SUPPORT = {
    "on", "on_top_of", "sitting_on", "standing_on", "riding", "ride",
    "sitting", "standing", "sit", "stand", "lie", "lying_on",
}
WEARING = {"wearing", "wear"}
INTERACTION = {
    "holding", "hold", "carrying", "carry", "attached", "touching", "touch",
    "eating", "eat", "playing", "play", "have", "has", "using", "pulling",
    "pushing", "push", "catch", "drinking", "drink", "feeding", "feed",
}


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
        average_rank = ((index + 1) + end) / 2.0
        rank_sum += average_rank * sum(label == 1 for _, label in pairs[index:end])
        index = end
    return float((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _box(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        box = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    return box if all(math.isfinite(item) for item in box) and box[2] > box[0] and box[3] > box[1] else None


def _primitive_scores(subject: Sequence[float], obj: Sequence[float]) -> dict[str, float]:
    sx1, sy1, sx2, sy2 = subject
    ox1, oy1, ox2, oy2 = obj
    s_area = max(box_area(subject), 1e-8)
    o_area = max(box_area(obj), 1e-8)
    inter = box_intersection_area(subject, obj)
    sc = centroid(subject)
    oc = centroid(obj)
    joint_diag = math.hypot(max(sx2, ox2) - min(sx1, ox1), max(sy2, oy2) - min(sy1, oy1)) + 1e-8
    proximity = max(0.0, 1.0 - float(math.hypot(*(sc - oc))) / joint_diag)
    overlap_min = inter / min(s_area, o_area)
    subject_inside = inter / s_area
    object_inside = inter / o_area
    width_s, width_o = sx2 - sx1, ox2 - ox1
    height_s, height_o = sy2 - sy1, oy2 - oy1
    x_overlap = max(0.0, min(sx2, ox2) - max(sx1, ox1)) / max(min(width_s, width_o), 1e-8)
    subject_above = 1.0 / (1.0 + math.exp((sc[1] - oc[1]) / max((height_s + height_o) / 2.0, 1e-8)))
    return {
        "proximity": proximity,
        "overlap_min": overlap_min,
        "subject_inside": subject_inside,
        "object_inside": object_inside,
        "support": 0.45 * proximity + 0.35 * x_overlap + 0.20 * subject_above,
        "interaction": 0.65 * proximity + 0.35 * overlap_min,
        "wearing": 0.50 * proximity + 0.50 * object_inside,
        "area_ratio": math.log(s_area / o_area),
        "height_ratio": math.log(height_s / height_o),
    }


def candidate_signal(row: dict[str, Any]) -> tuple[str, float] | None:
    relation = normalise_relation(row.get("relation", ""))
    subject = _box(row.get("box_subject"))
    obj = _box(row.get("box_object"))
    if relation == "present":
        return "object_presence", 1.0 if (subject is not None or obj is not None) else 0.0
    if subject is None or obj is None:
        return None
    scores = _primitive_scores(subject, obj)
    if relation in SIZE_AREA_POS:
        return "size_area", scores["area_ratio"]
    if relation in SIZE_AREA_NEG:
        return "size_area", -scores["area_ratio"]
    if relation in SIZE_HEIGHT_POS:
        return "size_height", scores["height_ratio"]
    if relation in SIZE_HEIGHT_NEG:
        return "size_height", -scores["height_ratio"]
    if relation in CONTAINED_BY:
        return "containment", scores["subject_inside"]
    if relation in CONTAINS:
        return "containment", scores["object_inside"]
    if relation in PROXIMITY:
        return "proximity", scores["proximity"]
    if relation in SUPPORT:
        return "support", scores["support"]
    if relation in WEARING:
        return "wearing", scores["wearing"]
    if relation in INTERACTION:
        return "interaction", scores["interaction"]
    return None


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--backbone", default="llava15")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    buckets: dict[tuple[str, str, str, str], list[tuple[int, float]]] = defaultdict(list)
    fallback_buckets: dict[tuple[str, str, str, str], list[tuple[int, float]]] = defaultdict(list)
    for dataset in ("amber_attr", "rbench", "mmrel"):
        path = cfg.path("processed") / f"claims_{args.backbone}_{dataset}.jsonl"
        for row in read_jsonl(path):
            if row.get("label") not in (0, 1):
                continue
            if not matched_signal_supported(row):
                predicate = (
                    normalise_relation(row.get("relation", ""))
                    if row.get("claim_type") == "rel"
                    else normalise_relation(row.get("attribute", ""))
                )
                channels = {
                    "global_clip_deficit": (
                        None if row.get("g_global") is None else 1.0 - float(row["g_global"])
                    ),
                    "predictive_entropy": row.get("lp_predictive_entropy"),
                    "semantic_entropy": row.get("semantic_entropy"),
                    "signed_conflict": row.get("d_conflict"),
                }
                for channel, value in channels.items():
                    try:
                        score = float(value)
                    except (TypeError, ValueError):
                        continue
                    if not math.isfinite(score):
                        continue
                    split = str(row.get("split"))
                    fallback_buckets[(dataset, split, channel, "__all__")].append(
                        (int(row["label"]), score)
                    )
                    fallback_buckets[(dataset, split, channel, predicate)].append(
                        (int(row["label"]), score)
                    )
            if row.get("claim_type") != "rel":
                continue
            candidate = candidate_signal(row)
            if candidate is None:
                continue
            family, signal = candidate
            split = str(row.get("split"))
            predicate = normalise_relation(row.get("relation", ""))
            buckets[(dataset, split, family, "__family__")].append((int(row["label"]), signal))
            buckets[(dataset, split, family, predicate)].append((int(row["label"]), signal))
    output: list[dict[str, Any]] = []
    for (dataset, split, family, predicate), pairs in sorted(buckets.items()):
        labels = [label for label, _ in pairs]
        signals = [signal for _, signal in pairs]
        output.append({
            "dataset": dataset,
            "split": split,
            "family": family,
            "predicate": predicate,
            "n": len(pairs),
            "n_hallucination": sum(label == 1 for label in labels),
            "n_correct": sum(label == 0 for label in labels),
            "grounding_deficit_auc": _auc(labels, [-signal for signal in signals]),
        })
    out_path = Path(args.output) if args.output else cfg.path("review") / "candidate_geometry.csv"
    _write(out_path, output)
    fallback_output: list[dict[str, Any]] = []
    for (dataset, split, channel, predicate), pairs in sorted(fallback_buckets.items()):
        labels = [label for label, _ in pairs]
        scores = [score for _, score in pairs]
        fallback_output.append({
            "dataset": dataset,
            "split": split,
            "channel": channel,
            "predicate": predicate,
            "n": len(pairs),
            "n_hallucination": sum(label == 1 for label in labels),
            "n_correct": sum(label == 0 for label in labels),
            "auroc": _auc(labels, scores),
        })
    fallback_path = out_path.with_name("candidate_fallback_channels.csv")
    _write(fallback_path, fallback_output)
    print(f"wrote {len(output)} rows to {out_path}")
    print(f"wrote {len(fallback_output)} rows to {fallback_path}")
    for row in output:
        if row["predicate"] == "__family__" and row["split"] in ("calib_cp", "test"):
            print(
                f"{row['dataset']}\t{row['split']}\t{row['family']}\t"
                f"n={row['n']}\tauc={row['grounding_deficit_auc']:.4f}"
            )
    for row in fallback_output:
        if row["predicate"] == "__all__" and row["split"] in ("calib_cp", "test"):
            print(
                f"{row['dataset']}\t{row['split']}\t{row['channel']}\t"
                f"n={row['n']}\tauc={row['auroc']:.4f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
