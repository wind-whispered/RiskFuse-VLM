"""Frozen-detector count consistency for AMBER number claims."""
from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterable, Sequence

from ..tools.geometry import box_area, box_intersection_area


_COUNT_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10,
}


def expected_count(attribute: Any) -> int | None:
    value = str(attribute or "").strip().lower()
    value = re.sub(r"^count[_\s-]*", "", value)
    if value.isdigit():
        return int(value)
    return _COUNT_WORDS.get(value)


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    inter = box_intersection_area(a, b)
    union = box_area(a) + box_area(b) - inter
    return float(inter / union) if union > 0 else 0.0


def deduplicate_detections(
    detections: Iterable[Dict[str, Any]], iou_threshold: float = 0.50
) -> list[Dict[str, Any]]:
    kept: list[Dict[str, Any]] = []
    ordered = sorted(detections, key=lambda item: float(item.get("score", 0.0)), reverse=True)
    for detection in ordered:
        box = detection.get("box")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        if all(_iou(box, prior["box"]) < float(iou_threshold) for prior in kept):
            kept.append(detection)
    return kept


def count_consistency(
    dino,
    image,
    obj: str,
    attribute: Any,
    cache_key: str | None = None,
    iou_threshold: float = 0.50,
) -> Dict[str, Any]:
    target = expected_count(attribute)
    if target is None:
        return {
            "g_count": None,
            "expected_count": None,
            "detected_count": None,
            "count_detections": [],
            "box_found": False,
            "signal_available": False,
        }
    detections = deduplicate_detections(
        dino.detect(image, obj, cache_key=cache_key), iou_threshold=iou_threshold
    )
    predicted = len(detections)
    score = math.exp(-abs(predicted - target) / max(float(target), 1.0))
    return {
        "g_count": float(score),
        "expected_count": int(target),
        "detected_count": int(predicted),
        "count_detections": detections,
        "box_found": bool(detections),
        "signal_available": True,
    }
