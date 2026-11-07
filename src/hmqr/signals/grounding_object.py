"""Frozen Grounding-DINO object-presence support."""
from __future__ import annotations

from typing import Any, Dict


def object_presence(
    dino, image, obj: str, cache_key: str | None = None
) -> Dict[str, Any]:
    detections = dino.detect(image, obj, cache_key=cache_key)
    best = detections[0] if detections else None
    return {
        "g_object": float(best["score"]) if best else 0.0,
        "box_object": best["box"] if best else None,
        "object_detections": detections,
        "box_found": bool(best),
        "signal_available": True,
    }
