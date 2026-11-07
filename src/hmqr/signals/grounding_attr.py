"""Attribute-level grounding signal (ALD).

    g_attr = cos(phi_v(R_o), phi_t(pi_{a,o})) - cos(phi_v(R_o), phi_t(pi_o))

The subtraction removes the object-identity contribution and isolates the
evidence for the attribute itself.
"""
from __future__ import annotations

from typing import Any, Dict, Optional


def g_attr(
    clip,
    dino,
    image,
    obj: str,
    attr: str,
    cache_key: Optional[str] = None,
) -> Dict[str, Any]:
    box = dino.best_box(image, obj, cache_key=cache_key)
    if box is None:
        return {
            "g_attr": None,
            "box_object": None,
            "box_found": False,
            "signal_available": False,
        }
    region = dino.crop(image, box)
    v = clip.encode_image(region)
    t_with = clip.encode_text(f"a photo of a {attr} {obj}")
    t_only = clip.encode_text(f"a photo of a {obj}")
    score = clip.cos(v, t_with) - clip.cos(v, t_only)
    return {
        "g_attr": float(score),
        "box_object": box,
        "box_found": True,
        "signal_available": True,
    }
