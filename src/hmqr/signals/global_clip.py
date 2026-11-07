"""Baseline global grounding signal: whole-image CLIP similarity.

This is the object-level style signal that the paper contrasts against the
granularity-matched signals: it is highly correlated with the language prior and
therefore has small residual variance.
"""
from __future__ import annotations

from typing import Any, Dict

from ..utils import claim_to_sentence


def g_global(clip, image, claim: Dict[str, Any]) -> float:
    v = clip.encode_image(image)
    t = clip.encode_text(claim_to_sentence(claim))
    return float(clip.cos(v, t))
