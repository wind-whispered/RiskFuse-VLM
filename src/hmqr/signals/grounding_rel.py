"""Relation-level grounding signal (RLD).

contact relations  : overlap ratio  |B_s and B_o| / min(|B_s|, |B_o|)
direction relations: centroid displacement projected on the relation unit vector
                     <c_s - c_o, u_r> / ||c_s - c_o||
size relations      : signed log area or height ratio
containment         : fraction of the contained box covered by the container

Residual non-redundancy and task utility are measured empirically; this module
does not assume that a geometric score is useful merely because it differs from
the language prior.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np

from ..tools.geometry import box_area, box_intersection_area, centroid
from .support import normalise_relation, relation_signal_kind


def rel_kind(relation: str, contact_keywords: Sequence[str], direction_vectors: Dict) -> str:
    del contact_keywords, direction_vectors
    return relation_signal_kind({"relation": relation})


def score_relation_boxes(
    subject_box: Sequence[float],
    object_box: Sequence[float],
    relation: str,
    contact_keywords: Sequence[str],
    direction_vectors: Dict[str, Sequence[float]],
) -> tuple[Optional[float], str]:
    """Score a relation from cached boxes without running Grounding DINO."""
    kind = rel_kind(relation, contact_keywords, direction_vectors)
    if kind == "global":
        return None, kind
    if kind == "contact":
        inter = box_intersection_area(subject_box, object_box)
        denom = min(box_area(subject_box), box_area(object_box)) + 1e-8
        return float(inter / denom), kind

    relation_key = normalise_relation(relation)
    if kind == "direction":
        u = direction_vectors.get(relation_key) or direction_vectors.get(
            relation_key.replace("_", " ")
        )
        if u is None:
            return None, kind
        u = np.asarray(u, dtype=np.float64)
        u = u / (np.linalg.norm(u) + 1e-8)
        displacement = centroid(subject_box) - centroid(object_box)
        score = float(np.dot(displacement, u) / (np.linalg.norm(displacement) + 1e-8))
        return score, kind
    if kind == "size_area":
        ratio = np.log((box_area(subject_box) + 1e-8) / (box_area(object_box) + 1e-8))
        return float(-ratio if relation_key == "smaller_than" else ratio), kind
    if kind == "size_height":
        subject_height = max(float(subject_box[3] - subject_box[1]), 1e-8)
        object_height = max(float(object_box[3] - object_box[1]), 1e-8)
        ratio = np.log(subject_height / object_height)
        return float(-ratio if relation_key == "shorter_than" else ratio), kind
    if kind == "containment":
        inter = box_intersection_area(subject_box, object_box)
        contained_area = (
            box_area(object_box)
            if relation_key in {"contains", "filled_with", "surrounded_by", "cover", "covering"}
            else box_area(subject_box)
        )
        return float(inter / (contained_area + 1e-8)), kind
    return None, kind


def g_rel(
    dino,
    image,
    subject: str,
    relation: str,
    obj: str,
    contact_keywords: Sequence[str],
    direction_vectors: Dict[str, Sequence[float]],
    cache_key: Optional[str] = None,
) -> Dict[str, Any]:
    kind = rel_kind(relation, contact_keywords, direction_vectors)
    if kind == "global":
        return {
            "g_rel": None,
            "rel_kind": kind,
            "box_subject": None,
            "box_object": None,
            "box_found": False,
            "signal_available": True,
        }
    b_s = dino.best_box(image, subject, cache_key=cache_key)
    b_o = dino.best_box(image, obj, cache_key=cache_key)

    if b_s is None or b_o is None:
        return {
            "g_rel": None,
            "rel_kind": kind,
            "box_subject": b_s,
            "box_object": b_o,
            "box_found": False,
            "signal_available": False,
        }

    score, kind = score_relation_boxes(
        b_s, b_o, relation, contact_keywords, direction_vectors
    )
    if score is None:
        return {
            "g_rel": None,
            "rel_kind": kind,
            "box_subject": b_s,
            "box_object": b_o,
            "box_found": True,
            "signal_available": False,
        }

    return {
        "g_rel": score,
        "rel_kind": kind,
        "box_subject": b_s,
        "box_object": b_o,
        "box_found": True,
        "signal_available": True,
    }
