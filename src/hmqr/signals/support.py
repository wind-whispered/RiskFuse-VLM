"""Evidence routing for frozen, training-free visual signals.

Claims with a valid local geometric primitive use it.  Count claims use
multi-box detector consistency.  Predicates without a defensible local
primitive fall back to global image--text compatibility instead of being
silently discarded.  The route is determined from predicate semantics and is
independent of benchmark labels.
"""
from __future__ import annotations

import re
from typing import Any, Dict


_PLANAR_DIRECTIONS = {
    "above", "below", "under", "left_of", "right_of", "on_top_of"
}
_DIRECT_CONTACT = {"touching", "touch", "direct_contact"}
_SIZE_AREA_POS = {"bigger_than", "larger_than"}
_SIZE_AREA_NEG = {"smaller_than"}
_SIZE_HEIGHT_POS = {"taller_than"}
_SIZE_HEIGHT_NEG = {"shorter_than"}
_CONTAINED_BY = {"in", "inside", "within"}
_CONTAINS = {"contains", "filled_with", "surrounded_by", "cover", "covering"}


def normalise_relation(value: Any) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", str(value).lower())
    return re.sub(r"_+", "_", value).strip("_")


def attribute_kind(row: Dict[str, Any]) -> str:
    question = str(row.get("question", "")).strip()
    attribute = str(row.get("attribute", "")).lower()
    if attribute.startswith("count_") or re.match(r"^(?:is|are) there\b", question, re.I):
        return "number"
    if re.match(r"^does the\b", question, re.I):
        return "action"
    return "state"


def relation_signal_kind(row: Dict[str, Any]) -> str:
    relation = normalise_relation(row.get("relation", ""))
    if relation in _PLANAR_DIRECTIONS:
        return "direction"
    if relation in _DIRECT_CONTACT:
        return "contact"
    if relation in _SIZE_AREA_POS | _SIZE_AREA_NEG:
        return "size_area"
    if relation in _SIZE_HEIGHT_POS | _SIZE_HEIGHT_NEG:
        return "size_height"
    if relation in _CONTAINED_BY | _CONTAINS:
        return "containment"
    return "global"


def matched_signal_group(row: Dict[str, Any]) -> str:
    claim_type = str(row.get("claim_type", ""))
    if claim_type == "attr":
        kind = attribute_kind(row)
        if kind == "state":
            return "attr|state"
        if kind == "number":
            return "attr|count"
        return "attr|global"
    if claim_type == "rel":
        return f"rel|{relation_signal_kind(row)}"
    return f"{claim_type or 'unknown'}|unsupported"


def matched_signal_supported(row: Dict[str, Any]) -> bool:
    return not matched_signal_group(row).endswith("|unsupported")
