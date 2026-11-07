"""Internal-confidence accessor, kept free of heavy dependencies.

The mitigation pipeline uses the backbone's own one-shot verification belief
computed in Step 02. Layer probes belong only to the mechanism analysis in
Figure 1 and must never replace or mix with this score.
"""
from __future__ import annotations

import math
from typing import Any, Dict


def has_valid_lp(row: Dict[str, Any]) -> bool:
    """Whether a row contains a parsed, explicitly computed one-shot belief."""
    if not row.get("lp_computed"):
        return False
    try:
        value = float(row.get("l_p"))
    except (TypeError, ValueError):
        return False
    return math.isfinite(value) and 0.0 <= value <= 1.0


def verification_adapter_matches(
    row: Dict[str, Any], expected_adapter_version: int
) -> bool:
    """Whether cached l_p came from the expected architecture adapter.

    Version 1 predates the explicit provenance field and is byte-for-byte the
    default path for non-Qwen backbones, so legacy missing stamps are accepted
    only for that version. Qwen's ChatML adapter is version 2 and must always be
    explicit.
    """
    value = row.get("verification_adapter_version")
    return bool(
        value == int(expected_adapter_version)
        or (int(expected_adapter_version) == 1 and value is None)
    )


def has_current_lp(row: Dict[str, Any], expected_adapter_version: int) -> bool:
    return bool(
        has_valid_lp(row)
        and verification_adapter_matches(row, expected_adapter_version)
    )


def get_lp(row: Dict[str, Any]) -> float:
    """Return the Step-02 self-verification confidence."""
    if not has_valid_lp(row):
        raise ValueError("row lacks a parsed Step-02 internal confidence")
    return float(row["l_p"])
