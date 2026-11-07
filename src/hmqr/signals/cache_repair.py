"""CPU-only repair of cached grounding metadata after split/schema migration."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .compute import SIGNALS_SCHEMA_VERSION, _route_cached_core, normalise_g
from .lp import has_valid_lp


def repair_rows(rows: List[Dict[str, Any]], cfg) -> Tuple[Dict[str, int], Dict]:
    counts = {
        "rows": len(rows), "available": 0, "unavailable": 0,
        "invalid_core": 0, "invalid_lp": 0,
    }
    for row in rows:
        # Legacy Step 03 wrote a sampled probe prediction into production claims.
        # The paper outline defines probes as mechanism-only, so remove it to
        # prevent mixed confidence semantics in downstream residual analyses.
        row.pop("l_p_probe", None)
        claim_type = row.get("claim_type")
        if claim_type in ("attr", "rel"):
            available = _route_cached_core(row, cfg)
            if not available:
                row["g_raw"] = None
                row["signal_available"] = False
        else:
            score = row.get("g_raw")
            available = row.get("box_object") is not None and score is not None
            row["g_raw"] = float(score) if available else None

        row["box_found"] = bool(
            row.get("box_object") is not None
            and (claim_type != "rel" or row.get("box_subject") is not None)
        )
        row["signal_available"] = bool(available)
        row["signal_computed"] = True
        row["signals_schema_version"] = SIGNALS_SCHEMA_VERSION
        if row.get("g_global") is None:
            counts["invalid_core"] += 1
        if not has_valid_lp(row):
            # Remove the old --no-lp neutral placeholder without pretending a
            # self-verification call was attempted.
            row["l_p"] = None
            row["lp_computed"] = False
            counts["invalid_lp"] += 1

    params = normalise_g(rows)
    for row in rows:
        usable = bool(row.get("signal_available")) and (
            row.get("claim_type") not in ("attr", "rel") or row.get("g") is not None
        )
        counts["available" if usable else "unavailable"] += 1
    return counts, params
