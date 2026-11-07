"""Deterministic, row-order-invariant image-level sampling helpers."""

from __future__ import annotations

import zlib
from typing import Any, Dict, List, Mapping, Sequence


def stable_claim_key(row: Dict[str, Any]) -> str:
    claim_id = str(row.get("claim_id", ""))
    return claim_id.split("|", 1)[1] if "|" in claim_id else claim_id


def image_unit_rows(
    rows: Sequence[Dict[str, Any]],
    seed: int,
    namespace: str,
) -> List[Dict[str, Any]]:
    """Select one claim/image without inspecting labels, scores or availability.

    Call separately for each split and semantic family. Missing identities and
    duplicate claim identities are errors, since either makes selection opaque.
    Selection must precede the finite-score and native-label filters.
    """
    by_image: Dict[str, List[int]] = {}
    seen = set()
    for i, row in enumerate(rows):
        identity = row.get("image_id")
        claim = stable_claim_key(row)
        if identity is None or not str(identity).strip() or not claim:
            raise ValueError("image-unit sampling requires image_id and claim_id")
        key = (str(identity), claim)
        if key in seen:
            raise ValueError(f"duplicate claim identity in image-unit sampling: {key}")
        seen.add(key)
        by_image.setdefault(str(identity), []).append(i)
    return [rows[i] for i in one_index_per_image(rows, by_image, seed, namespace)]


def one_index_per_image(
    rows: Sequence[Dict[str, Any]],
    image_to_indices: Mapping[str, List[int]],
    seed: int,
    namespace: str,
) -> List[int]:
    """Choose one claim/image without depending on source-file row order."""
    chosen: List[int] = []
    for image_id, indices in sorted(image_to_indices.items()):

        def rank(index: int) -> tuple[int, str]:
            key = stable_claim_key(rows[index])
            payload = f"{seed}|{namespace}|{image_id}|{key}".encode("utf-8")
            return zlib.crc32(payload), key

        chosen.append(min(indices, key=rank))
    return sorted(chosen, key=lambda index: stable_claim_key(rows[index]))
