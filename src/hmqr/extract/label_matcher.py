"""T1 [PROVENANCE]: assign hallucination labels from benchmark ground truth ONLY.

label = 1  hallucination (claim contradicts the ground truth)
label = 0  visually grounded
label = -1 unmatched -> excluded from every evaluation, counted as parsing noise

Two regimes
-----------
Discriminative (AMBER-attr / AMBER-rel / R-Bench / MMRel):
    The benchmark supplies the answer for the question the claim was parsed from.
    A claim asserted by the model is a hallucination iff the ground-truth answer
    is "no". No heuristics involved.

Generative (AMBER generative / CHAIR-500):
    The claim is matched against the ground-truth annotation of the image using
    the benchmark's own object/attribute/relation lists plus its synonym table
    (AMBER ships data/safe_words.txt and metrics.json). Claims whose object is not
    present in the annotation vocabulary cannot be adjudicated -> label = -1.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

from ..config import Config
from ..utils import get_logger, read_json

LOG = get_logger("hmqr.label")


# --------------------------------------------------------------------------- #
# synonym handling
# --------------------------------------------------------------------------- #
class SynonymTable:
    """Maps surface forms to a canonical concept id, using the benchmark's own list."""

    def __init__(self, groups: Sequence[Sequence[str]] = ()):
        self.canon: Dict[str, str] = {}
        for group in groups:
            if not group:
                continue
            head = self._norm(group[0])
            for word in group:
                self.canon[self._norm(word)] = head

    @staticmethod
    def _norm(word: str) -> str:
        w = str(word).strip().lower().replace("_", " ")
        w = re.sub(r"[^a-z0-9 ]", "", w)
        w = re.sub(r"\s+", " ", w).strip()
        if w.endswith("es") and len(w) > 4:
            singular = w[:-2]
        elif w.endswith("s") and len(w) > 3:
            singular = w[:-1]
        else:
            singular = w
        return singular

    def key(self, word: str) -> str:
        n = self._norm(word)
        return self.canon.get(n, n)

    def same(self, a: str, b: str) -> bool:
        return self.key(a) == self.key(b)

    @classmethod
    def from_amber(cls, amber_root: Path) -> "SynonymTable":
        groups: List[List[str]] = []
        for name in ("safe_words.txt", "synonyms.txt"):
            p = amber_root / "data" / name
            if p.exists():
                for line in p.read_text(encoding="utf-8").splitlines():
                    parts = [w.strip() for w in re.split(r"[,\t]", line) if w.strip()]
                    if parts:
                        groups.append(parts)
        metrics = amber_root / "data" / "metrics.json"
        if metrics.exists():
            try:
                m = read_json(metrics)
                for key, val in (m.items() if isinstance(m, dict) else []):
                    if isinstance(val, list) and all(isinstance(v, str) for v in val):
                        groups.append([key] + val)
            except Exception as exc:  # noqa: BLE001
                LOG.warning("could not read AMBER metrics.json: %s", exc)
        LOG.info("AMBER synonym groups loaded: %d", len(groups))
        return cls(groups)


# --------------------------------------------------------------------------- #
# discriminative labelling
# --------------------------------------------------------------------------- #
def label_discriminative(gt_answer: str) -> int:
    """The model asserts the claim in the question; GT "no" => hallucination."""
    a = str(gt_answer).strip().lower()
    if a == "no":
        return 1
    if a == "yes":
        return 0
    return -1


# --------------------------------------------------------------------------- #
# generative labelling
# --------------------------------------------------------------------------- #
def _gt_lists(gt: Dict[str, Any]) -> Dict[str, List]:
    """Normalise the ground-truth annotation into object/attribute/relation lists."""
    objects = gt.get("objects") or gt.get("truth") or gt.get("object") or []
    attributes = gt.get("attributes") or gt.get("attribute") or []
    relations = gt.get("relations") or gt.get("relation") or []
    hallu = gt.get("hallu") or gt.get("hallucination") or []
    return {
        "objects": list(objects),
        "attributes": list(attributes),
        "relations": list(relations),
        "hallu": list(hallu),
    }


def label_generative_attr(
    claim: Dict[str, Any], gt: Dict[str, Any], syn: SynonymTable
) -> int:
    lists = _gt_lists(gt)
    obj, attr = claim.get("object", ""), claim.get("attribute", "")
    if not lists["objects"] and not lists["attributes"]:
        return -1
    obj_present = any(syn.same(obj, o) for o in lists["objects"])
    if not obj_present:
        # the object itself is not in the annotation -> attribute cannot be adjudicated
        return -1
    if not lists["attributes"]:
        return -1
    for entry in lists["attributes"]:
        if isinstance(entry, dict):
            e_obj = entry.get("object", entry.get("obj", ""))
            e_attrs = entry.get("attributes", entry.get("attribute", []))
            if isinstance(e_attrs, str):
                e_attrs = [e_attrs]
            if syn.same(obj, e_obj):
                return 0 if any(syn.same(attr, a) for a in e_attrs) else 1
        elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
            if syn.same(obj, entry[0]):
                return 0 if syn.same(attr, entry[1]) else 1
        elif isinstance(entry, str):
            if syn.same(attr, entry):
                return 0
    return 1


def label_generative_rel(
    claim: Dict[str, Any], gt: Dict[str, Any], syn: SynonymTable
) -> int:
    lists = _gt_lists(gt)
    subj, rel, obj = claim.get("subject", ""), claim.get("relation", ""), claim.get("object", "")
    if not lists["relations"]:
        return -1
    subj_ok = any(syn.same(subj, o) for o in lists["objects"]) if lists["objects"] else True
    obj_ok = any(syn.same(obj, o) for o in lists["objects"]) if lists["objects"] else True
    if not (subj_ok and obj_ok):
        return -1
    for entry in lists["relations"]:
        triple = None
        if isinstance(entry, dict):
            triple = (
                entry.get("subject", entry.get("s", "")),
                entry.get("relation", entry.get("predicate", entry.get("r", ""))),
                entry.get("object", entry.get("o", "")),
            )
        elif isinstance(entry, (list, tuple)) and len(entry) >= 3:
            triple = (entry[0], entry[1], entry[2])
        elif isinstance(entry, str):
            parts = entry.split()
            if len(parts) >= 3:
                triple = (parts[0], " ".join(parts[1:-1]), parts[-1])
        if triple is None:
            continue
        if syn.same(subj, triple[0]) and syn.same(obj, triple[2]):
            return 0 if syn.same(rel, triple[1]) else 1
    return 1


def label_claim(
    claim: Dict[str, Any],
    sample_mode: str,
    gt_answer: Optional[str],
    gt: Dict[str, Any],
    syn: SynonymTable,
) -> int:
    if sample_mode == "discriminative":
        return label_discriminative(gt_answer) if gt_answer else -1
    if claim.get("claim_type") == "attr":
        return label_generative_attr(claim, gt, syn)
    return label_generative_rel(claim, gt, syn)


# --------------------------------------------------------------------------- #
# object-level claims (needed by the layer-wise probe, T2)
# --------------------------------------------------------------------------- #
def label_object_claim(obj: str, gt: Dict[str, Any], syn: SynonymTable) -> int:
    lists = _gt_lists(gt)
    if not lists["objects"]:
        return -1
    return 0 if any(syn.same(obj, o) for o in lists["objects"]) else 1


# --------------------------------------------------------------------------- #
def sanity_report(rows: List[Dict[str, Any]], dataset: str) -> Dict[str, Any]:
    """Positive-rate self-check quoted in the spec (attr ~50%, rel ~58.6%)."""
    kept = [r for r in rows if r.get("label", -1) in (0, 1)]
    n = len(kept)
    pos = sum(1 for r in kept if r["label"] == 1)
    rep = {
        "dataset": dataset,
        "n_claims_total": len(rows),
        "n_claims_labelled": n,
        "n_unmatched": len(rows) - n,
        "positive_rate": (pos / n) if n else 0.0,
    }
    LOG.info(
        "[%s] labelled=%d unmatched=%d positive_rate=%.3f",
        dataset,
        n,
        rep["n_unmatched"],
        rep["positive_rate"],
    )
    return rep
