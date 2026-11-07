"""Evaluate selective interventions with native external-dataset labels.

AMBER, R-Bench and MMRel already state whether the benchmark target claim is
hallucinated. They supervise whether a target should be acted on; they do not
provide truth labels for an unrestricted generated replacement. This module
reports target-error removal and incorrect intervention without inventing a
post-revision label or requiring an additional review dataset.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Sequence

from ..config import Config
from ..utils import get_logger, write_jsonl

LOG = get_logger("hmqr.post_revision")
POST_REVISION_SCHEMA_VERSION = 8


def target_retraction_sentence(row: Dict[str, Any]) -> str:
    """Return a grammatical statement that retracts only the target predicate."""
    if row.get("claim_type") == "attr":
        obj = str(row.get("object") or "object").replace("_", " ")
        return f"The visual state of the {obj} is left unspecified."
    subject = str(row.get("subject") or "subject").replace("_", " ")
    obj = str(row.get("object") or "object").replace("_", " ")
    return f"The relationship between the {subject} and the {obj} is left unspecified."


def _target_retraction(row: Dict[str, Any]) -> bool:
    return (
        str(row.get("action")) == "retract"
        and _normalise(row.get("revised_sentence", ""))
        == _normalise(target_retraction_sentence(row))
        and not str(row.get("replacement") or "").strip()
    )


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", str(text).lower())).strip()


def _target_pattern(target: str) -> str:
    tokens = [
        re.escape(token) for token in re.split(r"[\s_-]+", str(target).strip()) if token
    ]
    if not tokens:
        return ""
    return r"(?<!\w)" + r"[\s_-]+".join(tokens) + r"(?!\w)"


def generative_target_retraction(row: Dict[str, Any]) -> Dict[str, Any]:
    """Retract a unique generated target without changing any other character."""
    original = str(row.get("original_sentence", row.get("sentence", "")))
    target = str(
        row.get("target_span")
        or (
            row.get("attribute")
            if row.get("claim_type") == "attr"
            else row.get("relation")
        )
        or ""
    )
    pattern = _target_pattern(target)
    if not pattern or len(list(re.finditer(pattern, original, re.I))) != 1:
        return {
            "decision": "abstain",
            "action": "abstain",
            "replacement": "",
            "revision_error": "non_unique_target_span",
        }
    marker = (
        "[attribute unspecified]"
        if row.get("claim_type") == "attr"
        else "[relation unspecified]"
    )
    return {
        "decision": "revise",
        "action": "retract",
        "target_span": target,
        "replacement": marker,
        "revised_sentence": re.sub(
            pattern, lambda _: marker, original, count=1, flags=re.I
        ),
        "revision_error": None,
        "revision_parse_mode": "deterministic_generative_retraction",
    }


def _target_only_edit(row: Dict[str, Any]) -> bool:
    """Verify that the generated operation replaced exactly one target span."""
    original = str(row.get("original_sentence", row.get("sentence", "")))
    revised = str(row.get("revised_sentence", ""))
    target = str(
        row.get("target_span")
        or (
            row.get("attribute")
            if row.get("claim_type") == "attr"
            else row.get("relation")
        )
        or ""
    )
    replacement = str(row.get("replacement") or "")
    if not original or not revised or not target or not replacement:
        return False
    pattern = _target_pattern(target)
    if not pattern or len(list(re.finditer(pattern, original, flags=re.I))) != 1:
        return False
    expected = re.sub(pattern, lambda _: replacement, original, count=1, flags=re.I)
    return expected == revised


def _edit_ratio(original: str, revised: str) -> float:
    return float(1.0 - SequenceMatcher(None, str(original), str(revised)).ratio())


def verify_rows(
    cfg: Config,
    dataset: str,
    rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Attach native-label intervention outcomes to claim-level rows.

    ``target_hallucination_removed`` means that a benchmark-labelled target
    error received an auditable target retraction. An unrestricted replacement
    is never treated as correct merely because it changed the target span.
    """
    del cfg, dataset
    verified: List[Dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        original_label = int(row.get("original_label", row.get("label", -1)))
        decision = str(row.get("decision", "keep"))
        original = str(row.get("original_sentence", row.get("sentence", "")))
        revised = str(row.get("revised_sentence", original))
        target_only = decision != "revise" or _target_retraction(row)
        if decision == "revise" and not target_only:
            raise ValueError(
                f"{row.get('claim_id')}: completed revision is not a verified target-only edit"
            )
        acted = decision == "revise" and target_only and row.get("action") == "retract"
        row.update(
            original_label=original_label,
            evaluation_target="original_benchmark_claim",
            label_source=(
                "external_benchmark_gt" if original_label in (0, 1) else "unavailable"
            ),
            intervention_evaluable=original_label in (0, 1),
            target_only_edit=target_only,
            target_only_intervention=target_only,
            intervention_mode="target_retraction" if acted else "none",
            target_hallucination_removed=bool(acted and original_label == 1),
            incorrect_intervention=bool(acted and original_label == 0),
            non_target_preserved=bool(target_only),
            new_object_count=0 if acted else None,
            edit_ratio=_edit_ratio(original, revised),
            post_revision_schema_version=POST_REVISION_SCHEMA_VERSION,
        )
        verified.append(row)
    return verified


def verify_and_write(
    cfg: Config,
    backbone: str,
    dataset: str,
    strategy: str,
    rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    verified = verify_rows(cfg, dataset, rows)
    suffix = "" if strategy == "full" else f"_{strategy}"
    out = cfg.path("processed") / f"revised_labeled_{backbone}_{dataset}{suffix}.jsonl"
    write_jsonl(out, verified)
    LOG.info("wrote %s (%d rows)", out.name, len(verified))
    return verified


def write_quality_only(
    cfg: Config,
    backbone: str,
    dataset: str,
    strategy: str,
    rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Persist CHAIR-500 caption outputs for dataset-native quality metrics."""
    quality_rows: List[Dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        original = str(row.get("original_sentence", row.get("sentence", "")))
        revised = str(row.get("revised_sentence", original))
        row.update(
            original_label=-1,
            evaluation_target="caption_quality",
            label_source="chair500_annotations",
            intervention_evaluable=False,
            target_only_edit=(
                row.get("decision") != "revise" or _target_only_edit(row)
            ),
            target_hallucination_removed=False,
            incorrect_intervention=False,
            non_target_preserved=None,
            new_object_count=None,
            edit_ratio=_edit_ratio(original, revised),
            post_revision_schema_version=POST_REVISION_SCHEMA_VERSION,
        )
        quality_rows.append(row)
    suffix = "" if strategy == "full" else f"_{strategy}"
    out = cfg.path("processed") / f"revised_labeled_{backbone}_{dataset}{suffix}.jsonl"
    write_jsonl(out, quality_rows)
    LOG.info("wrote %s (%d quality-only rows)", out.name, len(quality_rows))
    return quality_rows
