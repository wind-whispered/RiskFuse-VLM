"""Structured, image-grounded minimal revision (L0 text interface)."""
from __future__ import annotations

import re
from typing import Any, Dict, List

from ..utils import claim_to_sentence, get_logger, parse_json_block

LOG = get_logger("hmqr.revise")

REVISION_ADAPTER_VERSION = 10

REVISION_PROMPT = (
    "Edit one target span using the supplied image as the only factual authority.\n"
    "Retrieved text is wording help only and may be wrong.\n"
    "Sentence: {sentence}\n"
    "Target type: {kind}\n"
    "Target span: {target_span}\n"
    "Target claim: {claim}\n"
    "Current-image grounding: {image_evidence}\n"
    "Optional wording references:\n{evidence}\n"
    "If the image clearly supports a correction, answer exactly one line as "
    "REPLACE: replacement words. Give only the words that replace the target span; "
    "do not repeat or rewrite the sentence. If no correction is visually clear, "
    "answer exactly ABSTAIN. Example: if red should be blue, answer REPLACE: blue."
)

INSTRUCTBLIP_REVISION_PROMPT = (
    "Look at the image and answer the editing question. The target {kind} words "
    "in this sentence may be wrong.\n"
    "Sentence: {sentence}\n"
    "Target words: {target_span}\n"
    "Question: what words should replace the target words? If the image clearly "
    "shows the answer, write REPLACE: followed by those words. Do not repeat the "
    "sentence. If the image does not show a clear answer, write ABSTAIN. Example: "
    "if red should be blue, write REPLACE: blue. Answer:"
)

_REFUSAL_PATTERN = re.compile(
    r"\b(?:abstain|unclear|unknown|cannot|can't|unable|not sure|insufficient)\b",
    flags=re.I,
)
_REPLACE_PATTERN = re.compile(r"^\s*replace\s*[:=\-]\s*(.+?)\s*$", flags=re.I | re.S)


def _normalised_tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", str(text).lower())


def _target_only_replacement(
    original: str, target_span: str, revised: str
) -> str:
    """Recover a replacement only when no non-target token changed."""
    original_tokens = _normalised_tokens(original)
    target_tokens = _normalised_tokens(target_span)
    revised_tokens = _normalised_tokens(revised)
    if not original_tokens or not target_tokens or not revised_tokens:
        return ""
    starts = [
        index for index in range(len(original_tokens) - len(target_tokens) + 1)
        if original_tokens[index:index + len(target_tokens)] == target_tokens
    ]
    if len(starts) != 1:
        return ""
    start = starts[0]
    prefix = original_tokens[:start]
    suffix = original_tokens[start + len(target_tokens):]
    if revised_tokens[:len(prefix)] != prefix:
        return ""
    if suffix and revised_tokens[-len(suffix):] != suffix:
        return ""
    end = len(revised_tokens) - len(suffix) if suffix else len(revised_tokens)
    replacement_tokens = revised_tokens[len(prefix):end]
    if not replacement_tokens or replacement_tokens == target_tokens:
        return ""
    return " ".join(replacement_tokens)


def _replace_target_span(original: str, target_span: str, replacement: str) -> str:
    """Apply one deterministic target-span replacement, or return an empty string."""
    tokens = [
        re.escape(token)
        for token in re.split(r"[\s_-]+", str(target_span).strip())
        if token
    ]
    if not tokens:
        return ""
    pattern = r"(?<!\w)" + r"[\s_-]+".join(tokens) + r"(?!\w)"
    matches = list(re.finditer(pattern, str(original), flags=re.I))
    if len(matches) != 1:
        return ""
    return re.sub(
        pattern, lambda _: str(replacement).strip(), str(original), count=1, flags=re.I
    )


def _clean_replacement(
    value: str, original: str, target_span: str, max_tokens: int = 8
) -> str:
    """Conservatively normalise a phrase without inventing model content."""
    candidate = str(value).strip()
    candidate = re.sub(r"\s*\[region[^\]]*\]\s*$", "", candidate, flags=re.I)
    candidate = re.sub(r"\s*<box>.*?</box>\s*$", "", candidate, flags=re.I | re.S)
    sentence_terminated = bool(re.search(r"[.!?]\s*$", candidate))
    candidate = re.sub(r"^```(?:json|text)?\s*|\s*```$", "", candidate, flags=re.I)
    candidate = candidate.strip().strip('"\'`').strip()
    candidate = re.sub(r"[.!?]+$", "", candidate).strip()
    if not candidate or _REFUSAL_PATTERN.search(candidate):
        return ""

    # A few backbones may still return a complete edited sentence. Recover the
    # replacement only if every non-target token is unchanged.
    derived = _target_only_replacement(original, target_span, candidate)
    if derived:
        candidate = derived
    elif _normalised_tokens(candidate) == _normalised_tokens(original):
        candidate = target_span
    elif sentence_terminated:
        return ""

    tokens = _normalised_tokens(candidate)
    if not tokens or len(tokens) > int(max_tokens):
        return ""
    if candidate.lower() in {
        "yes", "no", "true", "false", "action", "revise", "replace"
    }:
        return ""
    if re.fullmatch(r"[0-9]+", candidate):
        return ""
    if any(char in candidate for char in "{}[]") or "\n" in candidate:
        return ""
    return candidate


def _claim_aware_replacement(
    replacement: str,
    claim: Dict[str, Any],
    original: str,
    target_span: str,
) -> str:
    """Remove a repeated attribute object while preserving the model's answer."""
    value = str(replacement).strip()
    if claim.get("claim_type") != "attr":
        value_tokens = _normalised_tokens(value)
        if value_tokens == _normalised_tokens(original):
            return target_span
        for role in ("subject", "object"):
            role_tokens = _normalised_tokens(claim.get(role, ""))
            if role_tokens and any(
                value_tokens[index:index + len(role_tokens)] == role_tokens
                for index in range(len(value_tokens) - len(role_tokens) + 1)
            ):
                return ""
        return value
    object_name = str(claim.get("object", "")).strip()
    object_tokens = [
        re.escape(token) for token in re.split(r"[\s_-]+", object_name) if token
    ]
    if not object_tokens:
        return value
    object_pattern = r"[\s_-]+".join(object_tokens)
    prefixed = re.match(
        rf"^\s*{object_pattern}\s+(?:is|looks|appears)\s+(.+?)\s*$",
        value,
        flags=re.I,
    )
    if prefixed:
        value = prefixed.group(1).strip()
    else:
        value = re.sub(
            rf"\s+{object_pattern}\s*$", "", value, count=1, flags=re.I
        ).strip()
    return re.sub(r"^(?:a|an|the)\s+", "", value, count=1, flags=re.I).strip()


def _parsed_revision(
    text: str, original: str, target_span: str
) -> Dict[str, Any]:
    """Parse the compact adapter contract, retaining legacy JSON compatibility."""
    raw = str(text).strip()
    parsed = parse_json_block(raw)
    if isinstance(parsed, dict):
        action = str(parsed.get("action", "")).strip().lower()
        supplied_replacement = str(parsed.get("replacement", "")).strip()
        supplied_sentence = str(parsed.get("revised_sentence", "")).strip()
        if action == "abstain":
            unchanged = (
                not supplied_sentence
                or _normalised_tokens(supplied_sentence) == _normalised_tokens(original)
            )
            if not supplied_replacement and unchanged:
                return {"action": "abstain", "replacement": "", "parse_mode": "json"}
            # Preserve the old repair route only when the payload itself contains
            # an unambiguous target-only edit.
            repaired = _target_only_replacement(original, target_span, supplied_sentence)
            if repaired and (
                not supplied_replacement
                or _normalised_tokens(repaired) == _normalised_tokens(supplied_replacement)
            ):
                return {
                    "action": "revise", "replacement": repaired,
                    "parse_mode": "json_target_only_repair",
                }
            return {"error": "abstain_payload_inconsistent", "parse_ok": True}
        if action == "revise":
            replacement = _clean_replacement(
                supplied_replacement or supplied_sentence, original, target_span
            )
            if replacement:
                return {"action": "revise", "replacement": replacement, "parse_mode": "json"}
            return {"error": "incomplete_revision_payload", "parse_ok": True}
        return {"error": f"invalid_action:{action or 'empty'}", "parse_ok": True}

    compact = re.sub(r"^```(?:text)?\s*|\s*```$", "", raw, flags=re.I).strip()
    if re.fullmatch(r"abstain[.!]?", compact, flags=re.I):
        return {"action": "abstain", "replacement": "", "parse_mode": "compact"}
    match = _REPLACE_PATTERN.match(compact)
    if match:
        replacement = _clean_replacement(match.group(1), original, target_span)
        if replacement:
            return {"action": "revise", "replacement": replacement, "parse_mode": "compact"}
        return {"error": "invalid_replacement_phrase", "parse_ok": True}

    # Accept a bare short phrase because some instruction-tuned checkpoints drop
    # the requested prefix. Generic actions, binary answers and refusals remain
    # rejected above/by _clean_replacement.
    replacement = _clean_replacement(compact, original, target_span, max_tokens=6)
    if replacement:
        return {"action": "revise", "replacement": replacement, "parse_mode": "bare_phrase"}
    if _REFUSAL_PATTERN.search(compact):
        return {"action": "abstain", "replacement": "", "parse_mode": "refusal"}
    return {"error": "revision_parse_failed", "parse_ok": False}


def revise_claim(
    backbone,
    image,
    claim: Dict[str, Any],
    evidence_text: str,
    image_evidence: str,
    max_new_tokens: int = 32,
) -> Dict[str, Any]:
    kind = "attribute" if claim.get("claim_type") == "attr" else "relation"
    target_span = str(
        claim.get("attribute") if kind == "attribute" else claim.get("relation", "")
    ).strip()
    # ``pipeline._perform_actions`` canonicalises discriminative yes/no
    # questions into declarative claims before revision.  Prefer that value;
    # falling back to ``sentence`` preserves generative-caption behaviour.
    original = str(
        claim.get("original_sentence") or claim.get("sentence", "")
    ).strip()
    prompt_template = (
        INSTRUCTBLIP_REVISION_PROMPT
        if str(getattr(backbone, "name", "")) == "instructblip"
        else REVISION_PROMPT
    )
    prompt = prompt_template.format(
        kind=kind,
        target_span=target_span,
        claim=claim_to_sentence(claim),
        image_evidence=image_evidence,
        evidence=evidence_text,
        sentence=original,
    )
    try:
        # ``generate_verification`` is the common instruction/dialogue adapter.
        # It is identical to ``generate`` for LLaVA and InstructBLIP, while Qwen
        # overrides it to add the required ChatML assistant turn.
        generate = (
            getattr(backbone, "generate_revision", None)
            or getattr(backbone, "generate_verification", None)
            or backbone.generate
        )
        out = generate(image, prompt, max_new_tokens=max_new_tokens)
    except Exception as exc:  # noqa: BLE001
        LOG.warning("revision failed for %s: %s", claim.get("claim_id"), exc)
        return {"action": "abstain", "replacement": "",
                "revised_sentence": original, "target_span": target_span,
                "parse_ok": False, "raw_revision_output": "",
                "schema_ok": False,
                "revision_error": str(exc),
                "revision_adapter_version": REVISION_ADAPTER_VERSION}
    text = str(out.get("text", "")).strip()
    parsed = _parsed_revision(text, original, target_span)
    if "error" in parsed:
        return {
            "action": "abstain", "replacement": "",
            "revised_sentence": original, "target_span": target_span,
            "parse_ok": bool(parsed.get("parse_ok", False)), "schema_ok": False,
            "raw_revision_output": text,
            "revision_error": str(parsed["error"]),
            "revision_adapter_version": REVISION_ADAPTER_VERSION,
        }
    action = str(parsed["action"])
    replacement = _claim_aware_replacement(
        str(parsed.get("replacement", "")), claim, original, target_span
    )
    if action == "abstain":
        return {
            "action": "abstain", "replacement": "",
            "revised_sentence": original, "target_span": target_span,
            "parse_ok": True, "schema_ok": True,
            "raw_revision_output": text, "revision_error": None,
            "revision_parse_mode": parsed.get("parse_mode"),
            "revision_adapter_version": REVISION_ADAPTER_VERSION,
        }
    if _normalised_tokens(replacement) == _normalised_tokens(target_span):
        return {
            "action": "abstain", "replacement": "",
            "revised_sentence": original, "target_span": target_span,
            "parse_ok": True, "schema_ok": True,
            "raw_revision_output": text, "revision_error": None,
            "revision_parse_mode": "unchanged_target",
            "revision_adapter_version": REVISION_ADAPTER_VERSION,
        }
    if not replacement:
        return {
            "action": "abstain", "replacement": "",
            "revised_sentence": original, "target_span": target_span,
            "parse_ok": True, "schema_ok": False,
            "raw_revision_output": text,
            "revision_error": "invalid_claim_aware_replacement",
            "revision_adapter_version": REVISION_ADAPTER_VERSION,
        }
    revised = _replace_target_span(original, target_span, replacement)
    if not revised:
        return {
            "action": "abstain", "replacement": "",
            "revised_sentence": original, "target_span": target_span,
            "parse_ok": True, "schema_ok": False,
            "raw_revision_output": text,
            "revision_error": "target_span_not_unique",
            "revision_adapter_version": REVISION_ADAPTER_VERSION,
        }
    return {
        "action": "revise", "replacement": replacement,
        "revised_sentence": revised, "parse_ok": True, "schema_ok": True,
        "target_span": target_span, "structured_minimal_edit": True,
        "raw_revision_output": text, "revision_error": None,
        "revision_parse_mode": parsed.get("parse_mode"),
        "revision_adapter_version": REVISION_ADAPTER_VERSION,
    }


def revise_sentence(
    backbone, image, claim: Dict[str, Any], evidence_text: str, max_new_tokens: int = 96
) -> str:
    """Compatibility wrapper; new code should consume :func:`revise_claim`."""
    result = revise_claim(
        backbone,
        image,
        claim,
        evidence_text,
        image_evidence="Current image is supplied to the model.",
        max_new_tokens=max_new_tokens,
    )
    return str(result["revised_sentence"])


def apply_abstention(sentence: str, mode: str = "annotate") -> str:
    """Abstain: either drop the sentence or mark it as uncertain."""
    if mode == "drop":
        return ""
    return f"{sentence.rstrip('.')} (uncertain)."


def rebuild_caption(
    caption_sentences: List[str], edits: Dict[int, str]
) -> str:
    """Reassemble a caption after per-sentence keep / abstain / revise edits."""
    out: List[str] = []
    for idx, sent in enumerate(caption_sentences):
        new = edits.get(idx, sent)
        if new:
            out.append(new.strip())
    return " ".join(out)
