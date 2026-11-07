"""T1: structured claim extraction (L0 -- text interface only).

For generative samples the VLM is prompted to extract its own attribute and
relation claims as JSON. For discriminative samples the claim is already implied
by the question, so it is parsed directly from the question text.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..data.datasets import Sample, parse_attr_claim, parse_rel_claim
from ..utils import get_logger, parse_json_block, split_sentences

LOG = get_logger("hmqr.extract")

EXTRACTION_PROMPT = (
    "Read the following image description and list every attribute claim and every "
    "relation claim it makes.\n"
    "An attribute claim states that an object has a property (colour, material, state).\n"
    "A relation claim states a spatial or semantic relation between two objects.\n"
    "Return ONLY a JSON array. Each element must be one of:\n"
    '  {"type": "attr", "object": "<object>", "attribute": "<attribute>", '
    '"sentence_idx": <int>}\n'
    '  {"type": "rel", "subject": "<subject>", "relation": "<relation>", '
    '"object": "<object>", "sentence_idx": <int>}\n'
    "sentence_idx is the 0-based index of the sentence the claim comes from.\n\n"
    "Description:\n{caption}\n\nJSON:"
)
# NOTE: substitute with str.replace, never str.format. The prompt shows a JSON
# schema, so it contains literal braces; str.format reads `{"type": "attr", ...}`
# as a replacement field and raises KeyError: '"type"' before the model is ever
# called. Escaping the braces would work but breaks again the next time the
# schema is edited.


def extract_claims_from_caption(
    backbone, image, caption: str, max_new_tokens: int = 512
) -> List[Dict[str, Any]]:
    """Ask the VLM to structure its own output. Falls back to a rule-based parse."""
    sentences = split_sentences(caption)
    prompt = EXTRACTION_PROMPT.replace("{caption}", caption)
    audit = {'raw_response': None, 'error': None, 'route': 'pending'}
    try:
        out = backbone.generate(image, prompt, max_new_tokens=max_new_tokens)
        audit['raw_response'] = out['text']
        parsed = parse_json_block(out["text"])
    except Exception as exc:  # noqa: BLE001
        LOG.warning("extraction generate() failed: %s", exc)
        audit['error'] = str(exc)
        parsed = None

    claims: List[Dict[str, Any]] = []
    if isinstance(parsed, list):
        for item in parsed:
            if not isinstance(item, dict):
                continue
            ctype = str(item.get("type", "")).lower()
            idx = item.get("sentence_idx", 0)
            try:
                idx = int(idx)
            except (TypeError, ValueError):
                idx = 0
            idx = max(0, min(idx, max(0, len(sentences) - 1)))
            sentence = sentences[idx] if sentences else caption
            if ctype == "attr" and item.get("object") and item.get("attribute"):
                claims.append(
                    {
                        "claim_type": "attr",
                        "object": str(item["object"]).lower().strip(),
                        "attribute": str(item["attribute"]).lower().strip(),
                        "subject": None,
                        "relation": None,
                        "sentence_idx": idx,
                        "sentence": sentence,
                    }
                )
            elif ctype == "rel" and item.get("subject") and item.get("object"):
                claims.append(
                    {
                        "claim_type": "rel",
                        "object": str(item["object"]).lower().strip(),
                        "attribute": None,
                        "subject": str(item["subject"]).lower().strip(),
                        "relation": str(item.get("relation", "")).lower().strip().replace(" ", "_"),
                        "sentence_idx": idx,
                        "sentence": sentence,
                    }
                )
    if not claims:
        claims = rule_based_claims(sentences)
        audit['route'] = 'rule_fallback'
    else:
        audit['route'] = 'schema_prompt'
    audit['n_claims'] = len(claims)
    backbone.last_extraction_audit = audit
    return claims


def rule_based_claims(sentences: List[str]) -> List[Dict[str, Any]]:
    """Fallback parser: reuse the discriminative-question patterns per sentence."""
    claims: List[Dict[str, Any]] = []
    for idx, sent in enumerate(sentences):
        a = parse_attr_claim(sent)
        if a:
            claims.append(
                {
                    "claim_type": "attr",
                    "object": a["object"],
                    "attribute": a["attribute"],
                    "subject": None,
                    "relation": None,
                    "sentence_idx": idx,
                    "sentence": sent,
                }
            )
        r = parse_rel_claim(sent)
        if r:
            claims.append(
                {
                    "claim_type": "rel",
                    "object": r["object"],
                    "attribute": None,
                    "subject": r["subject"],
                    "relation": r["relation"],
                    "sentence_idx": idx,
                    "sentence": sent,
                }
            )
    return claims


def claim_from_question(sample: Sample) -> Optional[Dict[str, Any]]:
    """Discriminative sample -> a single structured claim."""
    parsed = sample.gt or {}
    if not parsed:
        parsed = (
            parse_attr_claim(sample.question or "")
            if sample.claim_type == "attr"
            else parse_rel_claim(sample.question or "")
        ) or {}
    if not parsed:
        return None
    if sample.claim_type == "attr":
        if not (parsed.get("object") and parsed.get("attribute")):
            return None
        return {
            "claim_type": "attr",
            "object": parsed["object"],
            "attribute": parsed["attribute"],
            "subject": None,
            "relation": None,
            "sentence_idx": 0,
            "sentence": sample.question,
        }
    if not (parsed.get("subject") and parsed.get("object")):
        return None
    return {
        "claim_type": "rel",
        "object": parsed["object"],
        "attribute": None,
        "subject": parsed["subject"],
        "relation": parsed.get("relation", ""),
        "sentence_idx": 0,
        "sentence": sample.question,
    }
