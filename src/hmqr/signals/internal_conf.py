"""Model self-verification confidence and entropy baselines.

``l_p`` is the backbone's one-shot yes/no belief and is the only internal
confidence used by the mitigation pipeline. Semantic entropy is separately
computed from repeated samples for the RQ1 baseline.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


LP_CONFIDENCE_DEFINITION = "signed_geometric_sequence_likelihood_v1"


def _verification_generate(backbone, image, question: str, **kwargs) -> Dict[str, Any]:
    """Use the architecture-matched verifier, with duck-typed test fallback."""
    generator = getattr(backbone, "generate_verification", backbone.generate)
    return generator(image, question, **kwargs)


def _verification_adapter_version(backbone) -> int:
    return int(getattr(backbone, "verification_adapter_version", 1))


def parse_binary_answer(answer: str) -> Optional[str]:
    """Parse the strict yes/no verifier response without accepting empty text."""
    match = re.match(r"^[^a-z]*(yes|no)\b", str(answer).strip().lower())
    return match.group(1) if match else None


# --------------------------------------------------------------------------- #
# semantic entropy
# --------------------------------------------------------------------------- #
def _cluster_answers(answers: Sequence[str]) -> List[int]:
    """Bidirectional-entailment clustering, approximated by normalised yes/no
    equivalence, which is exact for the binary questions used in the benchmarks."""
    clusters: List[int] = []
    canonical: Dict[str, int] = {}
    for ans in answers:
        a = str(ans).strip().lower()
        if a.startswith("yes") or a in {"true", "y"}:
            key = "yes"
        elif a.startswith("no") or a in {"false", "n"}:
            key = "no"
        elif any(phrase in a for phrase in (
            "unanswerable", "cannot answer", "can't answer", "cannot tell",
            "can't tell", "cannot determine", "can't determine", "unclear",
            "not enough information", "insufficient information",
        )):
            key = "unknown"
        else:
            key = a[:32]
        if key not in canonical:
            canonical[key] = len(canonical)
        clusters.append(canonical[key])
    return clusters


def semantic_answer_class(answer: str) -> str:
    """Expose the same answer normalisation used by semantic clustering."""
    value = str(answer).strip().lower()
    parsed = parse_binary_answer(value)
    if parsed is not None:
        return parsed
    if any(phrase in value for phrase in (
        "unanswerable", "cannot answer", "can't answer", "cannot tell",
        "can't tell", "cannot determine", "can't determine", "unclear",
        "not enough information", "insufficient information",
    )):
        return "unknown"
    return value[:32]


def semantic_entropy(answers: Sequence[str]) -> float:
    if not answers:
        return 0.0
    clusters = _cluster_answers(answers)
    counts = Counter(clusters)
    n = len(clusters)
    ent = -sum((c / n) * math.log(c / n + 1e-12) for c in counts.values())
    max_ent = math.log(max(2, len(counts)))
    return float(np.clip(ent / max_ent, 0.0, 1.0)) if max_ent > 0 else 0.0


def sample_answers(
    backbone,
    image,
    question: str,
    n_samples: int = 10,
    max_new_tokens: int = 8,
    temperature: float = 0.7,
    top_p: float = 0.95,
) -> List[str]:
    """Sample binary answers in one batched generation invocation."""
    try:
        output = _verification_generate(
            backbone,
            image,
            question,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=int(n_samples),
        )
    except Exception:  # noqa: BLE001
        return []
    return [str(value) for value in output.get("texts", [output.get("text", "")])]


def sample_answer_signals(
    backbone,
    image,
    question: str,
    n_samples: int = 10,
    max_return_sequences_per_generate: int | None = None,
    max_new_tokens: int = 8,
    temperature: float = 0.7,
    top_p: float = 0.95,
) -> Dict[str, Any]:
    """Return sampled answers and a full-vocabulary predictive-entropy baseline.

    Long visual sequences (notably LLaVA-1.6) cannot fit ten returned
    continuations in 32 GiB. ``max_return_sequences_per_generate`` chunks only
    that return-sequence dimension; it never reduces ``n_samples``. Mean
    normalised token predictive entropy is taken from the first generation,
    while every chunk contributes answers to SemanticEntropy. This is not the
    answer-vocabulary entropy used by Reefknot and must not be labelled as such.
    """
    chunk_size = int(max_return_sequences_per_generate or n_samples)
    if chunk_size <= 0:
        raise ValueError("max_return_sequences_per_generate must be positive")
    answers: List[str] = []
    reefknot_entropies: List[float] = []
    invocations = 0
    while len(answers) < int(n_samples):
        current = min(chunk_size, int(n_samples) - len(answers))
        output = _verification_generate(
            backbone,
            image,
            question,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=current,
        )
        invocations += 1
        chunk_answers = [
            str(value) for value in output.get("texts", [output.get("text", "")])
        ]
        if len(chunk_answers) != current:
            break
        answers.extend(chunk_answers)
        if invocations == 1:
            reefknot_entropies = [
                float(value) for value in output.get("token_entropies", [])
                if np.isfinite(float(value))
            ]
    return {
        "answers": answers,
        "parsed_answers": [parse_binary_answer(answer) for answer in answers],
        "semantic_answer_classes": [semantic_answer_class(answer) for answer in answers],
        "empty_answer_count": sum(not str(answer).strip() for answer in answers),
        "unparsed_answer_count": sum(
            parse_binary_answer(answer) is None for answer in answers
        ),
        "predictive_entropy": (
            float(np.mean(reefknot_entropies))
            if reefknot_entropies else float("nan")
        ),
        # Deprecated compatibility alias for caches created before the
        # baseline was given its methodologically accurate name.
        "reefknot_entropy": (
            float(np.mean(reefknot_entropies))
            if reefknot_entropies else float("nan")
        ),
        "model_generate_invocations": invocations,
        "verification_adapter_version": _verification_adapter_version(backbone),
    }


def confidence_from_entropy(entropy: float) -> float:
    """Larger entropy -> less confident."""
    return float(1.0 - np.clip(entropy, 0.0, 1.0))


# --------------------------------------------------------------------------- #
# (c) cheap fallback: sequence log-probability of the claim tokens
# --------------------------------------------------------------------------- #
def logprob_confidence(token_logprobs: Sequence[float]) -> float:
    if not token_logprobs:
        return 0.5
    mean_lp = float(np.mean(token_logprobs))
    return float(np.exp(np.clip(mean_lp, -20.0, 0.0)))


VERIFY_PROMPT = 'Is it true that {claim}? Answer with yes or no only.'


def verify_confidence(backbone, image, claim_sentence: str) -> float:
    """Uniform bootstrap l_p for every claim: the model's own yes/no belief.

    Returns a value in [0, 1]: p(the claim is true) as expressed by the backbone.
    This score remains fixed throughout the residual, detection, and mitigation
    stages; mechanism probes never overwrite it.
    """
    return verify_confidence_signals(backbone, image, claim_sentence)["l_p"]


def verify_confidence_signals(backbone, image, claim_sentence: str) -> Dict[str, Any]:
    """Return an auditable one-shot yes/no belief.

    An unparsable or failed generation is missing data, not neutral confidence.
    The old ``0.5`` fallback made a skipped Step-02 pass look scientifically
    complete and collapsed every downstream residual to a grounding-only score.
    """
    try:
        out = _verification_generate(
            backbone, image, VERIFY_PROMPT.format(claim=claim_sentence), max_new_tokens=4
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "l_p": None,
            "lp_computed": False,
            "lp_attempted": True,
            "lp_parse_status": f"generation_error:{type(exc).__name__}",
            "lp_answer": "",
            "lp_predictive_entropy": None,
            "verification_adapter_version": _verification_adapter_version(backbone),
            "lp_confidence_definition": LP_CONFIDENCE_DEFINITION,
        }
    text = str(out.get("text", "")).strip().lower()
    conf = logprob_confidence(out.get("token_logprobs", []))
    token_entropies = out.get("token_entropies", [])
    predictive_entropy = (
        float(token_entropies[0]) if token_entropies else float("nan")
    )
    answer = parse_binary_answer(text)
    if answer == "yes":
        belief = float(0.5 + 0.5 * conf)
    elif answer == "no":
        belief = float(0.5 - 0.5 * conf)
    else:
        belief = None
    return {
        "l_p": belief,
        "lp_computed": belief is not None,
        "lp_attempted": True,
        "lp_parse_status": answer or "unparsed",
        "lp_answer": text[:64],
        "lp_predictive_entropy": (
            predictive_entropy if np.isfinite(predictive_entropy) else None
        ),
        "verification_adapter_version": _verification_adapter_version(backbone),
        "lp_confidence_definition": LP_CONFIDENCE_DEFINITION,
    }
