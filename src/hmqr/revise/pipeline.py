"""Selective image-grounded correction with native-label risk calibration.

``calib_cp`` fits conformal detection thresholds, ``calib_revision`` fits the
residual gate from external benchmark labels, and ``test`` is evaluated once.
Visual Genome supplies retrieval evidence only and never evaluation labels.
"""

from __future__ import annotations

import re
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

from ..config import Config
from ..data.images import load_image
from ..detect.conformal import calibrate, fixed_threshold, flag_hallucination
from ..sampling import image_unit_rows, one_index_per_image, stable_claim_key
from ..signals.support import matched_signal_group, matched_signal_supported
from ..utils import (
    claim_to_sentence,
    get_logger,
    read_json,
    read_jsonl,
    split_sentences,
    write_json,
    write_jsonl,
)
from .post_revision import (
    _target_retraction,
    _target_only_edit,
    target_retraction_sentence,
    generative_target_retraction,
    verify_and_write,
    write_quality_only,
)
from .rag_retrieve import EvidenceRetriever
from .residual_gate import select_theta
from .reviser import (
    REVISION_ADAPTER_VERSION,
    apply_abstention,
    rebuild_caption,
    revise_claim,
)

LOG = get_logger("hmqr.pipeline")

STRATEGIES = [
    "full",
    "no_gate",
    "abstain_all",
    "no_cp",
    "no_residual",
    "no_rag",
    "no_image",
]
FROZEN_GATE_ABLATIONS = {"no_image"}
MIN_CLAIMS_FOR_CP = 10
PIPELINE_SCHEMA_VERSION = 22
REVISION_SAMPLING_SCHEMA_VERSION = 2
_SHARED: Dict[str, Any] = {"clip": None, "retriever": None}


def _shared_resources(cfg: Config):
    if _SHARED["retriever"] is None:
        from ..tools.clip_scorer import ClipScorer

        _SHARED["clip"] = ClipScorer(
            cfg.resource_id(cfg["tools"]["clip_hf_id"], cfg['tools'].get('clip_local_path')),
            cfg.get("device", "cuda"),
            cfg.get("dtype", "float16"),
            cache_dir=cfg.path("tool_cache"),
        )
        _SHARED["retriever"] = EvidenceRetriever(cfg, clip=_SHARED["clip"])
    return _SHARED["clip"], _SHARED["retriever"]


def _collect() -> None:
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _group_key(row: Dict[str, Any]) -> str:
    return matched_signal_group(row)


def _stable_claim_key(row: Dict[str, Any]) -> str:
    return stable_claim_key(row)


def mark_revision_evaluation_sample(rows: List[Dict[str, Any]], cfg: Config) -> None:
    """Select a decision-independent image sample shared by all strategies.

    CP thresholds still use the full calibration data. Expensive one-shot
    internal confidence and revision evaluation use predeclared image-level
    samples. Sampling ignores labels,
    scores, flags, and future decisions.
    """
    cap = int(cfg.get("revision_evaluation", {}).get("max_images_per_group_split", 100))
    seed = int(cfg.get("seed", 42))
    buckets: Dict[Tuple[str, str], Dict[str, List[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for index, row in enumerate(rows):
        split = str(row.get("split"))
        if split not in ("calib_revision", "test"):
            continue
        buckets[(_group_key(row), split)][str(row.get("image_id"))].append(index)

    selected = set()
    for stratum in sorted(buckets):
        candidates = one_index_per_image(
            rows,
            buckets[stratum],
            seed,
            "revision|" + "|".join(stratum),
        )
        stratum_seed = seed + zlib.crc32("|".join(stratum).encode("utf-8"))
        rng = np.random.default_rng(stratum_seed)
        order = rng.permutation(len(candidates))[:cap]
        selected.update(candidates[int(position)] for position in order)
    for index, row in enumerate(rows):
        row["revision_evaluation_selected"] = index in selected
        row["revision_evaluation_cap"] = cap
        row["revision_sampling_unit"] = "image_id"
        row["revision_sampling_schema_version"] = REVISION_SAMPLING_SCHEMA_VERSION


def _signal_value(row: Dict[str, Any]) -> Optional[float]:
    if not matched_signal_supported(row):
        return None
    if not bool(row.get("signal_available", row.get("g") is not None)):
        return None
    try:
        value = float(row.get("g"))
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def decide(
    flagged: bool,
    signal_available: bool,
    d_conflict: Optional[float],
    theta: Optional[float],
) -> str:
    if not flagged:
        return "keep"
    if not signal_available or d_conflict is None or not np.isfinite(float(d_conflict)):
        return "abstain"
    if theta is None or float(d_conflict) <= float(theta):
        return "abstain"
    return "revise"


def _current_image_summary(row: Dict[str, Any]) -> str:
    return (
        f"signal_available={bool(row.get('signal_available'))}; "
        f"matched_grounding={row.get('g')}; d_conflict={row.get('d_conflict')}; "
        f"relation_kind={row.get('rel_kind')}; object_box={row.get('box_object')}; "
        f"subject_box={row.get('box_subject')}"
    )


def _compose_final_caption(records: Sequence[Dict[str, Any]]) -> str:
    """Compose multiple target-level edits without last-write-wins sentence loss."""
    if not records:
        return ""
    caption = str(records[0].get("caption", "") or "")
    sentences = split_sentences(caption)
    if not sentences:
        return caption
    by_sentence: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        by_sentence[int(rec.get("sentence_idx", 0))].append(rec)
    edits: Dict[int, str] = {}
    for sentence_index, sentence_records in by_sentence.items():
        if not 0 <= sentence_index < len(sentences):
            continue
        current = sentences[sentence_index]
        uncertain = False
        for rec in sorted(
            sentence_records, key=lambda row: str(row.get("claim_id", ""))
        ):
            decision = str(rec.get("decision", "keep"))
            if decision == "abstain":
                uncertain = True
                continue
            if decision != "revise" or not rec.get("replacement"):
                continue
            target = str(
                rec.get("target_span")
                or (
                    rec.get("attribute")
                    if rec.get("claim_type") == "attr"
                    else rec.get("relation")
                )
                or ""
            )
            tokens = [
                re.escape(token)
                for token in re.split(r"[\s_-]+", target.strip())
                if token
            ]
            if not tokens:
                continue
            pattern = r"(?<!\w)" + r"[\s_-]+".join(tokens) + r"(?!\w)"
            current = re.sub(
                pattern,
                lambda _: str(rec["replacement"]),
                current,
                count=1,
                flags=re.I,
            )
        edits[sentence_index] = apply_abstention(current) if uncertain else current
    return rebuild_caption(sentences, edits)


def _prepare_records(
    rows: Sequence[Dict[str, Any]],
    strategy: str,
    alpha: float,
    cfg: Config,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, float]]:
    calibration_records: List[Dict[str, Any]] = []
    test_records: List[Dict[str, Any]] = []
    q_by_group: Dict[str, float] = {}
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(_group_key(row), []).append(row)

    for group_key, group in sorted(grouped.items()):
        cp_candidates = [row for row in group if row.get("split") == "calib_cp"]
        cp_candidates = image_unit_rows(
            cp_candidates, int(cfg.get("seed", 42)), "conformal"
        )
        cp_rows = [
            row
            for row in cp_candidates
            if row.get("split") == "calib_cp"
            and row.get("label") in (0, 1)
            and _signal_value(row) is not None
        ]
        revision_rows = [
            row
            for row in group
            if row.get("split") == "calib_revision"
            and bool(row.get("revision_evaluation_selected"))
        ]
        test_rows = [
            row
            for row in group
            if row.get("split") == "test"
            and bool(row.get("revision_evaluation_selected"))
        ]
        if not test_rows:
            continue
        if len(cp_rows) < MIN_CLAIMS_FOR_CP:
            LOG.warning(
                "[%s] insufficient supported calib_cp rows; evaluation rows fail closed",
                group_key,
            )
            for split_rows, destination in (
                (revision_rows, calibration_records),
                (test_rows, test_records),
            ):
                for row in split_rows:
                    rec = dict(row)
                    rec.update(
                        flagged=True,
                        signal_available=False,
                        q_hat=None,
                        strategy=strategy,
                        calibration_group=group_key,
                        original_label=int(row.get("label", -1)),
                        pipeline_schema_version=PIPELINE_SCHEMA_VERSION,
                        signal_unavailable_reason="unsupported_or_underpowered_group",
                    )
                    destination.append(rec)
            continue
        g_cp = np.asarray([_signal_value(row) for row in cp_rows], dtype=np.float64)
        labels_cp = [int(row["label"]) for row in cp_rows]
        if strategy == "no_cp":
            q_hat = fixed_threshold(g_cp, quantile=1.0 - alpha)
        else:
            try:
                q_hat = calibrate(
                    g_cp,
                    labels_cp,
                    alpha,
                    min_hallucinations=int(
                        cfg["cp"].get("min_hallucinations_per_group", 5)
                    ),
                )
            except ValueError as exc:
                LOG.warning("[%s] CP unavailable: %s", group_key, exc)
                for split_rows, destination in (
                    (revision_rows, calibration_records),
                    (test_rows, test_records),
                ):
                    for row in split_rows:
                        destination.append(
                            dict(
                                row,
                                flagged=True,
                                signal_available=False,
                                q_hat=None,
                                strategy=strategy,
                                calibration_group=group_key,
                                original_label=int(row.get("label", -1)),
                                pipeline_schema_version=PIPELINE_SCHEMA_VERSION,
                                signal_unavailable_reason="insufficient_calibration",
                            )
                        )
                continue
        q_by_group[group_key] = float(q_hat)

        for split_rows, destination in (
            (revision_rows, calibration_records),
            (test_rows, test_records),
        ):
            usable = [row for row in split_rows if _signal_value(row) is not None]
            flags = (
                flag_hallucination([_signal_value(row) for row in usable], q_hat)
                if usable
                else np.asarray([], dtype=bool)
            )
            flag_by_id = {
                str(row.get("claim_id")): bool(flag) for row, flag in zip(usable, flags)
            }
            for row in split_rows:
                rec = dict(row)
                available = _signal_value(row) is not None
                # No matched signal means no CP decision can be made. Fail closed
                # into Abstain rather than treating missing evidence as truth.
                rec["flagged"] = flag_by_id.get(str(row.get("claim_id")), True)
                rec["signal_available"] = available
                rec["q_hat"] = float(q_hat)
                rec["strategy"] = strategy
                rec["calibration_group"] = group_key
                rec["original_label"] = int(row.get("label", -1))
                rec["pipeline_schema_version"] = PIPELINE_SCHEMA_VERSION
                destination.append(rec)
    return calibration_records, test_records, q_by_group


def _replace_residual_with_raw_grounding_deficit(
    records: Sequence[Dict[str, Any]],
) -> None:
    """Ablate residualization while preserving CP and risk calibration."""
    for rec in records:
        value = _signal_value(rec)
        rec["d_conflict"] = (1.0 - float(value)) if value is not None else None
        rec["conflict_source"] = "raw_matched_grounding_deficit"


def _replace_residual_without_kb(records: Sequence[Dict[str, Any]]) -> None:
    """Use the calibration-fitted residual that excludes the VG prior."""
    for rec in records:
        rec["d_conflict"] = rec.get("d_conflict_no_kb")
        rec["conflict_source"] = "internal_belief_only_residual"


def _prepare_transfer_records(
    rows: Sequence[Dict[str, Any]],
    threshold_payloads: Dict[str, Dict[str, Any]],
    strategy: str,
    cfg: Optional[Config] = None,
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for row in rows:
        if row.get("claim_type") not in ("attr", "rel"):
            continue
        rec = dict(row)
        source_dataset = (
            "amber_attr" if row.get("claim_type") == "attr" else "amber_rel"
        )
        threshold_payload = threshold_payloads.get(source_dataset, {})
        q_by_group = threshold_payload.get("q_hat", {})
        theta_by_group = threshold_payload.get("theta", {})
        group_key = _group_key(row)
        q_hat = q_by_group.get(group_key)
        value = _signal_value(row)
        available = value is not None and q_hat is not None
        flagged = (
            bool(flag_hallucination([value], float(q_hat))[0]) if available else True
        )
        theta = theta_by_group.get(group_key)
        rec.update(
            calibration_group=group_key,
            q_hat=float(q_hat) if q_hat is not None else None,
            theta=float(theta) if theta is not None else None,
            flagged=flagged,
            signal_available=available,
            strategy=strategy,
            original_label=int(row.get("label", -1)),
            pipeline_schema_version=PIPELINE_SCHEMA_VERSION,
            threshold_source=source_dataset,
            deterministic_retraction=(
                cfg is not None
                and str(cfg.get("intervention", {}).get("generative_action", ""))
                == "target_retraction"
            ),
        )
        quality_policy = (
            str(cfg.get("quality_transfer", {}).get("policy", "selective_revision"))
            if cfg is not None
            else "selective_revision"
        )
        if quality_policy == "audit_only_keep":
            rec["decision"] = "keep"
            rec["quality_transfer_policy"] = quality_policy
        elif strategy == "no_gate":
            rec["decision"] = (
                "revise"
                if flagged and available
                else ("abstain" if flagged else "keep")
            )
        elif strategy == "abstain_all":
            rec["decision"] = "abstain" if flagged else "keep"
        else:
            rec["decision"] = decide(flagged, available, rec.get("d_conflict"), theta)
        output.append(rec)
    return output


def _perform_actions(
    cfg: Config,
    backbone_name: str,
    dataset: str,
    strategy: str,
    records: List[Dict[str, Any]],
    out_path: Path,
    backbone=None,
    retriever=None,
    overwrite: bool = False,
) -> List[Dict[str, Any]]:
    partial_path = out_path.with_name(out_path.name + ".partial")
    complete_path = out_path.with_name(out_path.name + ".complete.json")
    checkpoint = _reusable_generation_cache(records, out_path, overwrite)
    pending_generation = _enforce_generation_budget(
        cfg, records, out_path, overwrite=overwrite
    )
    max_generation_calls = int(
        cfg.get("revision_generation", {}).get("max_calls_per_job", 250)
    )
    LOG.info(
        "[%s] pending revision generations=%d (hard cap=%d)",
        out_path.name,
        pending_generation,
        max_generation_calls,
    )
    for index, rec in enumerate(
        tqdm(records, desc=f"revise {out_path.stem}", ncols=88), start=1
    ):
        rec["original_sentence"] = (
            claim_to_sentence(rec)
            if rec.get("mode") == "discriminative"
            else rec.get("sentence", "")
        )
        rec["revision_requested"] = rec.get("decision") == "revise"
        rec["revision_attempted"] = False
        prior = checkpoint.get(str(rec.get("claim_id")))
        if prior:
            for key in (
                "decision",
                "action",
                "replacement",
                "revised_sentence",
                "parse_ok",
                "schema_ok",
                "target_span",
                "raw_revision_output",
                "revision_error",
                "revision_adapter_version",
                "retrieval_ids",
                "current_image_evidence",
                "revision_attempt_complete",
                "revision_requested",
                "revision_attempted",
                "revision_parse_mode",
            ):
                if key in prior:
                    rec[key] = prior[key]
            rec["_revision_complete"] = True
            continue
        if rec["decision"] == "keep":
            rec["revised_sentence"] = rec["original_sentence"]
            rec["replacement"] = ""
            rec["retrieval_ids"] = []
        elif rec["decision"] == "abstain":
            rec["revised_sentence"] = apply_abstention(rec["original_sentence"])
            rec["replacement"] = ""
            rec["retrieval_ids"] = []
        elif (
            rec.get("mode") == "discriminative"
            and str(cfg.get("intervention", {}).get("discriminative_action"))
            == "target_retraction"
        ):
            rec["action"] = "retract"
            rec["revised_sentence"] = target_retraction_sentence(rec)
            rec["replacement"] = ""
            rec["retrieval_ids"] = []
            rec["parse_ok"] = True
            rec["schema_ok"] = True
            rec["revision_attempted"] = False
            rec["revision_parse_mode"] = "deterministic_target_retraction"
            rec["revision_error"] = None
        elif bool(rec.get("deterministic_retraction")):
            result = generative_target_retraction(rec)
            rec.update(result)
            if rec.get("decision") == "abstain":
                rec["revised_sentence"] = apply_abstention(rec["original_sentence"])
            rec["retrieval_ids"] = []
            rec["parse_ok"] = True
            rec["schema_ok"] = True
            rec["revision_attempted"] = False
        else:
            image = load_image(rec["image_path"])
            if image is None:
                rec["decision"] = "abstain"
                rec["action"] = "abstain"
                rec["revised_sentence"] = apply_abstention(rec["original_sentence"])
                rec["replacement"] = ""
                rec["retrieval_ids"] = []
                rec["parse_ok"] = False
                rec["schema_ok"] = False
                rec["revision_error"] = "image_unavailable"
            elif backbone is None:
                raise RuntimeError(
                    f"{rec.get('claim_id')}: pending revision has no loaded backbone"
                )
            else:
                if retriever is not None:
                    hits = retriever.retrieve(rec)
                    references = retriever.format_evidence(hits)
                    rec["retrieval_ids"] = [hit.get("image_id") for hit in hits]
                else:
                    references = "(retrieval-reference ablation)"
                    rec["retrieval_ids"] = []
                model_image = image
                if strategy == "no_image":
                    from PIL import Image

                    model_image = Image.new("RGB", image.size, color=(127, 127, 127))
                    rec["current_image_evidence"] = (
                        "(current-image evidence removed for ablation)"
                    )
                else:
                    rec["current_image_evidence"] = _current_image_summary(rec)
                rec["revision_attempted"] = True
                result = revise_claim(
                    backbone,
                    model_image,
                    rec,
                    references,
                    rec["current_image_evidence"],
                )
                rec.update(result)
                rec["revision_attempt_complete"] = True
                target_only = result.get("action") == "revise" and _target_only_edit(
                    rec
                )
                if not target_only:
                    if result.get("action") == "revise":
                        rec["revision_error"] = "non_target_only_edit_failed_closed"
                    rec["decision"] = "abstain"
                    rec["revised_sentence"] = apply_abstention(rec["original_sentence"])
                    rec["replacement"] = ""
        rec["_revision_complete"] = True
        if index % 10 == 0 or index == len(records):
            write_jsonl(partial_path, records)

    by_image: Dict[str, List[Dict[str, Any]]] = {}
    for rec in records:
        by_image.setdefault(str(rec.get("image_id")), []).append(rec)
    for recs in by_image.values():
        final = _compose_final_caption(recs)
        for rec in recs:
            rec["final_caption"] = final
            rec.pop("_revision_complete", None)
    write_jsonl(out_path, records)
    write_json(
        complete_path,
        {
            "row_count": len(records),
            "strategy": strategy,
            "schema_version": PIPELINE_SCHEMA_VERSION,
            "revision_adapter_version": REVISION_ADAPTER_VERSION,
        },
    )
    if partial_path.exists():
        partial_path.unlink()
    return records


def _pending_revision_generations(
    records: Sequence[Dict[str, Any]], out_path: Path, overwrite: bool = False
) -> int:
    """Count real VLM calls before a backbone or retriever is loaded."""
    checkpoint = _reusable_generation_cache(records, out_path, overwrite)
    return int(
        sum(
            rec.get("decision") == "revise"
            and rec.get("mode") != "discriminative"
            and not bool(rec.get("deterministic_retraction"))
            and str(rec.get("claim_id")) not in checkpoint
            for rec in records
        )
    )


def _reusable_generation_cache(
    records: Sequence[Dict[str, Any]], out_path: Path, overwrite: bool = False
) -> Dict[str, Dict[str, Any]]:
    """Return only cached generations that satisfy the current edit contract.

    A completion marker alone is insufficient: older runs may contain parsed
    placeholder text that is not an exact target-span replacement.  Valid rows
    remain reusable, while invalid rows are counted and regenerated.  Once a
    dataset run has completed under the current pipeline and adapter schemas,
    however, its failed-closed outcomes are also final experimental outcomes.
    Downstream RQ aggregation must reuse those outcomes instead of silently
    re-querying the VLM and changing both cost and run identity.
    """
    if overwrite:
        return {}
    complete_path = out_path.with_name(out_path.name + ".complete.json")
    complete_current = False
    if out_path.exists() and complete_path.exists():
        payload = read_json(complete_path)
        complete_current = (
            int(payload.get("row_count", -1)) == len(records)
            and int(payload.get("schema_version", -1)) == PIPELINE_SCHEMA_VERSION
            and int(payload.get("revision_adapter_version", -1))
            == REVISION_ADAPTER_VERSION
        )
    candidates: Dict[str, Dict[str, Any]] = {}
    if out_path.exists():
        candidates.update(
            {str(row.get("claim_id")): row for row in read_jsonl(out_path)}
        )
    partial_path = out_path.with_name(out_path.name + ".partial")
    if partial_path.exists() and not complete_current:
        candidates.update(
            {
                str(row.get("claim_id")): row
                for row in read_jsonl(partial_path)
                if bool(row.get("_revision_complete"))
            }
        )
    reusable: Dict[str, Dict[str, Any]] = {}
    for expected in records:
        if expected.get("decision") != "revise":
            continue
        claim_id = str(expected.get("claim_id"))
        cached = candidates.get(claim_id)
        if not cached:
            continue
        if expected.get("mode") == "discriminative" and not (
            cached.get("decision") == "revise"
            and cached.get("action") == "retract"
            and _target_retraction(cached)
        ):
            # Generated replacements from the superseded free-form revision
            # path are never reusable under the native-label retraction
            # contract, even if they happened to be target-only edits.
            continue
        expected_original = (
            claim_to_sentence(expected)
            if expected.get("mode") == "discriminative"
            else str(expected.get("sentence", ""))
        )
        expected_target = str(
            expected.get("attribute")
            if expected.get("claim_type") == "attr"
            else expected.get("relation")
        )
        normalise_target = lambda value: re.sub(  # noqa: E731
            r"[\s_-]+", "_", str(value).strip().lower()
        )
        adapter_current = (
            int(cached.get("revision_adapter_version", -1)) == REVISION_ADAPTER_VERSION
        )
        verified_revision = (
            cached.get("decision") == "revise"
            and cached.get("action") == "revise"
            and bool(cached.get("revision_attempted"))
            and bool(cached.get("parse_ok"))
            and bool(cached.get("schema_ok"))
            and _target_only_edit(cached)
        )
        valid_explicit_abstention = (
            cached.get("decision") == "abstain"
            and cached.get("action") == "abstain"
            and bool(cached.get("revision_attempted"))
            and bool(cached.get("parse_ok"))
            and bool(cached.get("schema_ok"))
            and cached.get("revision_error") in (None, "")
        )
        completed_run_outcome = (
            complete_current
            and int(cached.get("pipeline_schema_version", -1))
            == PIPELINE_SCHEMA_VERSION
            and bool(cached.get("revision_requested"))
            and cached.get("decision") in ("revise", "abstain")
        )
        if (
            (adapter_current or completed_run_outcome)
            and str(cached.get("original_sentence", "")) == expected_original
            and normalise_target(cached.get("target_span"))
            == normalise_target(expected_target)
            and (
                verified_revision or valid_explicit_abstention or completed_run_outcome
            )
        ):
            reusable[claim_id] = cached
    return reusable


def _enforce_generation_budget(
    cfg: Config,
    records: Sequence[Dict[str, Any]],
    out_path: Path,
    overwrite: bool = False,
) -> int:
    pending_generation = _pending_revision_generations(records, out_path, overwrite)
    max_generation_calls = int(
        cfg.get("revision_generation", {}).get("max_calls_per_job", 250)
    )
    if pending_generation > max_generation_calls:
        raise RuntimeError(
            f"revision-generation budget exceeded for {out_path.name}: "
            f"pending={pending_generation} > max_calls_per_job={max_generation_calls}; "
            "inspect the selection and explicitly raise "
            "revision_generation.max_calls_per_job before retrying"
        )
    return pending_generation


def _load_backbone_and_retriever(
    cfg: Config, backbone_name: str, strategy: str, records: Sequence[Dict[str, Any]]
):
    if not any(
        row.get("decision") == "revise"
        and row.get("mode") != "discriminative"
        and not bool(row.get("deterministic_retraction"))
        for row in records
    ):
        return None, None
    from ..backbones.registry import build_backbone

    backbone = build_backbone(cfg, backbone_name)
    retriever = None
    if strategy != "no_rag":
        _, retriever = _shared_resources(cfg)
    return backbone, retriever


def _decision_counts(records: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    return {
        decision: int(sum(str(row.get("decision")) == decision for row in records))
        for decision in ("keep", "abstain", "revise")
    }


def _select_gate_thresholds(
    calibration_rows: Sequence[Dict[str, Any]],
    q_by_group: Dict[str, float],
    gate_cfg: Dict[str, Any],
) -> tuple[Dict[str, Optional[float]], Dict[str, Dict[str, Any]]]:
    """Calibrate each group from native labels, failing closed if underpowered."""
    theta_by_group: Dict[str, Optional[float]] = {}
    status_by_group: Dict[str, Dict[str, Any]] = {}
    for group_key in sorted(q_by_group):
        group_rows = [
            row for row in calibration_rows if row.get("calibration_group") == group_key
        ]
        n_evaluable = sum(
            bool(row.get("signal_available"))
            and row.get("original_label", row.get("label")) in (0, 1)
            for row in group_rows
        )
        n_flagged_evaluable = sum(
            bool(row.get("flagged"))
            and bool(row.get("signal_available"))
            and row.get("original_label", row.get("label")) in (0, 1)
            for row in group_rows
        )
        try:
            theta = select_theta(
                group_rows,
                target_incorrect_intervention_rate=float(
                    gate_cfg.get("target_incorrect_intervention_rate", 0.05)
                ),
                min_calibration_claims=int(gate_cfg.get("min_calibration_claims", 10)),
                min_correct_claims_for_risk=int(
                    gate_cfg.get("min_correct_claims_for_risk", 5)
                ),
            )
        except ValueError as exc:
            theta_by_group[group_key] = None
            status_by_group[group_key] = {
                "status": "no_action",
                "reason": str(exc),
                "n_calibration_rows": len(group_rows),
                "n_native_labelled_claims": int(n_evaluable),
                "n_flagged_native_labelled_claims": int(n_flagged_evaluable),
            }
            LOG.warning(
                "residual gate group %s is fail-closed (no action): %s",
                group_key,
                exc,
            )
        else:
            theta_by_group[group_key] = float(theta)
            status_by_group[group_key] = {
                "status": "calibrated",
                "reason": "",
                "n_calibration_rows": len(group_rows),
                "n_native_labelled_claims": int(n_evaluable),
                "n_flagged_native_labelled_claims": int(n_flagged_evaluable),
            }
    return theta_by_group, status_by_group


def _load_full_gate_for_ablation(
    cfg: Config,
    backbone_name: str,
    dataset: str,
    q_by_group: Dict[str, float],
) -> Dict[str, Any]:
    """Load Full thresholds so content ablations change one component only."""
    path = cfg.path("logs") / f"gate_thresholds_{backbone_name}_{dataset}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} is required before running a revision-content ablation"
        )
    payload = read_json(path)
    if payload.get("schema_version") != PIPELINE_SCHEMA_VERSION:
        raise RuntimeError(
            f"{path.name} uses a stale pipeline schema; complete the Full strategy first"
        )
    stored_q = payload.get("q_hat", {})
    stored_theta = payload.get("theta", {})
    expected = set(q_by_group)
    if expected != set(stored_q) or expected != set(stored_theta):
        raise RuntimeError(
            f"{path.name} does not contain the same calibration groups as the ablation"
        )
    for group_key, value in q_by_group.items():
        if not np.isclose(
            float(stored_q[group_key]), float(value), rtol=0.0, atol=1e-12
        ):
            raise RuntimeError(
                f"{path.name} q_hat mismatch for {group_key}; refusing a confounded ablation"
            )
    return payload


def _apply_frozen_full_gate(
    test: Sequence[Dict[str, Any]], payload: Dict[str, Any]
) -> None:
    theta_by_group = payload["theta"]
    for rec in test:
        raw_theta = theta_by_group.get(str(rec.get("calibration_group")))
        theta = float(raw_theta) if raw_theta is not None else None
        rec["theta"] = theta
        rec["threshold_source_strategy"] = "full"
        rec["decision"] = decide(
            bool(rec["flagged"]),
            bool(rec["signal_available"]),
            rec.get("d_conflict"),
            theta,
        )


def preflight_one(
    cfg: Config,
    backbone_name: str,
    dataset: str,
    strategy: str = "full",
    alpha: Optional[float] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Read-only, model-free generation budget for one revision job."""
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}")
    src = cfg.path("processed") / f"claims_{backbone_name}_{dataset}.jsonl"
    if not src.exists():
        raise FileNotFoundError(src)
    if (
        dataset == "chair500"
        and strategy == "full"
        and str(cfg.get("quality_transfer", {}).get("policy"))
        == "object_presence_prune"
    ):
        from .chair_object import preflight as chair_preflight

        chair_report = chair_preflight(cfg, backbone_name, overwrite=overwrite)
        return {
            "backbone": backbone_name,
            "dataset": dataset,
            "strategy": strategy,
            "model_loaded": False,
            "files_written": False,
            "pending_test_generations": 0,
            "pending_next_run_generations": 0,
            "next_run_budget_is_upper_bound": False,
            **chair_report,
        }
    rows = [row for row in read_jsonl(src) if row.get("claim_type") in ("attr", "rel")]
    if any(row.get("split") == "calib" for row in rows):
        raise RuntimeError(f"{src.name} uses legacy calib/test splits")
    if dataset != "chair500":
        mark_revision_evaluation_sample(rows, cfg)
        selected_rows = [row for row in rows if row.get("revision_evaluation_selected")]
        missing_residual = sum(
            _signal_value(row) is not None and row.get("d_conflict") is None
            for row in selected_rows
        )
        if missing_residual:
            LOG.warning(
                "%s has %d selected claims without budgeted internal belief; "
                "the residual gate will fail them closed to Abstain",
                src.name,
                missing_residual,
            )

    suffix = "" if strategy == "full" else f"_{strategy}"
    report: Dict[str, Any] = {
        "backbone": backbone_name,
        "dataset": dataset,
        "strategy": strategy,
        "model_loaded": False,
        "files_written": False,
        "hard_cap_per_stage": int(
            cfg.get("revision_generation", {}).get("max_calls_per_job", 250)
        ),
    }
    if dataset == "chair500":
        payloads: Dict[str, Dict[str, Any]] = {}
        missing = []
        for source_dataset in ("amber_attr", "amber_rel"):
            path = (
                cfg.path("logs")
                / f"gate_thresholds_{backbone_name}_{source_dataset}.json"
            )
            payload = read_json(path) if path.exists() else {}
            if payload.get("schema_version") != PIPELINE_SCHEMA_VERSION:
                source_path = (
                    cfg.path("processed")
                    / f"claims_{backbone_name}_{source_dataset}.jsonl"
                )
                if not source_path.exists():
                    missing.append(
                        path.name + " (cannot derive: source claims missing)"
                    )
                    continue
                source_rows = [
                    row
                    for row in read_jsonl(source_path)
                    if row.get("claim_type") in ("attr", "rel")
                ]
                mark_revision_evaluation_sample(source_rows, cfg)
                source_calibration, _, source_q = _prepare_records(
                    source_rows,
                    "full",
                    float(alpha if alpha is not None else cfg["cp"]["alpha_default"]),
                    cfg,
                )
                source_theta, source_status = _select_gate_thresholds(
                    source_calibration, source_q, cfg["residual_gate"]
                )
                payload = {
                    "theta": source_theta,
                    "gate_status": source_status,
                    "q_hat": source_q,
                    "schema_version": PIPELINE_SCHEMA_VERSION,
                    "preflight_derived": True,
                }
            payloads[source_dataset] = payload
        if missing:
            report.update(
                status="deferred_thresholds",
                reason="; ".join(missing),
                pending_test_generations=None,
                pending_next_run_generations=None,
                next_run_budget_is_upper_bound=False,
            )
            return report
        test = _prepare_transfer_records(rows, payloads, strategy, cfg)
        out = cfg.path("processed") / f"revised_{backbone_name}_{dataset}{suffix}.jsonl"
        report.update(
            status="ready",
            test_decisions=_decision_counts(test),
            pending_test_generations=_pending_revision_generations(
                test, out, overwrite=overwrite
            ),
        )
        report["pending_next_run_generations"] = report["pending_test_generations"]
        report["next_run_budget_is_upper_bound"] = False
        return report

    alpha = float(alpha if alpha is not None else cfg["cp"]["alpha_default"])
    calibration, test, q_by_group = _prepare_records(rows, strategy, alpha, cfg)
    if strategy == "no_residual":
        _replace_residual_with_raw_grounding_deficit(calibration)
        _replace_residual_with_raw_grounding_deficit(test)
    elif strategy == "no_rag":
        _replace_residual_without_kb(calibration)
        _replace_residual_without_kb(test)
    out = cfg.path("processed") / f"revised_{backbone_name}_{dataset}{suffix}.jsonl"
    if strategy == "no_gate":
        for rec in test:
            rec["decision"] = (
                "revise"
                if rec["flagged"] and rec["signal_available"]
                else ("abstain" if rec["flagged"] else "keep")
            )
        report.update(
            status="ready",
            calibration_decisions={"keep": 0, "abstain": 0, "revise": 0},
            pending_calibration_generations=0,
            test_decisions=_decision_counts(test),
            pending_test_generations=_pending_revision_generations(
                test, out, overwrite=overwrite
            ),
        )
        report["pending_next_run_generations"] = report["pending_test_generations"]
        report["next_run_budget_is_upper_bound"] = False
        return report
    if strategy == "abstain_all":
        for rec in test:
            rec["decision"] = "abstain" if rec["flagged"] else "keep"
        report.update(
            status="ready",
            calibration_decisions={"keep": 0, "abstain": 0, "revise": 0},
            pending_calibration_generations=0,
            test_decisions=_decision_counts(test),
            pending_test_generations=0,
            pending_next_run_generations=0,
            next_run_budget_is_upper_bound=False,
        )
        return report
    if strategy in FROZEN_GATE_ABLATIONS:
        try:
            payload = _load_full_gate_for_ablation(
                cfg, backbone_name, dataset, q_by_group
            )
        except (FileNotFoundError, RuntimeError) as exc:
            report.update(
                status="deferred_full_threshold",
                reason=str(exc),
                calibration_decisions={"keep": 0, "abstain": 0, "revise": 0},
                pending_calibration_generations=0,
                pending_test_generations=None,
                pending_next_run_generations=None,
                next_run_budget_is_upper_bound=False,
            )
            return report
        _apply_frozen_full_gate(test, payload)
        pending_test = _pending_revision_generations(test, out, overwrite=overwrite)
        report.update(
            status="ready",
            threshold_source_strategy="full",
            calibration_decisions={"keep": 0, "abstain": 0, "revise": 0},
            pending_calibration_generations=0,
            test_decisions=_decision_counts(test),
            pending_test_generations=pending_test,
            pending_next_run_generations=pending_test,
            next_run_budget_is_upper_bound=False,
        )
        return report

    theta_by_group, gate_status = _select_gate_thresholds(
        calibration, q_by_group, cfg["residual_gate"]
    )
    for rec in test:
        theta = theta_by_group.get(str(rec.get("calibration_group")))
        rec["theta"] = theta
        rec["decision"] = decide(
            bool(rec["flagged"]),
            bool(rec["signal_available"]),
            rec.get("d_conflict"),
            theta,
        )
    report.update(
        status="ready",
        calibration_source="external_benchmark_gt",
        gate_status=gate_status,
        calibration_decisions={"keep": 0, "abstain": 0, "revise": 0},
        pending_calibration_generations=0,
        test_decisions=_decision_counts(test),
        pending_test_generations=_pending_revision_generations(
            test, out, overwrite=overwrite
        ),
    )
    report["pending_next_run_generations"] = report["pending_test_generations"]
    report["next_run_budget_is_upper_bound"] = False
    return report


def run_one(
    cfg: Config,
    backbone_name: str,
    dataset: str,
    strategy: str = "full",
    alpha: Optional[float] = None,
    overwrite: bool = False,
    do_revision: bool = True,
) -> List[Dict[str, Any]]:
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}")
    processed = cfg.path("processed")
    src = processed / f"claims_{backbone_name}_{dataset}.jsonl"
    if not src.exists():
        LOG.warning("missing %s", src.name)
        return []
    if (
        dataset == "chair500"
        and strategy == "full"
        and str(cfg.get("quality_transfer", {}).get("policy"))
        == "object_presence_prune"
    ):
        from .chair_object import run as run_chair_object

        return run_chair_object(cfg, backbone_name, overwrite=overwrite)
    rows = [row for row in read_jsonl(src) if row.get("claim_type") in ("attr", "rel")]
    if not rows:
        return []
    if any(row.get("split") == "calib" for row in rows):
        raise RuntimeError(
            f"{src.name} uses legacy calib/test splits; run migrate_claim_splits.py first"
        )
    if dataset != "chair500":
        mark_revision_evaluation_sample(rows, cfg)
        selected_rows = [row for row in rows if row.get("revision_evaluation_selected")]
        missing_residual = sum(
            _signal_value(row) is not None and row.get("d_conflict") is None
            for row in selected_rows
        )
        if missing_residual:
            LOG.warning(
                "%s has %d selected claims without budgeted internal belief; "
                "the residual gate will fail them closed to Abstain",
                src.name,
                missing_residual,
            )
        counts: Dict[str, int] = defaultdict(int)
        for row in selected_rows:
            counts[f"{_group_key(row)}|{row.get('split')}"] += 1
        write_json(
            cfg.path("logs") / f"revision_sampling_{backbone_name}_{dataset}.json",
            {
                "sampling_unit": "image_id",
                "decision_independent": True,
                "row_order_invariant": True,
                "revision_sampling_schema_version": REVISION_SAMPLING_SCHEMA_VERSION,
                "max_images_per_group_split": int(
                    cfg.get("revision_evaluation", {}).get(
                        "max_images_per_group_split", 100
                    )
                ),
                "counts": dict(sorted(counts.items())),
                "stable_claim_keys": sorted(
                    _stable_claim_key(row) for row in selected_rows
                ),
            },
        )

    alpha = float(alpha if alpha is not None else cfg["cp"]["alpha_default"])
    if dataset == "chair500":
        threshold_payloads = {}
        for source_dataset in ("amber_attr", "amber_rel"):
            threshold_path = (
                cfg.path("logs")
                / f"gate_thresholds_{backbone_name}_{source_dataset}.json"
            )
            if not threshold_path.exists():
                raise RuntimeError(
                    f"{threshold_path.name} is required before the CHAIR quality transfer run"
                )
            payload = read_json(threshold_path)
            if payload.get("schema_version") != PIPELINE_SCHEMA_VERSION:
                raise RuntimeError(
                    f"{threshold_path.name} uses stale pipeline schema; rerun the "
                    f"{source_dataset} calibration before CHAIR transfer"
                )
            threshold_payloads[source_dataset] = payload
        test = _prepare_transfer_records(rows, threshold_payloads, strategy, cfg)
        missing_residual = sum(
            row.get("d_conflict") is None and row.get("signal_available")
            for row in test
        )
        if missing_residual and strategy != "no_gate":
            LOG.warning(
                "[%s/%s] %d CHAIR claims lack a transferable residual; "
                "the decision rule fails them closed to Abstain",
                backbone_name,
                strategy,
                missing_residual,
            )
        suffix = "" if strategy == "full" else f"_{strategy}"
        out_path = processed / f"revised_{backbone_name}_{dataset}{suffix}.jsonl"
        pending_generation = _enforce_generation_budget(
            cfg, test, out_path, overwrite=overwrite
        )
        backbone, retriever = (
            _load_backbone_and_retriever(cfg, backbone_name, strategy, test)
            if pending_generation
            else (None, None)
        )
        completed = _perform_actions(
            cfg,
            backbone_name,
            dataset,
            strategy,
            test,
            out_path,
            backbone=backbone,
            retriever=retriever,
            overwrite=overwrite,
        )
        labeled = write_quality_only(cfg, backbone_name, dataset, strategy, completed)
        if backbone is not None:
            backbone.unload()
            _collect()
        return labeled

    calibration, test, q_by_group = _prepare_records(rows, strategy, alpha, cfg)
    if strategy == "no_residual":
        _replace_residual_with_raw_grounding_deficit(calibration)
        _replace_residual_with_raw_grounding_deficit(test)
    elif strategy == "no_rag":
        _replace_residual_without_kb(calibration)
        _replace_residual_without_kb(test)
    if not test:
        return []
    suffix = "" if strategy == "full" else f"_{strategy}"
    out_path = processed / f"revised_{backbone_name}_{dataset}{suffix}.jsonl"

    theta_by_group: Dict[str, Optional[float]] = {}
    if strategy in ("no_gate",):
        for rec in test:
            rec["decision"] = (
                "revise"
                if rec["flagged"] and rec["signal_available"]
                else ("abstain" if rec["flagged"] else "keep")
            )
            rec["theta"] = None
    elif strategy == "abstain_all":
        for rec in test:
            rec["decision"] = "abstain" if rec["flagged"] else "keep"
            rec["theta"] = None
    elif strategy in FROZEN_GATE_ABLATIONS:
        payload = _load_full_gate_for_ablation(cfg, backbone_name, dataset, q_by_group)
        _apply_frozen_full_gate(test, payload)
    else:
        theta_by_group, gate_status_by_group = _select_gate_thresholds(
            calibration, q_by_group, cfg["residual_gate"]
        )
        write_json(
            cfg.path("logs")
            / f"gate_thresholds_{backbone_name}_{dataset}{suffix}.json",
            {
                "theta": theta_by_group,
                "gate_status": gate_status_by_group,
                "q_hat": q_by_group,
                "schema_version": PIPELINE_SCHEMA_VERSION,
            },
        )
        for rec in test:
            theta = theta_by_group.get(str(rec.get("calibration_group")))
            rec["theta"] = theta
            rec["decision"] = decide(
                bool(rec["flagged"]),
                bool(rec["signal_available"]),
                rec.get("d_conflict"),
                theta,
            )

    if not do_revision:
        for rec in test:
            rec["original_sentence"] = rec.get("sentence", "")
            rec["revised_sentence"] = rec["original_sentence"]
            rec["replacement"] = ""
            rec["final_caption"] = rec.get("caption", "")
        return test

    pending_test_generation = _enforce_generation_budget(
        cfg, test, out_path, overwrite=overwrite
    )
    backbone, retriever = (
        _load_backbone_and_retriever(cfg, backbone_name, strategy, test)
        if pending_test_generation
        else (None, None)
    )
    completed = _perform_actions(
        cfg,
        backbone_name,
        dataset,
        strategy,
        test,
        out_path,
        backbone=backbone,
        retriever=retriever,
        overwrite=overwrite,
    )
    labeled = verify_and_write(cfg, backbone_name, dataset, strategy, completed)
    if backbone is not None:
        backbone.unload()
        _collect()
    return labeled


def run_all(cfg: Config, overwrite: bool = False) -> None:
    # The four discriminative evaluation tasks use their own calibration labels;
    # CHAIR-500 is the frozen AMBER-threshold caption-quality transfer.
    datasets = ["amber_attr", "amber_rel", "rbench", "mmrel", "chair500"]
    failures: List[str] = []
    for backbone_name in cfg.backbone_names:
        for dataset in datasets:
            try:
                run_one(cfg, backbone_name, dataset, "full", overwrite=overwrite)
            except Exception as exc:  # noqa: BLE001
                LOG.error("[%s/%s] pipeline failed: %s", backbone_name, dataset, exc)
                failures.append(f"{backbone_name}/{dataset}: {exc}")
        _collect()
    write_overhead(cfg)
    if failures:
        raise RuntimeError("revision pipeline failed:\n  " + "\n  ".join(failures))
    from ..utils import mark_done

    mark_done(cfg.path("processed"), "revise_done")


def write_overhead(cfg: Config) -> None:
    logs = cfg.path("logs")
    lines = [
        "# computational overhead (Section 5.5)",
        f"pipeline schema version   : {PIPELINE_SCHEMA_VERSION}",
        "",
    ]
    tool_calls = logs / "signal_tool_calls.json"
    if tool_calls.exists():
        calls = read_json(tool_calls)
        lines += [
            f"CLIP image encodings      : {calls.get('clip_image_calls', 0)}",
            f"CLIP text encodings       : {calls.get('clip_text_calls', 0)}",
            f"Grounding DINO forwards   : {calls.get('dino_forward', 0)}",
        ]
    n_revised = n_total = 0
    images_with_revision, all_images = set(), set()
    for path in cfg.path("processed").glob("revised_labeled_*.jsonl"):
        if any(
            token in path.stem
            for token in ("no_gate", "abstain_all", "no_cp", "no_rag", "no_image")
        ):
            continue
        for row in read_jsonl(path):
            n_total += 1
            key = (path.stem, row.get("image_id"))
            all_images.add(key)
            if row.get("decision") == "revise":
                n_revised += 1
                images_with_revision.add(key)
    fraction = len(images_with_revision) / len(all_images) if all_images else 0.0
    lines += [
        f"claims processed          : {n_total}",
        f"claims revised            : {n_revised}",
        f"images triggering an extra VLM generation : {len(images_with_revision)} "
        f"({fraction:.1%} of images)",
        "",
        "VLM, CLIP and Grounding DINO parameters remain frozen. Lightweight probes "
        "and statistical calibration are reported separately.",
    ]
    path = logs / "overhead.txt"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tmp.replace(path)
    LOG.info("overhead written to %s", path)
