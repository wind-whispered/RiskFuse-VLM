"""Fresh same-machine latency for the native atomic-claim evaluation path.

Model load and image decode are recorded separately. Full/visual-verifier runs
both recompute visual evidence with detector caches cleared. This does not time
free-form extraction; native claims are supplied by benchmark annotations.
"""

from __future__ import annotations

import hashlib
import json
import platform
import time

import numpy as np

from ..sampling import image_unit_rows
from ..utils import read_jsonl, write_json, claim_to_sentence
from ..signals.support import matched_signal_group, attribute_kind, relation_signal_kind
from ..eval.detection_metrics import select_f1_threshold
from .audit import output_dir, write_csv
from .overlap import sha256


def select_jobs(cfg, backbone):
    jobs = []
    n = cfg["reviewer_revision"]["latency_images_per_dataset"]
    warmup = cfg["reviewer_revision"]["latency_warmup_images_per_dataset"]
    from ..revise.pipeline import PIPELINE_SCHEMA_VERSION

    for dataset in cfg["datasets"]["detect"]:
        path = cfg.path("processed") / f"claims_{backbone}_{dataset}.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        rows = read_jsonl(path)
        threshold_path = cfg.path("logs") / f"gate_thresholds_{backbone}_{dataset}.json"
        if not threshold_path.exists():
            raise FileNotFoundError(threshold_path)
        thresholds = json.loads(threshold_path.read_text(encoding="utf-8"))
        if thresholds.get("schema_version") != PIPELINE_SCHEMA_VERSION:
            raise ValueError("latency requires current frozen gate thresholds")
        test = image_unit_rows(
            [
                r
                for r in rows
                if r.get("split") == "test" and r.get("claim_type") in ("attr", "rel")
            ],
            int(cfg.get("seed", 42)),
            "latency",
        )
        chosen = sorted(
            test,
            key=lambda r: hashlib.sha256(
                f"{cfg.get('seed',42)}|latency|{r['image_id']}".encode()
            ).hexdigest(),
        )[: n + warmup]
        if len(chosen) < n + warmup:
            raise ValueError(f"insufficient prespecified latency images: {dataset}")
        for i, row in enumerate(chosen):
            group = matched_signal_group(row)
            cp = [
                r
                for r in rows
                if r.get("split") == "calib_cp"
                and matched_signal_group(r) == group
                and r.get("g_raw") is not None
            ]
            if not cp:
                raise ValueError(f"no source scaling for {dataset}/{group}")
            lo = min(float(r["g_raw"]) for r in cp)
            hi = max(float(r["g_raw"]) for r in cp)
            if hi - lo < 1e-8:
                hi = lo + 1
            fixed = select_f1_threshold(
                [1 - float(r["g"]) for r in cp if r.get("g") is not None],
                [int(r["label"]) for r in cp if r.get("g") is not None],
            )
            jobs.append(
                dict(
                    row=row,
                    dataset=dataset,
                    warmup=i < warmup,
                    lo=lo,
                    hi=hi,
                    q=thresholds.get("q_hat", {}).get(group),
                    theta=thresholds.get("theta", {}).get(group),
                    fixed=fixed,
                    thresholds_sha256=sha256(threshold_path),
                )
            )
    return jobs


def run(cfg, backbone, apply=False, max_generation_calls=None):
    jobs = select_jobs(cfg, backbone)
    bound = 2 * len(jobs)
    report = dict(
        backbone=backbone,
        n_images_including_warmup=len(jobs),
        n_measured_images=sum(not j["warmup"] for j in jobs),
        generation_call_upper_bound=bound,
        gpu_launched=False,
        scope="native_atomic_claim_path; free_form_extraction_excluded; model_load_and_image_decode_separate",
    )
    if not apply:
        return report
    if max_generation_calls is None or max_generation_calls < bound:
        raise ValueError(
            "latency needs an explicit --max-generation-calls at least its prespecified upper bound"
        )
    from ..backbones.registry import build_backbone
    from ..tools.clip_scorer import ClipScorer
    from ..tools.grounding_dino import GroundingDino
    from ..signals.grounding_attr import g_attr
    from ..signals.grounding_count import count_consistency
    from ..signals.grounding_rel import g_rel
    from ..signals.internal_conf import verify_confidence_signals
    from ..data.images import load_image
    from ..revise.rag_retrieve import EvidenceRetriever
    from ..revise.post_revision import target_retraction_sentence
    import torch

    sync = lambda: (
        torch.cuda.synchronize(cfg.get("device", "cuda"))
        if str(cfg.get("device", "cuda")).startswith("cuda")
        else None
    )

    def measured(function):
        sync()
        start = time.perf_counter()
        value = function()
        sync()
        return value, time.perf_counter() - start

    started = time.perf_counter()
    model = build_backbone(cfg, backbone)
    clip = ClipScorer(
        cfg.resource_id(cfg["tools"]["clip_hf_id"], cfg['tools'].get('clip_local_path')), cfg["device"], cfg["dtype"], cfg.path("tool_cache")
    )
    dino = GroundingDino(
        cfg.resource_id(cfg["tools"]["grounding_dino_hf_id"], cfg['tools'].get('grounding_dino_local_path')),
        cfg["device"],
        cfg["dtype"],
        cfg["tools"]["grounding_dino_box_threshold"],
        cfg["tools"]["grounding_dino_text_threshold"],
        cfg.path("tool_cache"),
    )
    retriever = EvidenceRetriever(cfg, clip=None)
    report.update(
        model_and_tool_load_seconds=time.perf_counter() - started,
        hardware=(
            torch.cuda.get_device_name()
            if torch.cuda.is_available()
            else platform.processor()
        ),
        python_version=platform.python_version(),
        torch_version=torch.__version__,
        gpu_launched=True,
    )
    records = []
    calls = 0
    try:
        for job in jobs:
            row = job["row"]
            image, decode = measured(lambda: load_image(row["image_path"]))
            if image is None:
                raise FileNotFoundError(row["image_path"])
            _, generation = measured(
                lambda: model.generate(
                    image,
                    cfg["generation"]["caption_prompt"],
                    max_new_tokens=cfg["generation"]["max_new_tokens"],
                )
            )
            calls += 1

            def grounding():
                kind = (
                    attribute_kind(row)
                    if row["claim_type"] == "attr"
                    else relation_signal_kind(row)
                )
                if row["claim_type"] == "attr" and kind == "state":
                    value = g_attr(
                        clip,
                        dino,
                        image,
                        row["object"],
                        row["attribute"],
                        cache_key=row["image_id"],
                    ).get("g_attr")
                elif row["claim_type"] == "attr" and kind == "number":
                    value = count_consistency(
                        dino,
                        image,
                        row["object"],
                        row["attribute"],
                        cache_key=row["image_id"],
                    ).get("g_count")
                elif kind == "global" or row["claim_type"] == "attr":
                    value = clip.cos(
                        clip.encode_image(image),
                        clip.encode_text(claim_to_sentence(row)),
                    )
                else:
                    value = g_rel(
                        dino,
                        image,
                        row["subject"],
                        row["relation"],
                        row["object"],
                        cfg["relation_contact_keywords"],
                        cfg["relation_direction_vectors"],
                        cache_key=row["image_id"],
                    ).get("g_rel")
                return (
                    None
                    if value is None
                    else float(
                        np.clip((value - job["lo"]) / (job["hi"] - job["lo"]), 0, 1)
                    )
                )

            dino._cache.clear()

            def full():
                nonlocal calls
                g = grounding()
                if g is None or job["q"] is None:
                    return "abstain"
                if 1 - g < job["q"]:
                    return "keep"
                if job["theta"] is None:
                    return "abstain"
                confidence = verify_confidence_signals(
                    model, image, claim_to_sentence(row)
                )
                calls += 1
                if confidence["l_p"] is None or job["theta"] is None:
                    return "abstain"
                prior = retriever.plausibility_prior(row)
                needed = (
                    "residual_intercept",
                    "residual_beta",
                    "residual_beta_kb",
                    "residual_center",
                    "residual_mad",
                )
                if any(row.get(k) is None for k in needed):
                    raise ValueError(
                        "latency requires current saved residual coefficients"
                    )
                prediction = (
                    row["residual_intercept"]
                    + row["residual_beta"] * confidence["l_p"]
                    + row["residual_beta_kb"] * prior["kb_prior"]
                )
                prediction += (
                    row.get("residual_beta_kb_available", 0)
                    * prior["kb_prior_available"]
                )
                d = -(g - prediction - row["residual_center"]) / (
                    row["residual_mad"] + 1e-8
                )
                if d <= job["theta"]:
                    return "abstain"
                target_retraction_sentence(row)
                return "revise"

            decision, full_seconds = measured(full)
            dino._cache.clear()

            def baseline():
                g = grounding()
                if g is not None and 1 - g >= job["fixed"]:
                    target_retraction_sentence(row)
                    return "revise"
                return "abstain" if g is None else "keep"

            baseline_decision, baseline_seconds = measured(baseline)
            record = dict(
                backbone=backbone,
                dataset=job["dataset"],
                claim_id=row["claim_id"],
                image_id=row["image_id"],
                warmup=job["warmup"],
                image_decode_seconds=decode,
                backbone_generation_seconds=generation,
                full_atomic_intervention_seconds=full_seconds,
                visual_verifier_baseline_seconds=baseline_seconds,
                full_total_seconds=generation + full_seconds,
                baseline_total_seconds=generation + baseline_seconds,
                n_claims=1,
                decision=decision,
                baseline_decision=baseline_decision,
                cache_condition="detector_cache_cleared_for_each_method; text_encodings_recomputed; models_warm",
                thresholds_sha256=job["thresholds_sha256"],
            )
            records.append(record)
            write_csv(output_dir(cfg) / f"latency_records_{backbone}.csv", records)
    finally:
        model.unload()
    report.update(status="complete", actual_generation_calls=calls)
    write_json(output_dir(cfg) / f"latency_manifest_{backbone}.json", report)
    return report
