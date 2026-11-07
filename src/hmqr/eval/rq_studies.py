"""RQ2/RQ3/RQ4 analyses under the revised outcome contract."""
from __future__ import annotations

import json
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ..config import Config
from ..revise import pipeline
from ..utils import get_logger, read_jsonl
from . import caption_quality, mitigation_metrics

LOG = get_logger("hmqr.rq")
RQ_SCHEMA_VERSION = 20
RQ2_BACKBONES = ["llava15", "instructblip"]
RQ2_STRATEGIES = [
    ("no_gate", "revise-all (no gate)"),
    ("full", "full"),
    ("abstain_all", "abstain-all"),
]
RQ4_BACKBONE = "llava15"
RQ4_REVISION_ABLATIONS = [
    ("full", "Full"),
    ("no_gate", "w/o residual gate"),
    ("no_cp", "fixed threshold (w/o CP)"),
    ("no_residual", "w/o residualization"),
    ("no_rag", "w/o VG prior"),
    ("abstain_all", "abstain-all"),
]


def _ensure(
    cfg: Config, backbone: str, dataset: str, strategy: str, overwrite: bool = False
) -> List[Dict[str, Any]]:
    return pipeline.run_one(cfg, backbone, dataset, strategy, overwrite=overwrite)


def _summary_record(
    rows: List[Dict[str, Any]], seed: int, **identity: Any
) -> Dict[str, Any]:
    stats = mitigation_metrics.summarise(rows)
    ci_low, ci_high = mitigation_metrics.image_bootstrap_ci(rows, seed=seed)
    risk_ci_low, risk_ci_high = mitigation_metrics.image_bootstrap_ci(
        rows, metric="incorrect_intervention_rate", seed=seed + 1
    )
    return {
        **identity,
        **stats,
        "rq_schema_version": RQ_SCHEMA_VERSION,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "incorrect_intervention_ci_low": risk_ci_low,
        "incorrect_intervention_ci_high": risk_ci_high,
        "n_images": len({str(row.get("image_id")) for row in rows}),
    }


def _rbench_selection_risk_ci(
    cfg: Config,
    backbone: str,
    coverage_group: pd.DataFrame,
    seed: int,
    n_boot: int = 1000,
) -> tuple[float, float, float]:
    """Cluster-bootstrap R-Bench hallucination miss risk by image."""
    thresholds = {
        str(row.get("rel_kind") or "all"): float(row["q_hat"])
        for row in coverage_group.to_dict("records")
        if np.isfinite(float(row["q_hat"]))
    }
    rows = read_jsonl(
        cfg.path("processed") / f"claims_{backbone}_rbench.jsonl"
    )
    by_image: Dict[str, List[float]] = {}
    for row in rows:
        subtype = str(row.get("rel_kind") or "all")
        try:
            grounding = float(row.get("g"))
        except (TypeError, ValueError):
            continue
        if (
            row.get("split") != "test"
            or row.get("label") != 1
            or subtype not in thresholds
            or not np.isfinite(grounding)
        ):
            continue
        missed = float((1.0 - grounding) < thresholds[subtype])
        by_image.setdefault(str(row.get("image_id")), []).append(missed)
    image_ids = sorted(by_image)
    if not image_ids:
        return float("nan"), float("nan"), float("nan")
    point = float(np.mean([value for image in image_ids for value in by_image[image]]))
    if len(image_ids) < 2:
        return point, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(int(n_boot)):
        sampled = rng.choice(image_ids, size=len(image_ids), replace=True)
        values.append(float(np.mean([
            value for image in sampled for value in by_image[str(image)]
        ])))
    return point, float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def run_rq2(
    cfg: Config, overwrite: bool = False, recompute_revisions: bool = False
) -> pd.DataFrame:
    out = cfg.path("tables") / "rq2_gate.csv"
    if out.exists() and not overwrite:
        existing = pd.read_csv(out)
        if (
            {"target_hallucination_removal_rate", "incorrect_intervention_rate",
             "external_label_coverage", "revision_attempt_rate",
             "rq_schema_version"} <= set(existing)
            and existing["rq_schema_version"].eq(RQ_SCHEMA_VERSION).all()
        ):
            LOG.info("[skip] %s exists with current schema", out.name)
            return existing
    records: List[Dict[str, Any]] = []
    pooled_rows: Dict[tuple[str, str], List[Dict[str, Any]]] = {}
    seed = int(cfg.get("seed", 42))
    for backbone in RQ2_BACKBONES:
        if backbone not in cfg.backbone_names:
            continue
        for strategy, label in RQ2_STRATEGIES:
            for dataset, claim_type in (("amber_attr", "attr"), ("amber_rel", "rel")):
                rows = _ensure(
                    cfg, backbone, dataset, strategy, recompute_revisions
                )
                subset = [row for row in rows
                          if row.get("label", row.get("original_label")) in (0, 1)]
                if subset:
                    pooled_rows.setdefault((backbone, label), []).extend(subset)
                    records.append(_summary_record(
                        subset, seed,
                        backbone=backbone, dataset=dataset,
                        claim_type=claim_type, strategy=label, plot_scope="dataset",
                    ))
    for (backbone, label), rows in sorted(pooled_rows.items()):
        records.append(_summary_record(
            rows, seed,
            backbone=backbone, dataset="pooled",
            claim_type="attr+rel", strategy=label, plot_scope="pooled",
        ))
    df = pd.DataFrame(records)
    if not df.empty:
        df.to_csv(out, index=False)
        LOG.info("wrote %s", out.name)
    return df


def _quality_from_cached_chair(
    cfg: Config,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Read CHAIR quality only after a separately calibrated transfer run exists."""
    output: List[Dict[str, Any]] = []
    reduction_output: List[Dict[str, Any]] = []
    for backbone in cfg.backbone_names:
        claims_path = cfg.path("processed") / f"claims_{backbone}_chair500.jsonl"
        claims = read_jsonl(claims_path)
        revised: List[Dict[str, Any]] = []
        if claims:
            try:
                # Always pass through run_one: its completion marker checks the
                # current pipeline schema before allowing a cached transfer.
                revised = _ensure(cfg, backbone, "chair500", "full", False)
            except Exception as exc:  # noqa: BLE001
                LOG.warning("[%s] CHAIR transfer failed: %s", backbone, exc)
        if not claims or not revised:
            LOG.warning("[%s] CHAIR transfer output unavailable; quality metrics are N/A", backbone)
            continue
        final_by_image = {str(row["image_id"]): row.get("final_caption", "") for row in revised}
        # Claim extraction may legitimately return zero atomic attribute/relation
        # claims for a generated caption.  Such an image must still participate
        # in the fixed CHAIR-500 caption-quality denominator.  A small
        # supplemental cache stores captions for these zero-claim images without
        # forcing a full Stage-01 regeneration.
        caption_by_image = {
            str(row["image_id"]): row.get("caption", "")
            for row in claims
            if row.get("caption")
        }
        supplemental_path = (
            cfg.path("processed") / f"chair_captions_{backbone}.jsonl"
        )
        if supplemental_path.exists():
            for row in read_jsonl(supplemental_path):
                if row.get("caption"):
                    caption_by_image[str(row["image_id"])] = row["caption"]
        from ..data.datasets import load_dataset

        samples = {str(sample.image_id): sample for sample in load_dataset(cfg, "chair500")}
        vanilla, ours, objects, references = [], [], [], []
        missing_captions = []
        for image_id, sample in samples.items():
            caption = caption_by_image.get(image_id, "")
            if not caption:
                missing_captions.append(image_id)
                continue
            vanilla.append(caption)
            ours.append(final_by_image.get(image_id) or caption)
            image_references = sample.gt.get("references", [])
            objects.append(sorted(caption_quality.chair_ground_truth_objects(
                sample.gt.get("objects", []), image_references
            )))
            references.append(image_references)
        if not vanilla:
            continue
        expected_images = int(cfg["datasets"]["coco"].get("chair_num_images", 500))
        if len(vanilla) != expected_images:
            raise RuntimeError(
                f"{backbone}: CHAIR quality coverage={len(vanilla)}/{expected_images} "
                f"images; missing_caption_ids={missing_captions[:10]}"
            )
        vanilla_stats = caption_quality.summarise(vanilla, objects, references)
        ours_stats = caption_quality.summarise(ours, objects, references)
        chair_reduction, ci_low, ci_high = caption_quality.chair_i_reduction_ci(
            vanilla, ours, objects, seed=int(cfg.get("seed", 42))
        )
        for metric in ("CHAIR-i", "BLEU-4", "ROUGE-L", "obj_recall"):
            output.append({
                "backbone": backbone, "metric": metric,
                "vanilla": vanilla_stats[metric], "ours": ours_stats[metric],
                "n_images": len(vanilla), "rq_schema_version": RQ_SCHEMA_VERSION,
            })
        reduction_output.append({
            "backbone": backbone,
            "dataset": "chair500",
            "claim_type": "object",
            "base_hallu_rate": vanilla_stats["CHAIR-i"],
            "remaining_target_error_rate": ours_stats["CHAIR-i"],
            "target_hallucination_removal_rate": float("nan"),
            "incorrect_intervention_rate": float("nan"),
            "net_hallucination_reduction": chair_reduction,
            "object_hallucination_change": -chair_reduction,
            "selection_risk": float("nan"),
            "output_coverage": 1.0,
            "abstention_rate": float("nan"),
            "revision_request_rate": float("nan"),
            "revision_attempt_rate": float("nan"),
            "revision_rate": float("nan"),
            "valid_revision_yield": float("nan"),
            "failed_revision_attempt_rate": float("nan"),
            "non_target_preservation_rate": float("nan"),
            "signal_unavailable_rate": float("nan"),
            "external_label_coverage": 1.0,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "incorrect_intervention_ci_low": float("nan"),
            "incorrect_intervention_ci_high": float("nan"),
            "n_images": len(vanilla),
            "n_claims": len(vanilla),
            "rq_schema_version": RQ_SCHEMA_VERSION,
        })
    return output, reduction_output


def run_rq3(
    cfg: Config, overwrite: bool = False, recompute_revisions: bool = False
) -> Dict[str, pd.DataFrame]:
    tables = cfg.path("tables")
    reduction_path = tables / "rq3_reduction.csv"
    quality_path = tables / "rq3_quality.csv"
    if reduction_path.exists() and quality_path.exists() and not overwrite:
        existing = pd.read_csv(reduction_path)
        quality_existing = pd.read_csv(quality_path)
        expected_images = int(cfg["datasets"]["coco"].get("chair_num_images", 500))
        expected_pairs = {
            (backbone, dataset)
            for backbone in cfg.backbone_names
            for dataset in ("amber_attr", "amber_rel", "rbench", "mmrel", "chair500")
        }
        observed_pairs = set(zip(existing["backbone"], existing["dataset"]))
        if (
            {"net_hallucination_reduction", "rq_schema_version"} <= set(existing)
            and existing["rq_schema_version"].eq(RQ_SCHEMA_VERSION).all()
            and observed_pairs == expected_pairs
            and {"n_images", "rq_schema_version"} <= set(quality_existing)
            and quality_existing["rq_schema_version"].eq(RQ_SCHEMA_VERSION).all()
            and quality_existing["n_images"].eq(expected_images).all()
        ):
            return {"reduction": existing, "quality": quality_existing}

    seed = int(cfg.get("seed", 42))
    records: List[Dict[str, Any]] = []
    for backbone in cfg.backbone_names:
        for dataset, claim_type in (
            ("amber_attr", "attr"), ("amber_rel", "rel"),
            ("rbench", "rel"), ("mmrel", "rel")
        ):
            rows = _ensure(
                cfg, backbone, dataset, "full", recompute_revisions
            )
            subset = [row for row in rows
                      if row.get("label", row.get("original_label")) in (0, 1)]
            if subset:
                records.append(_summary_record(
                    subset, seed,
                    backbone=backbone, dataset=dataset,
                    claim_type=claim_type, selection_risk=float("nan"),
                    signal_unavailable_rate=float(np.mean([
                        not bool(row.get("signal_available")) for row in subset
                    ])),
                ))

    quality_records, chair_records = _quality_from_cached_chair(cfg)
    records.extend(chair_records)
    reduction = pd.DataFrame(records)
    quality = pd.DataFrame(quality_records)
    if not reduction.empty:
        reduction.to_csv(reduction_path, index=False)
    quality.to_csv(quality_path, index=False)
    LOG.info("wrote RQ3 tables")
    return {"reduction": reduction, "quality": quality}


def run_rq4(
    cfg: Config, overwrite: bool = False, recompute_revisions: bool = False
) -> pd.DataFrame:
    out = cfg.path("tables") / "rq4_ablation.csv"
    if out.exists() and not overwrite:
        existing = pd.read_csv(out)
        if (
            {"panel", "target_hallucination_removal_rate", "rq_schema_version"} <= set(existing)
            and existing["rq_schema_version"].eq(RQ_SCHEMA_VERSION).all()
        ):
            return existing
    backbone = RQ4_BACKBONE if RQ4_BACKBONE in cfg.backbone_names else cfg.backbone_names[0]
    records: List[Dict[str, Any]] = []

    detection_path = cfg.path("tables") / "rq1_detection.csv"
    coverage_path = cfg.path("tables") / "rq1_coverage.csv"
    if detection_path.exists():
        detection = pd.read_csv(detection_path)
        detection = detection[
            (detection["backbone"] == backbone)
            & detection["dataset"].isin(["amber_attr", "amber_rel"])
            & detection["method"].isin(["Ours", "GlobalCLIP-thr"])
        ]
        for row in detection.to_dict("records"):
            records.append({
                "panel": "A", "dataset": row["dataset"],
                "claim_type": row["claim_type"],
                "variant": (
                    "Granularity-matched signal"
                    if row["method"] == "Ours" else "Global CLIP"
                ),
                "auroc": row["auroc"], "ap": row["ap"],
                "empirical_recall": np.nan, "fpr": np.nan, "flag_rate": np.nan,
                "target_hallucination_removal_rate": np.nan,
                "incorrect_intervention_rate": np.nan,
                "net_hallucination_reduction": np.nan, "output_coverage": np.nan,
                "bleu4": np.nan, "external_label_coverage": np.nan,
                "n_images": row["n_images"], "n_claims": row["n_claims"],
                "rq_schema_version": RQ_SCHEMA_VERSION,
            })
    if coverage_path.exists():
        coverage = pd.read_csv(coverage_path)
        alpha = float(cfg["cp"]["alpha_default"])
        coverage = coverage[
            (coverage["backbone"] == backbone)
            & coverage["dataset"].isin(["amber_attr", "amber_rel"])
            & np.isclose(coverage["alpha"], alpha)
            & coverage["method"].isin(["Ours", "Ours-fixed"])
        ]
        for row in coverage.to_dict("records"):
            records.append({
                "panel": "A", "dataset": row["dataset"],
                "claim_type": row["claim_type"],
                "variant": "CP" if row["method"] == "Ours" else "Fixed threshold",
                "auroc": np.nan, "ap": np.nan,
                "empirical_recall": row["empirical_recall"],
                "fpr": row["fpr"], "flag_rate": row["flag_rate"],
                "target_hallucination_removal_rate": np.nan,
                "incorrect_intervention_rate": np.nan,
                "net_hallucination_reduction": np.nan, "output_coverage": np.nan,
                "bleu4": np.nan, "external_label_coverage": np.nan,
                "n_images": row["n_images"], "n_claims": np.nan,
                "rq_schema_version": RQ_SCHEMA_VERSION,
            })

    seed = int(cfg.get("seed", 42))
    for strategy, label in RQ4_REVISION_ABLATIONS:
        for dataset, claim_type in (("amber_attr", "attr"), ("amber_rel", "rel")):
            rows = _ensure(
                cfg, backbone, dataset, strategy, recompute_revisions
            )
            subset = [row for row in rows
                      if row.get("label", row.get("original_label")) in (0, 1)]
            if not subset:
                continue
            record = _summary_record(
                subset, seed, panel="B", dataset=dataset,
                claim_type=claim_type, variant=label,
            )
            record.update(auroc=np.nan, ap=np.nan, empirical_recall=np.nan,
                          fpr=np.nan, flag_rate=np.nan, bleu4=np.nan)
            records.append(record)

    result = pd.DataFrame(records)
    if not result.empty:
        result.to_csv(out, index=False)
        LOG.info("wrote %s", out.name)
    return result
