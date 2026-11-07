"""Shared-claim gate baselines, sensitivity, frozen transfer and paired inference."""

from __future__ import annotations

import json
import csv
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

from ..detect.conformal import calibrate, fixed_threshold, flag_hallucination
from ..eval.detection_metrics import select_f1_threshold
from ..residual.decomposition import RESIDUAL_SCHEMA_VERSION
from ..revise.pipeline import mark_revision_evaluation_sample
from ..revise.residual_gate import select_theta, select_theta_ltt
from ..sampling import image_unit_rows
from ..signals.lp import has_current_lp
from ..backbones.registry import verification_adapter_version
from ..signals.support import matched_signal_group
from ..utils import read_jsonl, write_json
from .audit import output_dir, write_csv
from .overlap import sha256


def finite(row, key):
    try:
        return bool(np.isfinite(float(row.get(key))))
    except (TypeError, ValueError):
        return False


def metric_vectors(labels, decisions):
    """Per-image additive terms, preserving abstentions in population denominators."""
    y = np.asarray(labels)
    revise = np.asarray(decisions) == "revise"
    release = np.asarray(decisions) != "abstain"
    removed = revise & (y == 1)
    incorrect = revise & (y == 0)
    return y, removed.astype(float), incorrect.astype(float), release.astype(float)


def point(labels, decisions):
    y, h, c, coverage = metric_vectors(labels, decisions)
    n1 = int((y == 1).sum())
    n0 = int((y == 0).sum())
    return dict(
        n_claims=len(y),
        n_hallucinated=n1,
        n_correct=n0,
        h_removed=int(h.sum()),
        c_intervened=int(c.sum()),
        nhr=float((h.sum() - c.sum()) / n1) if n1 else np.nan,
        iir=float(c.sum() / n0) if n0 else np.nan,
        removal=float(h.sum() / n1) if n1 else np.nan,
        coverage=float(coverage.mean()) if len(y) else np.nan,
        selection_risk=(
            float((((y == 1) & (h == 0)) | (c == 1))[coverage.astype(bool)].mean())
            if coverage.sum()
            else np.nan
        ),
    )


def paired_intervals(labels, decisions, reference, seed=42, n_boot=500):
    """Paired image bootstrap; callers supply at most one row per image."""
    y, h, c, _ = metric_vectors(labels, decisions)
    _, hr, cr, _ = metric_vectors(labels, reference)
    if len(y) < 2:
        return {}
    draws = np.random.default_rng(seed).integers(0, len(y), (n_boot, len(y)))
    n1 = (y[draws] == 1).sum(axis=1)
    n0 = (y[draws] == 0).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        samples = {
            "nhr": (h[draws].sum(axis=1) - c[draws].sum(axis=1)) / n1,
            "iir": c[draws].sum(axis=1) / n0,
            "nhr_difference_from_full": (
                (h - hr)[draws].sum(axis=1) - (c - cr)[draws].sum(axis=1)
            )
            / n1,
            "iir_difference_from_full": (c - cr)[draws].sum(axis=1) / n0,
        }
    out = {}
    for name, values in samples.items():
        values = values[np.isfinite(values)]
        if len(values) >= max(20, n_boot // 10):
            out[name + "_ci_low"], out[name + "_ci_high"] = map(
                float, np.quantile(values, [0.025, 0.975])
            )
    return out


def groups_for_study(rows, cfg):
    """Keep the prespecified revision claims and independent CP image units."""
    rows = [dict(row) for row in rows if row.get("claim_type") in ("attr", "rel")]
    mark_revision_evaluation_sample(rows, cfg)
    buckets = defaultdict(list)
    for row in rows:
        buckets[(matched_signal_group(row), str(row.get("split")))].append(row)
    output = {}
    for (group, split), items in buckets.items():
        if split == "calib_cp":
            items = image_unit_rows(items, int(cfg.get("seed", 42)), "conformal")
        else:
            items = [r for r in items if r.get("revision_evaluation_selected")]
        output.setdefault(group, {})[split] = items
    return output


def gate_scores(parts, seed=42):
    """Fit direct fusion on CP calibration only; no final-test label fitting."""
    cp = parts["calib_cp"]
    x = lambda rs: np.asarray(
        [[1 - float(r["g"]), float(r["lp_predictive_entropy"])] for r in rs]
    )
    clf = make_pipeline(
        StandardScaler(), LogisticRegression(C=1, max_iter=1000, random_state=seed)
    )
    clf.fit(x(cp), [int(r["label"]) for r in cp])
    result = {}
    for split, rows in parts.items():
        result[split] = {
            "Full": np.asarray([r["d_conflict"] for r in rows], float),
            "No VG prior": np.asarray([r["d_conflict_no_kb"] for r in rows], float),
            "Raw grounding": 1 - np.asarray([r["g"] for r in rows], float),
            "Self verification": 1 - np.asarray([r["l_p"] for r in rows], float),
            "Predictive entropy (verification)": np.asarray(
                [r["lp_predictive_entropy"] for r in rows], float
            ),
            "Grounding-entropy fusion": clf.predict_proba(x(rows))[:, 1],
        }
        if rows and all(finite(r, "probe_score") for r in rows):
            result[split]["Linear probe"] = np.asarray(
                [r["probe_score"] for r in rows], float
            )
    return result


def evaluate_group(parts, cfg, identity):
    required = ("calib_cp", "calib_revision", "test")
    supplied=[r for items in parts.values() for r in items]
    if any(r.get('split_schema_version')!=2 for r in supplied):
        return [],[],dict(identity,status='stale_split_metadata')
    if any(r.get('signals_schema_version')!=4 or not r.get('signal_computed') for r in supplied):
        return [],[],dict(identity,status='missing_current_visual_signal_pass')
    if any(r.get('internal_confidence_selected') and not r.get('lp_attempted') for r in supplied):
        return [],[],dict(identity,status='incomplete_internal_confidence_pass')
    if getattr(cfg,'_expected_kb_signature',None) and any(r.get('kb_prior_index_signature')!=cfg._expected_kb_signature for r in supplied):
        return [],[],dict(identity,status='missing_current_clean_prior')
    if any(not parts.get(split) for split in required):
        return [], [], dict(identity, status="missing_split")
    version = verification_adapter_version(cfg, identity["backbone"])
    complete = {}
    for split in required:
        complete[split] = [
            r
            for r in parts[split]
            if r.get("label") in (0, 1)
            and has_current_lp(r, version)
            and bool(r.get("signal_available"))
            and all(
                finite(r, k)
                for k in (
                    "g",
                    "d_conflict",
                    "d_conflict_no_kb",
                    "lp_predictive_entropy",
                )
            )
            and r.get("residual_schema_version") == RESIDUAL_SCHEMA_VERSION
            and (
                not getattr(cfg, '_expected_kb_signature', None)
                or r.get('kb_prior_index_signature') == cfg._expected_kb_signature
            )
        ]
    if any(len(complete[s]) < 10 for s in required):
        return (
            [],
            [],
            dict(
                identity,
                status="insufficient_current_common_scores",
                counts={s: len(complete[s]) for s in required},
            ),
        )
    if len({r["label"] for r in complete["calib_cp"]}) < 2:
        return [], [], dict(identity, status="single_class_learning_split")
    scores = gate_scores(complete, int(cfg.get('seed', 42)))
    settings = cfg["reviewer_revision"]
    result = []
    decisions_out = []
    for fraction in settings["calibration_fractions"]:
        # Row order is stable; image hash, not label or future outcome, sets order.
        ordered = sorted(
            range(len(complete["calib_revision"])),
            key=lambda i: __import__("hashlib")
            .sha256(
                f"{cfg.get('seed',42)}|gate_size|{complete['calib_revision'][i]['image_id']}".encode()
            )
            .hexdigest(),
        )
        selected = ordered[: max(1, int(len(ordered) * fraction))]
        revision = [complete["calib_revision"][i] for i in selected]
        test = complete["test"]
        cp = complete["calib_cp"]
        for alpha in settings["alpha_grid"]:
            try:
                q = calibrate([r["g"] for r in cp], [r["label"] for r in cp], alpha)
            except ValueError:
                continue
            flags = {
                s: flag_hallucination([r["g"] for r in complete[s]], q)
                for s in required
            }
            fixed_q = fixed_threshold([r["g"] for r in cp], 1 - alpha)
            for delta in settings["delta_grid"]:
                policy = {}
                metadata = {}
                for name in scores["test"]:
                    if name not in scores["calib_revision"]:
                        continue
                    train = [
                        dict(
                            r,
                            d_conflict=float(scores["calib_revision"][name][i]),
                            flagged=bool(flags["calib_revision"][i]),
                        )
                        for i, r in enumerate(complete["calib_revision"])
                        if i in selected
                    ]
                    try:
                        theta = select_theta(
                            train,
                            target_incorrect_intervention_rate=delta,
                            min_calibration_claims=cfg["residual_gate"][
                                "min_calibration_claims"
                            ],
                            min_correct_claims_for_risk=cfg["residual_gate"][
                                "min_correct_claims_for_risk"
                            ],
                        )
                        status = "empirical"
                    except ValueError:
                        theta = None
                        status = "no_action"
                    metadata[name] = {"theta": theta, "gate_status": status}
                    policy[name] = [
                        (
                            "keep"
                            if not f
                            else (
                                "revise"
                                if theta is not None and score > theta
                                else "abstain"
                            )
                        )
                        for f, score in zip(flags["test"], scores["test"][name])
                    ]
                cp_scores = scores["calib_cp"]["Full"]
                candidates = np.unique(
                    np.quantile(
                        cp_scores, np.linspace(0, 1, settings["threshold_grid_size"])
                    )
                )
                ltt_rows = [
                    dict(
                        r,
                        d_conflict=float(scores["calib_revision"]["Full"][i]),
                        flagged=bool(flags["calib_revision"][i]),
                    )
                    for i, r in enumerate(complete["calib_revision"])
                    if i in selected
                ]
                certificate = select_theta_ltt(
                    ltt_rows, candidates, delta, settings["ltt_failure_probability"]
                )
                theta = certificate["theta"]
                policy["LTT residual gate"] = [
                    (
                        "keep"
                        if not f
                        else "revise" if theta is not None and s > theta else "abstain"
                    )
                    for f, s in zip(flags["test"], scores["test"]["Full"])
                ]
                metadata["LTT residual gate"] = {
                    "theta": theta,
                    "gate_status": certificate["status"],
                    "risk_upper": certificate["risk_upper"],
                    "family_size": certificate["family_size"],
                }
                # Preserve the simple fixed threshold and no-gate comparators.
                policy["No gate"] = ["revise" if f else "keep" for f in flags["test"]]
                policy["Abstain all"] = [
                    "abstain" if f else "keep" for f in flags["test"]
                ]
                fixed_flags = flag_hallucination([r["g"] for r in revision], fixed_q)
                fixed_rows = [
                    dict(r, d_conflict=float(r["d_conflict"]), flagged=bool(f))
                    for r, f in zip(revision, fixed_flags)
                ]
                try:
                    fixed_theta = select_theta(
                        fixed_rows,
                        target_incorrect_intervention_rate=delta,
                        min_calibration_claims=cfg["residual_gate"][
                            "min_calibration_claims"
                        ],
                        min_correct_claims_for_risk=cfg["residual_gate"][
                            "min_correct_claims_for_risk"
                        ],
                    )
                except ValueError:
                    fixed_theta = None
                policy["Fixed threshold"] = [
                    (
                        "keep"
                        if not f
                        else (
                            "revise"
                            if fixed_theta is not None and s > fixed_theta
                            else "abstain"
                        )
                    )
                    for f, s in zip(
                        flag_hallucination([r["g"] for r in test], fixed_q),
                        scores["test"]["Full"],
                    )
                ]
                visual_q = select_f1_threshold(
                    1 - np.asarray([r["g"] for r in cp]), [r["label"] for r in cp]
                )
                policy["Visual verification retraction (simplified post-hoc)"] = [
                    "revise" if 1 - r["g"] >= visual_q else "keep" for r in test
                ]
                labels = [r["label"] for r in test]
                for name, decisions in policy.items():
                    stats = point(labels, decisions)
                    metrics = dict(
                        identity,
                        method=name,
                        alpha=alpha,
                        delta=delta,
                        calibration_fraction=fraction,
                        n_before_common_test=len(parts["test"]),
                        n_calibration_images=len(revision),
                        sampling_unit="one_claim_per_image",
                        evaluation_scope="current_common_available_verification_scores",
                        **stats,
                        **metadata.get(name, {}),
                        test_iir_exceeds_budget=bool(stats["iir"] > delta),
                    )
                    metrics.update(
                        paired_intervals(
                            labels, decisions, policy["Full"], int(cfg.get("seed", 42))
                        )
                    )
                    if name in scores["test"] and len(set(labels)) == 2:
                        metrics["score_auroc"] = roc_auc_score(
                            labels, scores["test"][name]
                        )
                        metrics["score_ap"] = average_precision_score(
                            labels, scores["test"][name]
                        )
                    result.append(metrics)
                    if (
                        alpha == cfg["cp"]["alpha_default"]
                        and delta
                        == cfg["residual_gate"]["target_incorrect_intervention_rate"]
                        and fraction == 1.0
                    ):
                        decisions_out.extend(
                            dict(
                                identity,
                                method=name,
                                claim_id=r["claim_id"],
                                image_id=r["image_id"],
                                label=r["label"],
                                decision=d,
                            )
                            for r, d in zip(test, decisions)
                        )
    return (
        result,
        decisions_out,
        dict(
            identity,
            status="analysed" if result else "insufficient_conformal_support",
            counts={s: len(complete[s]) for s in required},
        ),
    )


def run(cfg):
    out = output_dir(cfg)
    manifest_path = cfg.path('vg_index') / 'index_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {}
    if not manifest.get('identity_audit_complete'):
        report = dict(status='pending', n_operating_points=0, n_primary_decisions=0,
                      statuses=[dict(status='missing_complete_clean_index_manifest')])
        write_json(out / 'b3_analysis_status.json', report)
        return report
    cfg._expected_kb_signature = sha256(manifest_path)
    results = []
    decisions = []
    statuses = []
    provenance = {}
    for backbone in cfg.backbone_names:
        probe_metadata_path=out/f'probe_baseline_{backbone}.json'
        probe_metadata=json.loads(probe_metadata_path.read_text(encoding='utf-8')) if probe_metadata_path.exists() else {}
        probe_group_status={(r['dataset'],r['group']):r['status'] for r in probe_metadata.get('groups',[])}
        for dataset in cfg["datasets"]["detect"]:
            path = cfg.path("processed") / f"claims_{backbone}_{dataset}.jsonl"
            identity = dict(backbone=backbone, dataset=dataset)
            if not path.exists():
                statuses.append(dict(identity, status="missing_claim_cache"))
                continue
            rows = read_jsonl(path)
            provenance[str(path.relative_to(cfg.root))] = sha256(path)
            probe_path = out / f"probe_scores_{backbone}_{dataset}.jsonl"
            if probe_path.exists():
                lookup = {r["claim_id"]: r["score"] for r in read_jsonl(probe_path)}
                for row in rows:
                    row["probe_score"] = lookup.get(row["claim_id"])
            elif not any(d==dataset for d,g in probe_group_status):
                statuses.append(dict(identity, status="missing_probe_baseline"))
            for group, parts in groups_for_study(rows, cfg).items():
                probe_unsupported=probe_group_status.get((dataset,group))=='insufficient_development_support'
                if probe_unsupported:
                    statuses.append(dict(identity,group=group,status='insufficient_probe_development_support'))
                if probe_path.exists() and any(
                    not finite(row, "probe_score")
                    for items in parts.values()
                    for row in items
                ) and not probe_unsupported:
                    statuses.append(
                        dict(
                            identity,
                            group=group,
                            status="incomplete_probe_scores_for_group",
                        )
                    )
                metrics, raw, status = evaluate_group(
                    parts, cfg, dict(identity, group=group)
                )
                results.extend(metrics)
                decisions.extend(raw)
                statuses.append(status)
    write_csv(out / "b3_shared_protocol_metrics.csv", results)
    write_csv(out / "b3_primary_decisions.csv", decisions)
    transfer = frozen_transfer(cfg)
    write_csv(out / "b3_frozen_domain_transfer.csv", transfer)
    statistically_resolved={'analysed','insufficient_probe_development_support','missing_split',
                            'single_class_learning_split','insufficient_conformal_support','insufficient_current_common_scores'}
    resolved=bool(results) and all(s['status'] in statistically_resolved for s in statuses)
    unsupported=any(s['status']!='analysed' for s in statuses)
    summary_status='pending'
    analysis_status='pending'
    if resolved:
        summary_status='complete_with_unsupported_groups' if unsupported else 'complete'
        analysis_status='analysis_complete_with_unsupported_groups' if unsupported else 'analysis_complete'
    write_json(
        out / "b3_analysis_status.json",
        dict(
            status=summary_status,
            groups=statuses,
            input_sha256=provenance,
            primary_results_modified=False,
            entropy_definition="mean normalised vocabulary entropy of the same one-shot self-verification continuation; separate from RQ1 ten-sample predictive entropy",
            frozen_transfer_cells=len(transfer),
            additional_required_evidence=[
                "human_semantics_and_extraction",
                "clean_latency_logs",
            ],
        ),
    )
    return dict(
        status=analysis_status,
        n_operating_points=len(results),
        n_primary_decisions=len(decisions),
        statuses=statuses,
    )


def frozen_transfer(cfg):
    """Freeze source scaling, residuals, CP and gate; never fit target labels."""
    from ..residual.decomposition import (
        fit_residual_model,
        apply_residual_model,
        conflict_scores,
    )

    records = []
    identity_lookup = {}
    identity_path = output_dir(cfg) / 'vg_overlap_images.csv'
    if identity_path.exists():
        with identity_path.open(encoding='utf-8-sig', newline='') as stream:
            for identity in csv.DictReader(stream):
                identity_lookup[(identity['dataset'], identity['image_id'])] = {
                    value for value in identity.get('canonical_image_keys', '').split(';') if value
                }
    def image_keys(dataset, row):
        keys = set(identity_lookup.get((dataset, str(row.get('image_id'))), set()))
        if row.get('image_path'):
            keys.add('path:' + str(row['image_path']).replace('\\', '/'))
        return keys
    for backbone in cfg.backbone_names:
        adapter = verification_adapter_version(cfg, backbone)
        for source in ("amber_rel", "rbench", "mmrel"):
            source_rows = read_jsonl(
                cfg.path("processed") / f"claims_{backbone}_{source}.jsonl"
            )
            used_source_images = set()
            for row in source_rows:
                if row.get('split') in ('calib_cp', 'calib_revision'):
                    used_source_images.update(image_keys(source, row))
            for group, parts in groups_for_study(source_rows, cfg).items():
                raw_cp = [
                    r
                    for r in source_rows
                    if r.get("split") == "calib_cp"
                    and matched_signal_group(r) == group
                    and finite(r, "g_raw")
                ]
                if not raw_cp:
                    continue
                lo = min(float(r["g_raw"]) for r in raw_cp)
                hi = max(float(r["g_raw"]) for r in raw_cp)
                if hi - lo < 1e-8:
                    hi = lo + 1
                scale = lambda rows: np.clip(
                    (np.asarray([r["g_raw"] for r in rows], float) - lo) / (hi - lo),
                    0,
                    1,
                )
                usable = lambda rows: [
                    r
                    for r in rows
                    if finite(r, "g_raw")
                    and has_current_lp(r, adapter)
                    and r.get("label") in (0, 1)
                ]
                fit = usable(raw_cp)
                cp = usable(parts.get("calib_cp", []))
                revision = usable(parts.get("calib_revision", []))
                if (
                    len({r.get("image_id") for r in fit})
                    < cfg["residual_gate"]["min_residual_images"]
                    or not revision
                ):
                    continue
                prior = lambda rows: [float(r.get("kb_prior") or 0) for r in rows]
                available = lambda rows: [
                    bool(r.get("kb_prior_available")) for r in rows
                ]
                model = fit_residual_model(
                    scale(fit), [r["l_p"] for r in fit], prior(fit), available(fit)
                )

                def conflict(rows):
                    return conflict_scores(
                        apply_residual_model(
                            scale(rows),
                            [r["l_p"] for r in rows],
                            model,
                            prior(rows),
                            available(rows),
                        ),
                        model["center"],
                        model["mad"],
                    )

                try:
                    q = calibrate(
                        scale(cp), [r["label"] for r in cp], cfg["cp"]["alpha_default"]
                    )
                except ValueError:
                    continue
                rev = [
                    dict(r, flagged=bool(f), d_conflict=float(d))
                    for r, f, d in zip(
                        revision,
                        flag_hallucination(scale(revision), q),
                        conflict(revision),
                    )
                ]
                try:
                    theta = select_theta(
                        rev,
                        target_incorrect_intervention_rate=cfg["residual_gate"][
                            "target_incorrect_intervention_rate"
                        ],
                        min_calibration_claims=cfg["residual_gate"][
                            "min_calibration_claims"
                        ],
                        min_correct_claims_for_risk=cfg["residual_gate"][
                            "min_correct_claims_for_risk"
                        ],
                    )
                except ValueError:
                    theta = None
                for target in ("amber_rel", "rbench", "mmrel"):
                    if source == target:
                        continue
                    target_rows = read_jsonl(
                        cfg.path("processed") / f"claims_{backbone}_{target}.jsonl"
                    )
                    target_parts = groups_for_study(target_rows, cfg).get(group, {})
                    test = [
                        r
                        for r in target_parts.get("test", [])
                        if r.get("label") in (0, 1)
                    ]
                    n_before_overlap_exclusion = len(test)
                    test = [row for row in test if not (image_keys(target, row) & used_source_images)]
                    if not test:
                        continue
                    supported = usable(test)
                    flags = flag_hallucination(scale(supported), q) if supported else []
                    decision_by_id = {
                        r["claim_id"]: (
                            "keep"
                            if not f
                            else (
                                "revise"
                                if theta is not None and d > theta
                                else "abstain"
                            )
                        )
                        for r, f, d in zip(
                            supported, flags, conflict(supported) if supported else []
                        )
                    }
                    decisions = [
                        decision_by_id.get(r["claim_id"], "abstain") for r in test
                    ]
                    # Target labels are consulted only now, after all actions froze.
                    labels = [r["label"] for r in test]
                    records.append(
                        dict(
                            backbone=backbone,
                            source_dataset=source,
                            target_dataset=target,
                            group=group,
                            protocol="frozen_source_scaling_residual_cp_gate",
                            guarantee_under_shift=False,
                            source_q=q,
                            source_theta=theta,
                            n_unavailable=len(test) - len(supported),
                            n_excluded_source_calibration_image_overlap=n_before_overlap_exclusion-len(test),
                            **point(labels, decisions),
                        )
                    )
    return records
