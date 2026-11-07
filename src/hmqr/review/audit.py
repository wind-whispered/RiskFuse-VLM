"""Read-only B0 evidence inventory and launch checks."""

from __future__ import annotations

import csv
import importlib.util
import json
import re
from pathlib import Path

from ..utils import read_jsonl, write_json
from .overlap import sha256


def output_dir(cfg):
    path = cfg.path("review") / "reviewer_revision"
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def baseline(cfg):
    from ..residual.decomposition import RESIDUAL_SCHEMA_VERSION
    from ..detect.rq1 import RQ1_SCHEMA_VERSION
    from ..eval.rq_studies import RQ_SCHEMA_VERSION

    out = output_dir(cfg)
    requirements = {
        **{
            n: ("residual_schema_version", RESIDUAL_SCHEMA_VERSION)
            for n in (
                "residual_stats.csv",
                "residual_signal_gain.csv",
                "residual_gain_by_group.csv",
            )
        },
        **{
            n: ("rq1_schema_version", RQ1_SCHEMA_VERSION)
            for n in ("rq1_detection.csv", "rq1_coverage.csv")
        },
        **{
            n: ("rq_schema_version", RQ_SCHEMA_VERSION)
            for n in (
                "rq2_gate.csv",
                "rq3_reduction.csv",
                "rq3_quality.csv",
                "rq4_ablation.csv",
            )
        },
    }
    inventory = []
    for name, (column, expected) in requirements.items():
        path = cfg.path("tables") / name
        rows = []
        if path.exists():
            with path.open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
        versions = sorted({r.get(column, "missing") for r in rows})
        inventory.append(
            dict(
                file=str(path.relative_to(cfg.root)),
                rows=len(rows),
                version_column=column,
                observed_versions=";".join(versions),
                required_version=expected,
                sha256=sha256(path) if path.exists() else "",
                status=(
                    "schema_current_provenance_not_yet_verified"
                    if versions == [str(expected)]
                    else "rerun_required"
                ),
            )
        )
    write_csv(out / "b0_artifact_inventory.csv", inventory)
    cache = []
    for backbone in cfg.backbone_names:
        for dataset in [*cfg["datasets"]["detect"], *cfg["datasets"]["quality"]]:
            path = cfg.path("processed") / f"claims_{backbone}_{dataset}.jsonl"
            rows = read_jsonl(path)
            cache.append(
                dict(
                    backbone=backbone,
                    dataset=dataset,
                    exists=path.exists(),
                    n_claims=len(rows),
                    n_images=len({r.get("image_id") for r in rows}),
                    n_grounding=sum(r.get("g") is not None for r in rows),
                    n_self_verification=sum(r.get("l_p") is not None for r in rows),
                    n_predictive_entropy=sum(
                        r.get("predictive_entropy") is not None for r in rows
                    ),
                )
            )
    write_csv(out / "b0_upstream_cache_inventory.csv", cache)
    manuscript = cfg.abspath(
        cfg.get("reviewer_revision", {}).get(
            "manuscript_source", "review_materials/submitted_main.tex"
        )
    )
    mappings = []
    if manuscript.exists():
        tex = manuscript.read_text(encoding="utf-8-sig")
        for kind in ("figure", "table"):
            pattern = r"\\begin\{" + kind + r"\*?\}.*?\\end\{" + kind + r"\*?\}"
            number = 0
            for match in re.finditer(pattern, tex, re.S):
                block = match.group(0)
                labels = re.findall(r"\\label\{([^}]+)\}", block)
                images = re.findall(
                    r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", block
                )
                if kind == "figure" and not any(
                    label.startswith("fig:") for label in labels
                ):
                    continue  # An algorithm is wrapped in figure* without a figure caption.
                number += 1
                mappings.append(
                    dict(
                        kind=kind,
                        submitted_number=number,
                        labels=";".join(labels),
                        graphic_files=";".join(images),
                        source_line=tex.count("\n", 0, match.start()) + 1,
                    )
                )
        write_csv(out / "b0_manuscript_object_map.csv", mappings)
        historical_crosscheck(cfg, tex, out)
    # Keep the audit's own changes separate from the frozen historical inputs.
    tracked = [p for p in (cfg.root / "src").rglob("*.py")] + [
        cfg.root / "configs/default.yaml"
    ]
    tracked += [manuscript] if manuscript.exists() else []
    write_json(
        out / "b0_source_manifest.json",
        {str(p.relative_to(cfg.root)): sha256(p) for p in tracked},
    )
    report = dict(
        status="audited_not_submission_ready",
        artifacts=inventory,
        claims=cache,
        manuscript_available=manuscript.exists(),
        figure_table_note="Printed order is derived from environment order, not graphic filenames.",
        missing_claim_files=sum(not row["exists"] for row in cache),
        primary_results_modified=False,
    )
    write_json(out / "b0_summary.json", report)
    return report


def historical_crosscheck(cfg, tex, out):
    """Compare submitted numbers with historical CSVs; choose neither as truth."""
    import pandas as pd

    checks = []
    path = cfg.path("tables") / "rq1_detection.csv"
    if path.exists():
        detection = pd.read_csv(path)
        patterns = [
            (
                "Ours",
                r"grounding score reaches AUROC ([0-9.]+) on AMBER-Attribute, ([0-9.]+) on AMBER-Relation, ([0-9.]+) on MMRel, and ([0-9.]+) on R-Bench",
            ),
            (
                "GlobalCLIP-thr",
                r"corresponding global-CLIP values are ([0-9.]+), ([0-9.]+), ([0-9.]+), and ([0-9.]+)",
            ),
        ]
        for method, pattern in patterns:
            match = re.search(pattern, tex)
            if not match:
                continue
            for dataset, number in zip(
                ("amber_attr", "amber_rel", "mmrel", "rbench"), match.groups()
            ):
                hit = detection[
                    (detection["dataset"] == dataset)
                    & (detection["method"] == method)
                    & (detection["backbone"] == "Macro")
                ]
                if len(hit) != 1:
                    continue
                text_value = float(number.rstrip("."))
                csv_value = float(hit.iloc[0]["auroc"])
                checks.append(
                    dict(
                        scope="submitted_RQ1_vs_historical_CSV",
                        dataset=dataset,
                        method=method,
                        manuscript_value=text_value,
                        csv_value=csv_value,
                        numerical_agreement=abs(csv_value - text_value) < 0.0005,
                        validity="historical_diagnostic_only_no_authoritative_result_selected",
                    )
                )
    write_csv(out / "b0_historical_detection_crosscheck.csv", checks)
    count_checks = []
    if (
        "\\label{tbl:counts}" in tex
        and (path := cfg.path("tables") / "rq3_reduction.csv").exists()
    ):
        reduction = pd.read_csv(path)
        block = tex.split("\\label{tbl:counts}", 1)[1].split("\\end{table*}", 1)[0]
        aliases = {
            "LLaVA-1.5": "llava15",
            "LLaVA-1.6": "llava16",
            "InstructBLIP": "instructblip",
            "Qwen-VL": "qwenvl",
        }
        datasets = {
            "AMBER-Attribute": "amber_attr",
            "AMBER-Relation": "amber_rel",
            "R-Bench": "rbench",
            "MMRel": "mmrel",
        }
        backbone = None
        columns = [
            "n_images",
            "n_claims",
            "n_original_hallucinations",
            "n_original_correct",
            "n_target_hallucinations_removed",
            "n_incorrect_interventions",
        ]
        for line in block.splitlines():
            if "&" not in line:
                continue
            cells = [c.strip().rstrip("\\").strip() for c in line.split("&")]
            if len(cells) != 8:
                continue
            if cells[0] in aliases:
                backbone = aliases[cells[0]]
            if backbone is None or cells[1] not in datasets:
                continue
            try:
                values = [int(c) for c in cells[2:]]
            except ValueError:
                continue
            dataset = datasets[cells[1]]
            hit = reduction[
                (reduction["backbone"] == backbone) & (reduction["dataset"] == dataset)
            ]
            if len(hit) != 1:
                continue
            for key, old in zip(columns, values):
                actual = float(hit.iloc[0][key])
                count_checks.append(
                    dict(
                        backbone=backbone,
                        dataset=dataset,
                        quantity=key,
                        manuscript_value=old,
                        historical_csv_value=actual,
                        numerical_agreement=old == actual,
                        validity="requires_claim_ID_and_GT_provenance_audit",
                    )
                )
    write_csv(out / "b0_historical_count_crosscheck.csv", count_checks)


def preflight(cfg):
    from ..revise.pipeline import mark_revision_evaluation_sample
    from ..signals.lp_runner import mark_internal_confidence_selection, estimate_calls
    from ..signals.lp import has_current_lp
    from ..backbones.registry import verification_adapter_version

    out = output_dir(cfg)
    missing = []
    jobs = []
    split_errors = []
    for backbone in cfg.backbone_names:
        for dataset in [*cfg["datasets"]["detect"], *cfg["datasets"]["quality"]]:
            path = cfg.path("processed") / f"claims_{backbone}_{dataset}.jsonl"
            if not path.exists():
                missing.append(str(path.relative_to(cfg.root)))
                continue
            rows = read_jsonl(path)
            by_image = {}
            seen = set()
            for row in rows:
                image = str(row.get("image_id", ""))
                split = str(row.get("split", ""))
                key = str(row.get("claim_id", ""))
                if not image or not key or key in seen:
                    split_errors.append(f"{path.name}: missing or duplicate identity")
                seen.add(key)
                if image in by_image and by_image[image] != split:
                    split_errors.append(f"{path.name}: image crosses splits: {image}")
                by_image[image] = split
            if dataset != "chair500":
                mark_revision_evaluation_sample(rows, cfg)
            selected = mark_internal_confidence_selection(rows, cfg)
            version = verification_adapter_version(cfg, backbone)
            pending = estimate_calls(cfg, [backbone], [dataset])[path.name]
            jobs.append(
                dict(
                    backbone=backbone,
                    dataset=dataset,
                    n_claims=len(rows),
                    n_selected_confidence=len(selected),
                    pending_self_verification=int(pending),
                    n_invalid_selected=sum(
                        not has_current_lp(rows[i], version) for i in selected
                    ),
                    n_revision_eval=sum(
                        bool(r.get("revision_evaluation_selected")) for r in rows
                    ),
                )
            )
    report = dict(
        status=(
            "ready_for_budget_review"
            if not missing and not split_errors
            else "blocked_missing_or_invalid_inputs"
        ),
        missing_claim_files=missing,
        split_errors=sorted(set(split_errors)),
        jobs=jobs,
        total_pending_self_verification=(
            sum(j["pending_self_verification"] for j in jobs) if not missing else None
        ),
        gpu_launched=False,
    )
    write_json(out / "b3_preflight.json", report)
    write_csv(out / "b3_preflight_jobs.csv", jobs)
    return report
