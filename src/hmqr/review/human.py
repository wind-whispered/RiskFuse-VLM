"""Fixed-image human extractor/output audit; never synthesises human truth."""

from __future__ import annotations

import hashlib
import json
import random

from ..data.datasets import load_chair500
from ..utils import read_jsonl, write_json, write_jsonl
from .audit import output_dir, write_csv
from .overlap import sha256


def export_tasks(cfg):
    if cfg.get('reviewer_revision', {}).get('annotation_mode') == 'ai_assisted_human_review':
        from .assisted import export_review_workbook
        return export_review_workbook(cfg)
    import csv

    out = output_dir(cfg)
    folder = out / "human_audit"
    folder.mkdir(exist_ok=True)
    samples = load_chair500(cfg)
    seed = cfg.get("seed", 42)
    selected = sorted(
        samples,
        key=lambda s: hashlib.sha256(
            f"{seed}|human_audit|{s.image_id}".encode()
        ).hexdigest(),
    )[: cfg["reviewer_revision"]["annotation_images_per_backbone"]]
    tasks = []
    private = []
    pending = []
    for backbone in cfg.backbone_names:
        claims = read_jsonl(cfg.path("processed") / f"claims_{backbone}_chair500.jsonl")
        revised = read_jsonl(
            cfg.path("processed") / f"revised_labeled_{backbone}_chair500.jsonl"
        )
        by_image = {}
        for row in claims:
            by_image.setdefault(str(row["image_id"]), []).append(row)
        final = {
            str(row["image_id"]): row.get("final_caption")
            for row in revised
            if row.get("final_caption") is not None
        }
        for sample in selected:
            image = str(sample.image_id)
            rows = by_image.get(image, [])
            captions = {str(r["caption"]) for r in rows if r.get("caption") is not None}
            if len(captions) > 1:
                raise ValueError(f"conflicting original captions: {backbone}/{image}")
            task_id = hashlib.sha256(f"{seed}|{backbone}|{image}".encode()).hexdigest()[
                :16
            ]
            variants = []
            for method, text in [
                ("original", next(iter(captions), None)),
                ("full", final.get(image)),
            ]:
                if text is not None:
                    variant_id = hashlib.sha256(
                        f"{task_id}|{method}".encode()
                    ).hexdigest()[:10]
                    variants.append(dict(variant_id=variant_id, text=text))
                    private.append(
                        dict(task_id=task_id, variant_id=variant_id, method=method)
                    )
            random.Random(task_id).shuffle(variants)
            complete = bool(captions and image in final)
            if not complete:
                pending.append(task_id)
            tasks.append(
                dict(
                    task_id=task_id,
                    backbone=backbone,
                    image_id=image,
                    image_path=sample.image_path,
                    original_caption=next(iter(captions), None),
                    extracted_claims=[
                        {
                            k: r.get(k)
                            for k in (
                                "claim_id",
                                "claim_type",
                                "subject",
                                "relation",
                                "object",
                                "attribute",
                                "sentence",
                            )
                        }
                        for r in rows
                        if r.get("claim_type") in ("attr", "rel")
                    ],
                    blind_variants=variants,
                    status=(
                        "ready_for_annotation"
                        if complete
                        else "pending_generated_outputs"
                    ),
                )
            )
    for task in tasks:
        task["task_sha256"] = hashlib.sha256(
            json.dumps(task, sort_keys=True).encode()
        ).hexdigest()
    task_path = folder / "tasks.jsonl"
    if task_path.exists():
        old = read_jsonl(task_path)
        if [r["task_id"] for r in old] != [r["task_id"] for r in tasks]:
            raise ValueError(
                "human audit image set changed; archive the previous audit before replacing its task identity"
            )
    current_hashes = {t["task_id"]: t["task_sha256"] for t in tasks}
    refresh = []
    for name in ("annotator_A.csv", "annotator_B.csv", "adjudicated.csv"):
        path = folder / name
        if not path.exists():
            refresh.append(name)
            continue
        with path.open(encoding="utf-8-sig", newline="") as stream:
            existing = list(csv.DictReader(stream))
        filled = any(
            any(
                row.get(k, "").strip()
                for k in (
                    "annotator",
                    "status",
                    "gold_claims_json",
                    "matches_json",
                    "variant_judgements_json",
                    "notes",
                )
            )
            for row in existing
        )
        if filled and any(
            row.get("task_sha256") != current_hashes.get(row.get("task_id"))
            for row in existing
        ):
            raise ValueError(
                "changed generated outputs would invalidate existing human work; archive and explicitly re-audit"
            )
        if not filled:
            refresh.append(name)
    write_jsonl(task_path, tasks)
    write_jsonl(folder / "variant_key_private.jsonl", private)
    write_jsonl(
        folder / "output_blind_tasks.jsonl",
        [
            {
                k: t[k]
                for k in (
                    "task_id",
                    "image_id",
                    "image_path",
                    "blind_variants",
                    "status",
                    "task_sha256",
                )
            }
            for t in tasks
        ],
    )
    template = [
        dict(
            task_id=t["task_id"],
            task_sha256=t["task_sha256"],
            annotator="",
            status="",
            gold_claims_json="",
            matches_json="",
            variant_judgements_json="",
            notes="",
        )
        for t in tasks
    ]
    for name in refresh:
        write_csv(folder / name, template)
    write_json(
        folder / "sample_manifest.json",
        dict(
            seed=seed,
            n_shared_images=len(selected),
            n_caption_tasks=len(tasks),
            image_ids=[str(s.image_id) for s in selected],
            tasks_sha256=sha256(task_path),
            selection="label_score_and_action_blind_image_hash",
        ),
    )
    return dict(
        status="complete" if not pending else "pending_generated_outputs",
        n_shared_images=len(selected),
        n_caption_tasks=len(tasks),
        n_tasks_awaiting_outputs=len(pending),
        annotations_filled=False,
    )


def score_annotations(cfg):
    if cfg.get('reviewer_revision', {}).get('annotation_mode') == 'ai_assisted_human_review':
        from .assisted import score_review_workbook
        return score_review_workbook(cfg)
    import csv

    folder = output_dir(cfg) / "human_audit"
    tasks = read_jsonl(folder / "tasks.jsonl")
    if not tasks or any(t["status"] != "ready_for_annotation" for t in tasks):
        return dict(status="pending_generated_outputs", metrics_computed=False)
    expected = {t["task_id"] for t in tasks}
    tables = []
    for name in ("annotator_A.csv", "annotator_B.csv", "adjudicated.csv"):
        path = folder / name
        if not path.exists():
            return dict(status="pending_human_annotations", metrics_computed=False)
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != len(expected) or {r["task_id"] for r in rows} != expected:
            raise ValueError(f"{name}: task inventory does not match frozen sample")
        if any(
            r.get("status") != "complete" or not r.get("annotator", "").strip()
            for r in rows
        ):
            return dict(status="pending_human_annotations", metrics_computed=False)
        tables.append({r["task_id"]: r for r in rows})
    counts = {
        kind: dict(true_positive=0, n_predicted=0, n_gold=0) for kind in ("attr", "rel")
    }
    quality = []
    for task in tasks:
        a, b, adjudicated = [table[task["task_id"]] for table in tables]
        if any(
            row.get("task_sha256") != task["task_sha256"] for row in (a, b, adjudicated)
        ):
            raise ValueError(
                "annotation refers to a changed caption/task; re-audit the affected outputs"
            )
        if a["annotator"].strip() == b["annotator"].strip():
            raise ValueError(
                "independent annotations require different annotator identities"
            )
        if any(
            a.get(k) != b.get(k)
            for k in ("gold_claims_json", "matches_json", "variant_judgements_json")
        ) and adjudicated["annotator"].strip() in (
            a["annotator"].strip(),
            b["annotator"].strip(),
        ):
            raise ValueError("disagreements require an independent adjudicator")
        gold = json.loads(adjudicated["gold_claims_json"])
        matches = json.loads(adjudicated["matches_json"])
        predictions = {r["claim_id"]: r for r in task["extracted_claims"]}
        gold_map = {r["gold_id"]: r for r in gold}
        if len(gold_map) != len(gold):
            raise ValueError("duplicate gold-claim IDs")
        p_used = set()
        g_used = set()
        for m in matches:
            p_id, g_id = m["predicted_claim_id"], m["gold_id"]
            if (
                p_id not in predictions
                or g_id not in gold_map
                or p_id in p_used
                or g_id in g_used
            ):
                raise ValueError(
                    "semantic extraction matching must be one-to-one and refer to existing claims"
                )
            kind = predictions[p_id]["claim_type"]
            if kind != gold_map[g_id]["claim_type"]:
                raise ValueError("matched claim types differ")
            counts[kind]["true_positive"] += 1
            p_used.add(p_id)
            g_used.add(g_id)
        for p in predictions.values():
            counts[p["claim_type"]]["n_predicted"] += 1
        for g in gold:
            if g["claim_type"] not in counts:
                raise ValueError("gold claims must be attr or rel")
            if g.get("label") not in (-1, 0, 1):
                raise ValueError(
                    "gold visual truth must be explicitly 0, 1, or unresolved -1"
                )
            counts[g["claim_type"]]["n_gold"] += 1
        judgments = json.loads(adjudicated["variant_judgements_json"])
        if {j["variant_id"] for j in judgments} != {
            v["variant_id"] for v in task["blind_variants"]
        } or len(judgments) != len(task["blind_variants"]):
            raise ValueError(
                "every blinded output needs exactly one semantic and fluency judgement"
            )
        for j in judgments:
            for key in (
                "n_correct_claims",
                "n_incorrect_claims",
                "n_unresolved_claims",
            ):
                if type(j.get(key)) is not int or j[key] < 0:
                    raise ValueError(
                        "semantic counts must be explicit nonnegative integers"
                    )
            if type(j.get("fluency")) is not int or not 1 <= j["fluency"] <= 5:
                raise ValueError("fluency must be an integer from one to five")
            quality.append(
                dict(
                    task_id=task["task_id"],
                    backbone=task["backbone"],
                    image_id=task["image_id"],
                    **j,
                )
            )
    for values in counts.values():
        values["precision"] = (
            values["true_positive"] / values["n_predicted"]
            if values["n_predicted"]
            else None
        )
        values["recall"] = (
            values["true_positive"] / values["n_gold"] if values["n_gold"] else None
        )
    report = dict(
        status="complete",
        metrics_computed=True,
        n_caption_tasks=len(tasks),
        extraction=counts,
        annotation_sha256={
            name: sha256(folder / name)
            for name in ("annotator_A.csv", "annotator_B.csv", "adjudicated.csv")
        },
    )
    write_json(folder / "human_audit_results.json", report)
    write_csv(folder / "output_semantics_and_fluency.csv", quality)
    return report
