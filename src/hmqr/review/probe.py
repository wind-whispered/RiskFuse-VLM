"""Budgeted hidden-state baseline with development-only layer selection."""

from __future__ import annotations

import hashlib
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from ..utils import read_jsonl, write_json, write_jsonl
from ..sampling import stable_claim_key
from .audit import output_dir
from .overlap import sha256
from .studies import groups_for_study


def tasks(cfg, backbone):
    folder = output_dir(cfg) / "probe_features" / backbone
    folder.mkdir(parents=True, exist_ok=True)
    output = []
    for dataset in cfg["datasets"]["detect"]:
        path = cfg.path("processed") / f"claims_{backbone}_{dataset}.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        rows = read_jsonl(path)
        for group, parts in groups_for_study(rows, cfg).items():
            for split, items in parts.items():
                if split not in ("calib_cp", "calib_revision", "test"):
                    continue
                for row in items:
                    if row.get("label") not in (0, 1):
                        continue
                    image = cfg.abspath(row["image_path"])
                    signature = dict(
                        model=cfg.backbone_cfg(backbone)["hf_id"],
                        image_sha256=sha256(image) if image.is_file() else None,
                        sentence=row.get("sentence"),
                        claim_type=row["claim_type"],
                        attribute=row.get("attribute"),
                        relation=row.get("relation"),
                        subject=row.get("subject"),
                        object=row.get("object"),
                    )
                    key = hashlib.sha256(
                        json.dumps(signature, sort_keys=True).encode()
                    ).hexdigest()
                    output.append(
                        dict(
                            row=row,
                            dataset=dataset,
                            group=group,
                            path=folder / (key + ".npy"),
                            image_exists=image.is_file(),
                        )
                    )
    return output


def select_probe(features, rows, seed, development_fraction):
    """Use a label-blind image hash for development; never select on test AUROC."""
    train = np.asarray([r.get("split") == "calib_cp" for r in rows])
    dev = (
        np.asarray(
            [
                int(
                    hashlib.sha256(
                        f'{seed}|probe_dev|{r["image_id"]}'.encode()
                    ).hexdigest(),
                    16,
                )
                / 2**256
                < development_fraction
                for r in rows
            ]
        )
        & train
    )
    fit = train & ~dev
    labels = np.asarray([int(r["label"]) for r in rows])
    if (
        fit.sum() < 5
        or dev.sum() < 5
        or len(set(labels[fit])) < 2
        or len(set(labels[dev])) < 2
    ):
        raise ValueError(
            "insufficient image-disjoint probe training/development support"
        )
    best = None
    for layer in range(features.shape[1]):
        model = make_pipeline(
            StandardScaler(), LogisticRegression(C=1, max_iter=2000, random_state=seed)
        )
        model.fit(features[fit, layer], labels[fit])
        score = roc_auc_score(
            labels[dev], model.predict_proba(features[dev, layer])[:, 1]
        )
        if best is None or score > best[0]:
            best = (score, layer)
    model = make_pipeline(
        StandardScaler(), LogisticRegression(C=1, max_iter=2000, random_state=seed)
    )
    model.fit(features[train, best[1]], labels[train])
    return model.predict_proba(features[:, best[1]])[:, 1], dict(
        selected_layer=best[1],
        development_auroc=best[0],
        n_training_images=int(fit.sum()),
        n_development_images=int(dev.sum()),
        fit_split="calib_cp",
        layer_selection="label_blind_image_hash_within_calib_cp_development",
    )


def run(cfg, backbone, apply=False, max_forward_calls=None):
    jobs = tasks(cfg, backbone)
    pending = [job for job in jobs if not job["path"].exists()]
    report = dict(
        backbone=backbone,
        pending_forward_calls=len(pending),
        total_claims=len(jobs),
        missing_images=sum(not j["image_exists"] for j in jobs),
        gpu_launched=False,
    )
    if not apply:
        return report
    if max_forward_calls is None or len(pending) > max_forward_calls:
        raise ValueError(
            "probe feature job requires an explicit sufficient --max-forward-calls budget"
        )
    if report["missing_images"]:
        raise FileNotFoundError("probe inputs include missing images")
    from ..backbones.registry import build_backbone
    from ..data.images import load_image
    from ..mechanism.layer_probe import key_tokens

    backbone_model = build_backbone(cfg, backbone) if pending else None
    try:
        for job in pending:
            row = job["row"]
            keywords = key_tokens(row)
            state = backbone_model.forward_states(
                load_image(row["image_path"]),
                cfg["generation"]["caption_prompt"],
                row.get("sentence", ""),
                keywords=keywords,
                need_attentions=False,
            )
            hidden = state["hidden_states"]
            vectors = []
            for keyword in keywords:
                span = state["text_token_spans"].get(str(keyword))
                if span is not None and 0 <= span[0] < span[1] <= hidden.shape[1]:
                    vectors.append(hidden[:, span[0] : span[1], :].mean(axis=1))
            if not vectors:
                raise ValueError(f'probe target alignment failed: {row["claim_id"]}')
            temporary = job["path"].with_suffix(".partial.npy")
            np.save(temporary, np.asarray(np.mean(vectors, axis=0), dtype=np.float16))
            temporary.replace(job["path"])
    finally:
        if backbone_model is not None:
            backbone_model.unload()
    groups = {}
    for job in jobs:
        groups.setdefault((job["dataset"], job["group"]), []).append(job)
    scores = {}
    metadata = []
    for (dataset, group), entries in groups.items():
        rows = [j["row"] for j in entries]
        try:
            values, meta = select_probe(
                np.stack([np.load(j["path"]).astype(np.float32) for j in entries]),
                rows,
                int(cfg.get("seed", 42)),
                cfg["reviewer_revision"]["probe_development_fraction"],
            )
        except ValueError as exc:
            metadata.append(
                dict(
                    dataset=dataset,
                    group=group,
                    status="insufficient_development_support",
                    reason=str(exc),
                )
            )
            continue
        scores.setdefault(dataset, []).extend(
            dict(
                claim_id=r["claim_id"],
                image_id=r["image_id"],
                score=float(v),
                split=r["split"],
                group=group,
                **meta,
            )
            for r, v in zip(rows, values)
        )
        metadata.append(dict(dataset=dataset, group=group, status="complete", **meta))
    for dataset, records in scores.items():
        write_jsonl(
            output_dir(cfg) / f"probe_scores_{backbone}_{dataset}.jsonl", records
        )
    write_json(
        output_dir(cfg) / f"probe_baseline_{backbone}.json",
        dict(report, groups=metadata, gpu_launched=bool(pending)),
    )
    return dict(
        report,
        status=(
            "complete"
            if metadata and all(r["status"] == "complete" for r in metadata)
            else "complete_with_unsupported_groups"
            if metadata and all(r['status'] in ('complete','insufficient_development_support') for r in metadata)
            else "partial"
        ),
        n_scored=sum(map(len, scores.values())),
    )
