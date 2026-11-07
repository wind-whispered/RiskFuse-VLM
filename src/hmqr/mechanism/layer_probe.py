"""T2: layer-wise probing -> Figure 1.

For every claim we take the hidden state of the key token (attribute word /
relation word / object word), mean-pooled over its sub-word span, at every
Transformer layer. One L2-regularised logistic-regression probe is fitted per
(backbone, layer, claim class) on probe_train and scored by AUROC on probe_test.
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from ..backbones.registry import build_backbone
from ..config import Config
from ..data.images import load_image
from ..utils import get_logger, read_jsonl, write_json

LOG = get_logger("hmqr.probe")

CLASS_ORDER = ["obj", "attr", "rel"]
CLASS_LABEL = {"obj": "object", "attr": "attribute", "rel": "relation"}
PROBE_SCHEMA_VERSION = 2


def _balanced_indices(
    rows: List[Dict[str, Any]],
    max_rows: Optional[int],
    max_images_per_stratum: Optional[int] = None,
    seed: int = 42,
) -> List[int]:
    """Select a deterministic class/label/split-balanced smoke subset.

    Claim files are ordered by filename, so ``rows[:max_rows]`` previously took
    AMBER attributes first and yielded zero relation claims.  Round-robin over
    the probe strata exercises every available class without changing full runs.
    """
    eligible = [
        i for i, row in enumerate(rows)
        if row.get("claim_type") in CLASS_ORDER and row.get("label") in (0, 1)
    ]
    if max_images_per_stratum is not None:
        by_stratum: Dict[Tuple[str, int, str], Dict[str, int]] = defaultdict(dict)
        for i in eligible:
            row = rows[i]
            key = (
                str(row["claim_type"]),
                int(row["label"]),
                str(row.get("probe_split", "probe_train")),
            )
            by_stratum[key].setdefault(str(row.get("image_id")), i)
        sampled: List[int] = []
        for stratum_i, key in enumerate(sorted(by_stratum)):
            candidates = np.asarray(list(by_stratum[key].values()), dtype=int)
            rng = np.random.default_rng(seed + stratum_i)
            rng.shuffle(candidates)
            sampled.extend(candidates[:max_images_per_stratum].tolist())
        eligible = sampled

    if max_rows is None or len(eligible) <= max_rows:
        return eligible

    buckets: Dict[Tuple[str, int, str], Deque[int]] = defaultdict(deque)
    for i in eligible:
        row = rows[i]
        key = (
            str(row["claim_type"]),
            int(row["label"]),
            str(row.get("probe_split", "probe_train")),
        )
        buckets[key].append(i)

    selected: List[int] = []
    keys = sorted(buckets)
    while len(selected) < max_rows:
        added = False
        for key in keys:
            if buckets[key] and len(selected) < max_rows:
                selected.append(buckets[key].popleft())
                added = True
        if not added:
            break
    return selected


def key_tokens(row: Dict[str, Any]) -> List[str]:
    if row["claim_type"] == "attr":
        return [w for w in [row.get("attribute"), row.get("object")] if w]
    if row["claim_type"] == "rel":
        return [w for w in [row.get("relation"), row.get("subject"), row.get("object")] if w]
    return [w for w in [row.get("object")] if w]


def extract_features(
    cfg: Config, backbone_name: str, rows: List[Dict[str, Any]], max_rows: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[int]]:
    """Returns (features [N, L+1, H], labels [N], classes [N], splits [N], src_idx [N]).

    `src_idx` maps every kept feature row back to its source claim for sampling
    provenance and audit counts.
    """
    backbone = build_backbone(cfg, backbone_name)
    prompt = cfg["generation"]["caption_prompt"]

    feats: List[np.ndarray] = []
    labels: List[int] = []
    classes: List[str] = []
    splits: List[str] = []
    src_idx: List[int] = []

    mechanism = cfg.get("mechanism", {})
    paper_cap = mechanism.get("probe_max_images_per_stratum") if max_rows is None else None
    selected = _balanced_indices(
        rows,
        max_rows,
        max_images_per_stratum=int(paper_cap) if paper_cap is not None else None,
        seed=int(cfg.get("seed", 42)),
    )
    for row_i in tqdm(selected, desc=f"probe-features {backbone_name}", ncols=88):
        row = rows[row_i]
        image = load_image(row["image_path"])
        if image is None:
            continue
        kws = key_tokens(row)
        if not kws:
            continue
        target = row.get("sentence") or row.get("caption") or ""
        try:
            # hidden states only: the probe never reads attentions, and asking
            # for them makes LLaVA-1.6 try to allocate ~21 GB per forward
            out = backbone.forward_states(
                image, prompt, target, keywords=kws, need_attentions=False
            )
        except Exception as exc:  # noqa: BLE001
            LOG.warning("forward_states failed (%s): %s", row["claim_id"], exc)
            continue
        hidden = out["hidden_states"]          # [L+1, S, H]
        spans = out["text_token_spans"]
        seq_len = hidden.shape[1]
        vecs = []
        for kw in kws:
            span = spans.get(str(kw))
            if span is None:
                continue
            s, e = max(0, min(span[0], seq_len - 1)), max(1, min(span[1], seq_len))
            vecs.append(hidden[:, s:e, :].mean(axis=1))   # [L+1, H]
        if not vecs:
            continue
        feats.append(np.mean(vecs, axis=0))
        labels.append(int(row["label"]))
        classes.append(row["claim_type"])
        splits.append(row.get("probe_split", "probe_train"))
        src_idx.append(row_i)

    backbone.unload()
    if not feats:
        return np.empty((0, 0, 0)), np.empty((0,)), np.empty((0,), dtype=object), [], []
    return (
        np.stack(feats, axis=0),
        np.asarray(labels),
        np.asarray(classes, dtype=object),
        splits,
        src_idx,
    )


def fit_layer_probes(
    features: np.ndarray,
    labels: np.ndarray,
    classes: np.ndarray,
    splits: List[str],
    backbone_name: str,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Return per-layer AUROC and fitted peak probes for Figure 1 summaries."""
    rows: List[Dict[str, Any]] = []
    peak: Dict[str, Any] = {}
    split_arr = np.asarray(splits, dtype=object)
    n_layers = features.shape[1]

    for cls in CLASS_ORDER:
        sel = classes == cls
        if sel.sum() < 20:
            LOG.warning("[%s] too few %s claims (%d); skipped", backbone_name, cls, int(sel.sum()))
            continue
        tr = sel & (split_arr == "probe_train")
        te = sel & (split_arr == "probe_test")
        if tr.sum() < 10 or te.sum() < 10 or len(np.unique(labels[tr])) < 2:
            LOG.warning("[%s] %s: degenerate split; skipped", backbone_name, cls)
            continue

        best = (-1.0, -1, None, None)
        for layer in range(n_layers):
            x_tr, x_te = features[tr, layer, :], features[te, layer, :]
            scaler = StandardScaler().fit(x_tr)
            clf = LogisticRegression(
                C=1.0, max_iter=2000, random_state=seed
            ).fit(scaler.transform(x_tr), labels[tr])
            try:
                auroc = float(roc_auc_score(labels[te], clf.predict_proba(scaler.transform(x_te))[:, 1]))
            except ValueError:
                auroc = 0.5
            rows.append(
                {
                    "probe_schema_version": PROBE_SCHEMA_VERSION,
                    "backbone": backbone_name,
                    "layer": layer,
                    "class": CLASS_LABEL[cls],
                    "auroc": auroc,
                }
            )
            if auroc > best[0]:
                best = (auroc, layer, clf, scaler)
        peak[cls] = {"auroc": best[0], "layer": best[1], "clf": best[2], "scaler": best[3]}
        LOG.info("[%s] %s peak layer=%d auroc=%.3f", backbone_name, cls, best[1], best[0])

    return pd.DataFrame(rows), peak


def run(cfg: Config, overwrite: bool = False, max_rows: Optional[int] = None) -> pd.DataFrame:
    processed = cfg.path("processed")
    tables = cfg.path("tables")
    backbone_names = getattr(cfg, "_selected_backbones", None) or cfg.backbone_names
    suffix = "" if set(backbone_names) == set(cfg.backbone_names) else "_" + "_".join(backbone_names)
    out_csv = tables / f"layer_probe_auroc{suffix}.csv"
    peak_csv = tables / f"probe_peak_layers{suffix}.csv"
    if out_csv.exists() and not overwrite:
        existing = pd.read_csv(out_csv)
        if (
            "probe_schema_version" in existing.columns
            and not existing.empty
            and set(existing["probe_schema_version"].astype(int)) == {PROBE_SCHEMA_VERSION}
        ):
            LOG.info("[skip] %s exists with schema v%d", out_csv.name, PROBE_SCHEMA_VERSION)
            return existing
        LOG.info("[stale] %s lacks current probe schema; recomputing", out_csv.name)

    all_tables: List[pd.DataFrame] = []
    peak_rows: List[Dict[str, Any]] = []

    for backbone_name in backbone_names:
        files = sorted(processed.glob(f"claims_{backbone_name}_*.jsonl"))
        pooled: List[Dict[str, Any]] = []
        for path in files:
            if not any(k in path.stem for k in ("amber", "rbench")):
                continue
            rows = read_jsonl(path)
            for r in rows:
                pooled.append(r)
        if not pooled:
            continue

        feats, labels, classes, splits, index_map = extract_features(
            cfg, backbone_name, pooled, max_rows=max_rows
        )
        if feats.size == 0:
            LOG.warning("[%s] no probe features extracted", backbone_name)
            continue

        table, peak = fit_layer_probes(
            feats, labels, classes, splits, backbone_name, cfg.get("seed", 42)
        )
        all_tables.append(table)
        selected_rows = [pooled[i] for i in index_map]
        sampling_counts: Dict[str, int] = defaultdict(int)
        for row in selected_rows:
            key = "|".join(
                [str(row.get("claim_type")), str(row.get("label")), str(row.get("probe_split"))]
            )
            sampling_counts[key] += 1
        write_json(
            cfg.path("logs") / f"probe_sampling_{backbone_name}.json",
            {
                "probe_schema_version": PROBE_SCHEMA_VERSION,
                "max_images_per_stratum": cfg.get("mechanism", {}).get(
                    "probe_max_images_per_stratum"
                ),
                "selected_claims": len(selected_rows),
                "selected_images": len({str(row.get("image_id")) for row in selected_rows}),
                "strata": dict(sorted(sampling_counts.items())),
            },
        )
        for cls, info in peak.items():
            peak_rows.append(
                {
                    "probe_schema_version": PROBE_SCHEMA_VERSION,
                    "backbone": backbone_name,
                    "class": CLASS_LABEL[cls],
                    "peak_layer": info["layer"],
                    "peak_auroc": info["auroc"],
                }
            )

    if all_tables:
        df = pd.concat(all_tables, ignore_index=True)
        if max_rows is None:
            expected = {
                (backbone, cls)
                for backbone in backbone_names
                for cls in CLASS_LABEL.values()
            }
            observed = set(zip(df["backbone"], df["class"]))
            missing = sorted(expected - observed)
            if missing:
                raise RuntimeError(
                    "full layer-probe matrix is incomplete; missing backbone/class pairs: "
                    f"{missing}"
                )
        df.to_csv(out_csv, index=False)
        pd.DataFrame(peak_rows).to_csv(peak_csv, index=False)
        LOG.info("wrote %s and %s", out_csv.name, peak_csv.name)
        return df
    LOG.error("no probe results produced")
    return pd.DataFrame()
