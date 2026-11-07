"""T3: visual-attention-collapse analysis -> Figure 2.

For each claim we take the attention row of the key token, average over heads,
sum the mass that lands on image-patch tokens, normalise by the total mass, and
average across layers. Relation claims restrict the visual mass to the patches
covered by the subject and object boxes when those boxes are available.
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..backbones.registry import build_backbone
from ..config import Config
from ..data.images import load_image
from ..tools.geometry import box_to_patch_mask
from ..utils import get_logger, read_jsonl, write_jsonl

LOG = get_logger("hmqr.attention")
ATTENTION_SCHEMA_VERSION = 2


def _balanced_rows(
    rows: List[Dict[str, Any]],
    max_rows: Optional[int],
    max_images_per_stratum: Optional[int] = None,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Round-robin over (claim type, label) for a representative smoke subset."""
    if max_images_per_stratum is not None:
        by_stratum: Dict[Tuple[str, int], Dict[str, Dict[str, Any]]] = defaultdict(dict)
        for row in rows:
            key = (str(row["claim_type"]), int(row["label"]))
            by_stratum[key].setdefault(str(row.get("image_id")), row)
        sampled: List[Dict[str, Any]] = []
        for stratum_i, key in enumerate(sorted(by_stratum)):
            candidates = list(by_stratum[key].values())
            rng = np.random.default_rng(seed + stratum_i)
            order = rng.permutation(len(candidates))[:max_images_per_stratum]
            sampled.extend(candidates[i] for i in order)
        rows = sampled
    if max_rows is None or len(rows) <= max_rows:
        return rows
    buckets: Dict[Tuple[str, int], Deque[Dict[str, Any]]] = defaultdict(deque)
    for row in rows:
        buckets[(str(row["claim_type"]), int(row["label"]))].append(row)
    selected: List[Dict[str, Any]] = []
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


def visual_attention_statistics(attn: np.ndarray, visual_mask: np.ndarray) -> Dict[str, float]:
    """attn: [num_layers, num_heads, seq_len].

    Already the single row at the key position -- the backbone captures it per
    layer and releases the square map immediately, avoiding a retained
    [L, heads, S, S] tensor (about 21 GB on LLaVA-1.6).
    """
    row = attn.mean(axis=1)                                # [L, S] average over heads
    vis = row[:, visual_mask].sum(axis=1)                  # [L]
    tot = row.sum(axis=1) + 1e-8
    raw = float(np.mean(vis / tot))
    uniform = float(np.sum(visual_mask) / len(visual_mask)) if len(visual_mask) else float("nan")
    enrichment = raw / uniform if uniform > 0 else float("nan")
    return {
        "vattn_raw": raw,
        "uniform_baseline": uniform,
        "vattn_enrichment": enrichment,
        "n_visual_tokens": int(np.sum(visual_mask)),
        "effective_seq_len": int(len(visual_mask)),
    }


def visual_attention_ratio(attn: np.ndarray, visual_mask: np.ndarray) -> float:
    """Compatibility accessor for the unnormalised visual-attention mass."""
    return visual_attention_statistics(attn, visual_mask)["vattn_raw"]


def _region_mask(
    visual_mask: np.ndarray, boxes: List[Optional[List[float]]], image_size: Tuple[int, int]
) -> np.ndarray:
    """Restrict the visual mask to the patches covered by the given boxes."""
    idx = np.where(visual_mask)[0]
    n_patch = idx.size
    if n_patch == 0:
        return visual_mask
    grid = int(round(np.sqrt(n_patch)))
    if grid * grid != n_patch:
        return visual_mask  # non-square layout (e.g. anyres); fall back to all patches
    keep = np.zeros(n_patch, dtype=bool)
    any_box = False
    for box in boxes:
        if box is None:
            continue
        keep |= box_to_patch_mask(box, image_size, grid)
        any_box = True
    if not any_box or not keep.any():
        return visual_mask
    out = np.zeros_like(visual_mask)
    out[idx[keep]] = True
    return out


def collect(cfg: Config, max_rows: Optional[int] = None) -> pd.DataFrame:
    processed = cfg.path("processed")
    prompt = cfg["generation"]["caption_prompt"]
    records: List[Dict[str, Any]] = []

    backbone_names = getattr(cfg, "_selected_backbones", None) or cfg.backbone_names
    for backbone_name in backbone_names:
        rows: List[Dict[str, Any]] = []
        for path in sorted(processed.glob(f"claims_{backbone_name}_*.jsonl")):
            if any(k in path.stem for k in ("amber", "rbench")):
                rows.extend(read_jsonl(path))
        rows = [r for r in rows if r.get("claim_type") in ("attr", "rel") and r.get("label") in (0, 1)]
        if not rows:
            continue
        paper_cap = (
            cfg.get("mechanism", {}).get("attention_max_images_per_stratum")
            if max_rows is None else None
        )
        rows = _balanced_rows(
            rows,
            max_rows,
            max_images_per_stratum=int(paper_cap) if paper_cap is not None else None,
            seed=int(cfg.get("seed", 42)),
        )
        partial_path = cfg.path("tables") / f"attention_vattn_{backbone_name}.partial.jsonl"
        valid_claim_ids = {r.get("claim_id") for r in rows}
        prior = (
            [
                r for r in read_jsonl(partial_path)
                if r.get("claim_id") in valid_claim_ids
                and r.get("attention_schema_version") == ATTENTION_SCHEMA_VERSION
            ]
            if partial_path.exists() else []
        )
        completed = {r.get("claim_id") for r in prior}
        if prior:
            LOG.info(
                "[resume] attention %s: %d/%d claims checkpointed",
                backbone_name, len(completed), len(rows),
            )
            records.extend(prior)

        backbone = build_backbone(cfg, backbone_name)
        for row_i, row in enumerate(
            tqdm(rows, desc=f"attention {backbone_name}", ncols=88), start=1
        ):
            if row.get("claim_id") in completed:
                continue
            image = load_image(row["image_path"])
            if image is None:
                continue
            if row["claim_type"] == "attr":
                kws = [w for w in [row.get("attribute"), row.get("object")] if w]
            else:
                kws = [w for w in [row.get("relation"), row.get("subject"), row.get("object")] if w]
            if not kws:
                continue
            target = row.get("sentence") or row.get("caption") or ""
            try:
                out = backbone.forward_states(image, prompt, target, keywords=kws)
            except Exception as exc:  # noqa: BLE001
                LOG.warning("forward_states failed (%s): %s", row["claim_id"], exc)
                continue

            attn = out["attentions"]
            vmask = backbone.visual_token_mask()
            if row["claim_type"] == "rel":
                vmask = _region_mask(
                    vmask, [row.get("box_subject"), row.get("box_object")], image.size
                )
            record = {
                    "attention_schema_version": ATTENTION_SCHEMA_VERSION,
                    "backbone": backbone_name,
                    "image_id": row.get("image_id"),
                    "claim_id": row["claim_id"],
                    "claim_type": row["claim_type"],
                    "label": int(row["label"]),
            }
            record.update(visual_attention_statistics(attn, vmask))
            records.append(record)
            if row_i % 50 == 0 or row_i == len(rows):
                shard_rows = [r for r in records if r.get("backbone") == backbone_name]
                write_jsonl(partial_path, shard_rows)
        shard_rows = [r for r in records if r.get("backbone") == backbone_name]
        write_jsonl(partial_path, shard_rows)
        backbone.unload()

    return pd.DataFrame(records)


def _effect_and_ci(group: pd.DataFrame, seed: int = 42, n_boot: int = 500):
    def effect(frame: pd.DataFrame) -> float:
        hallu = frame[frame["label"] == 1]["vattn_enrichment"].dropna().to_numpy(float)
        grounded = frame[frame["label"] == 0]["vattn_enrichment"].dropna().to_numpy(float)
        if len(hallu) < 2 or len(grounded) < 2:
            return float("nan")
        pooled = np.sqrt((np.var(hallu, ddof=1) + np.var(grounded, ddof=1)) / 2.0)
        return float((np.mean(hallu) - np.mean(grounded)) / (pooled + 1e-8))

    observed = effect(group)
    image_ids = np.asarray(sorted(set(group["image_id"].astype(str))))
    if len(image_ids) < 2:
        return observed, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    values = []
    keyed = {image_id: group[group["image_id"].astype(str) == image_id] for image_id in image_ids}
    for _ in range(n_boot):
        sample = pd.concat([keyed[value] for value in rng.choice(image_ids, len(image_ids), True)])
        value = effect(sample)
        if np.isfinite(value):
            values.append(value)
    if len(values) < 20:
        return observed, float("nan"), float("nan")
    return observed, float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def descriptive_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Distribution summaries and image-level effect uncertainty for Figure 2."""
    rows: List[Dict[str, Any]] = []
    effects = {
        (backbone, ctype): _effect_and_ci(group)
        for (backbone, ctype), group in df.groupby(["backbone", "claim_type"], sort=True)
    }
    for (backbone, ctype, label), grp in df.groupby(
        ["backbone", "claim_type", "label"], sort=True
    ):
        values = grp["vattn_enrichment"].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
        rows.append(
            {
                "attention_schema_version": ATTENTION_SCHEMA_VERSION,
                "backbone": backbone,
                "claim_type": ctype,
                "label": int(label),
                "n_images": int(grp["image_id"].astype(str).nunique()),
                "n_claims": int(values.size),
                "mean": float(values.mean()),
                "median": float(median),
                "q1": float(q1),
                "q3": float(q3),
                "iqr": float(q3 - q1),
                "effect_size": effects[(backbone, ctype)][0],
                "effect_ci_low": effects[(backbone, ctype)][1],
                "effect_ci_high": effects[(backbone, ctype)][2],
            }
        )
    return pd.DataFrame(rows)


def run(cfg: Config, overwrite: bool = False, max_rows: Optional[int] = None) -> pd.DataFrame:
    tables = cfg.path("tables")
    backbone_names = getattr(cfg, "_selected_backbones", None) or cfg.backbone_names
    suffix = "" if set(backbone_names) == set(cfg.backbone_names) else "_" + "_".join(backbone_names)
    out = tables / f"attention_vattn{suffix}.csv"
    summary_out = tables / f"attention_summary{suffix}.csv"
    if out.exists() and not overwrite:
        existing = pd.read_csv(out)
        if (
            {"vattn_enrichment", "uniform_baseline", "image_id", "attention_schema_version"}
            <= set(existing.columns)
            and not existing.empty
            and set(existing["attention_schema_version"].astype(int))
            == {ATTENTION_SCHEMA_VERSION}
        ):
            LOG.info("[skip] %s exists with schema v%d", out.name, ATTENTION_SCHEMA_VERSION)
            # The summary is a deterministic CPU-only derivative. Refresh it
            # even when expensive attention extraction is safely reused, so a
            # stale summary can never accompany a current raw table.
            descriptive_summary(existing).to_csv(summary_out, index=False)
            return existing
        LOG.info("[stale] %s lacks current attention schema; recomputing", out.name)
    df = collect(cfg, max_rows=max_rows)
    if df.empty:
        LOG.error("no attention records produced")
        return df
    if max_rows is None:
        expected = {
            (backbone, ctype, label)
            for backbone in backbone_names
            for ctype in ("attr", "rel")
            for label in (0, 1)
        }
        counts = df.groupby(["backbone", "claim_type", "label"]).size().to_dict()
        missing = sorted(key for key in expected if counts.get(key, 0) < 10)
        if missing:
            raise RuntimeError(
                "full attention matrix needs at least 10 rows per backbone/type/label; "
                f"insufficient groups: {missing}"
            )
    df.to_csv(out, index=False)
    descriptive_summary(df).to_csv(summary_out, index=False)
    for backbone in backbone_names:
        partial = tables / f"attention_vattn_{backbone}.partial.jsonl"
        if partial.exists():
            partial.unlink()
    LOG.info("wrote %s and %s", out.name, summary_out.name)
    return df
