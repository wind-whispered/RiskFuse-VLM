#!/usr/bin/env python
"""Repair cached image identities and migrate claims to three-way image splits.

This repairs historical AMBER/MMRel question-level ``image_id`` values, then
assigns a shared image-disjoint split. It is metadata-only and never regenerates
captions or reloads a VLM. The default mode reports planned changes; pass
``--apply`` to write them.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hmqr.config import load_config  # noqa: E402
from hmqr.data.datasets import load_dataset  # noqa: E402
from hmqr.utils import read_jsonl, write_jsonl  # noqa: E402

SPLIT_SCHEMA_VERSION = 2


def canonical_image_id(row):
    dataset = str(row.get("dataset", ""))
    image_path = str(row.get("image_path", ""))
    if dataset.startswith("amber_") and image_path:
        return Path(image_path.split("?", 1)[0]).stem
    if dataset == "mmrel" and image_path:
        return image_path.replace("\\", "/").split("?", 1)[0]
    return str(row.get("image_id"))


def split_map(image_ids, seed: int, cp_ratio: float, revision_ratio: float):
    ids = sorted({str(value) for value in image_ids})
    random.Random(seed).shuffle(ids)
    n_cp = int(round(len(ids) * cp_ratio))
    n_revision = int(round(len(ids) * revision_ratio))
    return {
        image_id: (
            "calib_cp" if index < n_cp else
            "calib_revision" if index < n_cp + n_revision else "test"
        )
        for index, image_id in enumerate(ids)
    }


def probe_map(image_ids, seed: int, train_ratio: float = 0.6):
    ids = sorted({str(value) for value in image_ids})
    random.Random(seed + 1).shuffle(ids)
    n_train = int(round(len(ids) * train_ratio))
    train = set(ids[:n_train])
    return {image_id: ("probe_train" if image_id in train else "probe_test") for image_id in ids}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    cp = cfg["cp"]
    processed = cfg.path("processed")
    changed_files = 0
    claim_image_map = {}
    for path in sorted(processed.glob("claims_*.jsonl")):
        rows = read_jsonl(path)
        if not rows or all(
            row.get("split_schema_version") == SPLIT_SCHEMA_VERSION for row in rows
        ):
            continue
        dataset = str(rows[0].get("dataset"))
        samples = load_dataset(cfg, dataset)
        mapping = split_map(
            [sample.image_id for sample in samples],
            int(cfg.get("seed", 42)),
            float(cp.get("calib_cp_ratio", 0.25)),
            float(cp.get("calib_revision_ratio", 0.25)),
        )
        probe_roles = probe_map([sample.image_id for sample in samples], int(cfg.get("seed", 42)))
        counts = {"calib_cp": 0, "calib_revision": 0, "test": 0}
        for row in rows:
            row["image_id"] = canonical_image_id(row)
            claim_image_map[str(row.get("claim_id"))] = str(row["image_id"])
            role = mapping.get(str(row["image_id"]), "test")
            row["split"] = role
            row["probe_split"] = probe_roles.get(str(row["image_id"]), "probe_test")
            row["split_schema_version"] = SPLIT_SCHEMA_VERSION
            # Split-dependent normalisation/projection parameters are now stale.
            # Keep expensive raw model outputs so step 02 can resume without
            # recomputing rows whose grounding and entropy values are reusable.
            row.pop("signals_schema_version", None)
            if row.get("claim_type") in {"attr", "rel"}:
                row["signal_computed"] = False
            for key in (
                "g_perp", "d_conflict", "residual_center", "residual_mad",
                "residual_fit_split", "residual_group", "residual_schema_version",
            ):
                row.pop(key, None)
            counts[role] += 1
        changed_files += 1
        print(f"{path.name}: {counts}")
        if args.apply:
            write_jsonl(path, rows)
            signals_partial = path.with_name(path.name + ".signals.partial")
            if signals_partial.exists():
                signals_partial.unlink()
                print(f"  removed stale checkpoint: {signals_partial.name}")
    if args.apply and claim_image_map:
        import pandas as pd

        from hmqr.mechanism.attention_analysis import descriptive_summary

        for attention_path in sorted(cfg.path("tables").glob("attention_vattn*.csv")):
            frame = pd.read_csv(attention_path)
            required = {"claim_id", "image_id", "vattn_enrichment"}
            if not required <= set(frame.columns):
                print(f"  attention cache needs step 04 rebuild: {attention_path.name}")
                continue
            frame["image_id"] = frame["claim_id"].astype(str).map(claim_image_map).fillna(
                frame["image_id"].astype(str)
            )
            frame.to_csv(attention_path, index=False)
            suffix = attention_path.stem.removeprefix("attention_vattn")
            summary_path = cfg.path("tables") / f"attention_summary{suffix}.csv"
            descriptive_summary(frame).to_csv(summary_path, index=False)
            print(f"  refreshed attention image bootstrap metadata: {attention_path.name}")
    action = "updated" if args.apply else "would update"
    print(f"{action} {changed_files} claim files; no model inference performed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
