#!/usr/bin/env python
"""Validate and merge artifacts produced by independent per-backbone GPU shards."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hmqr.config import load_config  # noqa: E402
from hmqr.extract.build_claims import SPLIT_SCHEMA_VERSION  # noqa: E402
from hmqr.mechanism.attention_analysis import ATTENTION_SCHEMA_VERSION  # noqa: E402
from hmqr.mechanism.layer_probe import PROBE_SCHEMA_VERSION  # noqa: E402
from hmqr.revise.pipeline import PIPELINE_SCHEMA_VERSION  # noqa: E402
from hmqr.signals.compute import SIGNALS_SCHEMA_VERSION  # noqa: E402
from hmqr.utils import mark_done, read_json, read_jsonl, write_json  # noqa: E402


def datasets(cfg):
    return list(dict.fromkeys(
        list(cfg["datasets"]["detect"])
        + list(cfg["datasets"]["quality"])
    ))


def validate_claims(cfg, require_signals: bool = False) -> None:
    processed = cfg.path("processed")
    subset_limit = cfg["datasets"].get("subset_limit")
    errors = []
    all_ids = set()
    shared_roles = {}
    shared_probe_roles = {}
    for backbone in cfg.backbone_names:
        for dataset in datasets(cfg):
            path = processed / f"claims_{backbone}_{dataset}.jsonl"
            if not path.exists():
                errors.append(f"missing {path.name}")
                continue
            rows = read_jsonl(path)
            ids = [str(r.get("claim_id")) for r in rows]
            if len(ids) != len(set(ids)):
                errors.append(f"duplicate claim_id in {path.name}")
            overlap = all_ids.intersection(ids)
            if overlap:
                errors.append(f"cross-file duplicate claim_id in {path.name}: {next(iter(overlap))}")
            all_ids.update(ids)
            if not rows:
                errors.append(f"{path.name}: empty claim file")
            roles = {str(r.get("split")) for r in rows}
            if not roles <= {"calib_cp", "calib_revision", "test"}:
                errors.append(f"{path.name}: invalid/legacy split roles {sorted(roles)}")
            if any(r.get("split_schema_version") != SPLIT_SCHEMA_VERSION for r in rows):
                errors.append(f"{path.name}: stale/missing split_schema_version")
            image_roles = {}
            probe_roles = {}
            for row in rows:
                image_roles.setdefault(str(row.get("image_id")), set()).add(row.get("split"))
                probe_roles.setdefault(str(row.get("image_id")), set()).add(row.get("probe_split"))
                key = (dataset, str(row.get("image_id")))
                prior = shared_roles.setdefault(key, row.get("split"))
                if prior != row.get("split"):
                    errors.append(
                        f"cross-backbone split mismatch for {dataset}/{row.get('image_id')}"
                    )
                prior_probe = shared_probe_roles.setdefault(key, row.get("probe_split"))
                if prior_probe != row.get("probe_split"):
                    errors.append(
                        f"cross-backbone probe split mismatch for "
                        f"{dataset}/{row.get('image_id')}"
                    )
            leaked = [image_id for image_id, values in image_roles.items() if len(values) > 1]
            if leaked:
                errors.append(f"{path.name}: image split leakage for {leaked[0]}")
            probe_leaked = [
                image_id for image_id, values in probe_roles.items() if len(values) > 1
            ]
            if probe_leaked:
                errors.append(f"{path.name}: probe image leakage for {probe_leaked[0]}")
            if require_signals:
                missing = [
                    r.get("claim_id") for r in rows
                    if r.get("g_global") is None
                    or r.get("signal_available") is None
                    or r.get("signals_schema_version") != SIGNALS_SCHEMA_VERSION
                ]
                if missing:
                    errors.append(f"{path.name}: {len(missing)} rows missing signals")
    if errors:
        raise RuntimeError("artifact validation failed:\n  " + "\n  ".join(errors))
    mark_done(processed, "signals_done" if require_signals else "claims_done")


def merge_csv_shards(cfg) -> None:
    tables = cfg.path("tables")
    specs = (
        ("layer_probe_auroc", ["backbone", "layer", "class"], "probe_schema_version", PROBE_SCHEMA_VERSION),
        ("probe_peak_layers", ["backbone", "class"], "probe_schema_version", PROBE_SCHEMA_VERSION),
        ("attention_vattn", ["backbone", "claim_id"], "attention_schema_version", ATTENTION_SCHEMA_VERSION),
        ("attention_summary", ["backbone", "claim_type", "label"], "attention_schema_version", ATTENTION_SCHEMA_VERSION),
    )
    for stem, keys, schema_col, schema_version in specs:
        paths = [tables / f"{stem}_{backbone}.csv" for backbone in cfg.backbone_names]
        missing = [p.name for p in paths if not p.exists()]
        if missing:
            raise FileNotFoundError(f"cannot merge {stem}; missing: {missing}")
        frames = [pd.read_csv(path) for path in paths]
        if schema_col is not None:
            stale = [
                path.name for path, frame in zip(paths, frames)
                if schema_col not in frame.columns or frame.empty
                or set(frame[schema_col].astype(int)) != {schema_version}
            ]
            if stale:
                raise RuntimeError(
                    f"{stem}: stale/missing schema v{schema_version} in {stale}"
                )
        df = pd.concat(frames, ignore_index=True)
        if df.duplicated(keys).any():
            raise RuntimeError(f"duplicate {keys} rows while merging {stem}")
        observed = set(df["backbone"].astype(str))
        expected = set(cfg.backbone_names)
        if observed != expected:
            raise RuntimeError(f"{stem}: backbone matrix mismatch {observed} != {expected}")
        df.to_csv(tables / f"{stem}.csv", index=False)


def merge_signal_logs(cfg) -> None:
    logs = cfg.path("logs")
    totals = {"clip_image_calls": 0, "clip_text_calls": 0, "dino_forward": 0}
    for backbone in cfg.backbone_names:
        path = logs / f"signal_tool_calls_{backbone}.json"
        if not path.exists():
            raise FileNotFoundError(path)
        record = read_json(path)
        for key in totals:
            totals[key] += int(record.get(key, 0))
    write_json(logs / "signal_tool_calls.json", totals)


def validate_revisions(cfg) -> None:
    errors = []
    for backbone in cfg.backbone_names:
        for dataset in ("amber_attr", "amber_rel", "rbench", "mmrel", "chair500"):
            path = cfg.path("processed") / f"revised_labeled_{backbone}_{dataset}.jsonl"
            if not path.exists():
                errors.append(f"missing {path.name}")
                continue
            rows = read_jsonl(path)
            if not rows:
                errors.append(f"empty {path.name}")
            if any(
                int(r.get("pipeline_schema_version", -1)) != PIPELINE_SCHEMA_VERSION
                for r in rows
            ):
                errors.append(
                    f"stale/missing pipeline_schema_version in {path.name}; "
                    f"expected v{PIPELINE_SCHEMA_VERSION}"
                )
            if any(r.get("decision") not in {"keep", "abstain", "revise"} for r in rows):
                errors.append(f"invalid decision in {path.name}")
            revised = [r for r in rows if r.get("decision") == "revise"]
            if dataset != "chair500" and any(
                not r.get("intervention_evaluable")
                or not r.get("target_only_edit")
                for r in revised
            ):
                errors.append(f"missing native-label intervention audit in {path.name}")
    if errors:
        raise RuntimeError("revision validation failed:\n  " + "\n  ".join(errors))
    mark_done(cfg.path("processed"), "revise_done")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--phase", required=True, choices=("01", "02", "04", "08", "all"),
        help="validate/merge the artifacts produced through this phase",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.phase in {"01", "all"}:
        validate_claims(cfg, require_signals=False)
    if args.phase in {"02", "all"}:
        validate_claims(cfg, require_signals=True)
        merge_signal_logs(cfg)
    if args.phase in {"04", "all"}:
        merge_csv_shards(cfg)
    if args.phase in {"08", "all"}:
        validate_revisions(cfg)
    print(f"multi-GPU merge/validation PASS through phase {args.phase}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
