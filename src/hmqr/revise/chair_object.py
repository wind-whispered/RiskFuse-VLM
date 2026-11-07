"""Conservative object-presence correction for CHAIR-500 captions."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable

from ..config import Config
from ..data.datasets import load_dataset
from ..data.images import load_image
from ..eval.caption_quality import mask_object_mentions, mentioned_objects
from ..signals.grounding_object import object_presence
from ..tools.grounding_dino import GroundingDino
from ..utils import read_jsonl, write_json, write_jsonl


_DINO: GroundingDino | None = None


def _dino(cfg: Config) -> GroundingDino:
    global _DINO
    if _DINO is None:
        _DINO = GroundingDino(
            cfg["tools"]["grounding_dino_hf_id"], cfg.get("device", "cuda"),
            cfg.get("dtype", "float16"),
            float(cfg["tools"]["grounding_dino_box_threshold"]),
            float(cfg["tools"]["grounding_dino_text_threshold"]),
            cache_dir=cfg.path("tool_cache"),
        )
    return _DINO


def _captions(cfg: Config, backbone: str) -> dict[str, str]:
    processed = cfg.path("processed")
    output = {
        str(row["image_id"]): str(row.get("caption", ""))
        for row in read_jsonl(processed / f"claims_{backbone}_chair500.jsonl")
        if row.get("caption")
    }
    supplemental = processed / f"chair_captions_{backbone}.jsonl"
    if supplemental.exists():
        for row in read_jsonl(supplemental):
            if row.get("caption"):
                output[str(row["image_id"])] = str(row["caption"])
    return output


def preflight(cfg: Config, backbone: str, overwrite: bool = False) -> Dict[str, Any]:
    captions = _captions(cfg, backbone)
    samples = {str(sample.image_id): sample for sample in load_dataset(cfg, "chair500")}
    expected = int(cfg["datasets"]["coco"].get("chair_num_images", 500))
    missing = sorted(set(samples) - set(captions))
    if len(captions) != expected or missing:
        raise RuntimeError(
            f"{backbone}: CHAIR captions incomplete: {len(captions)}/{expected}; "
            f"missing={missing[:10]}"
        )
    out = cfg.path("processed") / f"revised_labeled_{backbone}_chair500.jsonl"
    if out.exists() and not overwrite:
        rows = read_jsonl(out)
        if len(rows) == expected and all(
            int(row.get("pipeline_schema_version", -1)) == 21 for row in rows
        ):
            return {"status": "complete", "pending_object_queries": 0, "n_images": expected}
    pairs = {
        (image_id, obj)
        for image_id, caption in captions.items()
        for obj in mentioned_objects(caption)
    }
    return {
        "status": "ready",
        "pending_object_queries": len(pairs),
        "n_images": expected,
        "frozen_box_threshold": float(cfg["tools"]["grounding_dino_box_threshold"]),
    }


def run(cfg: Config, backbone: str, overwrite: bool = False) -> list[Dict[str, Any]]:
    report = preflight(cfg, backbone, overwrite=overwrite)
    out = cfg.path("processed") / f"revised_labeled_{backbone}_chair500.jsonl"
    if report["status"] == "complete":
        return read_jsonl(out)
    captions = _captions(cfg, backbone)
    samples = {str(sample.image_id): sample for sample in load_dataset(cfg, "chair500")}
    detector = _dino(cfg)
    partial = out.with_name(out.name + ".partial")
    checkpoint = {
        str(row.get("image_id")): row
        for row in (read_jsonl(partial) if partial.exists() and not overwrite else [])
    }
    rows: list[Dict[str, Any]] = []
    for image_id in sorted(samples):
        if image_id in checkpoint:
            rows.append(checkpoint[image_id])
            continue
        sample = samples[image_id]
        caption = captions[image_id]
        image = load_image(sample.image_path)
        scores: Dict[str, float] = {}
        unsupported = []
        for obj in sorted(mentioned_objects(caption)):
            result = object_presence(
                detector, image, obj, cache_key=str(sample.image_path)
            )
            scores[obj] = float(result["g_object"])
            if not result["box_found"]:
                unsupported.append(obj)
        revised = mask_object_mentions(caption, unsupported)
        rows.append({
            "claim_id": f"{backbone}|chair500|{image_id}|object_presence",
            "image_id": image_id,
            "image_path": str(sample.image_path),
            "backbone": backbone,
            "dataset": "chair500",
            "mode": "generative",
            "caption": caption,
            "original_sentence": caption,
            "final_caption": revised,
            "decision": "revise" if unsupported else "keep",
            "action": "mask_unsupported_object" if unsupported else "keep",
            "unsupported_objects": unsupported,
            "object_support_scores": scores,
            "revision_requested": bool(unsupported),
            "revision_attempted": False,
            "retrieval_ids": [],
            "quality_transfer_policy": "object_presence_prune",
            "frozen_box_threshold": float(cfg["tools"]["grounding_dino_box_threshold"]),
            "pipeline_schema_version": 21,
        })
        if len(rows) % 25 == 0:
            write_jsonl(partial, rows)
    write_jsonl(out, rows)
    if partial.exists():
        partial.unlink()
    write_json(out.with_name(out.name + ".complete.json"), {
        "row_count": len(rows),
        "schema_version": 21,
        "policy": "object_presence_prune",
        "dino_forward": detector.n_forward,
    })
    return rows
