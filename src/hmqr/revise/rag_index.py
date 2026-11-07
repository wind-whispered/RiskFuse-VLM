"""T7a: build the Visual Genome evidence index (no training involved).

attribute index : key = (object_class, attribute_family)   -> candidate expressions
relation  index : key = (subject_class, object, family)    -> candidate relations

Each entry keeps the VG image id, region metadata and text. Retrieved items are
reference context only; they are not evidence about the current evaluation image.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from ..config import Config
from ..utils import get_logger, write_json

LOG = get_logger("hmqr.rag_index")

ATTR_INDEX = "attr_index.json"
REL_INDEX = "rel_index.json"
MAX_ENTRIES_PER_KEY = 20
INDEX_SCHEMA_VERSION = 3

_COLOR_WORDS = {
    "red",
    "blue",
    "green",
    "yellow",
    "black",
    "white",
    "brown",
    "gray",
    "grey",
    "orange",
    "purple",
    "pink",
}
_MATERIAL_WORDS = {"wooden", "wood", "metal", "metallic", "plastic", "glass", "stone"}
_STATE_WORDS = {
    "open",
    "closed",
    "wet",
    "dry",
    "empty",
    "full",
    "broken",
    "clean",
    "dirty",
}
_DIRECTION_WORDS = {
    "above",
    "below",
    "under",
    "behind",
    "left",
    "right",
    "front",
    "over",
}


def attribute_family(attribute: str) -> str:
    words = set(_norm(attribute).split())
    if words & _COLOR_WORDS:
        return "color"
    if words & _MATERIAL_WORDS:
        return "material"
    if words & _STATE_WORDS:
        return "state"
    return "other"


def relation_family(relation: str) -> str:
    from ..signals.support import relation_signal_kind

    return relation_signal_kind({"relation": relation})


def _norm(text: str) -> str:
    t = str(text).strip().lower()
    t = re.sub(r"[^a-z0-9 ]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _load_vg_json(vg_root: Path, name: str) -> Any:
    for cand in (vg_root / name, vg_root / name.replace(".json", "") / name):
        if cand.exists():
            LOG.info("loading %s", cand.name)
            with open(cand, "r", encoding="utf-8") as f:
                return json.load(f)
    LOG.warning("%s not found under %s", name, vg_root)
    return None


def build_attribute_index(
    vg_root: Path,
    limit_images: Optional[int] = None,
    excluded_image_ids: set[str] | None = None,
) -> Dict[str, List[Dict]]:
    data = _load_vg_json(vg_root, "attributes.json")
    index: Dict[str, List[Dict]] = defaultdict(list)
    if data is None:
        return index
    for entry in tqdm(
        data[:limit_images] if limit_images else data, desc="VG attributes", ncols=88
    ):
        image_id = entry.get("image_id", entry.get("id"))
        if str(image_id) in (excluded_image_ids or set()):
            continue
        for att in entry.get("attributes", []):
            names = att.get("names") or ([att["name"]] if "name" in att else [])
            attrs = att.get("attributes") or []
            if not names or not attrs:
                continue
            obj = _norm(names[0])
            box = [
                att.get("x", 0),
                att.get("y", 0),
                att.get("x", 0) + att.get("w", 0),
                att.get("y", 0) + att.get("h", 0),
            ]
            for a in attrs:
                key = f"{obj}|{attribute_family(a)}"
                if len(index[key]) >= MAX_ENTRIES_PER_KEY:
                    continue
                index[key].append(
                    {
                        "image_id": image_id,
                        "box": box,
                        "text": f"{_norm(a)} {obj}",
                        "object": obj,
                        "attribute": _norm(a),
                    }
                )
    LOG.info("attribute index keys: %d", len(index))
    return index


def build_relation_index(
    vg_root: Path,
    limit_images: Optional[int] = None,
    excluded_image_ids: set[str] | None = None,
) -> Dict[str, List[Dict]]:
    data = _load_vg_json(vg_root, "relationships.json")
    index: Dict[str, List[Dict]] = defaultdict(list)
    if data is None:
        return index
    for entry in tqdm(
        data[:limit_images] if limit_images else data, desc="VG relations", ncols=88
    ):
        image_id = entry.get("image_id", entry.get("id"))
        if str(image_id) in (excluded_image_ids or set()):
            continue
        for rel in entry.get("relationships", []):
            subj = rel.get("subject", {})
            obj = rel.get("object", {})
            pred = _norm(rel.get("predicate", ""))
            s_name = _norm((subj.get("names") or [subj.get("name", "")])[0])
            o_name = _norm((obj.get("names") or [obj.get("name", "")])[0])
            if not (s_name and o_name and pred):
                continue
            key = f"{s_name}|{o_name}|{relation_family(pred)}"
            if len(index[key]) >= MAX_ENTRIES_PER_KEY:
                continue
            index[key].append(
                {
                    "image_id": image_id,
                    "box_subject": [
                        subj.get("x", 0),
                        subj.get("y", 0),
                        subj.get("x", 0) + subj.get("w", 0),
                        subj.get("y", 0) + subj.get("h", 0),
                    ],
                    "box_object": [
                        obj.get("x", 0),
                        obj.get("y", 0),
                        obj.get("x", 0) + obj.get("w", 0),
                        obj.get("y", 0) + obj.get("h", 0),
                    ],
                    "text": f"{s_name} {pred} {o_name}",
                    "subject": s_name,
                    "relation": pred.replace(" ", "_"),
                    "object": o_name,
                }
            )
    LOG.info("relation index keys: %d", len(index))
    return index


def build_region_texts(
    vg_root: Path,
    limit_images: Optional[int] = None,
    excluded_image_ids: set[str] | None = None,
) -> List[Dict[str, Any]]:
    """Flat list of VG region descriptions, used as the BM25 corpus."""
    data = _load_vg_json(vg_root, "region_descriptions.json")
    out: List[Dict[str, Any]] = []
    if data is None:
        return out
    for entry in tqdm(
        data[:limit_images] if limit_images else data, desc="VG regions", ncols=88
    ):
        entry_id = entry.get("image_id", entry.get("id"))
        if str(entry_id) in (excluded_image_ids or set()):
            continue
        for reg in entry.get("regions", []):
            region_id = reg.get("image_id", entry_id)
            if str(region_id) in (excluded_image_ids or set()):
                continue
            phrase = _norm(reg.get("phrase", ""))
            if not phrase:
                continue
            out.append(
                {
                    "image_id": region_id,
                    "box": [
                        reg.get("x", 0),
                        reg.get("y", 0),
                        reg.get("x", 0) + reg.get("width", 0),
                        reg.get("y", 0) + reg.get("height", 0),
                    ],
                    "text": phrase,
                }
            )
    LOG.info("region descriptions: %d", len(out))
    return out


def run(
    cfg: Config, overwrite: bool = False, limit_images: Optional[int] = None
) -> None:
    vg_root = cfg.abspath(cfg["datasets"]["visual_genome"]["root"])
    out_dir = cfg.path("vg_index")
    out_dir.mkdir(parents=True, exist_ok=True)
    from ..review.overlap import audit, sha256

    overlap = audit(cfg)
    if (
        cfg.get("rag", {}).get("require_complete_identity_audit", True)
        and not overlap["identity_audit_complete"]
    ):
        raise RuntimeError(
            "incomplete evaluation-image identity audit; inspect reviewer_revision/vg_overlap_audit.json before building a formal index"
        )
    exclusions = set(overlap["excluded_vg_image_ids"])
    sources = {
        name: sha256(vg_root / name)
        for name in (
            "attributes.json",
            "relationships.json",
            "region_descriptions.json",
        )
    }

    manifest_path = out_dir / "index_manifest.json"
    if (out_dir / ATTR_INDEX).exists() and manifest_path.exists() and not overwrite:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("schema_version") == INDEX_SCHEMA_VERSION
            and manifest.get("overlap_signature") == overlap["signature"]
            and manifest.get("sources") == sources
            and manifest.get("limit_images") == limit_images
            and all(
                (out_dir / name).exists()
                for name in (ATTR_INDEX, REL_INDEX, "region_texts.json")
            )
        ):
            LOG.info("[skip] VG index already built in %s", out_dir)
            return

    attr_index = build_attribute_index(vg_root, limit_images, exclusions)
    rel_index = build_relation_index(vg_root, limit_images, exclusions)
    regions = build_region_texts(vg_root, limit_images, exclusions)

    write_json(out_dir / ATTR_INDEX, attr_index)
    write_json(out_dir / REL_INDEX, rel_index)
    write_json(out_dir / "region_texts.json", regions[:500000])
    write_json(
        manifest_path,
        {
            "schema_version": INDEX_SCHEMA_VERSION,
            "attribute_key": "object|attribute_family",
            "relation_key": "subject|object|relation_family",
            "role": "reference_context_only",
            "overlap_signature": overlap["signature"],
            "sources": sources,
            "limit_images": limit_images,
            "excluded_image_count": len(exclusions),
            "identity_audit_complete": overlap["identity_audit_complete"],
            "near_duplicate_status": overlap["near_duplicate_status"],
            "outputs": {
                name: sha256(out_dir / name)
                for name in (ATTR_INDEX, REL_INDEX, "region_texts.json")
            },
        },
    )
    LOG.info("VG index written to %s", out_dir)
