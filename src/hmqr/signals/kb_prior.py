"""Assign label-free Visual Genome priors to cached claims."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from ..config import Config
from ..revise.rag_index import ATTR_INDEX, REL_INDEX
from ..revise.rag_retrieve import EvidenceRetriever
from ..utils import get_logger, read_jsonl, write_jsonl


LOG = get_logger("hmqr.kb_prior")
KB_PRIOR_SCHEMA_VERSION = 2


def assign_kb_priors(cfg: Config) -> None:
    if "vg_index" not in cfg.raw.get("paths", {}):
        LOG.warning("vg_index path absent; assigning a zero unavailable KB prior")
        for path in sorted(cfg.path("processed").glob("claims_*.jsonl")):
            rows = read_jsonl(path)
            for row in rows:
                if row.get("claim_type") in ("attr", "rel"):
                    row.update(
                        kb_prior=0.0,
                        kb_prior_hits=0,
                        kb_prior_available=False,
                        kb_prior_schema_version=KB_PRIOR_SCHEMA_VERSION,
                    )
            write_jsonl(path, rows)
        return
    index_dir = cfg.path("vg_index")
    required = (index_dir / ATTR_INDEX, index_dir / REL_INDEX)
    if not all(path.exists() for path in required) or cfg.get("rag", {}).get(
        "require_complete_identity_audit", False
    ):
        from ..revise.rag_index import run as build_index

        LOG.info("Visual Genome indices missing; building them before residual fitting")
        build_index(cfg, overwrite=False)
    retriever = EvidenceRetriever(cfg, clip=None)
    from ..review.overlap import sha256

    manifest_path = index_dir / "index_manifest.json"
    signature = sha256(manifest_path) if manifest_path.exists() else None
    cache: Dict[tuple[Any, ...], Dict[str, Any]] = {}
    for path in sorted(cfg.path("processed").glob("claims_*.jsonl")):
        rows = read_jsonl(path)
        changed = False
        for row in rows:
            if row.get("claim_type") not in ("attr", "rel"):
                continue
            key = (
                row.get("claim_type"),
                row.get("subject"),
                row.get("object"),
                row.get("attribute"),
                row.get("relation"),
                row.get("rel_kind"),
            )
            prior = cache.get(key)
            if prior is None:
                prior = retriever.plausibility_prior(row)
                cache[key] = prior
            row.update(prior)
            row["kb_prior_schema_version"] = KB_PRIOR_SCHEMA_VERSION
            row["kb_prior_index_signature"] = signature
            changed = True
        if changed:
            write_jsonl(path, rows)
    LOG.info(
        "Visual Genome priors assigned to claim caches (%d unique claims)", len(cache)
    )
