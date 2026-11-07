"""T7c: hybrid BM25 + CLIP retrieval of claim-specific region-level evidence."""

from __future__ import annotations

import difflib
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..config import Config
from ..utils import get_logger, read_json
from .rag_index import (
    ATTR_INDEX,
    REL_INDEX,
    INDEX_SCHEMA_VERSION,
    _norm,
    attribute_family,
    relation_family,
)

LOG = get_logger("hmqr.rag")


class EvidenceRetriever:
    def __init__(self, cfg: Config, clip=None):
        index_dir = cfg.path("vg_index")
        if cfg.get("rag", {}).get("require_complete_identity_audit", False):
            manifest_path = index_dir / "index_manifest.json"
            manifest = read_json(manifest_path) if manifest_path.exists() else {}
            if manifest.get(
                "schema_version"
            ) != INDEX_SCHEMA_VERSION or not manifest.get("identity_audit_complete"):
                raise RuntimeError(
                    "VG index lacks a current complete identity-exclusion manifest"
                )
            from ..review.overlap import sha256

            for name in (ATTR_INDEX, REL_INDEX, "region_texts.json"):
                if not (index_dir / name).exists() or manifest.get("outputs", {}).get(
                    name
                ) != sha256(index_dir / name):
                    raise RuntimeError(f"VG index checksum mismatch: {name}")
        self.attr_index: Dict[str, List[Dict]] = (
            read_json(index_dir / ATTR_INDEX)
            if (index_dir / ATTR_INDEX).exists()
            else {}
        )
        self.rel_index: Dict[str, List[Dict]] = (
            read_json(index_dir / REL_INDEX) if (index_dir / REL_INDEX).exists() else {}
        )
        regions_path = index_dir / "region_texts.json"
        self.regions: List[Dict] = (
            read_json(regions_path) if regions_path.exists() else []
        )
        self.topk = int(cfg["rag"]["topk"])
        self.w_bm25 = float(cfg["rag"]["bm25_weight"])
        self.w_clip = float(cfg["rag"]["clip_weight"])
        self.clip = clip
        self._bm25 = None
        self._corpus: List[Dict] = []
        self.n_queries = 0

    # ------------------------------------------------------------------ #
    def _ensure_bm25(self) -> None:
        if self._bm25 is not None or not self.regions:
            return
        from rank_bm25 import BM25Okapi

        self._corpus = self.regions
        tokenised = [r["text"].split() for r in self._corpus]
        LOG.info("building BM25 over %d region descriptions", len(tokenised))
        self._bm25 = BM25Okapi(tokenised)

    # ------------------------------------------------------------------ #
    def _exact_key_hits(self, claim: Dict[str, Any]) -> List[Dict]:
        if claim.get("claim_type") == "attr":
            key = (
                f"{_norm(claim.get('object', ''))}|"
                f"{attribute_family(claim.get('attribute', ''))}"
            )
            return list(self.attr_index.get(key, []))
        key = (
            f"{_norm(claim.get('subject', ''))}|"
            f"{_norm(claim.get('object', ''))}|"
            f"{relation_family(claim.get('relation', ''))}"
        )
        return list(self.rel_index.get(key, []))

    @staticmethod
    def _lexical_similarity(left: Any, right: Any) -> float:
        a, b = _norm(left), _norm(right)
        if not a or not b:
            return 0.0
        a_tokens, b_tokens = set(a.split()), set(b.split())
        jaccard = len(a_tokens & b_tokens) / max(len(a_tokens | b_tokens), 1)
        return float(max(jaccard, difflib.SequenceMatcher(None, a, b).ratio()))

    def plausibility_prior(self, claim: Dict[str, Any]) -> Dict[str, Any]:
        """Return a label-free Visual Genome plausibility prior.

        The score is context only.  It is never interpreted as evidence about
        the current evaluation image and is used solely as a nuisance prior in
        the residual-conflict model.
        """
        hits = self._exact_key_hits(claim)
        if claim.get("claim_type") == "attr":
            target, field = claim.get("attribute", ""), "attribute"
        else:
            target, field = claim.get("relation", ""), "relation"
        support = max(
            (
                self._lexical_similarity(target, hit.get(field, hit.get("text", "")))
                for hit in hits
            ),
            default=0.0,
        )
        return {
            "kb_prior": float(support),
            "kb_prior_hits": int(len(hits)),
            "kb_prior_available": bool(hits),
            "kb_prior_status": "observed_match" if hits else "no_corpus_evidence",
        }

    def _bm25_hits(self, query: str, n: int) -> List[Dict]:
        self._ensure_bm25()
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(query.split())
        if scores.size == 0:
            return []
        top = np.argsort(scores)[::-1][: n * 4]
        out = []
        for i in top:
            item = dict(self._corpus[int(i)])
            item["bm25"] = float(scores[int(i)])
            out.append(item)
        return out

    def _rerank_with_clip(self, hits: List[Dict], query: str) -> List[Dict]:
        if self.clip is None or not hits:
            return hits
        texts = [h["text"] for h in hits]
        try:
            emb = self.clip.encode_texts(texts)
            q = self.clip.encode_text(query)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("CLIP rerank failed: %s", exc)
            return hits
        bm = np.asarray([h.get("bm25", 0.0) for h in hits], dtype=np.float64)
        bm = (bm - bm.min()) / (bm.max() - bm.min() + 1e-8)
        cs = emb @ np.asarray(q, dtype=np.float64)
        cs = (cs - cs.min()) / (cs.max() - cs.min() + 1e-8)
        combined = self.w_bm25 * bm + self.w_clip * cs
        order = np.argsort(combined)[::-1]
        out = []
        for rank, i in enumerate(order):
            item = dict(hits[int(i)])
            item["score"] = float(combined[int(i)])
            out.append(item)
        return out

    # ------------------------------------------------------------------ #
    def retrieve(self, claim: Dict[str, Any]) -> List[Dict[str, Any]]:
        self.n_queries += 1
        hits = self._exact_key_hits(claim)
        if claim.get("claim_type") == "attr":
            query = _norm(
                f"{claim.get('object', '')} {attribute_family(claim.get('attribute', ''))} attribute"
            )
        else:
            query = _norm(
                f"{claim.get('subject', '')} {claim.get('object', '')} "
                f"{claim.get('rel_kind', 'contact')} relation"
            )
        if len(hits) < self.topk:
            hits = hits + self._bm25_hits(query, self.topk)
        hits = self._rerank_with_clip(hits, query)
        return hits[: self.topk]

    @staticmethod
    def format_evidence(hits: List[Dict[str, Any]]) -> str:
        if not hits:
            return "(no external evidence retrieved)"
        lines = []
        for i, h in enumerate(hits, 1):
            box = h.get("box") or h.get("box_subject")
            loc = f" [region {', '.join(str(int(v)) for v in box)}]" if box else ""
            lines.append(f"{i}. {h['text']}{loc}")
        return "\n".join(lines)
