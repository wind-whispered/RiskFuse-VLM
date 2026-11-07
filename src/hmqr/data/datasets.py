"""Benchmark loaders -> one uniform `Sample` record.

Sample
------
    image_id     : str
    image_path   : str
    dataset      : str                     amber_attr | amber_rel | rbench | mmrel | chair500
    mode         : "discriminative" | "generative"
    question     : str                     (discriminative only)
    gt_answer    : "yes" | "no" | None     (discriminative only) [PROVENANCE]
    claim_type   : "attr" | "rel" | None
    gt           : dict                    (generative only) ground-truth annotation

SCHEMA ADAPTERS
---------------
Each public benchmark ships its own json layout, and those layouts change between
releases. Every loader below therefore goes through a small adapter that tries the
known key spellings and raises a descriptive error when none match, rather than
guessing and silently producing wrong labels. If a release changes, edit only the
`_KEYS_*` tuples at the top of the corresponding loader.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from ..config import Config
from ..utils import get_logger, read_json, read_jsonl

LOG = get_logger("hmqr.datasets")


@dataclass
class Sample:
    image_id: str
    image_path: str
    dataset: str
    mode: str
    question: Optional[str] = None
    gt_answer: Optional[str] = None
    claim_type: Optional[str] = None
    gt: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)


class SchemaError(RuntimeError):
    """Raised when a benchmark file does not match any known layout."""


def _first_key(d: Dict[str, Any], keys: Iterable[str]) -> Optional[str]:
    for k in keys:
        if k in d:
            return k
    return None


def _norm_yesno(value: Any) -> Optional[str]:
    s = str(value).strip().lower()
    if s in {"yes", "y", "true", "1"}:
        return "yes"
    if s in {"no", "n", "false", "0"}:
        return "no"
    return None


# --------------------------------------------------------------------------- #
# claim parsing from a discriminative question
# --------------------------------------------------------------------------- #
_REL_WORDS = [
    "on top of", "in front of", "next to", "on the left of", "on the right of",
    "left of", "right of", "above", "below", "under", "behind", "beside",
    "holding", "riding", "wearing", "carrying", "eating", "sitting on",
    "standing on", "lying on", "attached to", "covering", "inside", "near", "on", "in",
]
_ATTR_PATTERNS = [
    re.compile(r"is (?:there )?(?:a |an |the )?(?P<attr>[\w-]+) (?P<obj>[\w\s-]+?) (?:in|on) the (?:image|picture)", re.I),
    re.compile(r"is the (?P<obj>[\w\s-]+?) (?P<attr>[\w-]+)\s*\?", re.I),
    re.compile(r"does the (?P<obj>[\w\s-]+?) (?:look|appear) (?P<attr>[\w-]+)", re.I),
]


def parse_attr_claim(question: str) -> Optional[Dict[str, str]]:
    for pat in _ATTR_PATTERNS:
        m = pat.search(question)
        if m:
            return {
                "object": m.group("obj").strip().lower(),
                "attribute": m.group("attr").strip().lower(),
            }
    return None


_ARTICLES = r"(?:a|an|the)"


def _entity(text: str) -> str:
    """Normalise an entity phrase without destroying multi-word object names."""
    value = re.sub(r"\s+", " ", str(text).strip().lower().strip(" .?\"'"))
    value = re.sub(r"\bin (?:this|the) (?:image|picture)\b", "", value)
    value = re.sub(r"^(?:a|an|the|any|all)\s+", "", value)
    value = re.sub(r"^(?:one|two|three|four|five|six|seven|eight|nine|ten)\s+", "", value)
    return re.sub(r"\s+", " ", value).strip()


def _relation(text: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", str(text).strip().lower()).strip("_")
    aliases = {
        "left_to": "left_of",
        "right_to": "right_of",
        "above_to": "above",
        "over_to": "above",
        "under_to": "below",
        "on_to": "on",
        "in_to": "in",
        "direct_contact": "touching",
    }
    return aliases.get(value, value)


def parse_amber_attr_claim(question: str, subtype: str = "") -> Optional[Dict[str, str]]:
    """Parse AMBER's three controlled attribute templates.

    AMBER annotations identify each row as state, number, or action. The old
    generic regex only retained the 4,764 state rows and reversed object and
    attribute in questions such as ``Is the sky sunny ...?``.
    """
    q = re.sub(r"\s+", " ", question.strip())
    kind = subtype.lower()
    if "number" in kind or re.match(r"^(?:Is|Are) there\b", q, re.I):
        m = re.match(r"^(?:Is|Are) there (?P<body>.+?) in this image\?$", q, re.I)
        if not m:
            return None
        words = m.group("body").strip().split()
        if len(words) < 2:
            return None
        count = words[0].lower()
        obj = _entity(" ".join(words[1:]))
        return {"object": obj, "attribute": f"count_{count}"} if obj else None

    if "action" in kind or re.match(r"^Does the\b", q, re.I):
        m = re.match(r"^Does the (?P<object>\S+) (?P<attribute>.+?) in this image\?$", q, re.I)
    else:
        m = re.match(r"^(?:Is|Are) the (?P<object>\S+) (?P<attribute>.+?) in this image\?$", q, re.I)
    if not m:
        return None
    obj = _entity(m.group("object"))
    attr = _relation(m.group("attribute"))
    return {"object": obj, "attribute": attr} if obj and attr else None


def parse_amber_rel_claim(question: str) -> Optional[Dict[str, str]]:
    """Parse AMBER's controlled direct-contact relation template."""
    m = re.match(
        rf"^Is there direct contact between {_ARTICLES}?\s*(?P<subject>.+?) "
        rf"and {_ARTICLES}?\s*(?P<object>.+?)\?$",
        re.sub(r"\s+", " ", question.strip()),
        re.I,
    )
    if not m:
        return None
    subject, obj = _entity(m.group("subject")), _entity(m.group("object"))
    return {"subject": subject, "relation": "touching", "object": obj} if subject and obj else None


_MMREL_RELATIONS = {
    "spatial": (
        "right to", "left to", "under to", "over to", "above to", "on to", "in to",
        "right of", "left of", "in front of", "on top of", "next to", "under", "above",
        "below", "behind", "inside", "near", "on", "in",
    ),
    "comparative": (
        "bigger than", "smaller than", "taller than", "shorter than", "longer than",
        "older than", "younger than", "higher than", "lower than",
    ),
}


def parse_mmrel_claim(question: str, category: str) -> Optional[Dict[str, str]]:
    """Parse MMRel using the relation family encoded by its official split name."""
    q = re.sub(r"\?\s*Please answer with one word\.?$", "", question.strip(), flags=re.I)
    q = re.sub(r"\s+", " ", q)
    family = category.lower()
    if "action" in family:
        m = re.match(
            rf"^(?:Does|Do) {_ARTICLES} (?P<body>.+?) {_ARTICLES} (?P<object>.+?) in the image$",
            q,
            re.I,
        )
        if not m:
            return None
        words = m.group("body").split()
        if len(words) < 2:
            return None
        subject, relation, obj = _entity(" ".join(words[:-1])), words[-1], _entity(m.group("object"))
    else:
        m = re.match(
            rf"^Is(?: there)? {_ARTICLES} (?P<body>.+?) {_ARTICLES} (?P<object>.+?) in the image$",
            q,
            re.I,
        )
        if not m:
            return None
        relation_family = "comparative" if "comparative" in family else "spatial"
        body = m.group("body").lower()
        hit = next(
            (rel for rel in sorted(_MMREL_RELATIONS[relation_family], key=len, reverse=True)
             if body.endswith(" " + rel)),
            None,
        )
        if hit is None:
            return None
        subject, relation, obj = _entity(body[: -len(hit)]), hit, _entity(m.group("object"))
    relation = _relation(relation)
    return {"subject": subject, "relation": relation, "object": obj} if subject and relation and obj else None


_RBENCH_RELATIONS = (
    "in front of", "on the left of", "on the right of", "next to", "on top of",
    "far from", "away from", "outside of", "inside of", "close to", "attached to",
    "leaning against", "placed against", "looking directly at", "looking at", "facing",
    "surrounded by", "covered by", "covered with", "filled with", "topped with",
    "made of", "composed of", "pulled by", "pushed by", "held by", "worn by",
    "being used by", "used by", "serviced by", "overlooked by", "occupied by",
    "playing", "wearing", "holding", "riding", "carrying", "eating", "drinking",
    "watching", "hitting", "pulling", "pushing", "touching", "standing on",
    "sitting on", "lying on", "hanging from", "stuck through", "moving through",
    "crawling down", "outside", "inside", "above", "below", "under", "behind",
    "beside", "against", "near", "over", "through", "with", "on",
)


def parse_rbench_claim(question: str) -> Optional[Dict[str, str]]:
    """Deterministically adapt all strict-binary R-Bench Image(All) questions.

    R-Bench stores natural questions, not triples. Explicit binary relations are
    parsed first. Unary/existence rows in the official file are represented as a
    self relation so they remain evaluable instead of being silently discarded.
    """
    q = re.sub(r"\s+", " ", question.strip().rstrip("?"))
    q = re.sub(r"\s+in (?:this|the) (?:image|picture)$", "", q, flags=re.I)
    q = re.sub(r"^(?:Is|Are|Does|Do)\s+", "", q, flags=re.I)
    q = re.sub(r"^there\s+", "", q, flags=re.I)
    q = re.sub(r"^(?:any|all)\s+", "", q, flags=re.I)
    lower = q.lower()

    m = re.match(
        r"^(?:the )?image (?:showing|displaying) (?P<subject>.+?) with (?P<object>.+)$",
        q,
        re.I,
    )
    if m:
        subject, obj = _entity(m.group("subject")), _entity(m.group("object"))
        return {"subject": subject, "relation": "contains", "object": obj}

    m = re.match(r"^(?:the )?(?P<subject>.+?) facing (?:each other|the same direction)$", q, re.I)
    if m:
        subject = _entity(m.group("subject"))
        return {"subject": subject, "relation": "facing", "object": subject}

    for rel in sorted(_RBENCH_RELATIONS, key=len, reverse=True):
        m = re.search(rf"\b{re.escape(rel)}\b", lower)
        if not m:
            continue
        left, right = q[:m.start()], q[m.end():]
        left = re.sub(r"\b(?:is|are|being|was|were)\s*$", "", left, flags=re.I)
        subject, obj = _entity(left), _entity(right)
        if subject and obj:
            return {"subject": subject, "relation": _relation(rel), "object": obj}

    words = q.split()
    verb_i = next(
        (i for i, word in enumerate(words[1:], 1)
         if word.lower().endswith(("ing", "ed")) or word.lower() in {
             "have", "has", "hold", "holds", "wear", "wears", "play", "plays",
             "hit", "hits", "pull", "pulls", "push", "pushes", "watch", "watches",
             "contain", "contains", "display", "displays", "show", "shows",
         }),
        None,
    )
    if verb_i is not None:
        subject = _entity(" ".join(words[:verb_i]))
        relation = _relation(words[verb_i])
        obj = _entity(" ".join(words[verb_i + 1:])) or subject
        if subject:
            return {"subject": subject, "relation": relation, "object": obj}

    entity = _entity(q)
    if not entity:
        return None
    return {"subject": entity, "relation": "present", "object": entity}


def parse_rel_claim(question: str) -> Optional[Dict[str, str]]:
    q = question.strip().rstrip("?").lower()
    q = re.sub(r"^(is|are|does|do)\s+", "", q)
    q = re.sub(r"\s+in the (image|picture)$", "", q)
    for rel in sorted(_REL_WORDS, key=len, reverse=True):
        pat = re.compile(rf"\b{re.escape(rel)}\b")
        m = pat.search(q)
        if not m:
            continue
        subj = q[: m.start()].strip()
        obj = q[m.end() :].strip()
        subj = re.sub(r"^(the|a|an)\s+", "", subj).strip()
        obj = re.sub(r"^(the|a|an)\s+", "", obj).strip()
        if subj and obj:
            return {
                "subject": subj,
                "relation": rel.replace(" ", "_"),
                "object": obj,
            }
    return None


# --------------------------------------------------------------------------- #
# AMBER
# --------------------------------------------------------------------------- #
_AMBER_QUERY_FILES = {
    "attr": ["query_discriminative-attribute.json", "query_attribute.json"],
    "rel": ["query_discriminative-relation.json", "query_relation.json"],
    "generative": ["query_generative.json"],
}
_KEYS_ID = ("id", "index", "question_id", "qid")
_KEYS_IMAGE = ("image", "image_id", "img", "filename", "image_path")
_KEYS_QUERY = ("query", "question", "text", "prompt")
_KEYS_ANSWER = ("answer", "label", "gt", "ground_truth", "truth")


def _amber_dir(root: Path) -> Path:
    """Locate the directory holding the AMBER query json files.

    The upstream repo keeps them in `data/query/`, but older snapshots put them
    directly in `data/`, so the search is recursive rather than assuming a layout.
    """
    for cand in (root / "data" / "query", root / "data", root):
        if cand.is_dir() and any(cand.glob("query*.json")):
            return cand
    hit = next((p for p in sorted(root.glob("**/query*.json"))), None)
    if hit is not None:
        return hit.parent
    found = sorted(p.relative_to(root).as_posix() for p in root.glob("**/*.json"))[:20]
    raise SchemaError(
        f"No AMBER query json found under {root}. Expected e.g. "
        f"{root}/data/query/query_discriminative-attribute.json\n"
        f"  json files actually present: {found or '(none)'}\n"
        f"  If the list is empty the repo was cloned without its data/ payload; "
        f"re-clone https://github.com/junyangwang0410/AMBER."
    )


def _amber_annotations(data_dir: Path) -> Dict[str, Any]:
    """Load AMBER annotations; searches the query dir, its parents, then the tree."""
    candidates: List[Path] = []
    for base in (data_dir, data_dir.parent, data_dir.parent.parent):
        for name in ("annotations.json", "annotation.json"):
            candidates.append(base / name)
    hit = next((p for p in candidates if p.exists()), None)
    if hit is None:
        root = data_dir.parent.parent
        hit = next(
            (p for p in sorted(root.glob("**/annotation*.json"))), None
        )
    if hit is None:
        raise SchemaError(
            f"AMBER annotations.json not found near {data_dir}. Looked at: "
            f"{[str(c) for c in candidates]}"
        )
    ann = read_json(hit)
    if isinstance(ann, list):
        out = {}
        for item in ann:
            key = item.get("id", item.get("index"))
            out[str(key)] = item
        return out
    return {str(k): v for k, v in ann.items()}


def load_amber(cfg: Config, claim_type: str, limit: Optional[int] = None) -> List[Sample]:
    """claim_type in {"attr", "rel", "generative"}."""
    root = cfg.abspath(cfg["datasets"]["amber"]["root"])
    data_dir = _amber_dir(root)
    image_dir = root / "image"
    if not image_dir.is_dir():
        alt = next((p for p in root.glob("**/image") if p.is_dir()), None)
        if alt is None:
            raise SchemaError(f"AMBER image directory not found under {root}")
        image_dir = alt

    query_path = None
    for cand in _AMBER_QUERY_FILES[claim_type]:
        if (data_dir / cand).exists():
            query_path = data_dir / cand
            break
    if query_path is None:
        raise SchemaError(
            f"None of {_AMBER_QUERY_FILES[claim_type]} found in {data_dir}. "
            "Check the AMBER release layout."
        )
    rows = read_json(query_path)
    if isinstance(rows, dict):
        rows = list(rows.values())

    # Loaded once for every claim_type: the generative branch needs it too, and
    # re-reading annotations.json per row costs minutes over the 1004 AMBER
    # generative images.
    annotations = None
    try:
        annotations = _amber_annotations(data_dir)
    except SchemaError as exc:
        if claim_type != "generative":
            LOG.warning("%s -- discriminative labels must come from the query file", exc)

    samples: List[Sample] = []
    selected_image_ids: Set[str] = set()
    for row in rows:
        k_id = _first_key(row, _KEYS_ID)
        k_img = _first_key(row, _KEYS_IMAGE)
        if k_img is None:
            raise SchemaError(f"AMBER row without an image key: {list(row)[:8]}")
        img_name = str(row[k_img])
        if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
            img_name += ".jpg"
        sid = str(row[k_id]) if k_id else img_name
        image_id = Path(img_name).stem
        if limit and image_id not in selected_image_ids and len(selected_image_ids) >= limit:
            continue
        selected_image_ids.add(image_id)

        if claim_type == "generative":
            samples.append(
                Sample(
                    image_id=image_id,
                    image_path=str(image_dir / img_name),
                    dataset="amber_generative",
                    mode="generative",
                    gt=(annotations or {}).get(sid, {}),
                    meta={"sample_id": sid, "annotation_id": sid},
                )
            )
            # The discriminative path caps at the bottom of the loop; this branch
            # continues past it, so subset_limit has to be honoured here too --
            # without this a --smoke run captions all 1004 images per backbone.
            continue

        k_q = _first_key(row, _KEYS_QUERY)
        if k_q is None:
            raise SchemaError(f"AMBER row without a query key: {list(row)[:8]}")
        question = str(row[k_q])

        answer = None
        k_a = _first_key(row, _KEYS_ANSWER)
        if k_a is not None:
            answer = _norm_yesno(row[k_a])
        if answer is None and annotations is not None:
            ann = annotations.get(sid, {})
            if isinstance(ann, dict):
                k_aa = _first_key(ann, _KEYS_ANSWER)
                if k_aa is not None:
                    answer = _norm_yesno(ann[k_aa])
            else:
                answer = _norm_yesno(ann)
        if answer is None:
            raise SchemaError(
                f"No ground-truth answer for AMBER id={sid}. Inspect "
                f"{query_path.name} / annotations.json and extend _KEYS_ANSWER."
            )

        ann = (annotations or {}).get(sid, {})
        subtype = str(ann.get("type", "")) if isinstance(ann, dict) else ""
        parsed = (
            parse_amber_attr_claim(question, subtype)
            if claim_type == "attr"
            else parse_amber_rel_claim(question)
        )
        if not parsed:
            raise SchemaError(
                f"Could not parse AMBER {claim_type} claim id={sid}: {question!r}"
            )
        samples.append(
            Sample(
                image_id=image_id,
                image_path=str(image_dir / img_name),
                dataset=f"amber_{claim_type}",
                mode="discriminative",
                question=question,
                gt_answer=answer,
                claim_type=claim_type,
                gt=parsed or {},
                meta={"sample_id": sid, "question_id": sid},
            )
        )
    LOG.info("AMBER %s: %d samples", claim_type, len(samples))
    return samples


# --------------------------------------------------------------------------- #
# R-Bench
# --------------------------------------------------------------------------- #
def load_rbench(cfg: Config, limit: Optional[int] = None) -> List[Sample]:
    root = cfg.abspath(cfg["datasets"]["rbench"]["root"])
    cand_files = list(root.glob("**/*.json"))
    if not cand_files:
        raise SchemaError(f"No json annotation found under {root} for R-Bench")

    # The R-Bench release ships nocaps_pope_obj_{random,popular,adversarial}.json
    # alongside the relation files. Those are OBJECT-existence (POPE) probes, not
    # relation probes: selecting one would silently evaluate the wrong task, so
    # they are excluded outright rather than merely deprioritised.
    relation_files = [p for p in cand_files if not re.search(r"pope|obj", p.stem, re.I)]
    if not relation_files:
        raise SchemaError(
            f"Only object-level POPE files found under {root} "
            f"({[p.name for p in cand_files]}). R-Bench needs image-level_filterd.json "
            "or instance-level_filterd.json."
        )
    # prefer the documented relation annotation names, then the largest file
    cand_files = sorted(
        relation_files,
        key=lambda p: (
            0 if re.search(r"(image-level|instance-level)", p.stem, re.I) else 1,
            1 if re.search(r"instance-level", p.stem, re.I) else 0,  # image-level first
            -p.stat().st_size,
        ),
    )
    path = cand_files[0]
    LOG.info("R-Bench: using %s (candidates: %s)",
             path.name, [p.name for p in cand_files])
    rows = read_json(path)
    if isinstance(rows, dict):
        rows = rows.get("data", list(rows.values()))

    image_root = root / "images"
    samples: List[Sample] = []
    selected_image_ids: Set[str] = set()
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        k_img = _first_key(row, _KEYS_IMAGE + ("image_url", "url", "coco_url"))
        k_q = _first_key(row, _KEYS_QUERY)
        k_a = _first_key(row, _KEYS_ANSWER)
        if not (k_img and k_q and k_a):
            continue
        answer = _norm_yesno(row[k_a])
        if answer is None:
            continue
        img_val = str(row[k_img])
        # ``question_id`` is not guaranteed to be unique in the released
        # image-level file, while several questions can refer to one image.
        # Keep the image identity for leakage-free splitting and carry a
        # separate row-stable sample identity for claim IDs.
        image_id = Path(img_val.split("?", 1)[0]).stem
        if limit and image_id not in selected_image_ids and len(selected_image_ids) >= limit:
            continue
        selected_image_ids.add(image_id)
        question_id = str(row.get("question_id", row.get("id", i)))
        sample_id = f"{question_id}@{i}"
        if img_val.startswith("http"):
            img_path = img_val
        else:
            # R-Bench records carry an Open Images id that may or may not include
            # the extension; without this the path silently misses every file.
            if not img_val.lower().endswith((".jpg", ".jpeg", ".png")):
                img_val += ".jpg"
            img_path = str(image_root / img_val)
        question = str(row[k_q])
        parsed = parse_rbench_claim(question)
        if not parsed:
            raise SchemaError(
                f"Could not parse R-Bench relation id={row.get('question_id', i)}: "
                f"{question!r}"
            )
        samples.append(
            Sample(
                image_id=image_id,
                image_path=img_path,
                dataset="rbench",
                mode="discriminative",
                question=question,
                gt_answer=answer,
                claim_type="rel",
                gt=parsed,
                meta={"sample_id": sample_id, "question_id": question_id},
            )
        )
    if not samples:
        raise SchemaError(
            f"R-Bench file {path.name} parsed to 0 usable rows; inspect its keys "
            "and extend _KEYS_IMAGE/_KEYS_QUERY/_KEYS_ANSWER."
        )
    LOG.info("R-Bench: %d samples from %s", len(samples), path.name)
    return samples


# --------------------------------------------------------------------------- #
# MMRel
# --------------------------------------------------------------------------- #
_MMREL_IMAGE_INDEX: Dict[str, Path] = {}
_MMREL_INDEXED_ROOTS: set = set()
_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def _build_mmrel_image_index(roots: Sequence[Path]) -> None:
    """Index every image once by basename, so a path mismatch is still resolvable."""
    for root in roots:
        key = str(root)
        if key in _MMREL_INDEXED_ROOTS or not Path(root).is_dir():
            continue
        _MMREL_INDEXED_ROOTS.add(key)
        for p in Path(root).rglob("*"):
            if p.suffix.lower() in _IMAGE_EXTS:
                _MMREL_IMAGE_INDEX.setdefault(p.name, p)
                _MMREL_IMAGE_INDEX.setdefault(p.stem, p)


def _resolve_mmrel_image(
    img_val: str, bases: Sequence[Path], search_roots: Sequence[Path]
) -> str:
    """Resolve an MMRel image reference.

    The annotations live under json/Evaluation/... while the pictures sit under
    Dall-E_generated/... , so the reference is relative to the dataset root
    rather than to the json file. Extensions also vary (DALL-E images are png),
    hence the explicit candidate sweep before falling back to a basename index.
    """
    candidates: List[Path] = []
    for base in bases:
        candidates.append(Path(base) / img_val)
        if not img_val.lower().endswith(_IMAGE_EXTS):
            candidates.extend(Path(base) / (img_val + ext) for ext in _IMAGE_EXTS)
    for cand in candidates:
        if cand.exists():
            return str(cand)

    # Public MMRel namespaces are disjoint. In particular, SPEC/..../2.jpg
    # must never fall back to the unrelated Visual Genome image 2.jpg.
    normalised = img_val.replace('\\', '/').lstrip('./')
    if normalised.startswith(('Dall-E_generated/', 'SPEC/')):
        return str(Path(search_roots[0]) / normalised)
    if normalised.lower().startswith('vg/'):
        relative = normalised.split('/', 1)[1]
        for base in search_roots:
            candidate = Path(base) / relative
            if candidate.is_file():
                return str(candidate)
        return str(Path(search_roots[0]) / normalised)

    _build_mmrel_image_index(search_roots)
    name = Path(img_val).name
    hit = _MMREL_IMAGE_INDEX.get(name) or _MMREL_IMAGE_INDEX.get(Path(name).stem)
    if hit is not None:
        return str(hit)
    # Nothing matched: return the most plausible path so the loader still yields a
    # record; the image loader logs and skips it.
    return str(Path(bases[0]) / img_val)


def load_mmrel(cfg: Config, limit: Optional[int] = None) -> List[Sample]:
    root = cfg.abspath(cfg["datasets"]["mmrel"]["root"])
    snapshot = root / "hf_snapshot_path.txt"
    search_roots = [root]
    search_roots.extend(cfg.abspath(p) for p in cfg['datasets']['mmrel'].get('image_roots', []))
    if snapshot.exists():
        search_roots.append(Path(snapshot.read_text(encoding="utf-8").strip()))

    def _rank(path: Path) -> tuple:
        """Prefer the discriminative yes/no evaluation split.

        The release also ships generative open-ended QA and a fine-tuning split;
        only the binary yes/no questions yield the ground-truth labels this
        pipeline needs, so they are ranked first and fine-tuning data last.
        """
        s = path.as_posix().lower()
        return (
            0 if "discriminative" in s or "yn" in s else 1,
            1 if "fine-tuning" in s or "train" in s else 0,
            s,
        )

    rows: List[Dict[str, Any]] = []
    used: List[Path] = []
    for sr in search_roots:
        # The evaluation splits ship as .jsonl (one object per line); only the
        # fine-tuning dump is a .json array. Globbing *.json alone finds nothing
        # but the training data and the loader then refuses to run at all.
        candidates = sorted([*sr.glob("**/*.json"), *sr.glob("**/*.jsonl")], key=_rank)
        tier: Optional[tuple] = None
        for path in candidates:
            rank = _rank(path)[:2]  # tier only; the third element is a tiebreaker
            if tier is not None and rank != tier:
                break  # a whole tier is taken or none of it -- never mixed
            try:
                data = read_jsonl(path) if path.suffix.lower() == ".jsonl" else read_json(path)
            except Exception:  # noqa: BLE001
                continue
            if isinstance(data, dict):
                data = data.get("data", list(data.values()))
            if not (isinstance(data, list) and data and isinstance(data[0], dict)):
                continue
            if not (_first_key(data[0], _KEYS_QUERY) and _first_key(data[0], _KEYS_ANSWER)):
                continue
            # MMRel splits its 14,927 yes/no questions across one file per
            # relation category (the rest of the 22,441 evaluation QAs are
            # open-ended), so every file of the winning tier is concatenated;
            # stopping at the first would evaluate on a fraction of the benchmark
            # without saying so.
            tier = rank
            category = path.stem.lower()
            for item in data:
                copied = dict(item)
                copied["_mmrel_category"] = category
                rows.append(copied)
            used.append(path)
        if rows:
            break
    if not rows:
        raise SchemaError(f"No usable MMRel QA json/jsonl found under {search_roots}")
    LOG.info(
        "MMRel: %d rows from %d file(s) under %s",
        len(rows), len(used), used[0].parent,
    )

    # MMRel keeps annotations under json/Evaluation/... and images under
    # Dall-E_generated/... , so the image path is NOT relative to the json's own
    # directory. Resolve against several bases and fall back to a basename index.
    image_bases = [*{p.parent for p in used}, *search_roots]
    for sr in search_roots:
        image_bases.extend(p for p in sr.glob("*") if p.is_dir())
    samples: List[Sample] = []
    selected_image_ids: Set[str] = set()
    for i, row in enumerate(rows):
        k_img = _first_key(row, _KEYS_IMAGE + ("image_url", "url"))
        k_q = _first_key(row, _KEYS_QUERY)
        k_a = _first_key(row, _KEYS_ANSWER)
        if not (k_img and k_q and k_a):
            continue
        answer = _norm_yesno(row[k_a])
        if answer is None:
            continue
        img_val = str(row[k_img])
        if img_val.startswith("http"):
            img_path = img_val
        else:
            img_path = _resolve_mmrel_image(img_val, image_bases, search_roots)
        image_id = 'mmrel:' + img_val.replace("\\", "/").split("?", 1)[0].lstrip('./')
        if limit and image_id not in selected_image_ids and len(selected_image_ids) >= limit:
            continue
        selected_image_ids.add(image_id)
        question = str(row[k_q])
        category = str(row.get("_mmrel_category", row.get("category", row.get("type", ""))))
        parsed = parse_mmrel_claim(question, category)
        if not parsed:
            raise SchemaError(
                f"Could not parse MMRel relation ({category}) id={row.get('question_id', i)}: "
                f"{question!r}"
            )
        samples.append(
            Sample(
                image_id=image_id,
                image_path=img_path,
                dataset="mmrel",
                mode="discriminative",
                question=question,
                gt_answer=answer,
                claim_type="rel",
                gt=parsed,
                meta={
                    "relation_category": category,
                    "sample_id": f"{row.get('id', row.get('question_id', i))}@{i}",
                },
            )
        )
    LOG.info("MMRel: %d samples", len(samples))
    return samples


# --------------------------------------------------------------------------- #
# CHAIR-500 (COCO-2014 val)
# --------------------------------------------------------------------------- #
def load_chair500(cfg: Config, limit: Optional[int] = None) -> List[Sample]:
    import random

    root = cfg.abspath(cfg["datasets"]["coco"]["root"])
    n_images = int(cfg["datasets"]["coco"].get("chair_num_images", 500))
    ann_path = root / "annotations" / "instances_val2014.json"
    cap_path = root / "annotations" / "captions_val2014.json"
    if not ann_path.exists():
        raise SchemaError(f"{ann_path} missing; run scripts/00_download_data.py")

    inst = read_json(ann_path)
    id2name = {c["id"]: c["name"] for c in inst["categories"]}
    img_objects: Dict[int, set] = {}
    for a in inst["annotations"]:
        img_objects.setdefault(a["image_id"], set()).add(id2name[a["category_id"]])
    images = {im["id"]: im["file_name"] for im in inst["images"]}

    refs: Dict[int, List[str]] = {}
    if cap_path.exists():
        caps = read_json(cap_path)
        for a in caps["annotations"]:
            refs.setdefault(a["image_id"], []).append(a["caption"])

    ids = sorted(images)
    rng = random.Random(cfg.get("seed", 42))
    rng.shuffle(ids)
    ids = ids[: (limit or n_images)]

    samples = [
        Sample(
            image_id=str(i),
            image_path=str(root / "val2014" / images[i]),
            dataset="chair500",
            mode="generative",
            gt={"objects": sorted(img_objects.get(i, set())), "references": refs.get(i, [])},
        )
        for i in ids
    ]
    LOG.info("CHAIR-500: %d images", len(samples))
    return samples


# --------------------------------------------------------------------------- #
def load_dataset(cfg: Config, name: str, limit: Optional[int] = None) -> List[Sample]:
    limit = limit if limit is not None else cfg["datasets"].get("subset_limit")
    if name == "amber_attr":
        return load_amber(cfg, "attr", limit)
    if name == "amber_rel":
        return load_amber(cfg, "rel", limit)
    if name == "amber_generative":
        return load_amber(cfg, "generative", limit)
    if name == "rbench":
        return load_rbench(cfg, limit)
    if name == "mmrel":
        return load_mmrel(cfg, limit)
    if name == "chair500":
        return load_chair500(cfg, limit)
    raise ValueError(f"unknown dataset: {name}")
