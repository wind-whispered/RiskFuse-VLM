"""IO helpers, logging, jsonl utilities, idempotency flags."""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List

_LOGGER_READY = False


def get_logger(name: str = "hmqr", log_dir: Path | None = None) -> logging.Logger:
    global _LOGGER_READY
    logger = logging.getLogger(name)
    if not _LOGGER_READY:
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", "%H:%M:%S")
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)
        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_dir / "run.log", encoding="utf-8")
            fh.setFormatter(fmt)
            logger.addHandler(fh)
        _LOGGER_READY = True
    return logger


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    p = Path(path)
    if not p.exists():
        return rows
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def iter_jsonl(path: str | Path) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
    tmp.replace(p)


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.flush()
    tmp.replace(p)


def flag_path(processed_dir: Path, name: str) -> Path:
    return Path(processed_dir) / f"{name}.flag"


def is_done(processed_dir: Path, name: str, overwrite: bool = False) -> bool:
    if overwrite:
        return False
    return flag_path(processed_dir, name).exists()


def mark_done(processed_dir: Path, name: str) -> None:
    p = flag_path(processed_dir, name)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("done\n", encoding="utf-8")


def parse_json_block(text: str) -> Any:
    """Extract the first JSON array/object from free-form model output."""
    text = text.strip()
    if text.startswith("```"):
        # strip markdown fences
        parts = text.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("[") or part.startswith("{"):
                text = part
                break
    for opener, closer in (("[", "]"), ("{", "}")):
        start = text.find(opener)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(text)):
            if text[i] == opener:
                depth += 1
            elif text[i] == closer:
                depth -= 1
                if depth == 0:
                    fragment = text[start : i + 1]
                    # Some instruction-tuned models Markdown-escape underscores
                    # inside JSON keys (for example ``revised\_sentence``).  ``\_``
                    # is not a legal JSON escape.  Retry only this narrow,
                    # deterministic repair and retain the raw model output at the
                    # caller for auditability.
                    for candidate in (fragment, fragment.replace(r"\_", "_")):
                        try:
                            return json.loads(candidate)
                        except json.JSONDecodeError:
                            continue
                    break
    return None


def claim_to_sentence(claim: Dict[str, Any]) -> str:
    """Natural-language rendering of a structured claim (used by CLIP/BM25)."""
    if claim.get("claim_type") == "attr":
        return f"a {claim.get('attribute', '')} {claim.get('object', '')}".strip()
    return (
        f"a {claim.get('subject', '')} {str(claim.get('relation', '')).replace('_', ' ')} "
        f"a {claim.get('object', '')}"
    ).strip()


def split_sentences(text: str) -> List[str]:
    import re

    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]
