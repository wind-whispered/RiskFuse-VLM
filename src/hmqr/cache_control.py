"""Linux file-cache control for large HuggingFace checkpoints.

The AutoDL container has a 72 GiB cgroup memory limit and no swap.  Linux
charges clean model-weight pages to that limit: after several backbones have
been read, ``memory.stat`` can show roughly 70 GiB as ``active_file`` even when
the Python process is gone.  Loading the next Qwen shard then gets SIGKILL.

``POSIX_FADV_DONTNEED`` asks the kernel to discard only clean cached pages for
the checkpoint files.  It never deletes or changes a file and is a no-op on
platforms that do not provide ``posix_fadvise`` (notably Windows).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

from .utils import get_logger

LOG = get_logger("hmqr.cache")

_WEIGHT_SUFFIXES = {
    ".safetensors",
    ".bin",
    ".pt",
    ".pth",
    ".ckpt",
    ".h5",
    ".msgpack",
}


def _cgroup_memory_current() -> Optional[int]:
    for path in (
        Path("/sys/fs/cgroup/memory.current"),
        Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
    ):
        try:
            return int(path.read_text(encoding="ascii").strip())
        except (OSError, ValueError):
            continue
    return None


def reclaim_hf_file_cache(hf_home: Optional[str] = None) -> Dict[str, Any]:
    """Drop clean cached pages for backbone and frozen-tool checkpoints.

    Returns counters for logging/testing.  Snapshot paths are usually symlinks
    into ``hub/*/blobs``; inode de-duplication prevents advising the same blob
    more than once.
    """
    advise = getattr(os, "posix_fadvise", None)
    dontneed = getattr(os, "POSIX_FADV_DONTNEED", None)
    root_values = [hf_home] if hf_home else [
        os.environ.get("HF_HOME"),
        os.environ.get("HMQR_TOOL_CACHE"),
    ]
    before = _cgroup_memory_current()
    report: Dict[str, Any] = {
        "supported": bool(advise and dontneed is not None),
        "files": 0,
        "bytes": 0,
        "errors": 0,
        "memory_before": before,
        "memory_after": before,
    }
    if not report["supported"]:
        return report
    roots = [Path(value) for value in root_values if value]
    roots = list(dict.fromkeys(root.resolve() for root in roots if root.is_dir()))
    if not roots:
        return report

    seen: Set[Tuple[int, int]] = set()
    for root in roots:
        for path in root.rglob("*"):
            if path.suffix.lower() not in _WEIGHT_SUFFIXES:
                continue
            try:
                stat = path.stat()
                key = (stat.st_dev, stat.st_ino)
                if key in seen:
                    continue
                seen.add(key)
                fd = os.open(path, os.O_RDONLY)
                try:
                    advise(fd, 0, 0, dontneed)
                finally:
                    os.close(fd)
                report["files"] += 1
                report["bytes"] += stat.st_size
            except OSError:
                report["errors"] += 1

    report["memory_after"] = _cgroup_memory_current()
    if report["files"]:
        before_gib = (report["memory_before"] or 0) / (1024 ** 3)
        after_gib = (report["memory_after"] or 0) / (1024 ** 3)
        LOG.info(
            "released HF file cache: %d files / %.1f GiB advised; "
            "cgroup %.1f -> %.1f GiB (%d errors)",
            report["files"], report["bytes"] / (1024 ** 3),
            before_gib, after_gib, report["errors"],
        )
    return report
