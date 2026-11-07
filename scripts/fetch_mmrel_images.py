#!/usr/bin/env python
"""Fetch only third-party images referenced by MMRel discriminative QA.

MMRel redistributes DALL-E images but its Visual Genome and SPEC annotations
refer to the original datasets. This utility downloads the two official ZIP
archives, extracts only files named by MMRel, verifies them, and removes the
temporary archives unless ``--keep-archives`` is set. Partial HTTP archives are
resumed with Range requests.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, Set

import requests
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hmqr.config import load_config  # noqa: E402

VG_FALLBACK_URL = "https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip"
VG_REPO_PATH = (
    "datasets/BoyangZ/VisualGenome_VG_100K_1_and_2/resolve/main/images2.zip"
)
SPEC_REPO_PATH = "datasets/wjpoom/SPEC/resolve/main/data.zip"


def _required(mmrel_root: Path) -> Dict[str, Set[str]]:
    qa_dir = mmrel_root / "json" / "Evaluation" / "Discriminative_YN_QA"
    required: Dict[str, Set[str]] = {"spec": set(), "vg": set()}
    for path in sorted(qa_dir.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                ref = PurePosixPath(str(json.loads(line).get("image", "")))
                if not ref.parts:
                    continue
                if ref.parts[0].lower() == "spec":
                    required["spec"].add(PurePosixPath(*ref.parts[1:]).as_posix())
                elif ref.parts[0].lower() == "vg":
                    required["vg"].add(PurePosixPath(*ref.parts[1:]).as_posix())
    return required


def _download(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    existing = destination.stat().st_size if destination.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    with requests.get(url, headers=headers, stream=True, timeout=(30, 600)) as response:
        response.raise_for_status()
        resume = existing > 0 and response.status_code == 206
        mode = "ab" if resume else "wb"
        if existing and not resume:
            existing = 0
        total = existing + int(response.headers.get("content-length", 0) or 0)
        written = existing
        last_report = time.monotonic()
        with destination.open(mode) as handle:
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                if not chunk:
                    continue
                handle.write(chunk)
                written += len(chunk)
                if time.monotonic() - last_report >= 10:
                    pct = 100.0 * written / total if total else 0.0
                    print(
                        f"[download] {destination.name}: {written / 1024**3:.2f} GiB"
                        f" / {total / 1024**3:.2f} GiB ({pct:.1f}%)",
                        flush=True,
                    )
                    last_report = time.monotonic()
    print(f"[download] complete: {destination} ({destination.stat().st_size / 1024**3:.2f} GiB)")
    return destination


def _member_tail(member: str, source: str) -> str | None:
    parts = PurePosixPath(member).parts
    if source == "spec":
        for marker in ("relative_size", "relative_spatial"):
            if marker in parts:
                return PurePosixPath(*parts[parts.index(marker):]).as_posix()
    elif "VG_100K_2" in parts:
        return PurePosixPath(*parts[parts.index("VG_100K_2"):]).as_posix()
    return None


def _extract_selected(
    archive: Path, source: str, wanted: Set[str], destination_root: Path
) -> int:
    present = {rel for rel in wanted if (destination_root / rel).is_file()}
    remaining = wanted - present
    if not remaining:
        print(f"[extract] {source}: all {len(wanted)} required images already present")
        return len(wanted)
    extracted = 0
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            if member.is_dir():
                continue
            tail = _member_tail(member.filename, source)
            if tail not in remaining:
                continue
            target = destination_root / tail
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(member) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
            extracted += 1
    found = sum((destination_root / rel).is_file() for rel in wanted)
    print(f"[extract] {source}: extracted {extracted}; present {found}/{len(wanted)}")
    if found != len(wanted):
        missing = sorted(rel for rel in wanted if not (destination_root / rel).is_file())
        raise RuntimeError(f"{source} archive lacks {len(missing)} referenced files: {missing[:5]}")
    return found


def _verify_images(paths: Iterable[Path]) -> None:
    bad = []
    count = 0
    for path in paths:
        count += 1
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception as exc:  # noqa: BLE001
            bad.append(f"{path}: {exc}")
    if bad:
        raise RuntimeError(f"{len(bad)} corrupt images; first entries: {bad[:5]}")
    print(f"[verify] {count} source images are readable")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--only", choices=("spec", "vg"), default=None)
    parser.add_argument("--keep-archives", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    mmrel_root = cfg.abspath(cfg["datasets"]["mmrel"]["root"])
    required = _required(mmrel_root)
    print(f"[required] SPEC={len(required['spec'])}, VG={len(required['vg'])}")
    downloads = mmrel_root / ".downloads"
    sources = (args.only,) if args.only else ("spec", "vg")
    extracted_paths = []

    for source in sources:
        if source == "spec":
            endpoint = str(cfg.get("hf_endpoint", "https://huggingface.co")).rstrip("/")
            archive = downloads / "spec_data.zip"
            url = f"{endpoint}/{SPEC_REPO_PATH}"
            target = mmrel_root / "SPEC"
        else:
            endpoint = str(cfg.get("hf_endpoint", "https://huggingface.co")).rstrip("/")
            archive = downloads / "vg_images2_hf.zip"
            url = f"{endpoint}/{VG_REPO_PATH}"
            target = mmrel_root / "vg"
        try:
            _download(url, archive)
        except requests.RequestException:
            if source != "vg":
                raise
            print(f"[download] mirror failed; falling back to {VG_FALLBACK_URL}")
            archive.unlink(missing_ok=True)
            archive = downloads / "vg_images2_official.zip"
            _download(VG_FALLBACK_URL, archive)
        _extract_selected(archive, source, required[source], target)
        extracted_paths.extend(target / rel for rel in required[source])
        if not args.keep_archives:
            archive.unlink(missing_ok=True)

    _verify_images(extracted_paths)
    print("MMRel third-party image payload is complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
