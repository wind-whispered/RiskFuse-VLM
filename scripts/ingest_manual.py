#!/usr/bin/env python
"""Install manually downloaded dataset archives into the layout the loaders expect.

Why this exists
---------------
AMBER images and the R-Bench annotation bundle are distributed only through
Google Drive. Hosts that cannot reach Google (most servers in mainland China,
including AutoDL) fail there even when the HuggingFace mirror works, so those two
archives have to be fetched elsewhere and copied onto the machine. This script
takes whatever you copied over and puts it where the loaders look, then verifies
the result.

Usage
-----
Show what is missing and where each archive has to come from:

    python scripts/ingest_manual.py --status

Install an archive (a .zip, or an already-extracted directory):

    python scripts/ingest_manual.py --amber-images downloads/amber_images.zip
    python scripts/ingest_manual.py --rbench       downloads/R-Bench.zip

Both flags can be combined. Existing files are left alone unless --overwrite.

Download sources (fetch these on a machine with Google access, then upload)
--------------------------------------------------------------------------
AMBER images  : https://drive.google.com/file/d/1MaCHgtupcZUjf007anNl4_MV0o4DjXvl
                (file id 1MaCHgtupcZUjf007anNl4_MV0o4DjXvl, contains
                 AMBER_1.jpg .. AMBER_1004.jpg)
R-Bench data  : https://drive.google.com/file/d/1sqO0MWBg_HXp5cIKb-nstjNEEk5crUWH
                (contains image-level_filterd.json, instance-level_filterd.json,
                 nocaps_pope_obj_*.json, web_data/, instance_mask/)

Note on R-Bench images: the annotations reference the Open Images v4 validation
set, which is a separate multi-GB download from
https://storage.googleapis.com/openimages/web/download_v7.html . Place the jpgs
under data/datasets/rbench/images/. Without them R-Bench is skipped and its rows are
simply absent from RQ1/RQ3 -- AMBER still carries Table 1 and Figures 1/2/4/5.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

AMBER_DRIVE = "1MaCHgtupcZUjf007anNl4_MV0o4DjXvl"
RBENCH_DRIVE = "1sqO0MWBg_HXp5cIKb-nstjNEEk5crUWH"          # annotations
RBENCH_IMAGE_DRIVE = "10JXRdzTRMylWQA160qdoYITGO10g6N8k"    # images (curated subset)


def _diagnose_bad_archive(src: Path) -> str:
    """Explain *why* a file that should be a zip is not one."""
    size = src.stat().st_size
    head = src.open("rb").read(512)

    if size == 0:
        return "the file is empty (0 bytes) -- the download or upload produced nothing."
    if head[:1] in (b"<", b"\n", b"\r") or b"<!DOCTYPE html" in head or b"<html" in head:
        text = head.decode("utf-8", "replace")
        hint = ""
        if b"Google Drive" in head or b"virus" in head.lower() or b"drive.google" in head:
            hint = (
                "\n  This is Google Drive's 'cannot scan for viruses' confirmation page,"
                "\n  which is what a plain wget/curl of a large Drive file returns."
            )
        return (
            f"the file is HTML, not a zip ({size} bytes).{hint}\n"
            f"  first bytes: {text[:120]!r}\n"
            "  Fix: download it through a browser (click 'Download anyway'), or use\n"
            "       gdown, on a machine that can reach Google, then re-upload."
        )
    if head[:2] == b"PK":
        return (
            f"the file starts with a valid zip header but is not readable as an archive "
            f"({size / 1024**2:.1f} MB) -- it is almost certainly truncated by an "
            "interrupted download or upload.\n"
            "  Fix: re-transfer and compare `md5sum` on both machines."
        )
    return (
        f"unrecognised format ({size / 1024**2:.1f} MB, first bytes {head[:8]!r}). "
        "Check that you copied the right file."
    )


def _extract(src: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        print(f"  copying directory {src} -> {dest}")
        shutil.copytree(src, dest, dirs_exist_ok=True)
        return
    if not zipfile.is_zipfile(src):
        raise SystemExit(
            f"\n{src} is not a usable zip archive:\n  {_diagnose_bad_archive(src)}\n"
        )
    print(f"  extracting {src.name} -> {dest}")
    try:
        with zipfile.ZipFile(src) as z:
            z.extractall(dest)
    except zipfile.BadZipFile as exc:
        raise SystemExit(
            f"\n{src} failed to extract ({exc}).\n  {_diagnose_bad_archive(src)}\n"
        ) from exc


def _flatten_jpgs(target: Path) -> int:
    """Move jpgs out of a nested top-level folder (image.zip often has image/)."""
    direct = list(target.glob("*.jpg"))
    if direct:
        return len(direct)
    nested = sorted(p for p in target.glob("*/") if p.is_dir())
    for sub in nested:
        jpgs = list(sub.glob("*.jpg"))
        if jpgs:
            print(f"  flattening {len(jpgs)} images out of {sub.name}/")
            for p in jpgs:
                shutil.move(str(p), str(target / p.name))
            break
    return len(list(target.glob("*.jpg")))


def install_amber_images(cfg, archive: Path, overwrite: bool) -> None:
    root = cfg.abspath(cfg["datasets"]["amber"]["root"])
    image_dir = root / "image"
    existing = list(image_dir.glob("*.jpg")) if image_dir.is_dir() else []
    if existing and not overwrite:
        print(f"[amber] {len(existing)} images already present; use --overwrite to replace")
        return
    print(f"[amber] installing images into {image_dir}")
    _extract(archive, image_dir)
    n = _flatten_jpgs(image_dir)
    print(f"[amber] {n} jpg files now in {image_dir}")
    if n < 1004:
        print(f"[amber] WARNING: expected 1004 images (AMBER_1.jpg .. AMBER_1004.jpg), found {n}")
    else:
        print("[amber] OK")


def install_rbench(cfg, archive: Path, overwrite: bool) -> None:
    root = cfg.abspath(cfg["datasets"]["rbench"]["root"])
    existing = list(root.glob("**/*.json")) if root.is_dir() else []
    if existing and not overwrite:
        print(f"[rbench] {len(existing)} json files already present; use --overwrite to replace")
        return
    print(f"[rbench] installing annotations into {root}")
    _extract(archive, root)
    jsons = list(root.glob("**/*.json"))
    print(f"[rbench] {len(jsons)} json files now under {root}")
    for j in sorted(jsons)[:8]:
        print(f"           {j.relative_to(root).as_posix()}")
    imgs = list((root / "images").glob("*.jpg")) if (root / "images").is_dir() else []
    if not imgs:
        print(f"[rbench] NOTE: no images under {root / 'images'} yet. R-Bench ships its "
              "own curated image archive (a subset of Open Images val, NOT the full "
              "multi-GB release):")
        print(f"           https://drive.google.com/file/d/{RBENCH_IMAGE_DRIVE}")
        print("           install it with: python scripts/ingest_manual.py "
              "--rbench-images /path/to/images.zip")


def install_rbench_images(cfg, archive: Path, overwrite: bool) -> None:
    root = cfg.abspath(cfg["datasets"]["rbench"]["root"])
    image_dir = root / "images"
    existing = list(image_dir.glob("*.jpg")) if image_dir.is_dir() else []
    if existing and not overwrite:
        print(f"[rbench] {len(existing)} images already present; use --overwrite to replace")
        return
    print(f"[rbench] installing images into {image_dir}")
    _extract(archive, image_dir)
    n = _flatten_jpgs(image_dir)
    print(f"[rbench] {n} jpg files now in {image_dir}")


def status(cfg) -> None:
    ds = cfg["datasets"]
    amber = cfg.abspath(ds["amber"]["root"])
    rbench = cfg.abspath(ds["rbench"]["root"])

    n_amber = len(list((amber / "image").glob("*.jpg"))) if (amber / "image").is_dir() else 0
    n_rb = len(list(rbench.glob("**/*.json"))) if rbench.is_dir() else 0

    print("=" * 68)
    print("manual dataset status")
    print("=" * 68)
    print(f"\nAMBER images   : {n_amber} jpg in {amber / 'image'}")
    if n_amber < 1004:
        print("   REQUIRED. Without it nothing runs (AMBER carries Table 1 and most figures).")
        print(f"   1. download https://drive.google.com/file/d/{AMBER_DRIVE}")
        print("   2. upload image.zip to this machine")
        print("   3. python scripts/ingest_manual.py --amber-images /path/to/image.zip")

    print(f"\nR-Bench json   : {n_rb} json under {rbench}")
    if n_rb == 0:
        print("   OPTIONAL. Missing means R-Bench rows are absent from RQ1/RQ3.")
        print(f"   1. annotations: https://drive.google.com/file/d/{RBENCH_DRIVE}")
        print("      -> python scripts/ingest_manual.py --rbench /path/to/data.zip")
        print(f"   2. images:      https://drive.google.com/file/d/{RBENCH_IMAGE_DRIVE}")
        print("      (a curated Open Images val subset shipped by R-Bench, not the "
              "full release)")
        print("      -> python scripts/ingest_manual.py --rbench-images /path/to/images.zip")
    print()


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--status", action="store_true", help="report what is missing and exit")
    p.add_argument("--amber-images", type=Path, default=None,
                   help="path to the AMBER image.zip (or an extracted directory)")
    p.add_argument("--rbench", type=Path, default=None,
                   help="path to the R-Bench annotation zip (or an extracted directory)")
    p.add_argument("--rbench-images", type=Path, default=None,
                   help="path to the R-Bench image zip (or an extracted directory)")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    from hmqr.config import load_config

    cfg = load_config(args.config)

    supplied = (args.amber_images, args.rbench, args.rbench_images)
    if args.status or all(p is None for p in supplied):
        status(cfg)
        return 0

    for flag, path in (
        ("--amber-images", args.amber_images),
        ("--rbench", args.rbench),
        ("--rbench-images", args.rbench_images),
    ):
        if path is not None and not path.exists():
            raise SystemExit(f"{flag}: path does not exist: {path}")

    if args.amber_images:
        install_amber_images(cfg, args.amber_images, args.overwrite)
    if args.rbench:
        install_rbench(cfg, args.rbench, args.overwrite)
    if args.rbench_images:
        install_rbench_images(cfg, args.rbench_images, args.overwrite)

    print("\nRe-check with: python scripts/check_env.py --config " + args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
