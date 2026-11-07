#!/usr/bin/env python
"""Report disk usage under data/ and remove what the pipeline does not need.

Two kinds of waste accumulate:

1. Download archives. Every dataset zip is kept after extraction, so COCO,
   Visual Genome and the rest each occupy roughly twice their size.
2. Visual Genome images. The RAG index is built purely from the VG json files
   (region descriptions, attributes, relationships and their boxes); no code path
   ever opens a VG jpg. VG_100K/VG_100K_2 are therefore dead weight -- around
   15 GB extracted plus 15 GB of archives.

Nothing is deleted unless you pass --apply.

    python scripts/free_space.py                 # report only
    python scripts/free_space.py --apply         # delete archives + VG images
    python scripts/free_space.py --apply --keep-vg-images
    python scripts/free_space.py --apply --models   # also drop model weights
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def du(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            pass
    return total


def gb(n: int) -> str:
    return f"{n / 1024**3:8.2f} GB"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--apply", action="store_true", help="actually delete")
    p.add_argument("--keep-vg-images", action="store_true",
                   help="keep VG_100K/VG_100K_2 even though nothing reads them")
    p.add_argument("--models", action="store_true",
                   help="also delete downloaded model weights (they re-download)")
    args = p.parse_args()

    from hmqr.config import load_config

    cfg = load_config(args.config)
    datasets = cfg.path("datasets")

    print("=" * 72)
    print(f"disk usage under {datasets}")
    print("=" * 72)
    for child in sorted(datasets.iterdir()) if datasets.is_dir() else []:
        print(f"  {gb(du(child))}  {child.name}")

    try:
        total, used, free = shutil.disk_usage(datasets)
        print(f"\n  volume: {gb(used)} used, {gb(free)} free, {gb(total)} total")
    except OSError:
        pass

    # ---- build the removal list ------------------------------------------ #
    targets: list[tuple[Path, str]] = []

    for archive in sorted(datasets.rglob("*.zip")):
        targets.append((archive, "download archive (already extracted)"))

    if not args.keep_vg_images:
        vg_root = cfg.abspath(cfg["datasets"]["visual_genome"]["root"])
        for name in ("VG_100K", "VG_100K_2"):
            d = vg_root / name
            if d.exists():
                targets.append((d, "VG images -- never read; the RAG index uses json only"))

    if args.models:
        for cache, label in (
            (cfg.path("model_cache"), "VLM backbone cache (will re-download)"),
            (cfg.path("tool_cache"), "frozen tool cache (will re-download)"),
        ):
            if cache.exists():
                targets.append((cache, label))

    if not targets:
        print("\nNothing to reclaim.")
        return 0

    reclaim = 0
    print("\n" + "=" * 72)
    print("removable" if args.apply else "removable (dry run -- pass --apply to delete)")
    print("=" * 72)
    for path, why in targets:
        size = du(path)
        reclaim += size
        try:
            shown = path.relative_to(cfg.root)
        except ValueError:
            shown = path
        print(f"  {gb(size)}  {shown}\n              {why}")
    print(f"\n  total reclaimable: {gb(reclaim)}")

    if not args.apply:
        print("\nRe-run with --apply to delete.")
        return 0

    print("\ndeleting...")
    for path, _ in targets:
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            print(f"  removed {path.name}")
        except OSError as exc:
            print(f"  FAILED  {path}: {exc}")

    try:
        _, _, free = shutil.disk_usage(datasets)
        print(f"\nfree space now: {gb(free)}")
    except OSError:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
