#!/usr/bin/env python
"""Bootstrap: install dependencies, then download all datasets and models.

This script is deliberately stdlib-only until the dependency installation has
finished, so it can be the very first thing you run in a fresh environment:

    python scripts/00_download_data.py --config configs/default.yaml

Useful switches:
    --no-deps           skip pip install (dependencies already present)
    --skip-vg-images    skip the ~15 GB Visual Genome image archives
    --skip-models       skip HuggingFace model snapshots
    --only AMBER        run only the tasks whose name contains this substring
    --overwrite         redo tasks even when their marker exists
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def install_dependencies() -> None:
    req = ROOT / "requirements.txt"
    print(f"[deps] pip install -r {req}")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--upgrade", "pip"]
    )
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(req)])
    print("[deps] pip install -e . (editable install of the hmqr package)")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", str(ROOT)])
    # gdown is only needed for the Google-Drive-hosted AMBER images
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--no-deps", action="store_true")
    parser.add_argument("--skip-vg-images", action="store_true",
                        help="(default) skip the ~15 GB Visual Genome image archives; "
                             "retrieval needs only JSON; pixel overlap screening needs images")
    parser.add_argument("--vg-images", action="store_true",
                        help="do download the Visual Genome images anyway")
    parser.add_argument("--skip-models", action="store_true")
    parser.add_argument("--only", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.no_deps:
        install_dependencies()

    # import only after the dependencies exist
    sys.path.insert(0, str(ROOT / "src"))
    from hmqr.config import load_config, seed_everything
    from hmqr.data.download import run_download

    cfg = load_config(args.config)
    seed_everything(cfg.get("seed", 42))

    report = run_download(
        cfg,
        overwrite=args.overwrite,
        skip_vg_images=not args.vg_images,
        skip_models=args.skip_models,
        only=args.only,
    )

    print("\n================ download summary ================")
    print(f"  downloaded : {len(report.done)}")
    print(f"  skipped    : {len(report.skipped)}")
    print(f"  failed     : {len(report.failed)}")
    for name, err, hint in report.failed:
        print(f"\n  [FAILED] {name}\n     error: {err}\n     hint : {hint}")
    print("\n  manifest   : results/logs/data_manifest.txt")
    print("==================================================")
    if report.failed:
        print("\nSome sources need manual action (see hints above). Re-run this script "
              "after placing the files; completed tasks are skipped automatically.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
