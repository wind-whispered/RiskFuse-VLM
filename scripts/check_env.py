#!/usr/bin/env python
"""Preflight check: environment, hub reachability, dataset payloads.

Run this before the pipeline. It answers, in a few seconds, the three questions
that otherwise take a failed multi-hour run to discover:

    1. Can transformers actually see PyTorch?  (a version mismatch silently
       disables the torch backend, after which no model can be loaded)
    2. Is the HuggingFace endpoint reachable from this host?
    3. Are the dataset files really on disk, not just empty cloned directories?

    python scripts/check_env.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OK, BAD, WARN = "[ OK ]", "[FAIL]", "[WARN]"
problems: list[str] = []


def fail(msg: str, fix: str) -> None:
    print(f"{BAD} {msg}")
    print(f"       fix: {fix}")
    problems.append(msg)


def check_torch_transformers() -> None:
    print("\n--- 1. torch / transformers ---")
    try:
        import torch
    except ImportError:
        fail("torch is not installed", "pip install -r requirements.txt")
        return
    print(f"{OK} torch {torch.__version__}  cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"       GPU: {p.name}, {p.total_memory / 1024**3:.1f} GB")

    try:
        import transformers
    except ImportError:
        fail("transformers is not installed", "pip install -r requirements.txt")
        return
    print(f"{OK} transformers {transformers.__version__}")

    # The decisive test: transformers disables its torch backend on a version
    # mismatch and then only tokenizers/config utilities work.
    if not transformers.utils.is_torch_available():
        fail(
            "transformers cannot use PyTorch (torch backend disabled) -- "
            "no model can be loaded",
            'pip install -U "transformers>=4.44,<4.50"   '
            "(transformers v5 requires torch>=2.5; this host has "
            f"torch {torch.__version__})",
        )
    else:
        print(f"{OK} transformers torch backend is active")


def check_hub(cfg) -> None:
    print("\n--- 2. HuggingFace hub reachability ---")
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    print(f"       endpoint: {endpoint}")
    if os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in {"1", "true", "yes", "on"}:
        print("       HF_HUB_OFFLINE=1; validating local model snapshots")
        from huggingface_hub import snapshot_download

        repositories = [
            (cfg.backbone_cfg(name)["hf_id"], cfg.path("model_cache"))
            for name in cfg.backbone_names
        ]
        repositories += [
            (cfg["tools"]["clip_hf_id"], cfg.path("tool_cache")),
            (cfg["tools"]["grounding_dino_hf_id"], cfg.path("tool_cache")),
        ]
        for repo_id, cache_dir in repositories:
            try:
                path = snapshot_download(
                    repo_id=repo_id,
                    cache_dir=str(cache_dir),
                    local_files_only=True,
                )
                print(f"{OK} cached: {repo_id} ({path})")
            except Exception as exc:  # noqa: BLE001
                fail(
                    f"model snapshot is not fully available offline: {repo_id} ({exc})",
                    "temporarily disable HF_HUB_OFFLINE and complete the download from the mirror",
                )
        return
    try:
        import requests

        r = requests.get(f"{endpoint}/api/models/openai/clip-vit-large-patch14-336",
                         timeout=15)
        if r.status_code < 400:
            print(f"{OK} endpoint reachable (HTTP {r.status_code})")
        else:
            fail(f"endpoint returned HTTP {r.status_code}",
                 "check the mirror URL in configs/default.yaml -> hf_endpoint")
    except Exception as exc:  # noqa: BLE001
        fail(
            f"cannot reach {endpoint} ({type(exc).__name__}: {exc})",
            "this host cannot reach the hub directly. Set a mirror:\n"
            "            export HF_ENDPOINT=https://hf-mirror.com\n"
            "       (configs/default.yaml -> hf_endpoint already defaults to it)",
        )


def check_datasets(cfg) -> None:
    print("\n--- 3. dataset payloads on disk ---")
    ds = cfg["datasets"]

    def probe(label: str, root: Path, patterns: dict[str, str], required: bool) -> None:
        if not root.is_dir():
            (fail if required else print)(
                f"{label}: directory missing ({root})"
                if required else f"{WARN} {label}: directory missing ({root})",
                *([f"re-run: python scripts/00_download_data.py --only '{label}'"]
                  if required else []),
            )
            return
        missing = [name for name, pat in patterns.items()
                   if next(iter(root.glob(pat)), None) is None]
        if missing:
            msg = f"{label}: present but missing {missing}"
            if required:
                fail(msg, "see the hint printed by scripts/00_download_data.py")
            else:
                print(f"{WARN} {msg} (optional)")
        else:
            n = sum(1 for _ in root.rglob("*") if _.is_file())
            print(f"{OK} {label}: {n} files")

    # Every benchmark named in the paper is required for a full paper run.
    # "image/*.jpg" not "**/*.jpg": the repo ships README screenshots that would
    # otherwise mask a missing image set.
    probe("AMBER", cfg.abspath(ds["amber"]["root"]),
          {"query json": "**/query*.json", "images": "image/*.jpg"}, required=True)
    probe("R-Bench", cfg.abspath(ds["rbench"]["root"]),
          {"annotations": "**/*.json"}, required=True)
    probe("MMRel", cfg.abspath(ds["mmrel"]["root"]),
          {"evaluation QA jsonl": "json/Evaluation/Discriminative_YN_QA/*.jsonl"},
          required=True)
    probe("COCO val2014", cfg.abspath(ds["coco"]["root"]),
          {"images": "val2014/*.jpg", "annotations": "annotations/*.json"}, required=True)
    probe("Visual Genome", cfg.abspath(ds["visual_genome"]["root"]),
          {"region descriptions": "**/region_descriptions.json"}, required=True)


def check_dataset_loaders(cfg) -> None:
    """Run the real adapters and enforce the paper's full benchmark cardinalities."""
    print("\n--- 4. full benchmark cardinalities ---")
    subset = cfg["datasets"].get("subset_limit")
    if subset is not None:
        fail(
            f"datasets.subset_limit={subset}; this is not a full paper configuration",
            "set configs/default.yaml -> datasets.subset_limit: null",
        )
        return

    from hmqr.data.datasets import load_dataset

    # Exact adapter outputs for the official release files used in this paper.
    # R-Bench reports Image(All) only: instance-level is a separate protocol
    # requiring coloured boxes/masks and must not be pooled into one score.
    # MMRel's 22,441 evaluation QAs comprise 14,927 discriminative Y/N plus
    # 7,514 open-ended prompts; only the former have benchmark binary labels.
    expected = {
        "amber_attr": 7628,
        "amber_rel": 1664,
        "rbench": 7756,
        "mmrel": 14927,
        "chair500": 500,
    }
    for name, exact in expected.items():
        try:
            samples = load_dataset(cfg, name)
        except Exception as exc:  # noqa: BLE001
            fail(f"{name}: loader failed ({exc})", "repair or re-ingest this dataset before running")
            continue
        count = len(samples)
        if count != exact:
            fail(
                f"{name}: adapter returned {count} samples; paper protocol requires exactly {exact}",
                "verify the official release files and the loader-selected split/protocol",
            )
            continue
        missing_local = {
            sample.image_path for sample in samples
            if not str(sample.image_path).startswith("http") and not Path(sample.image_path).exists()
        }
        if missing_local:
            fix = (
                "run: python scripts/fetch_mmrel_images.py --config configs/default.yaml"
                if name == "mmrel"
                else f"first missing path: {next(iter(sorted(missing_local)))}"
            )
            fail(
                f"{name}: {len(missing_local)} referenced local images are missing",
                fix,
            )
        else:
            print(f"{OK} {name}: {count} samples, image references resolved")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    from hmqr.config import load_config

    cfg = load_config(args.config)   # also exports HF_ENDPOINT from the config

    print(f"[config] backbone cache={cfg.path('model_cache')}")
    print(f"[config] tool cache={cfg.path('tool_cache')}")

    print("=" * 68)
    print("hmqr preflight check")
    print("=" * 68)
    check_torch_transformers()
    check_hub(cfg)
    check_datasets(cfg)
    check_dataset_loaders(cfg)

    print("\n" + "=" * 68)
    if problems:
        print(f"{len(problems)} BLOCKING PROBLEM(S):")
        for p in problems:
            print(f"  - {p}")
        print("\nFix these before running the pipeline.")
        print("For datasets behind Google Drive, see: python scripts/ingest_manual.py --help")
        return 1
    print("All full-run checks passed. Safe to run: python run_all.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
