#!/usr/bin/env python
"""Migrate the legacy mixed ``data/raw`` layout to the structured v2 layout.

The command is a dry run unless ``--apply`` is supplied. It moves files on the
same volume (no copying of multi-GB checkpoints), splits backbone/tool HF cache
repositories, rewrites cached absolute paths atomically, and removes release
README/demo clutter while preserving licences in one central directory.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import stat
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
DATA = (ROOT / "data").resolve()
LEGACY = DATA / "raw"

DATASET_MOVES = {
    LEGACY / "AMBER": DATA / "datasets" / "amber",
    LEGACY / "R-Bench": DATA / "datasets" / "rbench",
    LEGACY / "MMRel": DATA / "datasets" / "mmrel",
    LEGACY / "coco": DATA / "datasets" / "coco2014",
    LEGACY / "visual_genome": DATA / "datasets" / "visual_genome",
}

TOOL_REPOS = {
    "models--openai--clip-vit-large-patch14-336",
    "models--IDEA-Research--grounding-dino-base",
}

BACKBONE_POINTERS = {
    "llava-hf__llava-1.5-7b-hf.path",
    "llava-hf__llava-v1.6-vicuna-7b-hf.path",
    "Salesforce__instructblip-vicuna-7b.path",
    "Qwen__Qwen-VL-Chat.path",
}

TOOL_POINTERS = {
    "openai__clip-vit-large-patch14-336.path",
    "IDEA-Research__grounding-dino-base.path",
}


def _inside_data(path: Path) -> Path:
    resolved = path.absolute() if path.is_symlink() else path.resolve()
    if not resolved.is_relative_to(DATA):
        raise RuntimeError(f"unsafe migration path outside {DATA}: {path} -> {resolved}")
    return path


def _size(path: Path) -> int:
    if path.is_symlink() or path.is_file():
        return path.lstat().st_size
    if not path.is_dir():
        return 0
    total = 0
    for item in path.rglob("*"):
        if item.is_file() and not item.is_symlink():
            total += item.stat().st_size
    return total


def _move_contents(source: Path, destination: Path) -> None:
    """Move a directory or merge it into an already-created empty skeleton."""
    _inside_data(source)
    _inside_data(destination)
    if not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        source.rename(destination)
        return
    if not source.is_dir() or not destination.is_dir():
        raise RuntimeError(f"cannot merge non-directories: {source} -> {destination}")
    for item in list(source.iterdir()):
        target = destination / item.name
        if target.exists() or target.is_symlink():
            if item.is_dir() and target.is_dir() and not item.is_symlink():
                _move_contents(item, target)
                continue
            raise RuntimeError(f"migration target already exists: {target}")
        item.rename(target)
    source.rmdir()


def _move_file(source: Path, destination: Path) -> None:
    _inside_data(source)
    _inside_data(destination)
    if not source.exists() and not source.is_symlink():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise RuntimeError(f"migration target already exists: {destination}")
    source.rename(destination)


def _remove(path: Path) -> None:
    _inside_data(path)
    if path.is_symlink() or path.is_file():
        try:
            path.chmod(stat.S_IWRITE)
        except OSError:
            pass
        path.unlink(missing_ok=True)
    elif path.is_dir():
        def _clear_readonly(func, item, _exc_info):
            os.chmod(item, stat.S_IWRITE)
            func(item)

        shutil.rmtree(path, onerror=_clear_readonly)


def _rewrite_text_paths(replacements: dict[bytes, bytes]) -> tuple[int, int]:
    roots = [DATA / "processed", ROOT / "results" / "tables", ROOT / "results" / "logs"]
    suffixes = {".json", ".jsonl", ".csv", ".txt", ".path"}
    updated_files = 0
    replacements_made = 0
    for directory in roots:
        if not directory.is_dir():
            continue
        for path in directory.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in suffixes:
                continue
            original = path.read_bytes()
            revised = original
            count = 0
            for old, new in replacements.items():
                occurrences = revised.count(old)
                if occurrences:
                    revised = revised.replace(old, new)
                    count += occurrences
            if revised == original:
                continue
            temporary = path.with_name(path.name + ".layout-v2.tmp")
            temporary.write_bytes(revised)
            os.replace(temporary, path)
            updated_files += 1
            replacements_made += count
    return updated_files, replacements_made


def _rewrite_pointer(path: Path, replacements: dict[bytes, bytes]) -> None:
    if not path.is_file():
        return
    value = path.read_bytes()
    for old, new in replacements.items():
        value = value.replace(old, new)
    temporary = path.with_name(path.name + ".layout-v2.tmp")
    temporary.write_bytes(value)
    os.replace(temporary, path)


def _cleanup_dataset_clutter() -> list[Path]:
    datasets = DATA / "datasets"
    licences = datasets / "_licenses"
    licences.mkdir(parents=True, exist_ok=True)
    moved_licences: list[Path] = []
    for name in ("amber", "rbench", "mmrel"):
        source = datasets / name / "LICENSE"
        target = licences / f"{name}.LICENSE"
        if source.is_file() and not target.exists():
            source.rename(target)
            moved_licences.append(target)

    clutter: list[Path] = []
    for root in [path for path in datasets.iterdir() if path.is_dir() and path.name != "_licenses"]:
        clutter.extend(root.rglob("README*"))
        clutter.extend(root.rglob(".cache"))
        clutter.extend(root.rglob(".DS_Store"))
        clutter.extend(root.rglob(".gitattributes"))
        clutter.extend(root.rglob(".git"))
    clutter.extend(
        [
            datasets / "amber" / "inference.py",
            datasets / "amber" / "README_File",
        ]
    )
    for path in sorted(set(clutter), key=lambda item: len(item.parts), reverse=True):
        _remove(path)
    return moved_licences


def _cleanup_unused_dataset_payloads() -> tuple[int, int]:
    """Remove payloads that no experiment in this repository can read.

    CHAIR uses a deterministic 500-image subset of COCO-2014 val. Visual
    Genome contributes JSON evidence only, so its image directories are pure
    storage overhead. The removals mirror the dataset loader and downloader,
    rather than relying on filename guesses.
    """
    config = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    removed_files = 0
    removed_bytes = 0

    coco = DATA / "datasets" / "coco2014"
    annotation = coco / "annotations" / "instances_val2014.json"
    if annotation.is_file():
        payload = json.loads(annotation.read_text(encoding="utf-8"))
        images = {int(row["id"]): str(row["file_name"]) for row in payload["images"]}
        ids = sorted(images)
        random.Random(int(config.get("seed", 42))).shuffle(ids)
        keep_n = int(config["datasets"]["coco"].get("chair_num_images", 500))
        keep = {images[image_id] for image_id in ids[:keep_n]}
        image_dir = coco / "val2014"
        present = {image.name for image in image_dir.glob("*.jpg")}
        missing = sorted(keep - present)
        if missing:
            raise RuntimeError(
                f"refusing to prune COCO: {len(missing)} configured CHAIR images are missing; "
                f"first={missing[:3]}"
            )
        for image in image_dir.glob("*.jpg"):
            if image.name in keep:
                continue
            removed_bytes += image.stat().st_size
            image.unlink()
            removed_files += 1

    for name in (
        "instances_train2014.json",
        "person_keypoints_train2014.json",
        "captions_train2014.json",
        "person_keypoints_val2014.json",
    ):
        path = coco / "annotations" / name
        if path.is_file():
            removed_bytes += path.stat().st_size
            path.unlink()
            removed_files += 1

    vg = DATA / "datasets" / "visual_genome"
    for name in ("VG_100K", "VG_100K_2", "images.zip", "images2.zip"):
        path = vg / name
        if not path.exists() and not path.is_symlink():
            continue
        removed_bytes += _size(path)
        if path.is_dir() and not path.is_symlink():
            removed_files += sum(1 for item in path.rglob("*") if item.is_file())
        else:
            removed_files += 1
        _remove(path)

    return removed_files, removed_bytes


def _preflight() -> None:
    """Fail before moving anything if the legacy root contains unknown data."""
    allowed = {
        "AMBER", "R-Bench", "MMRel", "coco", "visual_genome",
        "hf_cache", "models", "nltk_ok.flag", "url_image_cache", "image_cache",
    }
    if LEGACY.is_dir():
        unknown = sorted(path.name for path in LEGACY.iterdir() if path.name not in allowed)
        if unknown:
            raise RuntimeError(f"unclassified paths in legacy data/raw: {unknown}")

    legacy_pointers = LEGACY / "models"
    if legacy_pointers.is_dir():
        expected = BACKBONE_POINTERS | TOOL_POINTERS
        unknown = sorted(path.name for path in legacy_pointers.iterdir() if path.name not in expected)
        if unknown:
            raise RuntimeError(f"unclassified legacy model pointers: {unknown}")


def _plan() -> None:
    print(f"project={ROOT}")
    print(f"legacy={LEGACY} exists={LEGACY.exists()}")
    print("\nDataset moves")
    for source, target in DATASET_MOVES.items():
        print(f"  {source.relative_to(ROOT)} -> {target.relative_to(ROOT)}  { _size(source) / 1e9:.3f} GB")
    print("\nStorage moves")
    for source, target in (
        (LEGACY / "hf_cache", DATA / "models" / "hf"),
        (LEGACY / "models", DATA / "models" / "pointers"),
        (LEGACY / "nltk_ok.flag", DATA / "tools" / "resources" / "nltk_ok.flag"),
        (DATA / "vg_index", DATA / "indexes" / "visual_genome"),
    ):
        print(f"  {source.relative_to(ROOT)} -> {target.relative_to(ROOT)}  { _size(source) / 1e9:.3f} GB")
    print("\nTool repositories split from backbone cache")
    for repo in sorted(TOOL_REPOS):
        print(f"  {repo}")
    coco_images = LEGACY / "coco" / "val2014"
    vg_images = LEGACY / "visual_genome" / "VG_100K"
    vg_images_2 = LEGACY / "visual_genome" / "VG_100K_2"
    print("\nUnused payload cleanup")
    print("  Legacy URL/download cache directories -> removed")
    print(
        "  COCO val images: "
        f"{sum(1 for _ in coco_images.glob('*.jpg')) if coco_images.is_dir() else 0} -> "
        "configured CHAIR subset"
    )
    print(
        "  Visual Genome image payload: "
        f"{(_size(vg_images) + _size(vg_images_2)) / 1e9:.3f} GB -> removed "
        "(the index reads JSON only)"
    )


def _apply() -> None:
    if not DATA.is_dir() or DATA == Path(DATA.anchor):
        raise RuntimeError(f"invalid data root: {DATA}")
    _preflight()

    replacements: dict[bytes, bytes] = {}
    for source, target in DATASET_MOVES.items():
        replacements[str(source).encode()] = str(target).encode()
        replacements[source.relative_to(ROOT).as_posix().encode()] = target.relative_to(ROOT).as_posix().encode()
        _move_contents(source, target)

    old_hf = LEGACY / "hf_cache"
    new_hf = DATA / "models" / "hf"
    replacements[str(old_hf).encode()] = str(new_hf).encode()
    replacements[old_hf.relative_to(ROOT).as_posix().encode()] = new_hf.relative_to(ROOT).as_posix().encode()
    _move_contents(old_hf, new_hf)

    model_hub = DATA / "models" / "hf" / "hub"
    tool_cache = DATA / "tools" / "hf_cache"
    tool_cache.mkdir(parents=True, exist_ok=True)
    for repo in TOOL_REPOS:
        _move_contents(model_hub / repo, tool_cache / repo)
        _move_contents(model_hub / ".locks" / repo, tool_cache / ".locks" / repo)
        replacements[str(model_hub / repo).encode()] = str(tool_cache / repo).encode()

    legacy_pointers = LEGACY / "models"
    model_pointers = DATA / "models" / "pointers"
    tool_pointers = DATA / "tools" / "pointers"
    model_pointers.mkdir(parents=True, exist_ok=True)
    tool_pointers.mkdir(parents=True, exist_ok=True)
    for name in BACKBONE_POINTERS:
        _move_file(legacy_pointers / name, model_pointers / name)
    for name in TOOL_POINTERS:
        _move_file(legacy_pointers / name, tool_pointers / name)
    if legacy_pointers.is_dir() and not any(legacy_pointers.iterdir()):
        legacy_pointers.rmdir()
    for pointer in [*model_pointers.glob("*.path"), *tool_pointers.glob("*.path")]:
        _rewrite_pointer(pointer, replacements)

    _move_file(
        LEGACY / "nltk_ok.flag",
        DATA / "tools" / "resources" / "nltk_ok.flag",
    )
    _move_contents(DATA / "vg_index", DATA / "indexes" / "visual_genome")
    _remove(LEGACY / "url_image_cache")
    _remove(LEGACY / "image_cache")
    _remove(DATA / "cache")

    removed_files, removed_bytes = _cleanup_unused_dataset_payloads()
    moved_licences = _cleanup_dataset_clutter()
    updated_files, replacements_made = _rewrite_text_paths(replacements)

    if LEGACY.is_dir():
        leftovers = list(LEGACY.iterdir())
        if leftovers:
            raise RuntimeError(f"legacy data/raw still contains unclassified paths: {leftovers}")
        LEGACY.rmdir()

    manifest = {
        "layout_version": 2,
        "datasets": "data/datasets",
        "backbone_cache": "data/models/hf/hub",
        "tool_cache": "data/tools/hf_cache",
        "processed": "data/processed",
        "visual_genome_index": "data/indexes/visual_genome",
    }
    (DATA / "layout.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"moved_licences={len(moved_licences)}")
    print(f"removed_unused_files={removed_files}")
    print(f"removed_unused_gb={removed_bytes / 1e9:.3f}")
    print(f"updated_text_files={updated_files}")
    print(f"path_replacements={replacements_made}")
    print("layout_version=2")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="perform the migration")
    args = parser.parse_args()
    _plan()
    if not args.apply:
        print("\nDry run only. Re-run with --apply after reviewing the plan.")
        return 0
    _apply()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
