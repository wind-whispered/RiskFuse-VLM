"""Dataset and model acquisition.

Every item is described by a small task record so that the download is resumable
and idempotent: if the target marker already exists the task is skipped. Items
that cannot be fetched non-interactively (Google-Drive-gated releases) are
reported at the end with explicit manual instructions instead of failing silently.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from ..config import Config
from ..utils import get_logger

LOG = get_logger("hmqr.download")

# --------------------------------------------------------------------------- #
# generic helpers
# --------------------------------------------------------------------------- #


def _http_download(url: str, dest: Path, chunk: int = 1 << 20) -> None:
    import requests
    from tqdm import tqdm

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    resume_from = tmp.stat().st_size if tmp.exists() else 0
    headers = {"Range": f"bytes={resume_from}-"} if resume_from else {}
    with requests.get(url, stream=True, timeout=60, headers=headers) as r:
        if r.status_code == 416:  # already complete
            tmp.rename(dest)
            return
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0)) + resume_from
        mode = "ab" if resume_from else "wb"
        with open(tmp, mode) as f, tqdm(
            total=total or None,
            initial=resume_from,
            unit="B",
            unit_scale=True,
            desc=dest.name,
            ncols=88,
        ) as bar:
            for block in r.iter_content(chunk_size=chunk):
                if block:
                    f.write(block)
                    bar.update(len(block))
    tmp.rename(dest)


def _unzip(archive: Path, target_dir: Path, keep_archive: bool = False) -> None:
    """Extract and, by default, delete the archive.

    Keeping every zip alongside its extracted copy doubles the footprint of the
    raw data (COCO and Visual Genome alone account for ~20 GB of redundant
    archives), which is enough to exhaust a typical cloud data disk before the
    model weights are downloaded.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    LOG.info("extracting %s -> %s", archive.name, target_dir)
    with zipfile.ZipFile(archive) as z:
        z.extractall(target_dir)
    if not keep_archive:
        try:
            archive.unlink()
            LOG.info("removed archive %s (already extracted)", archive.name)
        except OSError as exc:
            LOG.warning("could not remove %s: %s", archive, exc)


def _git_clone(url: str, dest: Path, depth: int = 1) -> None:
    if dest.exists() and any(dest.iterdir()):
        LOG.info("git repo already present: %s", dest)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    LOG.info("git clone %s", url)
    subprocess.check_call(
        ["git", "clone", "--depth", str(depth), url, str(dest)],
        stdout=sys.stdout,
        stderr=sys.stderr,
    )


def _hf_snapshot(
    repo_id: str,
    repo_type: str = "model",
    allow_patterns=None,
    local_dir: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
) -> str:
    """Download a HF repo snapshot.

    `local_dir` places the real files in a directory of our choosing. Use it for
    data we read ourselves with plain filesystem globs, so the files land where a
    user would expect to find them.

    Model and tool snapshots receive separate explicit ``cache_dir`` values so
    the four VLM backbones are not mixed with CLIP / Grounding DINO weights.

    `resume_download` was deprecated in huggingface_hub 0.26 and removed in 1.x
    (resuming is the default now), so it is not passed here.
    """
    from huggingface_hub import snapshot_download

    LOG.info("huggingface snapshot: %s (%s)", repo_id, repo_type)
    kwargs = dict(
        repo_id=repo_id,
        repo_type=repo_type,
        allow_patterns=allow_patterns,
        max_workers=4,
    )
    if local_dir is not None:
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        kwargs["local_dir"] = str(local_dir)
    if cache_dir is not None:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        kwargs["cache_dir"] = str(cache_dir)
    return snapshot_download(**kwargs)


def _gdown(file_id_or_url: str, dest: Path) -> None:
    try:
        import gdown  # noqa: F401
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
    import gdown

    dest.parent.mkdir(parents=True, exist_ok=True)
    url = (
        file_id_or_url
        if file_id_or_url.startswith("http")
        else f"https://drive.google.com/uc?id={file_id_or_url}"
    )
    gdown.download(url, str(dest), quiet=False, fuzzy=True)


# --------------------------------------------------------------------------- #
# task table
# --------------------------------------------------------------------------- #


@dataclass
class Task:
    name: str
    marker: Path                 # existence of this path == task complete
    run: Callable[[], None]
    optional: bool = False
    manual_hint: str = ""
    # Optional content check. A bare `git clone` creates the target directory
    # even when the annotation payload is absent, which would make a pure
    # marker.exists() test report "skip" for a dataset that is in fact unusable.
    verify: Optional[Callable[[], bool]] = None

    def is_complete(self) -> bool:
        if not self.marker.exists():
            return False
        if self.verify is None:
            return True
        try:
            return bool(self.verify())
        except Exception:  # noqa: BLE001
            return False


def _has_any(root: Path, pattern: str) -> bool:
    """True when at least one file matches `pattern` anywhere under `root`."""
    return root.is_dir() and next(iter(root.glob(pattern)), None) is not None


@dataclass
class Report:
    done: List[str] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)
    failed: List[tuple] = field(default_factory=list)


COCO_URLS = {
    "val2014": "http://images.cocodataset.org/zips/val2014.zip",
    "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2014.zip",
}
VG_BASE = "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset"
VG_FILES = {
    "region_descriptions.json.zip": f"{VG_BASE}/region_descriptions.json.zip",
    "relationships.json.zip": f"{VG_BASE}/relationships.json.zip",
    "objects.json.zip": f"{VG_BASE}/objects.json.zip",
    "attributes.json.zip": f"{VG_BASE}/attributes.json.zip",
    "image_data.json.zip": f"{VG_BASE}/image_data.json.zip",
}
VG_IMAGE_FILES = {
    "images.zip": "https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip",
    "images2.zip": "https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip",
}

AMBER_REPO = "https://github.com/junyangwang0410/AMBER.git"
AMBER_IMAGE_DRIVE = "1MaCHgtupcZUjf007anNl4_MV0o4DjXvl"  # image.zip from the AMBER README
RBENCH_REPO = "https://github.com/mrwu-mac/R-Bench.git"
MMREL_REPO = "https://github.com/niejiahao1998/MMRel.git"
MMREL_HF = "jiahaonie/MMRel"  # official HF dataset release (CC-BY-4.0)


def build_tasks(cfg: Config, skip_vg_images: bool = False, skip_models: bool = False) -> List[Task]:
    tasks: List[Task] = []

    # ---------------- AMBER ---------------------------------------------- #
    amber_root = cfg.abspath(cfg["datasets"]["amber"]["root"])

    def _amber():
        _git_clone(AMBER_REPO, amber_root)
        img_dir = amber_root / "image"
        if not img_dir.exists() or not any(img_dir.glob("*.jpg")):
            zip_path = amber_root / "image.zip"
            if not zip_path.exists():
                _gdown(AMBER_IMAGE_DRIVE, zip_path)
            if zip_path.exists():
                _unzip(zip_path, amber_root)

    tasks.append(
        Task(
            name="AMBER (repo + images)",
            marker=amber_root / "data",
            run=_amber,
            # The image check must target the image directory specifically: the
            # repo ships four screenshots under README_File/, so a plain
            # "**/*.jpg" search matches even when the dataset images are absent.
            verify=lambda: _has_any(amber_root, "**/query*.json")
            and _has_any(amber_root / "image", "*.jpg"),
            manual_hint=(
                "Need BOTH: (1) the repo json payload -- clone "
                "https://github.com/junyangwang0410/AMBER so that "
                f"{amber_root}/data/query/query_discriminative-attribute.json exists; "
                "(2) the images -- download image.zip from the Google Drive link in "
                f"its README and unzip into {amber_root}/image/ (must contain *.jpg)."
            ),
        )
    )

    # ---------------- R-Bench -------------------------------------------- #
    rbench_root = cfg.abspath(cfg["datasets"]["rbench"]["root"])
    tasks.append(
        Task(
            name="R-Bench (repo)",
            marker=rbench_root,
            run=lambda: _git_clone(RBENCH_REPO, rbench_root),
            verify=lambda: _has_any(rbench_root, "**/*.json"),
            manual_hint=(
                "The R-Bench repo does not ship its annotations in git. Follow the "
                "data download link in https://github.com/mrwu-mac/R-Bench and place "
                f"the annotation json under {rbench_root} (any subdirectory is fine; "
                "the loader searches recursively). NoCaps images are fetched from the "
                "URLs inside the annotation file on first use."
            ),
        )
    )

    # ---------------- MMRel ---------------------------------------------- #
    mmrel_root = cfg.abspath(cfg["datasets"]["mmrel"]["root"])

    def _mmrel():
        # Downloaded straight into data/datasets/mmrel/ with the release's own
        # layout (json/... and Dall-E_generated/...), so the files are where you
        # would look for them rather than behind a hashed cache path.
        path = _hf_snapshot(
            MMREL_HF,
            repo_type="dataset",
            local_dir=mmrel_root,
        )
        LOG.info("MMRel data lives at: %s", path)
        # `local_dir` keeps a small transfer-metadata cache beside the finished
        # files. It is not needed by the loader once the snapshot is complete.
        shutil.rmtree(mmrel_root / ".cache", ignore_errors=True)
        # A stale pointer from an earlier cache-based run would send the loader
        # off to the old location; the data is local now.
        pointer = mmrel_root / "hf_snapshot_path.txt"
        if pointer.exists():
            pointer.unlink()

    def _mmrel_ok() -> bool:
        return _has_any(mmrel_root, "**/*.json")

    tasks.append(
        Task(
            name="MMRel (HuggingFace dataset jiahaonie/MMRel)",
            marker=mmrel_root,
            run=_mmrel,
            optional=True,
            verify=_mmrel_ok,
            manual_hint=(
                "MMRel is on HuggingFace (jiahaonie/MMRel) and works through the "
                "HF_ENDPOINT mirror, so no Google Drive is involved. Retry with:\n"
                f"         huggingface-cli download jiahaonie/MMRel --repo-type dataset "
                f"--local-dir {mmrel_root}\n"
                f"       or place the QA json under {mmrel_root} (searched recursively). "
                "MMRel only supplies the RQ1 zero-shot transfer row; the rest of the "
                "paper runs without it."
            ),
        )
    )

    # ---------------- COCO 2014 val (CHAIR-500) --------------------------- #
    coco_root = cfg.abspath(cfg["datasets"]["coco"]["root"])

    def _coco():
        for key, url in COCO_URLS.items():
            zip_path = coco_root / f"{key}.zip"
            target = coco_root / ("val2014" if key == "val2014" else "annotations")
            if target.exists() and any(target.iterdir()):
                continue
            if not zip_path.exists():
                _http_download(url, zip_path)
            _unzip(zip_path, coco_root)

        # This project evaluates the deterministic CHAIR-500 subset only. The
        # train annotations, keypoints, and other 40k val images are never read.
        for name in (
            "instances_train2014.json",
            "person_keypoints_train2014.json",
            "captions_train2014.json",
            "person_keypoints_val2014.json",
        ):
            (coco_root / "annotations" / name).unlink(missing_ok=True)

        import json
        import random

        annotation = coco_root / "annotations" / "instances_val2014.json"
        if annotation.is_file():
            payload = json.loads(annotation.read_text(encoding="utf-8"))
            images = {int(row["id"]): str(row["file_name"]) for row in payload["images"]}
            ids = sorted(images)
            random.Random(int(cfg.get("seed", 42))).shuffle(ids)
            keep_n = int(cfg["datasets"]["coco"].get("chair_num_images", 500))
            keep = {images[image_id] for image_id in ids[:keep_n]}
            for image in (coco_root / "val2014").glob("*.jpg"):
                if image.name not in keep:
                    image.unlink()

    tasks.append(
        Task(
            name="COCO-2014 val + annotations",
            marker=coco_root / "val2014",
            run=_coco,
            manual_hint="Download val2014.zip and annotations_trainval2014.zip from cocodataset.org",
        )
    )

    # ---------------- Visual Genome --------------------------------------- #
    vg_root = cfg.abspath(cfg["datasets"]["visual_genome"]["root"])

    def _vg_json():
        for fname, url in VG_FILES.items():
            stem = fname.replace(".zip", "")
            if (vg_root / stem).exists():
                continue
            zip_path = vg_root / fname
            if not zip_path.exists():
                _http_download(url, zip_path)
            _unzip(zip_path, vg_root)

    tasks.append(
        Task(
            name="Visual Genome annotations (regions/relationships/objects/attributes)",
            marker=vg_root / "region_descriptions.json",
            run=_vg_json,
            manual_hint="Download the json.zip files listed at visualgenome.org/api/v0/api_home.html",
        )
    )

    # Visual Genome IMAGES are intentionally not downloaded by default.
    # The RAG index is built from the VG json only (region descriptions,
    # attributes, relationships and their boxes) and no code path opens a VG
    # jpg, so these ~15 GB would sit unused. Pass --vg-images to fetch them.
    if not skip_vg_images:

        def _vg_images():
            for fname, url in VG_IMAGE_FILES.items():
                zip_path = vg_root / fname
                if not zip_path.exists():
                    _http_download(url, zip_path)
                _unzip(zip_path, vg_root)

        tasks.append(
            Task(
                name="Visual Genome images (VG_100K, VG_100K_2) ~15 GB",
                marker=vg_root / "VG_100K",
                run=_vg_images,
                optional=True,
                manual_hint="Download images.zip and images2.zip from visualgenome.org",
            )
        )

    # ---------------- models ---------------------------------------------- #
    if not skip_models:
        model_ids = [cfg.backbone_cfg(b)["hf_id"] for b in cfg.backbone_names]
        tool_ids = [cfg["tools"]["clip_hf_id"], cfg["tools"]["grounding_dino_hf_id"]]
        model_markers = cfg.path("model_pointers")
        tool_markers = cfg.path("tool_pointers")

        def _snapshots_ready(repo_ids: List[str], markers: Path) -> bool:
            for repo_id in repo_ids:
                pointer = markers / (repo_id.replace("/", "__") + ".path")
                if not pointer.is_file():
                    return False
                try:
                    snapshot = Path(pointer.read_text(encoding="utf-8").strip())
                except OSError:
                    return False
                if not snapshot.is_dir():
                    return False
            return True

        def _models():
            for rid in model_ids:
                path = _hf_snapshot(
                    rid, repo_type="model", cache_dir=cfg.path("model_cache")
                )
                (model_markers / (rid.replace("/", "__") + ".path")).write_text(
                    path, encoding="utf-8"
                )

        tasks.append(
            Task(
                name="VLM backbone weights (4 HuggingFace models)",
                marker=model_markers / (model_ids[-1].replace("/", "__") + ".path"),
                run=_models,
                verify=lambda: _snapshots_ready(model_ids, model_markers),
                manual_hint="Run `huggingface-cli download <repo_id>` for each model id.",
            )
        )

        def _tools():
            for rid in tool_ids:
                path = _hf_snapshot(
                    rid, repo_type="model", cache_dir=cfg.path("tool_cache")
                )
                (tool_markers / (rid.replace("/", "__") + ".path")).write_text(
                    path, encoding="utf-8"
                )

        tasks.append(
            Task(
                name="Frozen tool weights (CLIP + Grounding DINO)",
                marker=tool_markers / (tool_ids[-1].replace("/", "__") + ".path"),
                run=_tools,
                verify=lambda: _snapshots_ready(tool_ids, tool_markers),
                manual_hint="Run `huggingface-cli download <repo_id>` for each tool id.",
            )
        )

    # ---------------- NLTK data (BLEU/ROUGE tokenisers) ------------------- #
    def _nltk():
        import nltk

        nltk_dir = cfg.path("tool_resources") / "nltk_data"
        nltk_dir.mkdir(parents=True, exist_ok=True)
        for pkg in ("punkt", "punkt_tab", "wordnet", "omw-1.4"):
            try:
                nltk.download(pkg, download_dir=str(nltk_dir), quiet=True)
            except Exception as exc:  # noqa: BLE001
                LOG.warning("nltk %s failed: %s", pkg, exc)
        (cfg.path("tool_resources") / "nltk_ok.flag").write_text("ok\n", encoding="utf-8")

    tasks.append(
        Task(
            name="NLTK corpora",
            marker=cfg.path("tool_resources") / "nltk_ok.flag",
            run=_nltk,
            optional=True,
        )
    )

    return tasks


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #


def run_download(
    cfg: Config,
    overwrite: bool = False,
    skip_vg_images: bool = False,
    skip_models: bool = False,
    only: Optional[str] = None,
) -> Report:
    report = Report()
    tasks = build_tasks(cfg, skip_vg_images=skip_vg_images, skip_models=skip_models)
    for task in tasks:
        if only and only.lower() not in task.name.lower():
            continue
        if task.is_complete() and not overwrite:
            LOG.info("[skip] %s (already present: %s)", task.name, task.marker)
            report.skipped.append(task.name)
            continue
        if task.marker.exists() and task.verify is not None:
            LOG.warning(
                "[redo] %s -- the target directory exists but its payload is "
                "incomplete; retrying", task.name
            )
        LOG.info("[run ] %s", task.name)
        try:
            task.run()
            report.done.append(task.name)
        except Exception as exc:  # noqa: BLE001
            LOG.error("[fail] %s -> %s", task.name, exc)
            report.failed.append((task.name, str(exc), task.manual_hint))

    write_manifest(cfg, report)
    return report


def write_manifest(cfg: Config, report: Report) -> None:
    lines: List[str] = ["# data manifest", ""]
    for label, root in (
        ("datasets", cfg.path("datasets")),
        ("backbone_models", cfg.path("model_cache")),
        ("tool_weights", cfg.path("tool_cache")),
        ("tool_resources", cfg.path("tool_resources")),
        ("processed", cfg.path("processed")),
        ("indexes", cfg.path("vg_index")),
    ):
        n_files = sum(1 for path in root.rglob("*") if path.is_file())
        lines.append(f"{label}: {root.relative_to(cfg.root)}  files={n_files}")
    lines += ["", "## downloaded this run", *[f"  - {n}" for n in report.done]]
    lines += ["", "## skipped (already present)", *[f"  - {n}" for n in report.skipped]]
    if report.failed:
        lines += ["", "## FAILED -- manual action required"]
        for name, err, hint in report.failed:
            lines += [f"  - {name}", f"      error: {err}", f"      hint : {hint}"]
    out = cfg.path("logs") / "data_manifest.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOG.info("manifest written to %s", out)
