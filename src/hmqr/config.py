"""Configuration loading, seeding and path resolution."""
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


def project_root() -> Path:
    """Return the project root (the directory that contains configs/)."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "configs").is_dir() and (parent / "scripts").is_dir():
            return parent
    # fall back: src/hmqr/config.py -> src/hmqr -> src -> root
    return here.parents[2]


@dataclass
class Config:
    raw: Dict[str, Any]
    root: Path

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    def path(self, key: str) -> Path:
        """Resolve a path from cfg['paths'][key] against the project root."""
        p = Path(self.raw["paths"][key])
        return p if p.is_absolute() else self.root / p

    def abspath(self, value: str) -> Path:
        p = Path(value)
        return p if p.is_absolute() else self.root / p

    def backbone_cfg(self, name: str) -> Dict[str, Any]:
        return self.raw["backbones"][name]

    def resource_id(self, canonical: str, local_path: str | None = None) -> str:
        """Bind an existing local checkpoint without changing its canonical id."""
        if local_path and self.get('local_execution', {}).get('enabled', False):
            path = self.abspath(local_path)
            if not path.is_dir():
                raise FileNotFoundError(f'configured local checkpoint missing: {path}')
            return str(path)
        return canonical

    @property
    def backbone_names(self):
        return list(self.raw["backbones"]["order"])


def load_config(path: str | os.PathLike | None = None) -> Config:
    root = project_root()
    cfg_path = Path(path) if path else root / "configs" / "default.yaml"
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    raw.setdefault("paths", {}).setdefault("review", "results/review")
    cp = raw.get("cp", {})
    split_ratios = [
        float(cp.get("calib_cp_ratio", 0.0)),
        float(cp.get("calib_revision_ratio", 0.0)),
        float(cp.get("test_ratio", 0.0)),
    ]
    if any(value <= 0.0 for value in split_ratios) or abs(sum(split_ratios) - 1.0) > 1e-8:
        raise ValueError(
            "cp calib_cp_ratio/calib_revision_ratio/test_ratio must be positive and sum to 1"
        )
    cfg = Config(raw=raw, root=root)
    for key in (
        "datasets",
        "model_cache",
        "model_pointers",
        "tool_cache",
        "tool_pointers",
        "tool_resources",
        "processed",
        "vg_index",
        "figures",
        "tables",
        "logs",
        "review",
    ):
        cfg.path(key).mkdir(parents=True, exist_ok=True)
    _apply_hf_endpoint(raw)
    _apply_hf_home(cfg)
    _apply_resource_paths(cfg)
    return cfg


def _apply_hf_home(cfg: "Config") -> None:
    """Keep HuggingFace downloads inside the project instead of ~/.cache.

    Model and dataset snapshots are tens of gigabytes. On hosts where the home
    directory sits on a small system disk and the project on a large data disk
    (the standard AutoDL layout: / vs the project data directory), the default cache fills
    the system disk and the download dies part-way. Routing HF_HOME next to the
    other project data keeps everything on the same volume as the project.

    An HF_HOME already set in the environment always wins; set `hf_home: null`
    in the config to opt out and use the HuggingFace default.
    """
    if os.environ.get("HF_HOME"):
        return
    value = cfg.raw.get("hf_home", "data/models/hf")
    if not value:
        return
    path = cfg.abspath(str(value))
    path.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(path)
    print(f"[config] HF_HOME={path}")


def _apply_hf_endpoint(raw: Dict[str, Any]) -> None:
    """Route HuggingFace traffic through a mirror when the hub is unreachable.

    `hf_endpoint` in the config is applied only if HF_ENDPOINT is not already set
    in the environment, so an explicit `export HF_ENDPOINT=...` always wins. This
    must run before transformers/huggingface_hub read the variable, i.e. before
    any model is loaded, which is why it lives in load_config().
    """
    endpoint = raw.get("hf_endpoint")
    if endpoint and not os.environ.get("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = str(endpoint)
        print(f"[config] HF_ENDPOINT={endpoint}")


def _apply_resource_paths(cfg: "Config") -> None:
    """Expose separated tool resources to libraries that rely on env paths."""
    os.environ.setdefault("HMQR_TOOL_CACHE", str(cfg.path("tool_cache")))
    os.environ.setdefault(
        "NLTK_DATA", str(cfg.path("tool_resources") / "nltk_data")
    )


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
