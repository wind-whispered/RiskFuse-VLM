"""Backbone registry: build a VLMWrapper by config name.

The concrete wrappers import torch and transformers, so they are imported lazily
here. That keeps the analysis and figure modules importable on a machine without
a GPU stack (see tests/smoke_test.py).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Type

from ..config import Config

if TYPE_CHECKING:  # pragma: no cover
    from .base import VLMWrapper

_FAMILY_TO_CLASS = {
    "llava": ("hmqr.backbones.llava", "LlavaWrapper"),
    "llava_next": ("hmqr.backbones.llava", "LlavaNextWrapper"),
    "instructblip": ("hmqr.backbones.instructblip", "InstructBlipWrapper"),
    "qwenvl": ("hmqr.backbones.qwenvl", "QwenVLWrapper"),
}

_VERIFICATION_ADAPTER_VERSION = {
    "llava": 1,
    "llava_next": 1,
    "instructblip": 1,
    "qwenvl": 2,
}


def _resolve(family: str) -> Type["VLMWrapper"]:
    import importlib

    if family not in _FAMILY_TO_CLASS:
        raise KeyError(
            f"unknown backbone family '{family}'. Known: {sorted(_FAMILY_TO_CLASS)}"
        )
    module_name, class_name = _FAMILY_TO_CLASS[family]
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def build_backbone(cfg: Config, name: str, load: bool = True) -> "VLMWrapper":
    bc = cfg.backbone_cfg(name)
    cls = _resolve(bc["family"])
    wrapper = cls(
        hf_id=cfg.resource_id(bc["hf_id"], bc.get('local_path')),
        device=cfg.get("device", "cuda"),
        dtype=cfg.get("dtype", "float16"),
        cache_dir=cfg.path("model_cache"),
    )
    wrapper.name = name
    wrapper.canonical_hf_id = bc['hf_id']
    if cfg.get('local_execution', {}).get('enabled', False):
        options = dict(cfg['local_execution'].get('model_loading', {}))
        options.update(cfg['local_execution'].get('model_loading_by_backbone',{}).get(name,{}))
        if options:
            offload = cfg.abspath(cfg['local_execution']['offload_directory'])
            offload.mkdir(parents=True, exist_ok=True)
            options['offload_folder'] = str(offload)
        wrapper.runtime_load_options = options
    if load:
        wrapper.load()
    return wrapper


def verification_adapter_version(cfg: Config, name: str) -> int:
    """Return the expected verification adapter without importing model code."""
    backbones = cfg.get("backbones", {})
    family = str(
        backbones.get(name, {}).get(
            "family", "qwenvl" if "qwen" in str(name).lower() else "llava"
        )
    )
    return int(_VERIFICATION_ADAPTER_VERSION.get(family, 1))


__all__ = ["build_backbone", "verification_adapter_version"]
