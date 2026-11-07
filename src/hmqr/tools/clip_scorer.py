"""Frozen CLIP ViT-L/14-336 scorer (L2-normalised embeddings + cosine)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch


class ClipScorer:
    def __init__(
        self,
        hf_id: str,
        device: str = "cuda",
        dtype: str = "float16",
        cache_dir: Optional[str | Path] = None,
    ):
        from transformers import CLIPModel, CLIPProcessor

        self.device = device
        self.torch_dtype = getattr(torch, dtype)
        cache_kwargs = {"cache_dir": str(cache_dir)} if cache_dir is not None else {}
        self.processor = CLIPProcessor.from_pretrained(hf_id, **cache_kwargs)
        self.model = CLIPModel.from_pretrained(
            hf_id, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True, **cache_kwargs
        ).to(device)
        self.model.eval()
        self.n_image_calls = 0
        self.n_text_calls = 0

    @torch.no_grad()
    def encode_image(self, image) -> np.ndarray:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        inputs["pixel_values"] = inputs["pixel_values"].to(self.torch_dtype)
        feats = self.model.get_image_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        self.n_image_calls += 1
        return feats[0].float().cpu().numpy()

    @torch.no_grad()
    def encode_images(self, images: Sequence) -> np.ndarray:
        inputs = self.processor(images=list(images), return_tensors="pt").to(self.device)
        inputs["pixel_values"] = inputs["pixel_values"].to(self.torch_dtype)
        feats = self.model.get_image_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        self.n_image_calls += len(images)
        return feats.float().cpu().numpy()

    @torch.no_grad()
    def encode_text(self, text: str) -> np.ndarray:
        return self.encode_texts([text])[0]

    @torch.no_grad()
    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        inputs = self.processor(
            text=list(texts), return_tensors="pt", padding=True, truncation=True, max_length=77
        ).to(self.device)
        feats = self.model.get_text_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        self.n_text_calls += len(texts)
        return feats.float().cpu().numpy()

    @staticmethod
    def cos(a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
        return float(np.dot(a, b) / denom)
