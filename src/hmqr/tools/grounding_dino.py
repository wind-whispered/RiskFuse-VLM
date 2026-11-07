"""Frozen Grounding DINO-Base open-vocabulary detector.

detect(image, phrase) -> [{"box": [x1, y1, x2, y2], "score": float}, ...]
Boxes are absolute pixel coordinates, sorted by score (highest first).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


class GroundingDino:
    def __init__(
        self,
        hf_id: str,
        device: str = "cuda",
        dtype: str = "float16",
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
        cache_dir: Optional[str | Path] = None,
    ):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.device = device
        self.torch_dtype = getattr(torch, dtype)
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        # Weights stay in float32 even when the run is configured for fp16.
        # Grounding DINO's text enhancer adds sine position embeddings that
        # transformers builds in float32; with half weights the sum promotes to
        # float32 and the next Linear dies with "mat1 and mat2 must have the same
        # dtype, but got Float and Half". Autocast gives the speed back without
        # the mismatch, because it casts both operands of every matmul together.
        self._autocast = (
            self.torch_dtype in (torch.float16, torch.bfloat16)
            and str(device).startswith("cuda")
        )
        cache_kwargs = {"cache_dir": str(cache_dir)} if cache_dir is not None else {}
        self.processor = AutoProcessor.from_pretrained(hf_id, **cache_kwargs)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            hf_id, torch_dtype=torch.float32, low_cpu_mem_usage=True, **cache_kwargs
        ).to(device)
        self.model.eval()
        self.n_forward = 0
        self._cache: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}

    @staticmethod
    def _norm_phrase(phrase: str) -> str:
        p = str(phrase).strip().lower().replace("_", " ")
        return p if p.endswith(".") else p + "."

    @torch.no_grad()
    def detect(self, image, phrase: str, cache_key: Optional[str] = None) -> List[Dict[str, Any]]:
        key = (cache_key or "", self._norm_phrase(phrase))
        if cache_key is not None and key in self._cache:
            return self._cache[key]

        inputs = self.processor(
            images=image, text=self._norm_phrase(phrase), return_tensors="pt"
        ).to(self.device)
        if self._autocast:
            with torch.autocast(device_type="cuda", dtype=self.torch_dtype):
                outputs = self.model(**inputs)
        else:
            outputs = self.model(**inputs)
        self.n_forward += 1
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[image.size[::-1]],  # (h, w)
        )[0]
        dets: List[Dict[str, Any]] = []
        for box, score in zip(results["boxes"], results["scores"]):
            dets.append(
                {"box": [float(v) for v in box.tolist()], "score": float(score.item())}
            )
        dets.sort(key=lambda d: d["score"], reverse=True)
        if cache_key is not None:
            self._cache[key] = dets
        return dets

    def best_box(self, image, phrase: str, cache_key: Optional[str] = None) -> Optional[List[float]]:
        dets = self.detect(image, phrase, cache_key=cache_key)
        return dets[0]["box"] if dets else None

    @staticmethod
    def crop(image, box: Sequence[float], pad: float = 0.05):
        w, h = image.size
        x1, y1, x2, y2 = box
        bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
        x1 = max(0, int(x1 - pad * bw))
        y1 = max(0, int(y1 - pad * bh))
        x2 = min(w, int(x2 + pad * bw))
        y2 = min(h, int(y2 + pad * bh))
        if x2 <= x1 or y2 <= y1:
            return image
        return image.crop((x1, y1, x2, y2))


# geometry helpers live in a torch-free module so the analysis code can import them
from .geometry import (  # noqa: E402,F401
    box_area,
    box_intersection_area,
    box_to_patch_mask,
    centroid,
)
