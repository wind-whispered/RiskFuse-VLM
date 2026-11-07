"""Abstract VLM wrapper.

Access levels
-------------
L0 (black box) : `generate` only -- used by claim extraction (T1) and revision (T7).
L1 (white box) : `forward_states` + `visual_token_mask` -- used by the layer-wise
                 probe (T2) and the attention analysis (T3).

All subclasses must return, for `forward_states`:
    hidden_states : np.ndarray [num_layers + 1, seq_len, hidden]
    attentions    : np.ndarray [num_layers, num_heads, seq_len] -- the attention
                    landing on the key position only, or None when the caller
                    passed need_attentions=False
    attn_key_pos  : int, the position `attentions` was taken at
    token_types   : list[str], each in {"image", "text"}, length == seq_len
    text_token_spans : dict[str, tuple[int, int]]  keyword -> (start, end) in seq

Full square attention maps are never retained across layers. LLaVA-1.6 expands
an image to roughly 2900 tokens, which would otherwise keep about 21 GB alive at
once even though the only consumer (mechanism/attention_analysis.py) reads one
query row per layer.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


class VLMWrapper(ABC):
    """Unified interface over the four VLM backbones."""

    name: str = "base"
    family: str = "base"
    verification_adapter_version: int = 1

    def __init__(
        self,
        hf_id: str,
        device: str = "cuda",
        dtype: str = "float16",
        cache_dir: Optional[str | Path] = None,
    ):
        import torch

        self.hf_id = hf_id
        self.cache_dir = str(cache_dir) if cache_dir is not None else None
        self.device = device
        self.torch_dtype = getattr(torch, dtype)
        self.model = None
        self.processor = None
        self.tokenizer = None
        self._last_token_types: Optional[List[str]] = None

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    @abstractmethod
    def load(self) -> None:
        """Load weights onto the device (fp16)."""

    def unload(self) -> None:
        import gc

        import torch

        from ..cache_control import reclaim_hf_file_cache

        self.model = None
        self.processor = None
        self.tokenizer = None
        # a backbone is several GB; collect before the next one starts loading
        # rather than leaving it to the next allocation to trigger a collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        reclaim_hf_file_cache()

    @staticmethod
    def _prepare_load() -> None:
        """Make cgroup room before reading another multi-GB checkpoint."""
        from ..cache_control import reclaim_hf_file_cache

        reclaim_hf_file_cache()

    @property
    def _load_kwargs(self) -> Dict[str, Any]:
        """Shared from_pretrained arguments.

        `device_map` streams each shard straight onto the GPU. Without it,
        from_pretrained materialises the whole model in host RAM and `.to(cuda)`
        copies it over afterwards, so peak host usage is the full model size.
        The AutoDL SIGKILL investigated on 2026-08-18 was instead caused by
        roughly 70 GiB of clean checkpoint pages charged to the 72 GiB cgroup;
        `_prepare_load` / `unload` reclaim those pages explicitly.
        """
        kwargs = {
            "torch_dtype": self.torch_dtype,
            "low_cpu_mem_usage": True,
            "device_map": {"": self.device},
        }
        if self.cache_dir is not None:
            kwargs["cache_dir"] = self.cache_dir
        kwargs.update(getattr(self, 'runtime_load_options', {}))
        return kwargs

    @property
    def _cache_kwargs(self) -> Dict[str, Any]:
        """Cache routing shared by processors and tokenizers."""
        return {"cache_dir": self.cache_dir} if self.cache_dir is not None else {}

    # ------------------------------------------------------------------ #
    # L0 interface
    # ------------------------------------------------------------------ #
    @abstractmethod
    def generate(
        self,
        image,
        prompt: str,
        max_new_tokens: int = 256,
        do_sample: bool = False,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        num_return_sequences: int = 1,
    ) -> Dict[str, Any]:
        """Return text(s) plus first-sequence token statistics."""

    def generate_verification(
        self,
        image,
        prompt: str,
        max_new_tokens: int = 8,
        do_sample: bool = False,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        num_return_sequences: int = 1,
    ) -> Dict[str, Any]:
        """Generate an answer in the backbone's verification dialogue format.

        Most wrappers use the same formatting for free-form generation and
        self-verification. Chat-native backbones can override this method
        without changing the caption/claim-generation contract used upstream.
        """
        return self.generate(
            image,
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=num_return_sequences,
        )

    # ------------------------------------------------------------------ #
    # L1 interface
    # ------------------------------------------------------------------ #
    @abstractmethod
    def forward_states(
        self, image, prompt: str, target_text: str, keywords: Sequence[str] = (),
        need_attentions: bool = True,
    ) -> Dict[str, Any]:
        """Teacher-forcing forward pass with hidden states and attentions."""

    def visual_token_mask(self) -> np.ndarray:
        """Boolean mask over the sequence of the last `forward_states` call."""
        if self._last_token_types is None:
            raise RuntimeError("call forward_states() before visual_token_mask()")
        return np.array([t == "image" for t in self._last_token_types], dtype=bool)

    # ------------------------------------------------------------------ #
    # shared helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _stack_hidden(outputs) -> np.ndarray:
        import torch

        hs = torch.stack(outputs.hidden_states, dim=0)  # [L+1, B, S, H]
        return hs[:, 0].float().cpu().numpy()

    @staticmethod
    def _key_position(spans: Dict[str, Tuple[int, int]], keywords: Sequence[str],
                      seq_len: int) -> int:
        """Position whose attention row the analysis reads: end of the first keyword."""
        span = spans.get(str(keywords[0])) if keywords else None
        pos = int(span[1]) - 1 if span else seq_len - 1
        return max(0, min(pos, seq_len - 1))

    @staticmethod
    def _attn_key_row(attentions, key_pos: int) -> np.ndarray:
        """Attention landing on `key_pos`, per layer and head -> [L, heads, S].

        This helper is for models whose outputs have already been returned.  The
        memory-safe full-run route is `_capture_attention_rows`, which discards
        each square map before the parent decoder retains it.
        """
        import torch

        rows = [a[0, :, key_pos, :].float().cpu() for a in attentions]
        return torch.stack(rows, dim=0).numpy()

    @staticmethod
    def _capture_attention_rows(modules, forward_fn, key_pos: int) -> np.ndarray:
        """Capture one query row per attention layer without retaining L square maps.

        HuggingFace's model-level ``output_attentions=True`` normally appends the
        complete ``[heads, sequence, sequence]`` tensor from every decoder layer
        to the final output.  With LLaVA-NeXT any-resolution images this can hold
        more than 20 GiB at once.  Forward hooks copy only the requested query row
        to CPU and replace the full tensor with ``None`` before the parent decoder
        can retain it.  Peak GPU memory is therefore one layer's map, not all maps.
        """
        import torch

        captured: Dict[int, "torch.Tensor"] = {}
        handles = []

        def make_hook(layer_index: int):
            def hook(_module, _inputs, output):
                if not isinstance(output, (tuple, list)):
                    return output
                values = list(output)
                for position, value in enumerate(values):
                    if (
                        torch.is_tensor(value)
                        and value.ndim == 4
                        and value.shape[0] == 1
                        and key_pos < value.shape[-2]
                    ):
                        captured[layer_index] = value[0, :, key_pos, :].float().cpu()
                        values[position] = None
                        break
                return tuple(values) if isinstance(output, tuple) else values

            return hook

        for index, module in enumerate(modules):
            handles.append(module.register_forward_hook(make_hook(index)))
        try:
            forward_fn()
        finally:
            for handle in handles:
                handle.remove()

        missing = [index for index in range(len(modules)) if index not in captured]
        if missing:
            raise RuntimeError(f"attention row hook missed decoder layers: {missing}")
        return torch.stack([captured[index] for index in range(len(modules))], dim=0).numpy()

    def _token_strings(self, input_ids: "torch.Tensor") -> List[str]:
        tok = self.tokenizer if self.tokenizer is not None else self.processor.tokenizer
        return tok.convert_ids_to_tokens(input_ids[0].tolist())

    @staticmethod
    def _clean_piece(piece: Any) -> str:
        # Qwen-VL's remote tokenizer returns raw bytes for some byte-level
        # tokens.  str.replace with a text pattern then raises
        # "a bytes-like object is required, not 'str'" in white-box passes.
        if isinstance(piece, bytes):
            piece = piece.decode("utf-8", errors="replace")
        elif not isinstance(piece, str):
            piece = str(piece)
        return (
            piece.replace("▁", "")  # sentencepiece underline
            .replace("Ġ", "")            # byte-level BPE space
            .replace("</w>", "")
            .strip()
            .lower()
        )

    def _find_keyword_spans(
        self, token_strings: Sequence[str], keywords: Sequence[str]
    ) -> Dict[str, Tuple[int, int]]:
        """Locate each keyword's token span by greedy left-to-right matching."""
        cleaned = [self._clean_piece(t) for t in token_strings]
        spans: Dict[str, Tuple[int, int]] = {}
        for kw in keywords:
            if not kw:
                continue
            target = str(kw).lower().replace("_", " ").replace(" ", "")
            if not target:
                continue
            found = None
            for start in range(len(cleaned)):
                if not cleaned[start] or not target.startswith(cleaned[start]):
                    continue
                buf = ""
                for end in range(start, min(start + 12, len(cleaned))):
                    buf += cleaned[end]
                    if buf == target:
                        found = (start, end + 1)
                        break
                    if not target.startswith(buf):
                        break
                if found:
                    break
            if found is None:
                # fall back: last non-special token (keeps the pipeline running)
                found = (max(0, len(cleaned) - 2), max(1, len(cleaned) - 1))
            spans[str(kw)] = found
        return spans

    @staticmethod
    def _expand_token_types(
        base_types: List[str], seq_len: int, image_slot: int, n_image_tokens: int
    ) -> List[str]:
        """Insert `n_image_tokens` image entries where a single placeholder sat."""
        out = base_types[:image_slot] + ["image"] * n_image_tokens + base_types[image_slot + 1 :]
        if len(out) < seq_len:
            out += ["text"] * (seq_len - len(out))
        return out[:seq_len]

    @staticmethod
    def _shift_spans(
        spans: Dict[str, Tuple[int, int]], image_slot: int, delta: int
    ) -> Dict[str, Tuple[int, int]]:
        """Shift text spans that sit after the expanded image block."""
        shifted = {}
        for k, (s, e) in spans.items():
            if s > image_slot:
                shifted[k] = (s + delta, e + delta)
            else:
                shifted[k] = (s, e)
        return shifted

    @staticmethod
    def _logprobs_from_scores(scores, sequences, prompt_len: int) -> List[float]:
        """Per-generated-token log probability."""
        import torch

        out: List[float] = []
        gen_ids = sequences[0][prompt_len:]
        for step, score in enumerate(scores):
            if step >= len(gen_ids):
                break
            logp = torch.log_softmax(score[0].float(), dim=-1)
            out.append(float(logp[gen_ids[step]].item()))
        return out

    @staticmethod
    def _continuation_start(sequences, input_ids) -> int:
        """Return the echoed prompt length, or zero for continuation-only APIs."""
        import torch

        prompt_len = int(input_ids.shape[1])
        if sequences.shape[1] < prompt_len:
            return 0
        return prompt_len if torch.equal(sequences[0, :prompt_len], input_ids[0]) else 0

    @staticmethod
    def _entropies_from_scores(scores) -> List[float]:
        """Normalised predictive entropy for each generated-token distribution."""
        import math

        import torch

        out: List[float] = []
        for score in scores:
            logits = score[0].float()
            logp = torch.log_softmax(logits, dim=-1)
            p = torch.exp(logp)
            # Generation processors commonly mask tokens with -inf. The
            # mathematical limit 0*log(0)=0 must be applied explicitly;
            # raw floating-point multiplication would produce NaN.
            terms = torch.where(p > 0, p * logp, torch.zeros_like(p))
            entropy = float((-terms.sum()).item())
            out.append(entropy / max(math.log(int(logits.numel())), 1e-12))
        return out
