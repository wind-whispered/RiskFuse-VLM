"""Qwen-VL-Chat wrapper (trust_remote_code).

Image-token layout: the prompt encodes an image as
    "Picture 1: <img>{path}</img>\n"
and the tokenizer emits `image_start_id` .. `image_start_id+1` markers with a
fixed-length block of 256 visual tokens in between. We locate the block by the
`<img>` / `</img>` marker ids exposed by the tokenizer config.
"""
from __future__ import annotations

import copy
import importlib
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from .base import VLMWrapper


class QwenVLWrapper(VLMWrapper):
    name = "qwenvl"
    family = "qwenvl"
    verification_adapter_version = 2
    generation_adapter_version = 1
    generation_prompt_format = 'checkpoint_chatml'

    def load(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._prepare_load()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hf_id, trust_remote_code=True, **self._cache_kwargs
        )
        # Qwen-VL is the largest backbone (10 shards, ~19 GB) and loads last, so
        # it is the one that dies first when host RAM is tight -- see the
        # device_map note in VLMWrapper._load_kwargs.
        self.model = AutoModelForCausalLM.from_pretrained(
            self.hf_id,
            trust_remote_code=True,
            fp16=self.torch_dtype==torch.float16,
            bf16=self.torch_dtype==torch.bfloat16,
            fp32=self.torch_dtype==torch.float32,
            **self._load_kwargs,
        )
        self.model.eval()
        self.processor = None

    # ------------------------------------------------------------------ #
    def _build_query(self, image_path: str, question: str) -> str:
        return self.tokenizer.from_list_format(
            [{"image": str(image_path)}, {"text": question}]
        )

    @staticmethod
    def _image_path(image) -> str:
        """Qwen-VL consumes a path/URL; accept a PIL image with `.filename`."""
        if isinstance(image, str):
            return image
        path = getattr(image, "filename", None)
        if not path:
            raise ValueError("Qwen-VL requires an image path; pass a str or a PIL image "
                             "opened from disk (image.filename must be set).")
        return str(path)

    @torch.no_grad()
    def generate(
        self, image, prompt: str, max_new_tokens: int = 256,
        do_sample: bool = False, temperature: Optional[float] = None,
        top_p: Optional[float] = None, num_return_sequences: int = 1,
    ) -> Dict[str, Any]:
        return self.generate_verification(image,prompt,max_new_tokens=max_new_tokens,
                                         do_sample=do_sample,temperature=temperature,top_p=top_p,
                                         num_return_sequences=num_return_sequences)

    @torch.no_grad()
    def generate_verification(
        self, image, prompt: str, max_new_tokens: int = 8,
        do_sample: bool = False, temperature: Optional[float] = None,
        top_p: Optional[float] = None, num_return_sequences: int = 1,
    ) -> Dict[str, Any]:
        """Run self-verification through Qwen-VL-Chat's official ChatML path.

        Raw tokenisation of ``from_list_format`` omits the assistant turn. Qwen
        then often emits ``<|im_end|>``/EOS immediately, producing empty yes/no
        answers. This mirrors ``model.chat`` while retaining generation scores
        required by l_p and the Reefknot baseline.
        """
        query = self._build_query(self._image_path(image), prompt)
        remote = importlib.import_module(self.model.__class__.__module__)
        generation_config = copy.deepcopy(self.model.generation_config)
        chat_format = str(getattr(generation_config, "chat_format", "chatml"))
        if chat_format != "chatml":
            raise RuntimeError(f"Qwen verification requires chatml, got {chat_format!r}")
        max_window_size = int(getattr(generation_config, "max_window_size", 6144))
        raw_text, context_tokens = remote.make_context(
            self.tokenizer,
            query,
            history=[],
            system="You are a helpful assistant.",
            max_window_size=max_window_size,
            chat_format=chat_format,
        )
        stop_words_ids = remote.get_stop_words_ids(chat_format, self.tokenizer)
        input_ids = torch.tensor([context_tokens], dtype=torch.long, device=self.device)
        sampling: Dict[str, Any] = {"do_sample": bool(do_sample)}
        if do_sample:
            sampling.update(
                temperature=float(temperature if temperature is not None else 1.0),
                top_p=float(top_p if top_p is not None else 1.0),
            )
        out = self.model.generate(
            input_ids,
            stop_words_ids=stop_words_ids,
            generation_config=generation_config,
            max_new_tokens=int(max_new_tokens),
            output_scores=True,
            return_dict_in_generate=True,
            num_return_sequences=int(num_return_sequences),
            **sampling,
        )
        prompt_len = int(input_ids.shape[1])
        continuations = out.sequences[:, prompt_len:]
        texts = [
            remote.decode_tokens(
                sequence,
                self.tokenizer,
                raw_text_len=len(raw_text),
                context_length=len(context_tokens),
                chat_format=chat_format,
                verbose=False,
                errors="replace",
            ).strip()
            for sequence in out.sequences
        ]
        gen_ids = continuations[0]
        tokens = self.tokenizer.convert_ids_to_tokens(gen_ids.tolist())
        return {
            "text": texts[0],
            "texts": texts,
            "tokens": tokens,
            "token_logprobs": self._logprobs_from_scores(
                out.scores, out.sequences, prompt_len
            ),
            "token_entropies": self._entropies_from_scores(out.scores),
        }

    # ------------------------------------------------------------------ #
    def _visual_positions(self, ids: List[int], seq_len: int) -> List[str]:
        start_id = int(getattr(self.model.config, "visual", {}).get("image_start_id", 151857)) \
            if isinstance(getattr(self.model.config, "visual", None), dict) else 151857
        end_id = start_id + 1
        types = ["text"] * seq_len
        inside = False
        for i, t in enumerate(ids[:seq_len]):
            if t == start_id:
                inside = True
                continue
            if t == end_id:
                inside = False
                continue
            if inside:
                types[i] = "image"
        return types

    @torch.no_grad()
    def forward_states(
        self, image, prompt: str, target_text: str, keywords: Sequence[str] = (),
        need_attentions: bool = True,
    ) -> Dict[str, Any]:
        query = self._build_query(self._image_path(image), prompt) + " " + target_text
        inputs = self.tokenizer(query, return_tensors="pt").to(self.device)
        outputs = self.model(
            **inputs,
            output_hidden_states=not need_attentions,
            output_attentions=False,
            return_dict=True,
        )
        if need_attentions:
            seq_len = int(outputs.logits.shape[1])
            hidden = np.empty((0, seq_len, 0), dtype=np.float32)
        else:
            hidden = self._stack_hidden(outputs)
            seq_len = hidden.shape[1]

        ids = inputs["input_ids"][0].tolist()
        tok_strings = self._token_strings(inputs["input_ids"])
        spans = self._find_keyword_spans(tok_strings, keywords)
        token_types = self._visual_positions(ids, seq_len)

        key_pos = self._key_position(spans, keywords, seq_len)
        if need_attentions:
            blocks = self.model.transformer.h
            modules = [
                getattr(block, "attn", None) or getattr(block, "self_attn")
                for block in blocks
            ]
            attn = self._capture_attention_rows(
                modules,
                lambda: self.model(
                    **inputs,
                    output_hidden_states=False,
                    output_attentions=True,
                    return_dict=True,
                ),
                key_pos,
            )
        else:
            attn = None

        self._last_token_types = token_types
        return {
            "hidden_states": hidden,
            "attentions": attn,
            "attn_key_pos": key_pos,
            "token_types": token_types,
            "text_token_spans": spans,
        }
