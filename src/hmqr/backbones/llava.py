"""LLaVA-1.5 / LLaVA-1.6 (LLaVA-NeXT) wrappers, HF format.

Image-token layout: a single `<image>` placeholder in the prompt is expanded into
a contiguous block of patch tokens (576 for 1.5; variable for 1.6 due to anyres
tiling). We detect the block by comparing the forward sequence length with the
input length, which is robust across transformers versions.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from .base import VLMWrapper


class LlavaWrapper(VLMWrapper):
    name = "llava15"
    family = "llava"
    PROMPT_TMPL = "USER: <image>\n{q} ASSISTANT:"

    def load(self) -> None:
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        self._prepare_load()
        self.processor = AutoProcessor.from_pretrained(self.hf_id, **self._cache_kwargs)
        self.tokenizer = self.processor.tokenizer
        # eager attention: sdpa/flash cannot return attention weights, and
        # output_attentions=True (needed by T3) then either warns-and-falls-back
        # or raises depending on the transformers version.
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.hf_id,
            attn_implementation="eager",
            **self._load_kwargs,
        )
        self.model.eval()

    # ------------------------------------------------------------------ #
    def build_prompt(self, question: str) -> str:
        return self.PROMPT_TMPL.format(q=question)

    @torch.no_grad()
    def generate(
        self, image, prompt: str, max_new_tokens: int = 256,
        do_sample: bool = False, temperature: Optional[float] = None,
        top_p: Optional[float] = None, num_return_sequences: int = 1,
    ) -> Dict[str, Any]:
        full = self.build_prompt(prompt)
        inputs = self.processor(images=image, text=full, return_tensors="pt").to(
            self.device, self.torch_dtype
        )
        prompt_len = inputs["input_ids"].shape[1]
        sampling = {"do_sample": do_sample}
        if do_sample:
            sampling.update(
                temperature=float(temperature if temperature is not None else 1.0),
                top_p=float(top_p if top_p is not None else 1.0),
            )
        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            output_scores=True,
            return_dict_in_generate=True,
            num_return_sequences=int(num_return_sequences),
            **sampling,
        )
        continuations = out.sequences[:, prompt_len:]
        texts = [
            value.strip() for value in self.processor.tokenizer.batch_decode(
                continuations, skip_special_tokens=True
            )
        ]
        gen_ids = continuations[0]
        text = texts[0]
        tokens = self.processor.tokenizer.convert_ids_to_tokens(gen_ids.tolist())
        logprobs = self._logprobs_from_scores(out.scores, out.sequences, prompt_len)
        entropies = self._entropies_from_scores(out.scores)
        return {"text": text, "texts": texts, "tokens": tokens, "token_logprobs": logprobs,
                "token_entropies": entropies}

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def forward_states(
        self, image, prompt: str, target_text: str, keywords: Sequence[str] = (),
        need_attentions: bool = True,
    ) -> Dict[str, Any]:
        full = self.build_prompt(prompt) + " " + target_text
        inputs = self.processor(images=image, text=full, return_tensors="pt").to(
            self.device, self.torch_dtype
        )
        # Attention analysis uses a memory-safe two-pass route.  The first pass
        # determines the expanded multimodal sequence length.  The second pass
        # captures and frees one key-token row per decoder layer via hooks.
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

        input_ids = inputs["input_ids"]
        tok_strings = self._token_strings(input_ids)
        image_token_id = getattr(self.model.config, "image_token_index", 32000)
        ids = input_ids[0].tolist()
        image_positions = [i for i, t in enumerate(ids) if t == image_token_id]

        spans = self._find_keyword_spans(tok_strings, keywords)

        if len(ids) == seq_len and image_positions:
            # processor already expanded the image tokens in input_ids
            token_types = ["image" if t == image_token_id else "text" for t in ids]
        elif image_positions:
            # model expands internally: one placeholder -> n_image tokens
            slot = image_positions[0]
            n_image = seq_len - len(ids) + 1
            base_types = ["text"] * len(ids)
            token_types = self._expand_token_types(base_types, seq_len, slot, n_image)
            spans = self._shift_spans(spans, slot, n_image - 1)
        else:
            token_types = ["text"] * seq_len

        # after the span shift, so the key position refers to the final layout
        key_pos = self._key_position(spans, keywords, seq_len)
        if need_attentions:
            modules = [layer.self_attn for layer in self.model.language_model.model.layers]
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


class LlavaNextWrapper(LlavaWrapper):
    name = "llava16"
    family = "llava_next"

    def load(self) -> None:
        from transformers import AutoProcessor, LlavaNextForConditionalGeneration

        self._prepare_load()
        self.processor = AutoProcessor.from_pretrained(self.hf_id, **self._cache_kwargs)
        self.tokenizer = self.processor.tokenizer
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            self.hf_id,
            attn_implementation="eager",
            **self._load_kwargs,
        )
        self.model.eval()
