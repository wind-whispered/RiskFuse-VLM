"""InstructBLIP-Vicuna-7B wrapper.

Image-token layout: the Q-Former emits `num_query_tokens` (32) query embeddings
that are PREPENDED to the language-model sequence. Therefore the first
`num_query_tokens` positions of the LM sequence are the visual tokens.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch

from .base import VLMWrapper


class InstructBlipWrapper(VLMWrapper):
    name = "instructblip"
    family = "instructblip"

    def load(self) -> None:
        from transformers import InstructBlipForConditionalGeneration, InstructBlipProcessor

        self._prepare_load()
        self.processor = InstructBlipProcessor.from_pretrained(
            self.hf_id, **self._cache_kwargs
        )
        self.tokenizer = self.processor.tokenizer
        # eager attention so that output_attentions=True (T3) is supported
        self.model = InstructBlipForConditionalGeneration.from_pretrained(
            self.hf_id,
            attn_implementation="eager",
            **self._load_kwargs,
        )
        self.model.eval()

    def _num_query_tokens(self) -> int:
        return int(getattr(self.model.config, "num_query_tokens", 32))

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def generate(
        self, image, prompt: str, max_new_tokens: int = 256,
        do_sample: bool = False, temperature: Optional[float] = None,
        top_p: Optional[float] = None, num_return_sequences: int = 1,
        min_new_tokens: int = 0,
    ) -> Dict[str, Any]:
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(
            self.device
        )
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.torch_dtype)
        sampling = {"do_sample": do_sample}
        if do_sample:
            sampling.update(
                temperature=float(temperature if temperature is not None else 1.0),
                top_p=float(top_p if top_p is not None else 1.0),
            )
        generation_length = {"max_new_tokens": max_new_tokens}
        if int(min_new_tokens) > 0:
            generation_length["min_new_tokens"] = int(min_new_tokens)
        out = self.model.generate(
            **inputs,
            **generation_length,
            output_scores=True,
            return_dict_in_generate=True,
            num_return_sequences=int(num_return_sequences),
            **sampling,
        )
        seq = out.sequences
        # Decoder-only InstructBLIP checkpoints echo the textual prompt in
        # ``sequences``. Older wrapper code decoded that prompt as the answer,
        # making every yes/no parse fail and every sampled semantic entropy 0.
        continuation_start = self._continuation_start(seq, inputs["input_ids"])
        continuations = seq[:, continuation_start:]
        texts = [
            value.strip() for value in self.processor.tokenizer.batch_decode(
                continuations, skip_special_tokens=True
            )
        ]
        text = texts[0]
        tokens = self.processor.tokenizer.convert_ids_to_tokens(
            continuations[0].tolist()
        )
        logprobs = self._logprobs_from_scores(
            out.scores, seq, continuation_start
        )
        entropies = self._entropies_from_scores(out.scores)
        return {"text": text, "texts": texts, "tokens": tokens, "token_logprobs": logprobs,
                "token_entropies": entropies}

    def generate_revision(
        self, image, prompt: str, max_new_tokens: int = 32,
        do_sample: bool = False, temperature: Optional[float] = None,
        top_p: Optional[float] = None, num_return_sequences: int = 1,
    ) -> Dict[str, Any]:
        """Revision-only decoding that prevents an immediate empty EOS answer."""
        return self.generate(
            image,
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=num_return_sequences,
            min_new_tokens=1,
        )

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def forward_states(
        self, image, prompt: str, target_text: str, keywords: Sequence[str] = (),
        need_attentions: bool = True,
    ) -> Dict[str, Any]:
        full = f"{prompt} {target_text}"
        inputs = self.processor(images=image, text=full, return_tensors="pt").to(self.device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.torch_dtype)
        labels = inputs["input_ids"].clone()
        outputs = self.model(
            **inputs,
            labels=labels,
            output_hidden_states=not need_attentions,
            output_attentions=False,
            return_dict=True,
        )
        lm_out = outputs.language_model_outputs
        if need_attentions:
            seq_len = int(lm_out.logits.shape[1])
            hidden = np.empty((0, seq_len, 0), dtype=np.float32)
        else:
            hidden = torch.stack(lm_out.hidden_states, dim=0)[:, 0].float().cpu().numpy()
            seq_len = hidden.shape[1]

        n_query = self._num_query_tokens()
        tok_strings = self._token_strings(inputs["input_ids"])
        spans = self._find_keyword_spans(tok_strings, keywords)
        # query tokens are prepended -> all text spans shift right by n_query
        spans = {k: (s + n_query, e + n_query) for k, (s, e) in spans.items()}

        token_types = ["image"] * min(n_query, seq_len) + ["text"] * max(0, seq_len - n_query)

        key_pos = self._key_position(spans, keywords, seq_len)
        if need_attentions:
            modules = [layer.self_attn for layer in self.model.language_model.model.layers]
            attn = self._capture_attention_rows(
                modules,
                lambda: self.model(
                    **inputs,
                    labels=labels,
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
