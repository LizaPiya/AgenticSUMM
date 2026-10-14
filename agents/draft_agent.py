import os
import sys
from typing import List

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "focus"))
from prompts import DRAFT_MAX_NEW_TOKENS, build_draft_prompt  # noqa: E402


class DraftAgent:
    """Greedy draft generation. Dataset-specific prompt (hospital-stay summary or SOAP summary); only the
    newly generated tokens are returned, so a 'Summary:' inside the note cannot cut the output."""

    def __init__(self, model, tokenizer, dataset: str, max_new_tokens: int = DRAFT_MAX_NEW_TOKENS):
        self.model, self.tokenizer, self.dataset, self.max_new_tokens = model.eval(), tokenizer, dataset, max_new_tokens
        for p in self.model.parameters():
            p.requires_grad = False
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def generate(self, compressed_sentences: List[str]) -> str:
        if not compressed_sentences:
            return ""
        prompt = build_draft_prompt(self.dataset, " ".join(compressed_sentences))
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                                  eos_token_id=self.tokenizer.eos_token_id, pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(out[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip()
