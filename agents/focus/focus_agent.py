"""
FocusAgent for the final pipeline: conditional FOCUS (position-calibrated received attention).

Uses exactly the code that produced the sensitivity analysis (focus_scoring / focus_fast):
exact one-layer attention from the frozen model, no truncation, baseline table estimated once on
the calibration split, token-budget selection.

Short inputs (fewer than `min_sentences` sentences) and inputs above `max_tokens` are returned whole
and flagged, so every note is processed and the share affected can be reported.
"""
import os
import sys
from typing import Any, Dict, List

import numpy as np
import torch
from nltk.tokenize import sent_tokenize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from focus_fast import bin_index_matrix, fast_sentence_scores, sentence_index  # noqa: E402
from focus_scoring import (  # noqa: E402
    BaselineTable, build_text_and_spans, capture_layer_inputs, layer_attention_chunks,
    select_by_token_budget,
)


def _head_chunk(n: int) -> int:
    return 4 if n <= 4096 else (2 if n <= 8192 else 1)


class FocusAgent:
    def __init__(self, model, tokenizer, layer: int, rho: float, tables_path: str,
                 min_count: int = 200, min_sentences: int = 8, max_tokens: int = 12000,
                 verbose: bool = False):
        self.model, self.tokenizer = model, tokenizer
        self.layer, self.rho = layer, rho
        self.min_sentences, self.max_tokens, self.verbose = min_sentences, max_tokens, verbose
        self.device = next(model.parameters()).device
        self.H = model.config.num_attention_heads
        blob = torch.load(tables_path)[layer]
        table = BaselineTable(self.H)
        table.total, table.count = blob["total"], blob["count"]
        table.finalize(min_count)
        self.table_b = table.b.to(self.device)

    def _whole(self, sentences: List[str], reason: str, n_tokens: int = 0) -> Dict[str, Any]:
        return {"sentences": sentences, "sentence_indices": list(range(len(sentences))),
                "sentence_scores": [float("nan")] * len(sentences), "fallback_used": True,
                "fallback_reason": reason, "total_tokens": n_tokens, "retained_tokens": n_tokens}

    @torch.no_grad()
    def score_sentences(self, sentences: List[str]):
        """Conditional FOCUS score of every sentence: (scores, tok_counts, n_tokens), or None if the input is too long."""
        info = build_text_and_spans(sentences, self.tokenizer, self.device)
        n, m = info["n"], len(sentences)
        if n > self.max_tokens:
            return None
        cap = capture_layer_inputs(self.model, info["ids"], self.layer)
        sid, endk = sentence_index(n, info["spans"], self.device)
        idx = bin_index_matrix(n, self.device)
        acc = torch.zeros(m, device=self.device)
        for h0, A in layer_attention_chunks(self.model, cap, self.layer, _head_chunk(n)):
            acc += fast_sentence_scores(A, h0, sid, endk, m, ["conditional"], self.table_b, idx)["conditional"].sum(0)
            del A
        del cap
        scores = (acc / self.H).cpu().numpy()
        tok_counts = np.array([e - s for s, e in info["spans"]])
        torch.cuda.empty_cache()
        return scores, tok_counts, n

    @torch.no_grad()
    def compress(self, document: str) -> Dict[str, Any]:
        document = document.strip()
        sentences = sent_tokenize(document) or [document]
        if len(sentences) < self.min_sentences:
            return self._whole(sentences, "short_input")
        res = self.score_sentences(sentences)
        if res is None:
            return self._whole(sentences, "too_long")
        scores, tok_counts, n = res

        chosen = select_by_token_budget(scores, tok_counts, self.rho)
        if not chosen:
            return self._whole(sentences, "empty_selection", n)
        return {"sentences": [sentences[i] for i in chosen], "sentence_indices": list(chosen),
                "sentence_scores": [float(scores[i]) for i in chosen], "fallback_used": False,
                "fallback_reason": "", "total_tokens": int(tok_counts.sum()),
                "retained_tokens": int(tok_counts[list(chosen)].sum())}
