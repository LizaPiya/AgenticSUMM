"""
HallucinationDetectorAgent: AURA grounding and span-level entailment.

AURA (attention-to-underlying-reference alignment) is computed post hoc on the finished summary:
the summary is appended to the draft prompt (instruction + full source note), the frozen model
runs one forward pass, and for every summary token the fraction of its attention that falls on
the source-note tokens is measured at one attention layer, averaged over heads.

Properties:
  * no skipping on long notes and no truncation: only the summary rows of one layer's attention are computed
    (exact, from the layer's q/k), so memory is small for any length;
  * the start token (BOS), an attention sink, is excluded from numerator and denominator;
  * the scored prompt is the draft prompt (same instruction text, dataset-specific);
  * AURA and spans are recomputed for every summary version (no caching across rounds);
  * a span-level semantic check with a short answer budget (the verdict is the first words).
"""
import os
import sys
from typing import Any, Dict, List

import torch
from nltk.tokenize import sent_tokenize

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "focus"))
from focus_scoring import capture_layer_inputs  # noqa: E402
from prompts import build_draft_prompt  # noqa: E402


class ShortAnswerEntailmentJudge:
    """
    Span-level entailment check with a fixed instruction, applied through the chat template.
    The verdict is read from the model's first answer token (SUPPORTED vs NOT), so no text is generated.
    The note is encoded once (its key/value cache is reused for every sentence and every round of the same
    note); each sentence only adds its own tokens.
    """

    INSTR = "Verify if this sentence is supported by the document."

    def __init__(self, model, tokenizer, device: str = "cuda"):
        self.model, self.tokenizer = model.eval(), tokenizer
        self.device = next(model.parameters()).device
        self.id_sup = tokenizer.encode("SUPPORTED", add_special_tokens=False)[0]
        self.id_not = tokenizer.encode("NOT", add_special_tokens=False)[0]
        assert self.id_sup != self.id_not
        self._doc_key = None
        self._past = None
        self._prefix_len = 0
        self._cache = {}

    def reset(self):
        self._doc_key, self._past, self._prefix_len = None, None, 0
        self._cache.clear()
        torch.cuda.empty_cache()

    def _text(self, document: str, span: str):
        msgs = [{"role": "user", "content":
                 f"{self.INSTR}\n\nDocument:\n{document}\n\nSentence:\n{span}\n\nAnswer ONLY: SUPPORTED or NOT SUPPORTED"}]
        full = self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        cut = full.rfind("Sentence:\n") + len("Sentence:\n")
        return full[:cut], full[cut:]

    @torch.no_grad()
    def judge(self, document: str, span: str) -> Dict[str, Any]:
        key = (hash(document), hash(span))
        if key in self._cache:
            return self._cache[key]
        prefix, suffix = self._text(document, span)
        if self._doc_key != hash(document):
            ids = self.tokenizer(prefix, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
            self._past = self.model(input_ids=ids, use_cache=True).past_key_values
            self._prefix_len, self._doc_key = ids.shape[1], hash(document)
        s_ids = self.tokenizer(suffix, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
        logits = self.model(input_ids=s_ids, past_key_values=self._past, use_cache=True).logits[0, -1].float()
        self._past.crop(self._prefix_len)
        p_not = torch.softmax(logits[[self.id_sup, self.id_not]], dim=0)[1].item()
        res = {"is_supported": p_not < 0.5, "p_not_supported": p_not, "explanation": f"P(not supported)={p_not:.3f}",
               "raw_response": ""}
        self._cache[key] = res
        return res


@torch.no_grad()
def summary_rows_attention(model, cap: dict, layer: int, q0: int, head_chunk: int = 8):
    """Yield (h0, A) with A: (Hc, n - q0, n) causal softmax attention of query rows q0..n-1 at one layer."""
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

    attn = model.model.layers[layer].self_attn
    cfg = model.config
    Hq, Hkv = cfg.num_attention_heads, cfg.num_key_value_heads
    hd = attn.head_dim
    h = cap["h"]
    n = h.shape[1]
    q = attn.q_proj(h).view(1, n, Hq, hd).transpose(1, 2)
    k = attn.k_proj(h).view(1, n, Hkv, hd).transpose(1, 2)
    cos, sin = cap["pe"]
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    k = k.repeat_interleave(Hq // Hkv, dim=1)
    scale = getattr(attn, "scaling", hd ** -0.5)
    qpos = torch.arange(q0, n, device=h.device)
    kpos = torch.arange(n, device=h.device)
    allowed = kpos[None, :] <= qpos[:, None]
    for h0 in range(0, Hq, head_chunk):
        qs = q[0, h0:h0 + head_chunk, q0:].float()
        ks = k[0, h0:h0 + head_chunk].float()
        logits = (qs @ ks.transpose(-1, -2)) * scale
        logits = logits.masked_fill(~allowed, float("-inf"))
        yield h0, torch.softmax(logits, dim=-1)


class HallucinationDetectorAgent:
    def __init__(self, model, tokenizer, semantic_judge, dataset: str, aura_layer: int = -1,
                 aura_threshold: float = 0.42, use_aura: bool = True, use_semantic: bool = True,
                 max_tokens: int = 12000, epsilon: float = 1e-8, entail_cut: float = 0.5):
        assert use_aura or use_semantic
        self.model, self.tokenizer, self.semantic_judge = model.eval(), tokenizer, semantic_judge
        self.dataset = dataset
        L = model.config.num_hidden_layers
        self.layer = aura_layer if aura_layer >= 0 else L + aura_layer
        self.tau, self.use_aura, self.use_semantic = aura_threshold, use_aura, use_semantic
        self.entail_cut = entail_cut                           # flag a span when P(not supported) exceeds this
        self.max_tokens, self.eps = max_tokens, epsilon
        self.device = next(model.parameters()).device

    def reset(self):
        self.semantic_judge.reset()

    @torch.no_grad()
    def token_aura(self, source_document: str, summary: str):
        """Per-summary-token AURA and the summary's token character offsets, or None if the input is too long."""
        prompt = build_draft_prompt(self.dataset, source_document)
        c0 = prompt.find(source_document)
        c1 = c0 + len(source_document)
        enc = self.tokenizer(prompt, return_tensors="pt", return_offsets_mapping=True)
        offsets = enc["offset_mapping"][0].tolist()
        p_ids = enc["input_ids"]
        s_enc = self.tokenizer(" " + summary.strip(), return_tensors="pt", add_special_tokens=False,
                               return_offsets_mapping=True)
        s_ids = s_enc["input_ids"]
        T_p, T_s = p_ids.shape[1], s_ids.shape[1]
        n = T_p + T_s
        if T_s == 0 or n > self.max_tokens:
            return None
        ids = torch.cat([p_ids, s_ids], dim=1).to(self.device)

        src = torch.zeros(n, dtype=torch.bool, device=self.device)
        for t, (a, b) in enumerate(offsets):
            if b > a and a >= c0 and b <= c1:
                src[t] = True
        valid = torch.ones(n, dtype=torch.bool, device=self.device)
        if self.tokenizer.bos_token_id is not None and ids[0, 0].item() == self.tokenizer.bos_token_id:
            valid[0] = False                                   # attention sink, excluded from both terms
        src &= valid

        cap = capture_layer_inputs(self.model, ids, self.layer)
        ratio = torch.zeros(T_s, device=self.device)
        H = self.model.config.num_attention_heads
        for h0, A in summary_rows_attention(self.model, cap, self.layer, T_p):
            num = (A * src[None, None, :]).sum(-1)
            den = (A * valid[None, None, :]).sum(-1) + self.eps
            ratio += (num / den).sum(0)
            del A
        del cap
        torch.cuda.empty_cache()
        return (ratio / H).cpu().tolist(), s_enc["offset_mapping"][0].tolist()

    def _span_aura(self, summary: str, spans: List[str], tok_aura, offsets) -> Dict[int, float]:
        text = " " + summary.strip()
        ranges, cur = [], 0
        for sp in spans:
            i = text.find(sp, cur)
            if i < 0:
                ranges.append((cur, cur))
            else:
                ranges.append((i, i + len(sp)))
                cur = i + len(sp)
        tot = {j: 0.0 for j in range(len(spans))}
        cnt = {j: 0 for j in range(len(spans))}
        for t, (a, b) in enumerate(offsets):
            if t >= len(tok_aura) or text[a:b].strip() == "":
                continue
            for j, (sa, sb) in enumerate(ranges):
                if a >= sa and b <= sb:
                    tot[j] += tok_aura[t]
                    cnt[j] += 1
                    break
        return {j: (tot[j] / cnt[j] if cnt[j] else 0.0) for j in range(len(spans))}

    @torch.no_grad()
    def analyze(self, source_document: str, draft_summary: str) -> Dict[str, Any]:
        spans = sent_tokenize(draft_summary)
        details: Dict[int, Dict[str, Any]] = {}
        span_aura: Dict[int, float] = {j: float("nan") for j in range(len(spans))}
        aura_skipped = False

        if self.use_aura and spans:
            res = self.token_aura(source_document, draft_summary)
            if res is None:
                aura_skipped = True
            else:
                span_aura = self._span_aura(draft_summary, spans, res[0], res[1])

        sem = {j: 0 for j in range(len(spans))}
        p_not = {j: float("nan") for j in range(len(spans))}
        if self.use_semantic:
            for j, sp in enumerate(spans):
                r = self.semantic_judge.judge(document=source_document, span=sp)
                p_not[j] = r.get("p_not_supported", float("nan"))
                sem[j] = int(p_not[j] > self.entail_cut)
                if sem[j]:
                    details[j] = {"span": sp, "explanation": r.get("explanation", ""), "verdict": "UNSUPPORTED"}

        mask = {}
        for j in range(len(spans)):
            weak = self.use_aura and not aura_skipped and span_aura[j] < self.tau
            mask[j] = int(bool(sem[j]) or weak)
        return {"spans": spans, "span_aura_scores": span_aura, "semantic_p_not": p_not, "hallucination_mask": mask,
                "hallucination_details": details, "aura_skipped": aura_skipped}
