"""
FOCUS scoring utilities.

All scoring modes are pure functions of an attention tensor and sentence token
spans, so they can be unit-tested on synthetic matrices (see test_focus_scoring.py).

Modes (a mode string may carry a ":cross" suffix, e.g. "harmonic:cross"):
    original     C_k = sum_{i>=k} A[i,k]                       (column sum)
    horizon      C_k / (n - k)                                  (divide by #queries)
    harmonic     C_k / sum_{i>=k} 1/i                           (uniform causal null)
    conditional  mean_{i>=k} A[i,k] / b_h(dist, row-position)   (calibrated null)

":cross" restricts the queries for sentence j to tokens AFTER sentence j ends
(attention from other, later sentences only). The last sentence then has no
queries and scores NaN; the selection rule decides what to do with it.

Indexing: tokens are 0-based, row i has i+1 visible keys, so the uniform null
for row i is 1/(i+1). Token 0 is BOS when the tokenizer adds one.
"""
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch

N_BINS = 28  # 2 bins per octave, covers distances / positions up to 2**14 tokens
MODES = ("original", "horizon", "harmonic", "conditional")


# ----------------------------------------------------------------------------
# Binning for the conditional baseline b_h(distance, row position)
# ----------------------------------------------------------------------------
def _log_bin(x: torch.Tensor) -> torch.Tensor:
    """Half-octave log2 bin index of x >= 1."""
    return torch.clamp((2.0 * torch.log2(x.float())).floor().long(), 0, N_BINS - 1)


class BaselineTable:
    """
    Mean attention per (head, row-position bin, distance bin), estimated on a
    calibration split. Non-sink keys only (keys with index >= n_sink).
    Cells with fewer than `min_count` observations fall back to the uniform
    null 1/(row position) and are flagged in `is_fallback`.
    """

    def __init__(self, n_heads: int):
        self.n_heads = n_heads
        self.total = torch.zeros(n_heads, N_BINS * N_BINS, dtype=torch.float64)
        self.count = torch.zeros(n_heads, N_BINS * N_BINS, dtype=torch.float64)
        self.b: Optional[torch.Tensor] = None
        self.is_fallback: Optional[torch.Tensor] = None

    @torch.no_grad()
    def update(self, A: torch.Tensor, h0: int, n_sink: int = 1) -> None:
        """A: (Hc, n, n) attention for heads h0 .. h0+Hc-1."""
        Hc, n, _ = A.shape
        i = torch.arange(n, device=A.device)
        d = i[:, None] - i[None, :]
        valid = (d >= 0) & (i[None, :] >= n_sink)
        idx = _log_bin(i + 1)[:, None] * N_BINS + _log_bin(d.clamp(min=0) + 1)
        idx_v = idx[valid]
        for c in range(Hc):
            vals = A[c][valid].double()
            self.total[h0 + c] += torch.bincount(
                idx_v, weights=vals, minlength=N_BINS * N_BINS
            ).cpu()
            self.count[h0 + c] += torch.bincount(
                idx_v, minlength=N_BINS * N_BINS
            ).double().cpu()

    def finalize(self, min_count: int = 200) -> None:
        mean = self.total / self.count.clamp(min=1)
        ok = self.count >= min_count
        rep_i = 2.0 ** ((torch.arange(N_BINS, dtype=torch.float64) + 0.5) / 2.0)
        fallback = (
            (1.0 / rep_i)[None, :, None]
            .expand(self.n_heads, N_BINS, N_BINS)
            .reshape(self.n_heads, -1)
        )
        self.b = torch.where(ok, mean, fallback).clamp(min=1e-8).float()
        self.is_fallback = ~ok

    def save(self, path: str) -> None:
        torch.save(
            {"total": self.total, "count": self.count, "n_heads": self.n_heads}, path
        )

    @classmethod
    def load(cls, path: str, min_count: int = 200) -> "BaselineTable":
        blob = torch.load(path)
        t = cls(blob["n_heads"])
        t.total, t.count = blob["total"], blob["count"]
        t.finalize(min_count)
        return t


# ----------------------------------------------------------------------------
# Scoring from an attention chunk
# ----------------------------------------------------------------------------
def _suffix_inverse(n: int, device) -> torch.Tensor:
    """suffix[x] = sum_{i >= x} 1/(i+1), length n+1 (suffix[n] = 0)."""
    inv = 1.0 / torch.arange(1, n + 1, dtype=torch.float64, device=device)
    return torch.cat(
        [inv.flip(0).cumsum(0).flip(0), torch.zeros(1, dtype=torch.float64, device=device)]
    )


def parse_mode(spec: str) -> Tuple[str, bool]:
    base, _, flag = spec.partition(":")
    if base not in MODES or flag not in ("", "cross"):
        raise ValueError(f"bad mode spec: {spec}")
    return base, flag == "cross"


@torch.no_grad()
def sentence_scores_chunk(
    A: torch.Tensor,
    h0: int,
    spans: Sequence[Tuple[int, int]],
    spec: str,
    table: Optional[BaselineTable] = None,
    stats: Optional[Dict[str, float]] = None,
) -> torch.Tensor:
    """
    Per-head sentence scores for a chunk of heads.

    A: (Hc, n, n) float32 attention. spans: [(tok_start, tok_end_exclusive)] per
    sentence. Returns (Hc, m); NaN where a sentence has no eligible queries or no
    tokens. The sentence score is the mean over its tokens of the token score.
    """
    mode, cross = parse_mode(spec)
    Hc, n, _ = A.shape
    dev = A.device
    m = len(spans)
    out = torch.full((Hc, m), float("nan"), device=dev, dtype=torch.float32)
    suffix = _suffix_inverse(n, dev)
    if mode == "conditional":
        assert table is not None and table.b is not None, "conditional needs a finalized table"
        b_all = table.b[h0:h0 + Hc].to(dev)
        fb_all = table.is_fallback[h0:h0 + Hc].to(dev)

    for j, (s, e) in enumerate(spans):
        if e <= s:
            continue
        if cross:
            r0 = e
            q0 = torch.full((e - s,), e, device=dev)
        else:
            r0 = s
            q0 = torch.arange(s, e, device=dev)
        if r0 >= n:
            continue
        blk = A[:, r0:n, s:e]                                  # (Hc, nr, nk)

        if mode == "conditional":
            rows = torch.arange(r0, n, device=dev)
            keys = torch.arange(s, e, device=dev)
            d = (rows[:, None] - keys[None, :]).clamp(min=0)
            idx = (_log_bin(rows + 1)[:, None] * N_BINS + _log_bin(d + 1)).reshape(-1)
            b = b_all[:, idx].reshape(Hc, len(rows), len(keys))
            if stats is not None:
                eligible = (rows[:, None] >= keys[None, :]).reshape(-1)
                fb = fb_all[:, idx][:, eligible]
                stats["elements"] = stats.get("elements", 0) + fb.numel()
                stats["fallback"] = stats.get("fallback", 0) + int(fb.sum().item())
            num = (blk / b).sum(1)
        else:
            num = blk.sum(1)

        if mode == "original":
            tok = num
        elif mode in ("horizon", "conditional"):
            tok = num / (n - q0).clamp(min=1).float()[None, :]
        else:  # harmonic
            tok = num / suffix[q0].float()[None, :]
        out[:, j] = tok.mean(1)
    return out


def sink_diagnostics(A: torch.Tensor, n_sink: int, n_first: int = 5) -> Dict[str, float]:
    """Mean attention mass on BOS and on the first `n_first` real tokens."""
    Hc, n, _ = A.shape
    if n <= n_sink + n_first + 1:
        return {"bos_mass": float("nan"), "first_tokens_mass": float("nan")}
    rows = slice(n_sink + n_first, n)
    bos = A[:, rows, 0].mean().item() if n_sink else 0.0
    first = A[:, rows, n_sink:n_sink + n_first].sum(-1).mean().item()
    return {"bos_mass": bos, "first_tokens_mass": first}


# ----------------------------------------------------------------------------
# Exact single-layer attention (no output_attentions for all layers, no truncation)
# ----------------------------------------------------------------------------
class _StopForward(Exception):
    pass


@torch.no_grad()
def capture_layer_inputs(model, input_ids: torch.Tensor, layer_idx: int) -> dict:
    """Run the model up to layer `layer_idx` and capture that layer's attention inputs."""
    attn_mod = model.model.layers[layer_idx].self_attn
    cap: dict = {}

    def pre_hook(module, args, kwargs):
        cap["h"] = (args[0] if args else kwargs["hidden_states"]).detach()
        cap["pe"] = kwargs.get("position_embeddings")
        raise _StopForward()

    handle = attn_mod.register_forward_pre_hook(pre_hook, with_kwargs=True)
    try:
        model(input_ids=input_ids, use_cache=False)
    except _StopForward:
        pass
    finally:
        handle.remove()
    if cap.get("pe") is None:
        raise RuntimeError(
            "position_embeddings not passed to self_attn; this transformers version "
            "computes rotary inside attention. Upgrade transformers or adapt the hook."
        )
    return cap


@torch.no_grad()
def layer_attention_chunks(
    model, cap: dict, layer_idx: int, head_chunk: int = 4
) -> Iterator[Tuple[int, torch.Tensor]]:
    """Yield (head_offset, A) with A: (Hc, n, n) float32 causal softmax attention."""
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

    attn_mod = model.model.layers[layer_idx].self_attn
    cfg = model.config
    Hq, Hkv = cfg.num_attention_heads, cfg.num_key_value_heads
    hd = attn_mod.head_dim
    h = cap["h"]
    n = h.shape[1]
    q = attn_mod.q_proj(h).view(1, n, Hq, hd).transpose(1, 2)
    k = attn_mod.k_proj(h).view(1, n, Hkv, hd).transpose(1, 2)
    cos, sin = cap["pe"]
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    k = k.repeat_interleave(Hq // Hkv, dim=1)
    scale = getattr(attn_mod, "scaling", hd ** -0.5)
    causal = torch.ones(n, n, dtype=torch.bool, device=h.device).tril()
    for h0 in range(0, Hq, head_chunk):
        qs = q[0, h0:h0 + head_chunk].float()
        ks = k[0, h0:h0 + head_chunk].float()
        logits = (qs @ ks.transpose(-1, -2)) * scale
        logits = logits.masked_fill(~causal, float("-inf"))
        yield h0, torch.softmax(logits, dim=-1)


# ----------------------------------------------------------------------------
# Document-level helpers
# ----------------------------------------------------------------------------
def build_text_and_spans(sentences: List[str], tokenizer, device) -> dict:
    """Join sentences with one space and map each sentence to a contiguous token span."""
    text = " ".join(sentences)
    char_spans, cur = [], 0
    for s in sentences:
        char_spans.append((cur, cur + len(s)))
        cur += len(s) + 1
    enc = tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
    offsets = enc["offset_mapping"][0].tolist()
    ids = enc["input_ids"].to(device)
    n = ids.shape[1]
    tok_sent = [-1] * n
    for t, (ts, te) in enumerate(offsets):
        if te <= ts:
            continue
        pos = te - 1
        for j, (cs, ce) in enumerate(char_spans):
            if cs <= pos < ce:
                tok_sent[t] = j
                break
    spans = []
    for j in range(len(sentences)):
        idx = [t for t, sj in enumerate(tok_sent) if sj == j]
        spans.append((min(idx), max(idx) + 1) if idx else (0, 0))
    n_sink = 1 if (tokenizer.bos_token_id is not None and ids[0, 0].item() == tokenizer.bos_token_id) else 0
    return {"ids": ids, "spans": spans, "n": n, "n_sink": n_sink}


@torch.no_grad()
def score_document(
    model,
    tokenizer,
    sentences: List[str],
    layer: int,
    specs: Sequence[str],
    table: Optional[BaselineTable] = None,
    head_chunk: int = 4,
    max_tokens: int = 12000,
    calibrate_into: Optional[BaselineTable] = None,
) -> Optional[dict]:
    """
    One forward pass up to `layer`; returns sentence scores for every mode in `specs`.
    Returns None if the document exceeds max_tokens (never silently truncated).
    If `calibrate_into` is given, the attention also updates that BaselineTable.
    """
    device = next(model.parameters()).device
    info = build_text_and_spans(sentences, tokenizer, device)
    if info["n"] > max_tokens:
        return None
    cap = capture_layer_inputs(model, info["ids"], layer)
    H = model.config.num_attention_heads
    m = len(sentences)
    sums = {s: torch.zeros(m, device=device) for s in specs}
    stats: Dict[str, float] = {}
    sink = {"bos_mass": 0.0, "first_tokens_mass": 0.0}
    for h0, A in layer_attention_chunks(model, cap, layer, head_chunk):
        Hc = A.shape[0]
        if calibrate_into is not None:
            calibrate_into.update(A, h0, info["n_sink"])
        for s in specs:
            sums[s] += sentence_scores_chunk(A, h0, info["spans"], s, table, stats).sum(0)
        sd = sink_diagnostics(A, info["n_sink"])
        for key in sink:
            sink[key] += sd[key] * Hc / H
        del A
    return {
        "scores": {s: (sums[s] / H).cpu().numpy() for s in specs},
        "tok_counts": np.array([e - s for s, e in info["spans"]]),
        "n_tokens": info["n"],
        "sink": sink,
        "baseline_stats": stats,
    }


# ----------------------------------------------------------------------------
# Selection under an exact token budget + baselines + small stats helpers
# ----------------------------------------------------------------------------
def select_by_token_budget(
    scores: np.ndarray,
    tok_counts: np.ndarray,
    budget_frac: float,
    last_policy: str = "none",
) -> List[int]:
    """
    Rank sentences by score (NaN last), add until retained tokens >= budget_frac of
    all tokens, return indices in document order. Retained tokens can exceed the
    target by at most one sentence; callers must report the actual count.
    last_policy="keep" force-includes the final sentence first.
    """
    m = len(scores)
    target = int(np.ceil(budget_frac * tok_counts.sum()))
    s = np.where(np.isnan(scores), -np.inf, scores)
    order = list(np.argsort(-s, kind="mergesort"))
    chosen, total = [], 0
    if last_policy == "keep" and m > 0:
        chosen.append(m - 1)
        total += int(tok_counts[m - 1])
        order = [i for i in order if i != m - 1]
    for i in order:
        if total >= target:
            break
        chosen.append(int(i))
        total += int(tok_counts[i])
    return sorted(chosen)


def baseline_scores(kind: str, sentences: List[str], rng: np.random.Generator) -> np.ndarray:
    m = len(sentences)
    if kind == "random":
        return rng.random(m)
    if kind == "first":
        return -np.arange(m, dtype=float)
    if kind == "last":
        return np.arange(m, dtype=float)
    if kind == "tfidf":
        import re
        words = [re.findall(r"[a-z0-9]+", s.lower()) for s in sentences]
        df: Dict[str, int] = {}
        for w in words:
            for x in set(w):
                df[x] = df.get(x, 0) + 1
        idf = {x: np.log((m + 1) / (c + 1)) + 1.0 for x, c in df.items()}
        return np.array([sum(idf[x] for x in w) / max(len(w), 1) for w in words])
    raise ValueError(kind)


def rankdata(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    ranks[order] = np.arange(len(x))
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    return (np.bincount(inv, weights=ranks) / cnt)[inv]


def spearman(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    ra, rb = rankdata(a[ok]), rankdata(b[ok])
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])
