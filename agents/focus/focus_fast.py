"""
Vectorized FOCUS scoring (all modes, all layers) -- no Python loop over sentences.

Equivalent to focus_scoring.sentence_scores_chunk (the loop version), which is kept as the
reference implementation; test_focus_fast.py and run_layer_sweep.py --verify compare the two.

Token score for key token k, with q0(k) the first eligible query row:
    non-cross: q0 = k                 cross: q0 = end of k's sentence (queries after the sentence)
    original     sum_{i>=q0} A[i,k]
    horizon      sum_{i>=q0} A[i,k] / (n - q0)
    harmonic     sum_{i>=q0} A[i,k] / sum_{i>=q0} 1/(i+1)
    conditional  sum_{i>=q0} A[i,k]/b_h(i,k) / (n - q0)
Sentence score = mean over its tokens and over heads. Cross scores are NaN when q0 == n.

Suffix sums over rows are taken with a reversed cumulative sum (not total minus prefix), to
avoid float32 cancellation for sentences near the end of the note.
"""
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from focus_scoring import N_BINS, _log_bin, _suffix_inverse, parse_mode


def sentence_index(n: int, spans: Sequence[Tuple[int, int]], device) -> Tuple[torch.Tensor, torch.Tensor]:
    """sid[k] = sentence id of token k (-1 if none); endk[k] = exclusive end of that sentence."""
    sid = torch.full((n,), -1, dtype=torch.long, device=device)
    endk = torch.zeros(n, dtype=torch.long, device=device)
    for j, (s, e) in enumerate(spans):
        if e > s:
            sid[s:e] = j
            endk[s:e] = e
    return sid, endk


def bin_index_matrix(n: int, device) -> torch.Tensor:
    """(n, n) long: flat index (row-position bin, distance bin) into a BaselineTable row."""
    i = torch.arange(n, device=device)
    d = (i[:, None] - i[None, :]).clamp(min=0)
    return _log_bin(i + 1)[:, None] * N_BINS + _log_bin(d + 1)


@torch.no_grad()
def fast_sentence_scores(
    A: torch.Tensor,
    h0: int,
    sid: torch.Tensor,
    endk: torch.Tensor,
    n_sent: int,
    specs: Sequence[str],
    table_b: Optional[torch.Tensor] = None,
    idx: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    A: (Hc, n, n) float32 attention for heads h0..h0+Hc-1.
    table_b: (H, N_BINS*N_BINS) finalized baseline for ALL heads (needed for conditional modes).
    idx: bin_index_matrix(n). Returns {spec: (Hc, n_sent)} per-head sentence scores.
    """
    Hc, n, _ = A.shape
    dev = A.device
    k_ar = torch.arange(n, device=dev)
    valid = sid >= 0
    nan = float("nan")
    suffix = _suffix_inverse(n, dev).float()
    parsed = {s: parse_mode(s) for s in specs}
    need_cond = any(m == "conditional" for m, _ in parsed.values())
    need_plain = any(m != "conditional" for m, _ in parsed.values())
    need_cross = any(c for _, c in parsed.values())

    def reduce_rows(X: torch.Tensor):
        """(non-cross, cross) column quantities, each (Hc, n)."""
        nc = X.sum(1)
        cr = None
        if need_cross:
            Q = X.flip(1).cumsum(1)               # Q[:, t, k] = sum_{i >= n-1-t} X[:, i, k]
            rows = n - 1 - endk                   # row for sum_{i >= e}; e == n -> -1 (no queries)
            ok = rows >= 0
            cr = Q[:, rows.clamp(min=0), k_ar]
            cr = torch.where(ok[None, :], cr, torch.full_like(cr, nan))
            del Q
        return nc, cr

    red_A = reduce_rows(A) if need_plain else None
    red_R = None
    if need_cond:
        assert table_b is not None and idx is not None, "conditional needs table_b and idx"
        b = table_b[h0:h0 + Hc].to(dev)
        bm = b[:, idx.reshape(-1)].reshape(Hc, n, n)
        red_R = reduce_rows(A / bm)
        del bm

    cnt = torch.bincount(sid[valid], minlength=n_sent).float()
    out = {}
    for spec, (mode, cross) in parsed.items():
        num = (red_R if mode == "conditional" else red_A)[1 if cross else 0]
        q0 = endk if cross else k_ar
        if mode == "original":
            tok = num
        elif mode in ("horizon", "conditional"):
            tok = num / (n - q0).clamp(min=1).float()[None, :]
        else:  # harmonic
            tok = num / suffix[q0].clamp(min=1e-12)[None, :]
        sums = torch.zeros(Hc, n_sent, device=dev).index_add_(1, sid[valid], tok[:, valid])
        res = sums / cnt.clamp(min=1)
        res[:, cnt == 0] = nan
        out[spec] = res
    return out


# ----------------------------------------------------------------------------
# One forward pass capturing every layer's attention input
# ----------------------------------------------------------------------------
class _StopForward(Exception):
    pass


@torch.no_grad()
def capture_all_layers(model, input_ids: torch.Tensor) -> dict:
    """
    Run the model once and capture, for every layer li, the hidden states that enter its
    self-attention (post input-layernorm): caps[li] with shape (1, n, d). caps["pe"] holds the
    shared rotary (cos, sin). The forward is stopped at the last layer's attention input.
    """
    layers = model.model.layers
    last = len(layers) - 1
    caps: dict = {}
    handles = []

    def make_hook(li):
        def pre_hook(module, args, kwargs):
            caps[li] = (args[0] if args else kwargs["hidden_states"]).detach()
            if li == 0:
                caps["pe"] = kwargs.get("position_embeddings")
            if li == last:
                raise _StopForward()
        return pre_hook

    for li, layer in enumerate(layers):
        handles.append(layer.self_attn.register_forward_pre_hook(make_hook(li), with_kwargs=True))
    try:
        model(input_ids=input_ids, use_cache=False)
    except _StopForward:
        pass
    finally:
        for h in handles:
            h.remove()
    if caps.get("pe") is None:
        raise RuntimeError("position_embeddings not passed to self_attn in this transformers version")
    return caps
