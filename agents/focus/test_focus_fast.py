"""
CPU tests: the vectorized scorer (focus_fast) must match the loop reference (focus_scoring).
Run: python test_focus_fast.py
"""
import numpy as np
import torch

from focus_scoring import BaselineTable, sentence_scores_chunk
from focus_fast import sentence_index, bin_index_matrix, fast_sentence_scores

BASE = ["original", "horizon", "harmonic", "conditional"]
SPECS = BASE + [b + ":cross" for b in BASE]


def random_attention(H, n, seed):
    g = torch.Generator().manual_seed(seed)
    raw = torch.rand(H, n, n, generator=g).tril() + 1e-4 * torch.ones(H, n, n).tril()
    return raw / raw.sum(-1, keepdim=True)


def spans_with_gaps(n, sent_len, gap=1, start=1):
    """Sentences of sent_len tokens separated by `gap` unassigned tokens (like stray spaces)."""
    spans, s = [], start
    while s < n:
        e = min(s + sent_len, n)
        spans.append((s, e))
        s = e + gap
    return spans


def close(a, b, rtol=1e-4, atol=1e-6):
    if not torch.equal(torch.isnan(a), torch.isnan(b)):
        return False
    m = ~torch.isnan(a)
    return torch.allclose(a[m], b[m], rtol=rtol, atol=atol)


def build(H=3, n=150, sent_len=13, seed=0):
    A = random_attention(H, n, seed)
    table = BaselineTable(H)
    table.update(A, 0, n_sink=1)
    table.finalize(min_count=5)            # small threshold: exercises both table cells and fallback
    spans = spans_with_gaps(n, sent_len)
    sid, endk = sentence_index(n, spans, "cpu")
    return A, table, spans, sid, endk


def test_fast_matches_loop_all_specs():
    A, table, spans, sid, endk = build()
    idx = bin_index_matrix(A.shape[1], "cpu")
    fast = fast_sentence_scores(A, 0, sid, endk, len(spans), SPECS, table.b, idx)
    for spec in SPECS:
        ref = sentence_scores_chunk(A, 0, spans, spec, table)
        assert close(fast[spec], ref), (spec, fast[spec][0, :4], ref[0, :4])


def test_head_chunking_consistent():
    A, table, spans, sid, endk = build(H=4)
    idx = bin_index_matrix(A.shape[1], "cpu")
    full = fast_sentence_scores(A, 0, sid, endk, len(spans), SPECS, table.b, idx)
    a = fast_sentence_scores(A[:2], 0, sid, endk, len(spans), SPECS, table.b, idx)
    b = fast_sentence_scores(A[2:], 2, sid, endk, len(spans), SPECS, table.b, idx)
    for spec in SPECS:
        assert close(torch.cat([a[spec], b[spec]], 0), full[spec]), spec


def test_uniform_attention_properties():
    n, H = 300, 2
    A = torch.zeros(H, n, n)
    for i in range(n):
        A[:, i, : i + 1] = 1.0 / (i + 1)
    spans = spans_with_gaps(n, 20, gap=0)
    sid, endk = sentence_index(n, spans, "cpu")
    out = fast_sentence_scores(A, 0, sid, endk, len(spans), ["harmonic", "harmonic:cross"])
    h = out["harmonic"].mean(0)
    assert torch.allclose(h, torch.ones_like(h), atol=1e-4)
    c = out["harmonic:cross"].mean(0)
    assert torch.isnan(c[-1]) and torch.allclose(c[:-1], torch.ones_like(c[:-1]), atol=1e-4)


def test_empty_sentence_is_nan():
    A, table, spans, sid, endk = build()
    spans = spans + [(0, 0)]                       # a sentence with no tokens
    sid, endk = sentence_index(A.shape[1], spans, "cpu")
    out = fast_sentence_scores(A, 0, sid, endk, len(spans), ["original", "harmonic:cross"])
    assert torch.isnan(out["original"][:, -1]).all()


def test_cross_end_precision_float32():
    # A late sentence receives tiny attention from the few rows after it; the reversed cumsum
    # must keep that small sum accurate (a total-minus-prefix scheme would not).
    n, H = 2000, 1
    A = random_attention(H, n, 3)
    spans = spans_with_gaps(n, 40, gap=0)
    sid, endk = sentence_index(n, spans, "cpu")
    got = fast_sentence_scores(A, 0, sid, endk, len(spans), ["original:cross"])["original:cross"]
    ref = sentence_scores_chunk(A.double().float(), 0, spans, "original:cross", None)
    assert close(got, ref, rtol=1e-4, atol=1e-7)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} tests passed")
