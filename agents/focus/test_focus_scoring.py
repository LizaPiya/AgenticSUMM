"""
CPU unit tests for focus_scoring.py on synthetic attention matrices.
Run: python test_focus_scoring.py   (plain asserts, no pytest needed)
"""
import numpy as np
import torch

from focus_scoring import (
    BaselineTable, sentence_scores_chunk, select_by_token_budget, spearman, rankdata,
)


def uniform_attention(H, n):
    """A[h,i,k] = 1/(i+1) for k <= i (content-free causal attention)."""
    A = torch.zeros(H, n, n)
    for i in range(n):
        A[:, i, : i + 1] = 1.0 / (i + 1)
    return A


def make_spans(n, sent_len, start=1):
    """Token 0 is BOS; the rest is partitioned into sentences of sent_len tokens."""
    spans, s = [], start
    while s < n:
        spans.append((s, min(s + sent_len, n)))
        s += sent_len
    return spans


def test_tensor_sanity():
    A = uniform_attention(2, 50)
    assert torch.allclose(A.sum(-1), torch.ones(2, 50), atol=1e-6), "rows must sum to 1"
    assert torch.all(torch.triu(A, diagonal=1) == 0), "must be causal"
    assert torch.allclose(A.sum(1).sum(-1), torch.full((2,), 50.0), atol=1e-4), "sum of column sums = n"


def test_harmonic_is_one_under_uniform():
    n = 400
    A = uniform_attention(3, n)
    spans = make_spans(n, 20)
    sc = sentence_scores_chunk(A, 0, spans, "harmonic").mean(0)
    assert torch.allclose(sc, torch.ones_like(sc), atol=1e-4), sc[:5]


def test_horizon_is_not_constant_under_uniform():
    # This is the analytic argument against horizon division alone.
    n = 400
    A = uniform_attention(1, n)
    spans = make_spans(n, 20)
    sc = sentence_scores_chunk(A, 0, spans, "horizon").mean(0)
    ratio = (sc[0] / sc[-1]).item()
    assert ratio > 2.0, f"expected strong residual early bias, got ratio {ratio:.2f}"


def test_original_is_biased_under_uniform():
    n = 400
    A = uniform_attention(1, n)
    spans = make_spans(n, 20)
    sc = sentence_scores_chunk(A, 0, spans, "original").mean(0)
    assert sc[0] > 3 * sc[-1]


def test_cross_harmonic_one_and_last_nan():
    n = 400
    A = uniform_attention(1, n)
    spans = make_spans(n, 20)
    sc = sentence_scores_chunk(A, 0, spans, "harmonic:cross").mean(0)
    assert torch.isnan(sc[-1]), "last sentence has no later queries"
    assert torch.allclose(sc[:-1], torch.ones_like(sc[:-1]), atol=1e-4)


def test_against_reference_loop():
    rng = np.random.default_rng(0)
    n, H = 120, 2
    raw = torch.tensor(rng.random((H, n, n)), dtype=torch.float32).tril()
    A = raw / raw.sum(-1, keepdim=True)
    spans = make_spans(n, 15)
    for spec in ("original", "horizon", "harmonic"):
        got = sentence_scores_chunk(A, 0, spans, spec).mean(0).numpy()
        ref = []
        for (s, e) in spans:
            vals = []
            for k in range(s, e):
                col = A[:, k:, k].sum(1).mean().item()        # sum over queries i>=k, mean over heads
                if spec == "original":
                    vals.append(col)
                elif spec == "horizon":
                    vals.append(col / (n - k))
                else:
                    vals.append(col / sum(1.0 / (i + 1) for i in range(k, n)))
            ref.append(np.mean(vals))
        assert np.allclose(got, ref, rtol=1e-4, atol=1e-6), (spec, got[:3], ref[:3])


def test_conditional_close_to_one_on_calibration_data():
    n = 600
    A = uniform_attention(2, n)
    table = BaselineTable(2)
    table.update(A, 0, n_sink=1)
    table.finalize(min_count=20)
    spans = make_spans(n, 20)
    sc = sentence_scores_chunk(A, 0, spans, "conditional", table).mean(0)
    assert torch.all((sc > 0.6) & (sc < 1.6)), (sc.min().item(), sc.max().item())


def test_sink_misspecification_documented():
    # 90% of every row on key 0 (BOS), 10% spread uniformly over the other visible keys.
    n, H = 300, 1
    A = torch.zeros(H, n, n)
    A[:, 0, 0] = 1.0
    for i in range(1, n):
        A[:, i, 0] = 0.9
        A[:, i, 1 : i + 1] = 0.1 / i
    spans = make_spans(n, 20)
    sc = sentence_scores_chunk(A, 0, spans, "harmonic").mean(0)
    # Harmonic is NOT 1 here: this test records how far the uniform null drifts under a sink.
    assert (sc < 0.5).all(), sc[:5]


def test_selection_budget_and_order():
    scores = np.array([0.1, 0.9, 0.5, np.nan, 0.7])
    toks = np.array([10, 10, 10, 10, 10])
    chosen = select_by_token_budget(scores, toks, 0.4)
    assert chosen == sorted(chosen) and sum(toks[chosen]) >= 20
    assert chosen == [1, 4]
    kept = select_by_token_budget(scores, toks, 0.4, last_policy="keep")
    assert 4 in kept and 1 in kept


def test_spearman():
    assert abs(spearman([1, 2, 3, 4], [10, 20, 30, 40]) - 1) < 1e-9
    assert abs(spearman([1, 2, 3, 4], [4, 3, 2, 1]) + 1) < 1e-9
    assert np.allclose(rankdata([1, 1, 2]), [0.5, 0.5, 2.0])


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} tests passed")
