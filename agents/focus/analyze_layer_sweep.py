"""
No-generation layer diagnostics from the files saved by run_layer_sweep.py (CPU only).

For every layer and scoring mode, over all development notes:
  * position bias       : Spearman correlation of sentence score with relative position;
  * early share         : fraction of kept tokens from the first 30% of the note (neutral = 0.30);
  * coverage gain       : unigram recall of the reference summary's content words in the kept text,
                          minus the same for random selection at the same token budget;
  * layer x position heatmap of the mean within-note rank percentile of the score;
  * attention mass on the beginning-of-sequence token, by layer.
Outputs a CSV of aggregates and two figures. Confidence bands are 95% bootstrap intervals over notes.
"""
import argparse
import os
import re

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

import plot_style as ps
from focus_scoring import baseline_scores, rankdata, select_by_token_budget, spearman
from sweep_io import load_sweep, relative_positions

STOP = set("""a about above after again all also am an and any are as at be because been before being below between both
but by can could did do does doing down during each few for from further had has have having he her here hers him his how
i if in into is it its just me more most my no nor not of off on once only or other our out over own same she should so
some such than that the their them then there these they this those through to too under until up very was we were what
when where which while who whom why will with would you your""".split())

LABELS = {"original": "Column-sum scoring", "horizon": "Horizon division", "harmonic": "Harmonic",
          "harmonic:cross": "Harmonic, cross-sentence", "conditional": "Conditional",
          "conditional:cross": "Conditional, cross-sentence"}
# accent colors for the candidates, gray + distinct line style for the rest
KEY = {"conditional": "focus", "conditional:cross": "aqua", "original": "original",
       "harmonic": "tfidf", "harmonic:cross": "first", "horizon": "random"}


def words(text):
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) >= 3 and w not in STOP}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mimic", "soap"], required=True)
    ap.add_argument("--sweep-dir", default="outputs")
    ap.add_argument("--out", default="outputs/figures")
    ap.add_argument("--rhos", type=float, nargs="+", default=[0.3, 0.7])
    ap.add_argument("--heat-modes", nargs="+", default=["original", "conditional"])
    ap.add_argument("--theme", choices=["light", "dark"], default="light")
    ap.add_argument("--n-random", type=int, default=3)
    args = ap.parse_args()

    meta, notes, scores_fn, bos_fn = load_sweep(args.sweep_dir, args.dataset, "dev")
    layers, modes = meta["layers"], meta["modes"]
    L, M, R, N = len(layers), len(modes), len(args.rhos), len(notes)
    spear = np.full((N, L, M), np.nan)
    early = np.full((N, L, M), np.nan)
    gain = np.full((N, L, M), np.nan)
    heat_sum = np.zeros((L, M, 10))
    heat_cnt = np.zeros((L, M, 10))
    bos = np.zeros((N, L))
    r0 = args.rhos[0]
    rng_seed = 123

    for ni, note in enumerate(notes):
        tc = np.array(note["tok_counts"])
        pos = relative_positions(tc)
        dec = np.minimum((pos * 10).astype(int), 9)
        ref_w = words(note["reference"])
        if not ref_w:
            continue
        sw = [words(s) for s in note["sentences"]]
        S = scores_fn(ni)
        bos[ni] = bos_fn(ni)

        def coverage(kept):
            u = set()
            for i in kept:
                u |= sw[i]
            return len(ref_w & u) / len(ref_w)

        rand_cov = np.mean([coverage(select_by_token_budget(
            baseline_scores("random", note["sentences"], np.random.default_rng(rng_seed + 1000 * s + ni)),
            tc, r0, "none")) for s in range(args.n_random)])
        for li in range(L):
            for mi in range(M):
                sc = S[li, mi]
                spear[ni, li, mi] = spearman(sc, pos)
                kept = select_by_token_budget(sc, tc, r0, "none")
                mask = np.zeros(len(tc), bool)
                mask[kept] = True
                early[ni, li, mi] = tc[mask & (pos < 0.3)].sum() / max(tc[mask].sum(), 1)
                gain[ni, li, mi] = coverage(kept) - rand_cov
                fin = np.isfinite(sc)
                if fin.sum() >= 3:
                    pct = np.full(len(sc), np.nan)
                    pct[fin] = (rankdata(sc[fin]) / (fin.sum() - 1))
                    for d in range(10):
                        sel = (dec == d) & fin
                        heat_sum[li, mi, d] += pct[sel].sum()
                        heat_cnt[li, mi, d] += sel.sum()

    rng = np.random.default_rng(0)
    rows = []
    summary = {}
    for name, arr in (("position_spearman", spear), ("early_share", early), ("coverage_gain", gain)):
        for mi, mode in enumerate(modes):
            mat = arr[:, :, mi].T                       # (layers, notes)
            m, lo, hi = ps.boot_ci(mat, rng)
            sd = np.nanstd(mat, axis=1, ddof=1)
            summary[(name, mode)] = (m, lo, hi)
            rows += [dict(dataset=args.dataset, metric=name, mode=mode, layer=layers[i], rho=r0, mean=m[i], std=sd[i],
                          mean_pm_std=f"{m[i]:.3f} \u00b1 {sd[i]:.3f}", ci_lo=lo[i], ci_hi=hi[i]) for i in range(L)]
    os.makedirs(args.out, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(args.out, f"layer_diag_{args.dataset}.csv"), index=False)

    # ---- Figure 1: layer curves ----
    t = ps.setup(args.theme)
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(15.2, 4.0))
    panels = [("position_spearman", "Position bias", "Spearman(score, position); 0 = none", None),
              ("early_share", f"Kept tokens from first 30% (rho={r0:g})", "fraction; 0.30 = neutral", 0.30),
              ("coverage_gain", f"Reference coverage vs random (rho={r0:g})", "unigram recall gain; 0 = random", 0.0)]
    for ax, (name, title, ylab, ref) in zip(axes[:3], panels):
        ps.tidy_axes(ax, t)
        for mode in modes:
            m, lo, hi = summary[(name, mode)]
            acc = KEY[mode] in ("focus", "aqua", "original")
            ps.draw_series(ax, t, layers, m, lo if acc else None, hi if acc else None, KEY[mode], LABELS[mode])
        if ref is not None:
            ax.axhline(ref, color=t["ink2"], linewidth=1.0, linestyle=(0, (4, 3)), zorder=2)
        ax.set_title(title, loc="left")
        ax.set_xlabel("Attention layer")
        ax.set_ylabel(ylab)
    ax = axes[3]
    ps.tidy_axes(ax, t)
    bm = bos.mean(0)
    ps.draw_series(ax, t, layers, bm, None, None, "focus", "BOS attention mass")
    ax.set_title("Attention on the BOS token", loc="left")
    ax.set_xlabel("Attention layer")
    ax.set_ylabel("mean attention mass")
    handles = ps.legend_handles(t, [(ps.STYLE[KEY[m]][0], ps.STYLE[KEY[m]][1], ps.STYLE[KEY[m]][2], LABELS[m]) for m in modes])
    fig.tight_layout(rect=(0, 0.14, 1, 0.93))
    fig.legend(handles=handles, loc="lower center", ncol=6, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"Layer diagnostics without generation ({args.dataset.upper()})", x=0.01, ha="left", fontsize=11)
    p1 = os.path.join(args.out, f"layer_diagnostics_{args.dataset}_{args.theme}.png")
    fig.savefig(p1, dpi=200, bbox_inches="tight")
    fig.savefig(p1[:-4] + ".pdf", bbox_inches="tight")

    # ---- Figure 2: layer x position heatmaps (diverging around 0.5) ----
    hm = [m for m in args.heat_modes if m in modes]
    fig2, axs = plt.subplots(1, len(hm), figsize=(4.4 * len(hm) + 1.2, 5.2), constrained_layout=True, squeeze=False)
    cmap = LinearSegmentedColormap.from_list("div", [t["div_low"], t["div_mid"], t["div_high"]])
    grids = {m: heat_sum[:, modes.index(m), :] / np.maximum(heat_cnt[:, modes.index(m), :], 1) for m in hm}
    d = max(float(np.nanmax(np.abs(g - 0.5))) for g in grids.values())
    im = None
    for ax, m in zip(axs[0], hm):
        im = ax.imshow(grids[m], aspect="auto", cmap=cmap, vmin=0.5 - d, vmax=0.5 + d, origin="upper")
        ax.set_title(LABELS[m], loc="left")
        ax.set_xlabel("Position in note (decile)")
        ax.set_xticks(range(10))
        ax.set_xticklabels([str(i + 1) for i in range(10)])
        ax.set_yticks(range(0, L, 2))
        ax.set_yticklabels([str(layers[i]) for i in range(0, L, 2)])
        ax.set_ylabel("Attention layer")
    cb = fig2.colorbar(im, ax=axs[0].tolist(), shrink=0.8)
    cb.set_label("mean rank percentile of score (0.5 = no preference)")
    fig2.suptitle(f"Where each layer's score points ({args.dataset.upper()})", x=0.01, ha="left", fontsize=11)
    p2 = os.path.join(args.out, f"layer_position_heatmap_{args.dataset}_{args.theme}.png")
    fig2.savefig(p2, dpi=200, bbox_inches="tight")
    fig2.savefig(p2[:-4] + ".pdf", bbox_inches="tight")
    print(f"saved {p1}\nsaved {p2}\nsaved layer_diag_{args.dataset}.csv ({N} notes)")


if __name__ == "__main__":
    main()
