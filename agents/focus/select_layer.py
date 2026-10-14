"""
Apply SELECTION_RULES.md step 1 (layer choice) with paired bootstrap, exactly as written there.

  gain(layer) = mean over the two rho values of the average of [mean(FOCUS - random) on ROUGE-L,
                mean(FOCUS - random) on BERTScore F1], differences taken per note against the mean of the
                random seeds on the same note.
  1. drop layers whose judge hallucination is worse than random by more than 0.10 (if judge files exist);
  2. best layer = largest gain; layers whose gain is not below the lower end of the best layer's 95% paired
     bootstrap interval are "tied"; among tied layers choose the smallest |position correlation| from
     layer_diag_<dataset>.csv (the conditional mode).
Also prints a standardized-gain check (each metric's gain divided by the std of its per-note differences) so a
scale effect between ROUGE-L and BERTScore would be visible. Everything is mean +- std across notes.
"""
import argparse
import os

import numpy as np
import pandas as pd

from plot_sensitivity import group_name, load_frames


def per_note(df, note_ids, metric, rho, method, layer=None):
    sub = df[df["method"].map(group_name) == method]
    if layer is not None:
        sub = sub[sub["layer"].astype(str) == str(layer)]
    sub = sub[np.isclose(sub["rho"].astype(float), rho)]
    s = sub.groupby("note_id")[metric].mean()
    return np.array([s.get(n, np.nan) for n in note_ids])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mimic")
    ap.add_argument("--met-dir", default="outputs/metrics")
    ap.add_argument("--judge-dir", default="outputs/judge/phi35_mini")
    ap.add_argument("--diag", default=None)
    ap.add_argument("--spec", default="conditional")
    ap.add_argument("--judge-metric", default="hallucination", choices=["factual", "hallucination"],
                    help="judge rating used for the exclusion check (factual: higher is better; see SELECTION_RULES.md amendment)")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--rhos", type=float, nargs="+", default=[0.3, 0.7],
                    help="budgets of the layer-plot design; other budgets (the rho plot) are ignored here")
    ap.add_argument("--out", default="outputs/figures")
    args = ap.parse_args()

    df = load_frames(args.met_dir, args.judge_dir, args.dataset)
    note_ids = sorted(df["note_id"].unique())
    focus = df[df["method"] == "focus"]
    layers = sorted({int(float(l)) for l in focus["layer"].astype(str) if l not in ("", "nan")})
    rhos = [r for r in sorted(focus["rho"].astype(float).unique()) if any(abs(r - x) < 1e-9 for x in args.rhos)]
    # only layers that have every layer-plot budget (the rho plot adds other budgets at a single layer)
    layers = [L for L in layers if all(
        ((focus["layer"].astype(str) == str(L)) & (abs(focus["rho"].astype(float) - r) < 1e-9)).any() for r in rhos)]
    jm = args.judge_metric
    have_judge = jm in df.columns and df[jm].notna().any()

    # per-note differences FOCUS - random, shape (layers, rhos, notes) per metric
    diff = {}
    for metric in ("rouge_l", "bert_f1") + ((jm,) if have_judge else ()):
        arr = np.full((len(layers), len(rhos), len(note_ids)), np.nan)
        for li, L in enumerate(layers):
            for ri, r in enumerate(rhos):
                arr[li, ri] = per_note(df, note_ids, metric, r, "focus", L) - per_note(df, note_ids, metric, r, "random")
        diff[metric] = arr

    rng = np.random.default_rng(0)
    n = len(note_ids)
    idx = rng.integers(0, n, size=(args.n_boot, n))

    def gain(d_r, d_b, std_r=None, std_b=None):
        gr, gb = np.nanmean(d_r, axis=-1), np.nanmean(d_b, axis=-1)          # (layers, rhos)
        if std_r is not None:
            gr, gb = gr / std_r, gb / std_b
        return ((gr + gb) / 2).mean(axis=1)                                   # (layers,)

    g = gain(diff["rouge_l"], diff["bert_f1"])
    # bootstrap the gain per layer (paired over notes)
    bg = np.zeros((args.n_boot, len(layers)))
    for b in range(args.n_boot):
        i = idx[b]
        bg[b] = gain(diff["rouge_l"][:, :, i], diff["bert_f1"][:, :, i])
    lo, hi = np.percentile(bg, 2.5, axis=0), np.percentile(bg, 97.5, axis=0)

    sr = np.nanstd(diff["rouge_l"], axis=-1, ddof=1).mean()
    sb = np.nanstd(diff["bert_f1"], axis=-1, ddof=1).mean()
    g_std = gain(diff["rouge_l"], diff["bert_f1"], sr, sb)

    keep = np.ones(len(layers), bool)
    hall = np.full(len(layers), np.nan)
    if have_judge:
        hall = np.nanmean(np.nanmean(diff[jm], axis=-1), axis=1)       # FOCUS - random on the judge rating
        keep = ~(hall > 0.10) if jm == "hallucination" else ~(hall < -0.10)

    diag_path = args.diag or os.path.join(args.out, f"layer_diag_{args.dataset}.csv")
    pos = {}
    if os.path.exists(diag_path):
        d = pd.read_csv(diag_path)
        d = d[(d["metric"] == "position_spearman") & (d["mode"] == args.spec)]
        pos = {int(r["layer"]): abs(float(r["mean"])) for _, r in d.iterrows()}

    cand = [i for i in range(len(layers)) if keep[i]]
    best = max(cand, key=lambda i: g[i])
    tied = [i for i in cand if g[i] >= lo[best]]
    chosen = min(tied, key=lambda i: pos.get(layers[i], np.inf)) if pos else best

    out = pd.DataFrame({
        "layer": layers, "gain_over_random": g, "ci_lo": lo, "ci_hi": hi, "gain_standardized": g_std,
        f"{jm}_vs_random": hall, "kept_by_judge_check": keep,
        "abs_position_corr": [pos.get(L, np.nan) for L in layers], "tied_with_best": [i in tied for i in range(len(layers))],
    })
    # per-note gain (average over the two metrics and the rho values), so the table can show mean +- std over notes
    per_note_gain = ((diff["rouge_l"] + diff["bert_f1"]) / 2).mean(axis=1)               # (layers, notes)
    sd = np.nanstd(per_note_gain, axis=1, ddof=1)
    out["gain_mean_pm_std"] = [f"{np.nanmean(per_note_gain[i]):.2f} \u00b1 {sd[i]:.2f}" for i in range(len(layers))]
    os.makedirs(args.out, exist_ok=True)
    out.to_csv(os.path.join(args.out, f"layer_selection_{args.dataset}.csv"), index=False)
    print(out.round(3).to_string(index=False))
    print(f"\nbest by gain: layer {layers[best]}; tied with best: {[layers[i] for i in tied]}")
    print(f"chosen by rule (smallest |position correlation| among tied): layer {layers[chosen]}  "
          f"[judge check {'applied' if have_judge else 'PENDING (no judge files yet)'}]")
    print(f"standardized-gain argmax: layer {layers[int(np.argmax(np.where(keep, g_std, -np.inf)))]}")


if __name__ == "__main__":
    main()
