"""
Sensitivity plots from per-note metric files.

  --plot rho   : metric vs retained-token fraction; FOCUS (ours) and column-sum scoring in color,
                 TF-IDF / first-k / random as gray controls, full note as a reference mark.
  --plot layer : metric vs attention layer at each rho; random selection at the same rho (dotted),
                 full note as a dashed reference line.

Panels: ROUGE-L, BERTScore F1 and (if judge files exist) the LLM-judge hallucination rating.
Confidence bands are 95% paired bootstrap intervals over notes. A tidy CSV of every plotted value is
written next to each figure (the table view).
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

import plot_style as ps

METRICS = [("rouge_l", "ROUGE-L", "higher is better"),
           ("bert_f1", "BERTScore F1", "higher is better"),
           ("factual", "Judge: factual consistency (1-5)", "higher is better"),
           ("hallucination", "Judge: hallucination (1-5)", "lower is better")]


def canon_layer(v):
    """'' / NaN -> '' ; 20, 20.0, '20' -> '20' (so metric and judge files merge)."""
    s = str(v).strip()
    if s in ("", "nan", "None"):
        return ""
    return str(int(float(s)))


def load_frames(met_dir, judge_dir, dataset):
    mets = [pd.read_csv(f, keep_default_na=False) for f in sorted(glob.glob(os.path.join(met_dir, f"met_{dataset}_*.csv")))]
    if not mets:
        raise SystemExit(f"no metric files in {met_dir}")
    df = pd.concat(mets, ignore_index=True)
    for c in ("rouge_l", "bleu1", "bleu2", "bert_f1", "actual_frac", "rho"):   # empty cells (no score) become NaN
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["layer"] = df["layer"].map(canon_layer)
    judges = [pd.read_csv(f) for f in sorted(glob.glob(os.path.join(judge_dir, f"judge_{dataset}_*.csv")))]
    if judges:
        j = pd.concat(judges, ignore_index=True)
        if "faithfulness_exp" in j.columns:          # scored judge: map onto the four dimensions
            from judge_core import to_rating_dims
            j = to_rating_dims(j)
        j = j[["note_id", "method", "layer", "rho", "hallucination", "factual"]]
        j["layer"] = j["layer"].map(canon_layer)
        df = df.merge(j, on=["note_id", "method", "layer", "rho"], how="left")
    else:
        df["hallucination"] = np.nan
        df["factual"] = np.nan
    return df


def group_name(method):
    return "random" if method.startswith("random_s") else method


def per_note_matrix(df, note_ids, metric, rhos, method_group, layer=None):
    """(n_rho, n_notes) matrix; random is averaged over seeds within each note."""
    sub = df[df["method"].map(group_name) == method_group]
    if layer is not None:
        sub = sub[sub["layer"].astype(str) == str(layer)]
    out = np.full((len(rhos), len(note_ids)), np.nan)
    pos = {n: i for i, n in enumerate(note_ids)}
    for r_i, r in enumerate(rhos):
        s = sub[np.isclose(sub["rho"].astype(float), r)]
        g = s.groupby("note_id")[metric].mean()
        for nid, v in g.items():
            out[r_i, pos[nid]] = v
    return out


def frac_vector(df, rhos, method_group, layer=None):
    sub = df[df["method"].map(group_name) == method_group]
    if layer is not None:
        sub = sub[sub["layer"].astype(str) == str(layer)]
    return np.array([sub[np.isclose(sub["rho"].astype(float), r)]["actual_frac"].astype(float).mean() for r in rhos])


def plot_rho(df, dataset, focus_label, out_png, theme, seed=0):
    t = ps.setup(theme)
    rng = np.random.default_rng(seed)
    note_ids = sorted(df["note_id"].unique())
    rhos = sorted(r for r in df["rho"].astype(float).unique() if r < 1.0)
    panels = [m for m in METRICS if df[m[0]].notna().any()]
    fig, axes = plt_subplots(len(panels))
    rows = []
    series = [("focus", focus_label), ("original", "Column-sum scoring"), ("tfidf", "TF-IDF"),
              ("first", "First sentences"), ("random", "Random")]
    for ax, (metric, title, direction) in zip(axes, panels):
        ps.tidy_axes(ax, t)
        full = per_note_matrix(df, note_ids, metric, [1.0], "full")
        for key, label in series:
            mat = per_note_matrix(df, note_ids, metric, rhos, key)
            if np.isnan(mat).all():
                continue
            m, lo, hi = ps.boot_ci(mat, rng)
            sd = np.nanstd(mat, axis=1, ddof=1)
            x = frac_vector(df, rhos, key)
            ps.draw_series(ax, t, x, m, lo if key in ("focus", "original") else None,
                           hi if key in ("focus", "original") else None, key, label)
            rows += [dict(plot="rho", dataset=dataset, metric=metric, series=label, rho=r, x_actual_frac=xx,
                          mean=mm, std=ss, mean_pm_std=f"{mm:.2f} \u00b1 {ss:.2f}", ci_lo=ll, ci_hi=hh) for r, xx, mm, ss, ll, hh in zip(rhos, x, m, sd, lo, hi)]
        if not np.isnan(full).all():
            fm, fl, fh = ps.boot_ci(full, rng)
            fsd = float(np.nanstd(full, ddof=1))
            ax.axhline(fm[0], color=t["ink2"], linewidth=1.0, linestyle=(0, (4, 3)), zorder=2)
            ax.errorbar([1.0], fm, yerr=[fm - fl, fh - fm], fmt="*", color=t["ink"], markersize=9,
                        markeredgecolor=t["surface"], capsize=0, zorder=4)
            rows.append(dict(plot="rho", dataset=dataset, metric=metric, series="Full note", rho=1.0, x_actual_frac=1.0,
                             mean=fm[0], std=fsd, mean_pm_std=f"{fm[0]:.2f} \u00b1 {fsd:.2f}", ci_lo=fl[0], ci_hi=fh[0]))
        ax.set_title(f"{title}", loc="left")
        ax.set_xlabel("Retained tokens (fraction of note)")
        ax.set_ylabel(direction)
        ax.set_xlim(0, 1.04)
    handles = ps.legend_handles(t, [("focus", "-", "o", focus_label), ("original", "-", "s", "Column-sum scoring"),
                                    ("gray", "-.", "^", "TF-IDF"), ("gray", ":", "v", "First sentences"),
                                    ("gray", "--", "x", "Random (3 seeds)"), (t["ink"], "none", "*", "Full note")])
    fig.legend(handles=handles, loc="lower center", ncol=6, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"Sensitivity to the token budget ({dataset.upper()})", x=0.01, ha="left", fontsize=11)
    save(fig, rows, out_png)


def plot_layer(df, dataset, focus_label, out_png, theme, seed=0):
    t = ps.setup(theme)
    rng = np.random.default_rng(seed)
    note_ids = sorted(df["note_id"].unique())
    focus = df[df["method"] == "focus"]
    layers = sorted({int(float(l)) for l in focus["layer"].astype(str) if l not in ("", "nan")})
    rhos = [r for r in sorted(focus["rho"].astype(float).unique()) if any(abs(r - x) < 1e-9 for x in (0.3, 0.7))]
    layers = [L for L in layers if all(
        ((focus["layer"].astype(str) == str(L)) & (abs(focus["rho"].astype(float) - r) < 1e-9)).any() for r in rhos)]
    colors = ["focus", "original", "aqua"]
    panels = [m for m in METRICS if df[m[0]].notna().any()]
    fig, axes = plt_subplots(len(panels))
    rows = []
    for ax, (metric, title, direction) in zip(axes, panels):
        ps.tidy_axes(ax, t)
        full = per_note_matrix(df, note_ids, metric, [1.0], "full")
        for r_i, rho in enumerate(rhos):
            key = colors[r_i % len(colors)]
            mat = np.vstack([per_note_matrix(df, note_ids, metric, [rho], "focus", layer=L) for L in layers])
            m, lo, hi = ps.boot_ci(mat, rng)
            sd = np.nanstd(mat, axis=1, ddof=1)
            ps.draw_series(ax, t, layers, m, lo, hi, key, f"{focus_label}, rho={rho:g}")
            rows += [dict(plot="layer", dataset=dataset, metric=metric, series=f"{focus_label} rho={rho:g}", layer=L,
                          mean=mm, std=ss, mean_pm_std=f"{mm:.2f} \u00b1 {ss:.2f}", ci_lo=ll, ci_hi=hh)
                     for L, mm, ss, ll, hh in zip(layers, m, sd, lo, hi)]
            rmat = per_note_matrix(df, note_ids, metric, [rho], "random")
            if not np.isnan(rmat).all():
                rm = np.nanmean(rmat)
                ax.axhline(rm, color=t[ps.STYLE[key][0]], linewidth=1.2, linestyle=":", zorder=2)
                rows.append(dict(plot="layer", dataset=dataset, metric=metric, series=f"Random rho={rho:g}", layer=-1,
                                 mean=rm, ci_lo=np.nan, ci_hi=np.nan))
        if not np.isnan(full).all():
            fm = np.nanmean(full)
            ax.axhline(fm, color=t["ink2"], linewidth=1.0, linestyle=(0, (4, 3)), zorder=2)
            rows.append(dict(plot="layer", dataset=dataset, metric=metric, series="Full note", layer=-1, mean=fm,
                             ci_lo=np.nan, ci_hi=np.nan))
        ax.set_title(title, loc="left")
        ax.set_xlabel("Attention layer")
        ax.set_ylabel(direction)
        ax.set_xticks(layers)
    items = [(colors[i % len(colors)], "-", ps.STYLE[colors[i % len(colors)]][2], f"{focus_label}, rho={r:g}")
             for i, r in enumerate(rhos)]
    items += [("gray", ":", "none", "Random selection (same rho)"), ("gray", (0, (4, 3)), "none", "Full note")]
    fig.legend(handles=ps.legend_handles(t, items), loc="lower center", ncol=len(items), bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"Sensitivity to the attention layer ({dataset.upper()})", x=0.01, ha="left", fontsize=11)
    save(fig, rows, out_png)


def plt_subplots(n):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n + 0.4, 3.9))
    return fig, np.atleast_1d(axes)


def save(fig, rows, out_png):
    fig.tight_layout(rect=(0, 0.13, 1, 0.93))
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(os.path.splitext(out_png)[0] + ".pdf", bbox_inches="tight")
    pd.DataFrame(rows).to_csv(os.path.splitext(out_png)[0] + "_values.csv", index=False)
    print(f"saved {out_png} (+ .pdf and _values.csv)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mimic", "soap"], default="mimic")
    ap.add_argument("--plot", choices=["rho", "layer"], required=True)
    ap.add_argument("--focus-label", default="FOCUS")
    ap.add_argument("--met-dir", default="outputs/metrics")
    ap.add_argument("--judge-dir", default="outputs/judge")
    ap.add_argument("--out", default="outputs/figures")
    ap.add_argument("--theme", choices=["light", "dark"], default="light")
    args = ap.parse_args()

    df = load_frames(args.met_dir, args.judge_dir, args.dataset)
    tag = ""
    out_png = os.path.join(args.out, f"{tag}sensitivity_{args.plot}_{args.dataset}_{args.theme}.png")
    (plot_rho if args.plot == "rho" else plot_layer)(df, args.dataset, args.focus_label, out_png, args.theme)


if __name__ == "__main__":
    main()
