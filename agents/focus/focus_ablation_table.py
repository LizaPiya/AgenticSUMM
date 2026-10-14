"""
FOCUS scoring ablation (CPU only): original column sum, horizon, harmonic and conditional scoring at the
chosen layer, on the development notes of each dataset. Reads the saved layer diagnostics.

  position_spearman : correlation of a token/sentence score with its relative position in the note (0 = no trend)
  early_share       : share of retained tokens that come from the first 30% of the note (0.30 = no trend)
  coverage_gain     : reference-word coverage of the retained text minus that of random sentences of equal size
Cells are mean +- std across development notes.
"""
import argparse
import os

import pandas as pd

MODES = [("original", "Column sum"), ("horizon", "Per-reader (horizon)"), ("harmonic", "Harmonic"),
         ("conditional", "Conditional (FOCUS)"), ("conditional:cross", "Conditional, later readers only")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fig-dir", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    frames = []
    for ds in ("mimic", "soap"):
        d = pd.read_csv(os.path.join(args.fig_dir, f"layer_diag_{ds}.csv"))
        frames.append(d[d.layer == args.layer])
    df = pd.concat(frames)
    print("metrics:", sorted(df.metric.unique()), "| rho values:", sorted(df.rho.dropna().unique()))
    rows, tex = [], []
    for ds in ("mimic", "soap"):
        for mode, label in MODES:
            rec = {"dataset": ds, "mode": mode, "label": label}
            for metric, key in (("position_spearman", "spearman"), ("early_share", "early30"), ("coverage_gain", "cov_gain")):
                sub = df[(df.dataset == ds) & (df["mode"] == mode) & (df.metric == metric)]
                for _, r in sub.iterrows():
                    suffix = "" if pd.isna(r.rho) else f"_rho{r.rho:g}"
                    rec[key + suffix] = r.mean_pm_std
            rows.append(rec)
    out = pd.DataFrame(rows)
    os.makedirs(args.out, exist_ok=True)
    out.to_csv(os.path.join(args.out, f"focus_ablation_L{args.layer}.csv"), index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    for ds in ("mimic", "soap"):
        print(f"\n=== {ds}, layer {args.layer}")
        print(out[out.dataset == ds].drop(columns=["dataset", "mode"]).to_string(index=False))


if __name__ == "__main__":
    main()
