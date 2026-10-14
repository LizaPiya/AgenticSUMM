"""
Which detector signal flags the sentences? Reads the pipeline CSV (detect_log column) and prints, per round,
the AURA and entailment-probability distributions, the flag rate of each signal alone, and the flag rate
under a range of AURA thresholds. CPU only.
"""
import argparse
import json

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--tau", type=float, default=0.42)
    ap.add_argument("--p-cut", type=float, default=0.5)
    args = ap.parse_args()

    df = pd.read_csv(args.csv, keep_default_na=False)
    rows = []
    for _, r in df.iterrows():
        if not r["detect_log"]:
            continue
        for rd, d in enumerate(json.loads(r["detect_log"])):
            for a, p, m in zip(d["aura"], d["p_not"], d["mask"]):
                rows.append((r["note_id"], rd, a, p, m))
    s = pd.DataFrame(rows, columns=["note", "round", "aura", "p_not", "mask"])
    print(f"{len(df)} notes, {len(s)} span checks, rounds {sorted(s['round'].unique())}")
    q = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]
    for name in ("aura", "p_not"):
        print(f"\n{name} quantiles {q}:")
        for rd, g in s.groupby("round"):
            print(f"  round {rd} n={len(g)}:", np.round(g[name].quantile(q).values, 3).tolist())
    print(f"\nflag rate by signal (tau={args.tau}, p_cut={args.p_cut}), by round:")
    for rd, g in s.groupby("round"):
        print(f"  round {rd}: aura<tau {np.mean(g['aura'] < args.tau):.2f} | p_not>cut {np.mean(g['p_not'] > args.p_cut):.2f} "
              f"| either {np.mean(g['mask']):.2f}")
    print("\nflag rate of AURA alone at other thresholds (all rounds):")
    for t in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.42):
        print(f"  tau {t:.2f}: {np.mean(s['aura'] < t):.2f}")
    print("\nflag rate of entailment alone at other cuts (all rounds):")
    for c in (0.5, 0.7, 0.9, 0.97, 0.99):
        print(f"  p_not > {c}: {np.mean(s['p_not'] > c):.2f}")


if __name__ == "__main__":
    main()
