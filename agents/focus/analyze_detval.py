"""
From detector_validation output: for a false-alarm budget on matched sentences, the threshold of each signal,
the share of mismatched sentences it catches, and the union (OR) rates. CPU only.
"""
import argparse

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    args = ap.parse_args()
    df = pd.read_csv(args.csv)
    m, x = df[df.matched == 1], df[df.matched == 0]
    print(f"{len(m)} matched, {len(x)} mismatched sentences")
    print(f"{'matched budget':>14} | {'AURA tau':>8} {'catch':>6} | {'p_not cut':>9} {'catch':>6} | OR: matched flagged, mismatched caught")
    for b in (0.02, 0.05, 0.08, 0.10, 0.15, 0.20):
        tau = np.quantile(m["aura"], b)                    # flag aura < tau  -> b of matched flagged
        cut = np.quantile(m["p_not"], 1 - b)               # flag p_not > cut -> b of matched flagged
        fa_a, fa_e = (m["aura"] < tau).mean(), (m["p_not"] > cut).mean()
        ca, ce = (x["aura"] < tau).mean(), (x["p_not"] > cut).mean()
        u_m = ((m["aura"] < tau) | (m["p_not"] > cut)).mean()
        u_x = ((x["aura"] < tau) | (x["p_not"] > cut)).mean()
        print(f"{b:>14.2f} | {tau:>8.5f} {ca:>6.2f} | {cut:>11.8f} {ce:>6.2f} | {u_m:.2f}, {u_x:.2f}")
    print("\nentailment alone at its own budget, matched flagged -> mismatched caught:")
    for b in (0.02, 0.05, 0.10, 0.15):
        cut = np.quantile(m["p_not"], 1 - b)
        print(f"  {b:.2f}: cut {cut:.4f}, caught {(x['p_not'] > cut).mean():.2f}")


if __name__ == "__main__":
    main()
