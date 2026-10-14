"""
Cross-dataset step of SELECTION_RULES.md, step 1: if the datasets pick different layers on their own, use the layer
with the largest *smaller* per-dataset gain over random (the layer that is least bad on both). Reads the per-dataset
tables written by select_layer.py. Layers excluded by the judge check in either dataset are dropped.
"""
import argparse
import os

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="outputs/figures")
    ap.add_argument("--datasets", nargs="+", default=["mimic", "soap"])
    args = ap.parse_args()
    tabs = {d: pd.read_csv(os.path.join(args.dir, f"layer_selection_{d}.csv")).set_index("layer") for d in args.datasets}
    layers = sorted(set.intersection(*[set(t.index) for t in tabs.values()]))
    rows = []
    for L in layers:
        rec = {"layer": L}
        for d, t in tabs.items():
            rec[f"gain_{d}"] = t.loc[L, "gain_over_random"]
            rec[f"ci_{d}"] = f"[{t.loc[L, 'ci_lo']:.2f}, {t.loc[L, 'ci_hi']:.2f}]"
            rec[f"judge_ok_{d}"] = bool(t.loc[L, "kept_by_judge_check"])
        rec["min_gain"] = min(rec[f"gain_{d}"] for d in tabs)
        rec["all_judge_ok"] = all(rec[f"judge_ok_{d}"] for d in tabs)
        rows.append(rec)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(args.dir, "layer_selection_joint.csv"), index=False)
    print(out.round(3).to_string(index=False))
    own = {d: int(t["gain_over_random"].idxmax()) for d, t in tabs.items()}
    cand = out[out["all_judge_ok"]]
    chosen = int(cand.loc[cand["min_gain"].idxmax(), "layer"])
    print(f"\nbest layer by gain within each dataset: {own}")
    print(f"layer chosen by the cross-dataset rule (largest smaller gain): {chosen}")


if __name__ == "__main__":
    main()
