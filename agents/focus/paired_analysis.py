"""
Paired analysis of AgenticSum against every other system (CPU only, no model).

For each rating dimension of the Phi-only core jury (same judge for every system, so ratings are comparable) and
for ROUGE-L / BERT-F, report the mean per-note difference AgenticSum - system with a 95% paired bootstrap interval
over notes, and the share of notes where AgenticSum is strictly better. Hallucination is lower-is-better, so its
sign is flipped (positive always means AgenticSum is better).

Also: the notes where the repair loop actually ran (at least one sentence flagged in round 0), comparing AgenticSum
(final) with its own draft (C2) on the same notes.
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from judge_core import to_rating_dims

DIMS = ["hallucination", "factual", "complete", "coherent"]
rng = np.random.default_rng(0)


def boot(diff, n=2000):
    diff = np.asarray(diff, float)
    diff = diff[~np.isnan(diff)]
    if len(diff) == 0:
        return np.nan, np.nan, np.nan
    idx = rng.integers(0, len(diff), (n, len(diff)))
    means = diff[idx].mean(1)
    return diff.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jury-dir", required=True)
    ap.add_argument("--judge", default="phi35_mini")
    ap.add_argument("--table1-dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--pipeline", required=True, help="merged pipeline CSV (for the repair subset)")
    ap.add_argument("--ref", default="agenticsum")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    frames = [pd.read_csv(f) for f in sorted(glob.glob(os.path.join(args.jury_dir, f"{args.judge}__*.csv")))]
    piv = None
    if frames:                                              # the jury part is skipped when no ratings exist yet
        j = to_rating_dims(pd.concat(frames, ignore_index=True))
        bad_rows = set(j[j["skipped_long"] == 1]["row"])
        j = j[~j["row"].isin(bad_rows)]
        piv = {d: j.pivot(index="row", columns="system", values=d) for d in DIMS}

    ev = {}
    for f in glob.glob(os.path.join(args.table1_dir, f"evalsys_{args.tag}_*.csv")):
        name = os.path.basename(f)[len(f"evalsys_{args.tag}_"):-4]
        ev[name] = pd.read_csv(f).reset_index(drop=True)

    rows = []
    systems = sorted((set(piv["hallucination"].columns) if piv else set(ev)) - {args.ref})
    for s in systems:
        rec = {"system": s}
        for d in (DIMS if piv else []):
            sign = -1.0 if d == "hallucination" else 1.0
            diff = sign * (piv[d][args.ref] - piv[d][s])
            m, lo, hi = boot(diff)
            rec[f"{d}_diff"] = f"{m:+.3f} [{lo:+.3f},{hi:+.3f}]"
            rec[f"{d}_better_share"] = round(float((diff > 0).mean()), 2)
        for k in ("rouge_l", "bert_f1"):
            if args.ref in ev and s in ev:
                a, b = ev[args.ref][k].values, ev[s][k].values
                n = min(len(a), len(b))
                m, lo, hi = boot(a[:n] - b[:n])
                rec[f"{k}_diff"] = f"{m:+.2f} [{lo:+.2f},{hi:+.2f}]"
        rows.append(rec)
    out = pd.DataFrame(rows)
    os.makedirs(args.out, exist_ok=True)
    out.to_csv(os.path.join(args.out, "paired_vs_agenticsum.csv"), index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 40)
    print(f"Positive = {args.ref} better. Judge: {args.judge}. Mean difference [95% bootstrap CI] over notes.\n")
    if piv:
        print(out[["system"] + [f"{d}_diff" for d in DIMS]].to_string(index=False))
        print("\nShare of notes where AgenticSum is strictly better:")
        print(out[["system"] + [f"{d}_better_share" for d in DIMS]].to_string(index=False))
    print("\nOverlap metrics (same paired bootstrap):")
    cols = [c for c in ("rouge_l_diff", "bert_f1_diff") if c in out.columns]
    print(out[["system"] + cols].to_string(index=False))
    if not piv:
        return

    pipe = pd.read_csv(args.pipeline, keep_default_na=False)
    pipe = pipe.reset_index().rename(columns={"index": "row"})
    flagged = pipe[pd.to_numeric(pipe["n_flagged_first"], errors="coerce") > 0]["row"]
    print(f"\nNotes where the repair loop ran (>=1 sentence flagged in round 0): {len(flagged)} of {len(pipe)}")
    for d in DIMS:
        sign = -1.0 if d == "hallucination" else 1.0
        sub = piv[d].loc[piv[d].index.isin(flagged)]
        diff = sign * (sub[args.ref] - sub["agenticsum_c2_focus_draft"])
        m, lo, hi = boot(diff)
        print(f"  {d:14s} final vs draft on those notes: {m:+.3f} [{lo:+.3f},{hi:+.3f}]  (positive = repair helped)")


if __name__ == "__main__":
    main()
