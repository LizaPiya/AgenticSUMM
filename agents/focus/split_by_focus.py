"""
Table I (and the Phi-core jury where available) split by whether FOCUS compressed the note or passed it whole
(notes with fewer than 8 sentences are returned whole). CPU only.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

from judge_core import to_rating_dims

METRICS = ["rouge_l", "bleu2", "bert_f1"]
DIMS = ["hallucination", "factual", "complete", "coherent"]
rng = np.random.default_rng(0)


def ms(x):
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna()
    return f"{x.mean():.2f} ± {x.std():.2f}" if len(x) else "-"


def boot(d, n=2000):
    d = np.asarray(d, float)
    d = d[~np.isnan(d)]
    if not len(d):
        return "-"
    m = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return f"{d.mean():+.2f} [{np.percentile(m, 2.5):+.2f},{np.percentile(m, 97.5):+.2f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table1-dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--pipeline", required=True)
    ap.add_argument("--jury-dir", default=None)
    ap.add_argument("--judge", default="phi35_mini")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pipe = pd.read_csv(args.pipeline, keep_default_na=False)
    pipe["subset"] = np.where(pipe["focus_fallback"] == "", "FOCUS applied", "passed whole")
    subset = dict(zip(pipe["note_id"], pipe["subset"]))
    print(pipe["subset"].value_counts().to_dict())

    ev = {}
    for f in glob.glob(os.path.join(args.table1_dir, f"evalsys_{args.tag}_*.csv")):
        ev[os.path.basename(f)[len(f"evalsys_{args.tag}_"):-4]] = pd.read_csv(f)
    rows = []
    for name, d in ev.items():
        d = d.copy()
        d["subset"] = d["note_id"].map(subset)
        for sub, g in d.groupby("subset"):
            rec = {"system": name, "subset": sub, "n": len(g)}
            for m in METRICS:
                rec[m] = ms(g[m])
            rows.append(rec)
    out = pd.DataFrame(rows).sort_values(["subset", "system"])
    os.makedirs(args.out, exist_ok=True)
    out.to_csv(os.path.join(args.out, f"split_{args.tag}_overlap.csv"), index=False)
    pd.set_option("display.width", 200, "display.max_columns", 20)
    print("\nOverlap metrics by subset (mean ± std):")
    print(out.to_string(index=False))

    if "agenticsum" in ev and "llama32_3b" in ev:
        a, b = ev["agenticsum"].set_index("note_id"), ev["llama32_3b"].set_index("note_id")
        print("\nAgenticSum minus plain Llama-3.2-3B, paired bootstrap, by subset:")
        for sub in sorted(set(subset.values())):
            ids = [i for i in a.index if subset.get(i) == sub and i in b.index]
            print(f"  {sub:14s} (n={len(ids)}): " + "  ".join(f"{m} {boot(a.loc[ids, m].values - b.loc[ids, m].values)}" for m in METRICS))

    if args.jury_dir and glob.glob(os.path.join(args.jury_dir, f"{args.judge}__*.csv")):
        j = to_rating_dims(pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(args.jury_dir, f"{args.judge}__*.csv"))]))
        order = {r: n for r, n in zip(range(len(pipe)), pipe["note_id"])}
        j["subset"] = j["row"].map(order).map(subset)
        print(f"\nJury ({args.judge}) by subset:")
        for sub, g in j.groupby("subset"):
            print(f"  {sub}")
            print(g.groupby("system")[DIMS].mean().round(2).to_string())


if __name__ == "__main__":
    main()
