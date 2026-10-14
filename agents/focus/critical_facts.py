"""
Critical-fact retention under compression (CPU only, development notes of the layer sweep).

For every note, the reference summary supplies the facts that matter. A reference term counts as a fact if it also
occurs in the source note (so the compressed input could have kept it). Three term sets are tracked:
  all      : content words (not stop words) of the reference found in the source
  numeric  : terms containing a digit (doses, lab values, dates, counts)
  specific : terms that occur in at most 10% of the source notes (diagnoses, drug names, procedures)
Retention = share of those terms present in the sentences a selector keeps at a given token budget. FOCUS (conditional,
chosen layer) is compared with column-sum scoring, harmonic scoring, first sentences, TF-IDF sentences and random
sentences (mean of 3 seeds). All selectors keep the same token budget. Mean +- std across notes, and the paired
difference between FOCUS and random with a 95% bootstrap interval.
"""
import argparse
import json
import os
import re

import numpy as np
import pandas as pd

from focus_scoring import baseline_scores, select_by_token_budget

STOP = set("""a about above after again all also am an and any are as at be because been before being below between both but by can
could did do does doing down during each few for from further had has have having he her here hers him his how i if in into is it
its just me more most my no nor not now of off on once only or other our out over own same she should so some such than that the
their them then there these they this those through to too under until up very was we were what when where which while who whom why
will with would you your patient pt mg po daily day days given also however s per x admission admitted discharge discharged""".split())
rng = np.random.default_rng(0)


def toks(s):
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def boot(d, n=2000):
    d = np.asarray(d, float)
    d = d[~np.isnan(d)]
    m = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return f"{d.mean():+.3f} [{np.percentile(m, 2.5):+.3f},{np.percentile(m, 97.5):+.3f}]"


def run(ds, fig_dir, layer, budgets):
    meta = json.load(open(os.path.join(fig_dir, f"layer_meta_{ds}_dev.json")))
    z = np.load(os.path.join(fig_dir, f"layer_scores_{ds}_dev.npz"))
    notes = [json.loads(l) for l in open(os.path.join(fig_dir, f"layer_notes_{ds}_dev.jsonl"))]
    li = meta["layers"].index(layer)
    mi = {m: meta["modes"].index(m) for m in ("conditional", "original", "harmonic")}

    src = [toks(" ".join(n["sentences"])) for n in notes]
    dfreq = {}
    for s in src:
        for t in s:
            dfreq[t] = dfreq.get(t, 0) + 1
    N = len(notes)

    methods = ["FOCUS", "column_sum", "harmonic", "first", "tfidf", "random"]
    recs = []
    for i, n in enumerate(notes):
        sents, tc = n["sentences"], np.array(n["tok_counts"])
        ref = [t for t in re.findall(r"[a-z0-9]+", n["reference"].lower()) if t not in STOP and (len(t) >= 3 or any(c.isdigit() for c in t))]
        facts = {t for t in ref if t in src[i]}
        sets = {"all": facts,
                "numeric": {t for t in facts if any(c.isdigit() for c in t)},
                "specific": {t for t in facts if dfreq.get(t, 0) <= 0.10 * N}}
        if not sets["all"]:
            continue
        per = [toks(s) for s in sents]
        sc = {"FOCUS": z[f"n{i}"][li, mi["conditional"]], "column_sum": z[f"n{i}"][li, mi["original"]],
              "harmonic": z[f"n{i}"][li, mi["harmonic"]], "first": baseline_scores("first", sents, rng),
              "tfidf": baseline_scores("tfidf", sents, rng)}
        for rho in budgets:
            for meth in methods:
                reps = 3 if meth == "random" else 1
                for rep in range(reps):
                    s_ = sc[meth] if meth != "random" else baseline_scores("random", sents, np.random.default_rng(1000 * i + rep))
                    chosen = select_by_token_budget(np.asarray(s_, float), tc, rho)
                    kept = set().union(*[per[j] for j in chosen]) if chosen else set()
                    rec = {"note": i, "rho": rho, "method": meth, "rep": rep, "kept_token_share": tc[chosen].sum() / max(tc.sum(), 1)}
                    for k, v in sets.items():
                        rec[k] = len(v & kept) / len(v) if v else np.nan
                    recs.append(rec)
    df = pd.DataFrame(recs)
    df = df.groupby(["note", "rho", "method"], as_index=False).mean(numeric_only=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fig-dir", required=True, help="folder with layer_scores/notes/meta files")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    pd.set_option("display.width", 200, "display.max_columns", 20)
    for ds in ("mimic", "soap"):
        df = run(ds, args.fig_dir, args.layer, [0.3, 0.5, 0.7, 0.9])
        df.to_csv(os.path.join(args.out, f"critical_facts_{ds}.csv"), index=False)
        print(f"\n=== {ds}: share of reference facts kept (mean ± std over notes), layer {args.layer}")
        for k in ("all", "numeric", "specific"):
            tab = df.groupby(["rho", "method"])[k].agg(["mean", "std"]).reset_index()
            tab["cell"] = tab.apply(lambda r: f"{r['mean']:.3f} ± {r['std']:.3f}", axis=1)
            print(f"\n[{k}]")
            print(tab.pivot(index="method", columns="rho", values="cell").reindex(["FOCUS", "column_sum", "harmonic", "first", "tfidf", "random"]).to_string())
        print("\nFOCUS minus random, paired over notes [95% CI]:")
        for rho in (0.3, 0.5, 0.7):
            a = df[(df.rho == rho) & (df.method == "FOCUS")].set_index("note")
            b = df[(df.rho == rho) & (df.method == "random")].set_index("note")
            print(f"  rho {rho}: " + "  ".join(f"{k} {boot(a[k] - b[k])}" for k in ("all", "numeric", "specific")))


if __name__ == "__main__":
    main()
