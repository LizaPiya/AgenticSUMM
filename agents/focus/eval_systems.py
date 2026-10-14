"""
Table I for every system in a manifest: ROUGE-L, BLEU-1, BLEU-2 and BERTScore F1 against the reference, using the
metric code in agents/Evaluation.py, reported as mean +- std across notes as mean +- std cells.

Manifest CSV columns: system, family, csv, source_col, summary_col, reference_col (reference_col defaults to "target").
The same manifest feeds judge_jury.py, so Table I and Table II list the same systems.

Outputs (per dataset, under --out):
  evalsys_<system>.csv        per-note metrics (resumable: existing files are reused)
  table1_<tag>.csv            one row per system, "mean ± std" cells and the number of scored notes
  table1_<tag>.tex            LaTeX rows in a table style ($x_{\\pm s}$), best value per column in bold
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from Evaluation import clean_text, compute_bert_score_batched, compute_bleu_scores, compute_rouge_l  # noqa: E402

from tables import cell, mean_std, plain  # noqa: E402

METRICS = [("rouge_l", "ROUGE-L"), ("bleu1", "BLEU-1"), ("bleu2", "BLEU-2"), ("bert_f1", "BERT-F")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--tag", required=True, help="e.g. mimic_test")
    ap.add_argument("--out", default="outputs/table1")
    ap.add_argument("--display", default=None, help="optional CSV: system,display_name for the LaTeX rows")
    ap.add_argument("--min-rows", type=int, default=500, help="skip systems whose output is still incomplete")
    args = ap.parse_args()

    man = pd.read_csv(args.manifest)
    os.makedirs(args.out, exist_ok=True)
    names = {}
    if args.display:
        d = pd.read_csv(args.display)
        names = dict(zip(d["system"], d["display_name"]))

    rows, per_system = [], {}
    for _, s in man.iterrows():
        if not os.path.exists(s["csv"]):
            print(f"skip {s['system']}: {s['csv']} not found yet", flush=True)
            continue
        n_src = len(pd.read_csv(s["csv"], usecols=[0], keep_default_na=False))
        if n_src < args.min_rows:
            print(f"skip {s['system']}: only {n_src} of {args.min_rows} notes written so far", flush=True)
            continue
        path = os.path.join(args.out, f"evalsys_{args.tag}_{s['system']}.csv")
        ref_col = s["reference_col"] if "reference_col" in man.columns and isinstance(s.get("reference_col"), str) else "target"
        if os.path.exists(path):
            m = pd.read_csv(path)
        else:
            df = pd.read_csv(s["csv"], keep_default_na=False)
            refs = [clean_text(t) for t in df[ref_col]]
            cands = [clean_text(t) for t in df[s["summary_col"]]]
            b1, b2, rl = [], [], []
            for r, c in zip(refs, cands):
                if not r or not c:
                    b1.append(np.nan); b2.append(np.nan); rl.append(np.nan)
                else:
                    x, y = compute_bleu_scores(r, c)
                    b1.append(x); b2.append(y); rl.append(compute_rouge_l(r, c))
            _, _, bf = compute_bert_score_batched(refs, cands)
            m = pd.DataFrame({"note_id": df["note_id"], "rouge_l": rl, "bleu1": b1, "bleu2": b2, "bert_f1": bf,
                              "empty_summary": [int(not c) for c in cands]})
            m.to_csv(path, index=False)
        per_system[s["system"]] = m
        rec = {"system": s["system"], "family": s.get("family", ""), "n_notes": len(m), "n_empty": int(m["empty_summary"].sum())}
        for key, label in METRICS:
            rec[label] = plain(m[key].dropna())
            rec[label + "_mean"] = mean_std(m[key])[0]
        rows.append(rec)
        print(f"{s['system']:22s} n={len(m)}  " + "  ".join(f"{lab} {rec[lab]}" for _, lab in METRICS), flush=True)

    if not rows:
        raise SystemExit("no systems found")
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(args.out, f"table1_{args.tag}.csv"), index=False)

    best = {lab: out[lab + "_mean"].max() for _, lab in METRICS}
    lines = []
    for r in rows:
        cells = []
        for key, lab in METRICS:
            cells.append(cell(per_system[r["system"]][key].dropna(), bold=abs(r[lab + "_mean"] - best[lab]) < 1e-12))
        lines.append(f"\\texttt{{{names.get(r['system'], r['system'])}}} & " + " & ".join(cells) + " \\\\")
    with open(os.path.join(args.out, f"table1_{args.tag}.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"saved table1_{args.tag}.csv and .tex in {args.out}")


if __name__ == "__main__":
    main()
