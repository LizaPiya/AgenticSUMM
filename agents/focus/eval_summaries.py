"""
Score generated summaries with the metric code in agents/Evaluation.py: ROUGE-L, BLEU-1/2
and BERTScore F1 against the reference. One resumable metrics CSV per generated-summary file.
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from Evaluation import clean_text, compute_bert_score_batched, compute_bleu_scores, compute_rouge_l  # noqa: E402

from sweep_io import load_sweep  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mimic", "soap"], required=True)
    ap.add_argument("--sweep-dir", default="outputs")
    ap.add_argument("--gen-dir", default="outputs/gen")
    ap.add_argument("--met-dir", default="outputs/metrics")
    args = ap.parse_args()

    _, notes, _, _ = load_sweep(args.sweep_dir, args.dataset, "dev")
    ref_by_id = {n["note_id"]: n["reference"] for n in notes}
    os.makedirs(args.met_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.gen_dir, f"gen_{args.dataset}_*.csv")))
    todo = [f for f in files if not os.path.exists(os.path.join(args.met_dir, "met_" + os.path.basename(f)[4:]))]
    print(f"{len(files)} generated files, {len(todo)} to score", flush=True)
    for f in todo:
        df = pd.read_csv(f, keep_default_na=False)
        refs = [clean_text(ref_by_id[i]) for i in df["note_id"]]
        cands = [clean_text(s) for s in df["summary"]]
        b1, b2, rl = [], [], []
        for r, c in zip(refs, cands):
            if not r or not c:
                b1.append(np.nan); b2.append(np.nan); rl.append(np.nan)
            else:
                x, y = compute_bleu_scores(r, c)
                b1.append(x); b2.append(y); rl.append(compute_rouge_l(r, c))
        _, _, bf = compute_bert_score_batched(refs, cands)
        df["rouge_l"], df["bleu1"], df["bleu2"], df["bert_f1"] = rl, b1, b2, bf
        df["empty_summary"] = [int(not c) for c in cands]
        df.drop(columns=["summary"]).to_csv(os.path.join(args.met_dir, "met_" + os.path.basename(f)[4:]), index=False)
        print(f"scored {os.path.basename(f)}: ROUGE-L {np.nanmean(rl):.2f}  BERT-F {np.nanmean(bf):.2f}", flush=True)


if __name__ == "__main__":
    main()
