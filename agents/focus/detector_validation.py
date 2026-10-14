"""
Validate the two detector signals (AURA and entailment) and choose their thresholds on development notes.

For N dev notes: write a draft summary of each note with the pipeline's FOCUS and DraftAgent; then score every
summary sentence against (a) its own source note (matched, should look supported) and (b) another note's
source (mismatched, should look unsupported). Report AUROC of each signal, and the threshold that maximizes
Youden's J = (flag rate on mismatched) - (flag rate on matched). No human labels, no test notes.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from nltk.tokenize import sent_tokenize
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

from data_utils import load_split  # noqa: E402
from draft_agent import DraftAgent  # noqa: E402
from focus_agent import FocusAgent  # noqa: E402
from hallucination_detector import HallucinationDetectorAgent, ShortAnswerEntailmentJudge  # noqa: E402


def auroc(pos_scores, neg_scores):
    """P(score of a positive > score of a negative), ties count half. Positives are the sentences to be flagged."""
    s = pd.Series(np.concatenate([pos_scores, neg_scores]))
    r = s.rank().values
    n1, n0 = len(pos_scores), len(neg_scores)
    return (r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def best_threshold(matched, mismatched, flag_low):
    """Threshold t maximizing J. flag_low: flag when score < t (AURA); else flag when score > t (entailment)."""
    cands = np.unique(np.concatenate([matched, mismatched]))
    best = (-1.0, None, None, None)
    for t in cands:
        if flag_low:
            fm, fx = np.mean(matched < t), np.mean(mismatched < t)
        else:
            fm, fx = np.mean(matched > t), np.mean(mismatched > t)
        if fx - fm > best[0]:
            best = (fx - fm, float(t), float(fm), float(fx))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mimic", "soap"], required=True)
    ap.add_argument("--n-notes", type=int, default=40)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--rho", type=float, default=0.9)
    ap.add_argument("--shift", type=int, default=7, help="note i is paired with the summary of note (i+shift) mod N")
    ap.add_argument("--aura-layer", type=int, default=-1)
    ap.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-free-gb", type=float, default=16.0)
    args = ap.parse_args()

    hf = os.environ["HUGGINGFACE_HUB_TOKEN"]
    if torch.cuda.mem_get_info()[0] / 1e9 < args.min_free_gb:
        sys.exit("Not enough free GPU memory; resubmit later.")
    tok = AutoTokenizer.from_pretrained(args.model, token=hf)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map={"": 0}, token=hf).eval()
    tables = os.path.join(HERE, "..", "..", "..", "outputs", "focus", f"layer_tables_{args.dataset}.pt")
    focus = FocusAgent(model, tok, args.layer, args.rho, tables)
    draft = DraftAgent(model, tok, args.dataset)
    judge = ShortAnswerEntailmentJudge(model, tok)
    det = HallucinationDetectorAgent(model, tok, judge, args.dataset, aura_layer=args.aura_layer)

    notes = load_split(args.dataset, "dev", args.n_notes, 42)
    texts = [r["text"].strip() for r in notes]
    summaries = []
    for i, t in enumerate(texts):
        summaries.append(draft.generate(focus.compress(t)["sentences"]))
        if (i + 1) % 10 == 0:
            print(f"drafted {i + 1}/{len(texts)}", flush=True)

    rows = []
    N = len(texts)
    for i in range(N):
        for matched, j in ((1, i), (0, (i + args.shift) % N)):
            summ = summaries[j]
            spans = sent_tokenize(summ)
            if not spans:
                continue
            judge.reset()
            res = det.token_aura(texts[i], summ)
            if res is None:
                continue
            aura = det._span_aura(summ, spans, res[0], res[1])
            for k, sp in enumerate(spans):
                p = judge.judge(texts[i], sp)["p_not_supported"]
                rows.append((i, j, matched, k, aura[k], p))
        if (i + 1) % 5 == 0:
            print(f"scored {i + 1}/{N}", flush=True)

    df = pd.DataFrame(rows, columns=["source_note", "summary_note", "matched", "span", "aura", "p_not"])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out + ".csv", index=False)
    m, x = df[df.matched == 1], df[df.matched == 0]
    out = {"dataset": args.dataset, "n_notes": N, "n_matched_spans": len(m), "n_mismatched_spans": len(x)}
    # AURA: low means unsupported, so the mismatched sentences are the positives of "low"; negate for AUROC.
    out["auroc_aura"] = float(auroc(-x["aura"].values, -m["aura"].values))
    out["auroc_entailment"] = float(auroc(x["p_not"].values, m["p_not"].values))
    for name, flag_low in (("aura", True), ("p_not", False)):
        j, t, fm, fx = best_threshold(m[name].values, x[name].values, flag_low)
        out[f"best_{name}"] = {"youden_j": j, "threshold": t, "flag_rate_matched": fm, "flag_rate_mismatched": fx}
    with open(args.out + ".json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
