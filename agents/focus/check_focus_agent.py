"""
Check that FocusAgent (the pipeline's FOCUS) reproduces the scores saved by the layer sweep, for a few
development notes: same sentences, same conditional scores at the chosen layer, and report the selection.
Run through run_stage.sbatch (STAGE=check_agent).
"""
import argparse
import json
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from focus_agent import FocusAgent
from focus_scoring import select_by_token_budget


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--rho", type=float, default=0.3)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--out", default="../../outputs")
    ap.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    args = ap.parse_args()

    hf = os.environ["HUGGINGFACE_HUB_TOKEN"]
    tok = AutoTokenizer.from_pretrained(args.model, token=hf)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map={"": 0}, token=hf).eval()
    agent = FocusAgent(model, tok, args.layer, args.rho, os.path.join(args.out, f"layer_tables_{args.dataset}.pt"))

    meta = json.load(open(os.path.join(args.out, f"layer_meta_{args.dataset}_dev.json")))
    li = meta["layers"].index(args.layer)
    mi = meta["modes"].index("conditional")
    z = np.load(os.path.join(args.out, f"layer_scores_{args.dataset}_dev.npz"))
    notes = [json.loads(l) for l in open(os.path.join(args.out, f"layer_notes_{args.dataset}_dev.jsonl"))]

    worst = 0.0
    for i in range(min(args.n, len(notes))):
        sents = notes[i]["sentences"]
        scores, tc, n = agent.score_sentences(sents)
        ref = z[f"n{i}"][li, mi]
        ok = np.isfinite(ref)
        rel = float(np.max(np.abs(scores[ok] - ref[ok]) / (np.abs(ref[ok]) + 1e-9)))
        worst = max(worst, rel)
        chosen = select_by_token_budget(scores, tc, args.rho)
        print(f"note {i}: {len(sents)} sentences, {n} tokens, max rel diff {rel:.2e}, kept {len(chosen)} sentences "
              f"({int(tc[chosen].sum())}/{int(tc.sum())} tokens)", flush=True)
        assert tc.tolist() == notes[i]["tok_counts"], "token counts differ from the sweep"
    print(f"worst relative difference {worst:.2e}")
    assert worst < 5e-3, "FocusAgent disagrees with the sweep scores"
    print("check_agent OK")


if __name__ == "__main__":
    main()
