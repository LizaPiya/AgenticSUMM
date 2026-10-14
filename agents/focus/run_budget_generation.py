"""
Generate DraftAgent summaries from the sentences each selection method keeps (sensitivity analysis).

Reads the files saved by run_layer_sweep.py, so no attention is recomputed. For every
(method, layer, rho) configuration it writes one resumable CSV; finished configurations are skipped.

  --plot rho   : one FOCUS variant at one layer, swept over rho, next to column-sum scoring (last layer),
                 first-k, TF-IDF, random (3 seeds) and the full note (rho = 1.0).
  --plot layer : one FOCUS variant at the chosen layers for rho in config["layer_rhos"], plus random
                 selection at the same rho and the full note.

Selection uses an exact token budget (focus_scoring.select_by_token_budget). Decoding is greedy with
the DraftAgent settings; only the newly generated tokens are decoded.
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from focus_scoring import baseline_scores, select_by_token_budget
from prompts import DRAFT_MAX_NEW_TOKENS, build_draft_prompt
from sweep_io import load_sweep


def sanitize(x) -> str:
    return str(x).replace(":", "-").replace(".", "p")


def make_configs(plot, cfg, last_layer, focus_spec, layer):
    """List of (method, spec, layer, rho). method in focus|original|first|tfidf|random_sK|full."""
    cfgs = []
    seeds = cfg["random_seeds"]
    if plot == "rho":
        for rho in cfg["rhos_generation"]:
            cfgs.append(("focus", focus_spec, layer, rho))
            cfgs.append(("original", "original", last_layer, rho))
            cfgs.append(("first", None, None, rho))
            cfgs.append(("tfidf", None, None, rho))
            cfgs += [(f"random_s{s}", None, None, rho) for s in seeds]
    else:
        for L in cfg["layers_generation"]:
            for rho in cfg["layer_rhos"]:
                cfgs.append(("focus", focus_spec, L, rho))
        for rho in cfg["layer_rhos"]:
            cfgs += [(f"random_s{s}", None, None, rho) for s in seeds]
    cfgs.append(("full", None, None, 1.0))
    seen, out = set(), []
    for c in cfgs:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def select_indices(method, spec, layer, rho, ni, note, scores_fn, meta, base_seed):
    m = len(note["sentences"])
    tc = np.array(note["tok_counts"])
    if method == "full":
        return list(range(m))
    if method in ("focus", "original"):
        li = meta["layers"].index(layer)
        si = meta["modes"].index(spec)
        sc = scores_fn(ni)[li, si]
    elif method.startswith("random_s"):
        s = int(method.split("_s")[1])
        sc = baseline_scores("random", note["sentences"], np.random.default_rng(1000 * (base_seed + s) + ni))
    else:  # first / tfidf
        sc = baseline_scores(method, note["sentences"], np.random.default_rng(0))
    return select_by_token_budget(sc, tc, rho, "none")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mimic", "soap"], required=True)
    ap.add_argument("--plot", choices=["rho", "layer"], required=True)
    ap.add_argument("--focus-spec", required=True, help="e.g. conditional, conditional:cross, self (once added)")
    ap.add_argument("--layer", type=int, default=None, help="layer for --plot rho")
    ap.add_argument("--sweep-dir", default="outputs")
    ap.add_argument("--gen-dir", default="outputs/gen")
    ap.add_argument("--config", default="sensitivity_config.json")
    ap.add_argument("--n-notes", type=int, default=None)
    ap.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--min-free-gb", type=float, default=12.0)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    meta, notes, scores_fn, _ = load_sweep(args.sweep_dir, args.dataset, "dev")
    n_notes = min(args.n_notes or cfg["n_generation_notes"], len(notes))
    notes = notes[:n_notes]                              # same notes for every configuration
    last_layer = meta["layers"][-1]
    if args.plot == "rho":
        assert args.layer is not None, "--plot rho needs --layer"
        assert args.layer in meta["layers"], "layer not in the sweep"
    os.makedirs(args.gen_dir, exist_ok=True)

    cfgs = make_configs(args.plot, cfg, last_layer, args.focus_spec, args.layer)
    todo = []
    for method, spec, layer, rho in cfgs:
        tag = f"{method}_{sanitize(spec)}_L{layer}_r{sanitize(rho)}"
        path = os.path.join(args.gen_dir, f"gen_{args.dataset}_{tag}.csv")
        if not os.path.exists(path):
            todo.append((method, spec, layer, rho, path))
    print(f"{len(cfgs)} configurations, {len(todo)} to generate, {n_notes} notes each", flush=True)
    if not todo:
        return

    hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
    assert hf_token, "HUGGINGFACE_HUB_TOKEN not set"
    assert torch.cuda.is_available(), "CUDA required"
    free_gb = torch.cuda.mem_get_info()[0] / 1e9
    if free_gb < args.min_free_gb:
        sys.exit(f"Only {free_gb:.1f} GB of GPU memory free (need {args.min_free_gb} GB); resubmit later.")
    tok = AutoTokenizer.from_pretrained(args.model, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map={"": 0}, token=hf_token
    )
    model.eval()

    cache = {}
    t0 = time.time()
    for ci, (method, spec, layer, rho, path) in enumerate(todo):
        rows = []
        for ni, note in enumerate(notes):
            kept = select_indices(method, spec, layer, rho, ni, note, scores_fn, meta, cfg["seed"])
            key = (ni, tuple(kept))
            tc = np.array(note["tok_counts"])
            if key not in cache:
                context = " ".join(note["sentences"][i] for i in kept)
                prompt = build_draft_prompt(args.dataset, context)
                inputs = tok(prompt, return_tensors="pt", truncation=True).to(model.device)
                with torch.no_grad():
                    out = model.generate(
                        **inputs, max_new_tokens=DRAFT_MAX_NEW_TOKENS, do_sample=False,
                        eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id,
                    )
                cache[key] = tok.decode(out[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip()
            rows.append({
                "note_id": note["note_id"], "method": method, "spec": spec or "", "layer": "" if layer is None else layer,
                "rho": rho, "actual_frac": float(tc[kept].sum() / max(tc.sum(), 1)), "n_kept": len(kept),
                "kept_idx": json.dumps(kept), "summary": cache[key],
            })
        tmp = path + ".tmp"
        with open(tmp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        os.replace(tmp, path)
        print(f"[{ci + 1}/{len(todo)}] {os.path.basename(path)} done, {time.time() - t0:.0f}s elapsed", flush=True)
        torch.cuda.empty_cache()
    print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
