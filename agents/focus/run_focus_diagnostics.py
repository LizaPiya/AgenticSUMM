"""
FOCUS diagnostics harness ( scoring modes, position behaviour, shuffle
sensitivity, token-budget selection against baselines).

NOT included here (stage 2, needs a pre-agreed spec): critical-fact retention and the
downstream hallucination/omission evaluation. See README.md.

Splits are patient-disjoint: the patient id is the part of note_id before the first '-',
hashed into calibration (0-19), development (20-49) and test (50-99).

Typical use (via sbatch):
    python run_focus_diagnostics.py --split cal --calibrate --n-notes 200
    python run_focus_diagnostics.py --split dev --n-notes 200 --table table.pt
The test split is refused unless --final is given together with exactly one --modes entry.
"""
import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
import nltk
from nltk.tokenize import sent_tokenize
from transformers import AutoModelForCausalLM, AutoTokenizer

from focus_scoring import (
    BaselineTable, baseline_scores, score_document, select_by_token_budget, spearman,
)

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
csv.field_size_limit(sys.maxsize)

SPLITS = {"cal": (0, 20), "dev": (20, 50), "test": (50, 100)}
BASELINES = ["random", "first", "last", "tfidf"]


def patient_bucket(note_id: str) -> int:
    pid = note_id.split("-")[0]
    return int(hashlib.md5(pid.encode()).hexdigest(), 16) % 100


def soap_bucket(dialogue: str) -> int:
    """SOAP ids carry no patient information; bucket by a hash of the dialogue text, which
    also keeps exact-duplicate dialogues in the same split."""
    return int(hashlib.md5(dialogue.strip().encode()).hexdigest(), 16) % 100


def iter_mimic(path: str):
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            yield row["note_id"], row["input"], patient_bucket(row["note_id"])


def iter_soap(path: str):
    """`path` is a directory holding soap_{train,val,test}.jsonl; all 1,473 records are pooled
    and exact-duplicate dialogues are dropped."""
    seen = set()
    for part in ("train", "val", "test"):
        with open(os.path.join(path, f"soap_{part}.jsonl")) as f:
            for line in f:
                r = json.loads(line)
                key = r["dialogue"].strip()
                if key in seen:
                    continue
                seen.add(key)
                yield r["ID"], r["dialogue"], soap_bucket(r["dialogue"])


def load_split(dataset: str, path: str, split: str, n_notes: int, seed: int):
    lo, hi = SPLITS[split]
    rng = random.Random(seed)
    kept, seen = [], 0
    for note_id, text, bucket in (iter_mimic(path) if dataset == "mimic" else iter_soap(path)):
        if not (lo <= bucket < hi):
            continue
        seen += 1
        if len(kept) < n_notes:
            kept.append((note_id, text))
        else:
            j = rng.randrange(seen)
            if j < n_notes:
                kept[j] = (note_id, text)
    return kept


def relative_positions(tok_counts: np.ndarray) -> np.ndarray:
    cum = np.cumsum(tok_counts) - tok_counts
    return (cum + tok_counts / 2.0) / max(tok_counts.sum(), 1)


def selection_metrics(chosen, tok_counts, pos_rel):
    mask = np.zeros(len(tok_counts), bool)
    mask[chosen] = True
    kept_tok = tok_counts[mask].sum()
    dec = np.minimum((pos_rel * 10).astype(int), 9)
    out = {
        "actual_frac": kept_tok / max(tok_counts.sum(), 1),
        "share_first30": tok_counts[mask & (pos_rel < 0.3)].sum() / max(kept_tok, 1),
    }
    for d in range(10):
        out[f"dec{d}"] = tok_counts[mask & (dec == d)].sum() / max(kept_tok, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mimic", "soap"], default="mimic")
    ap.add_argument("--data", default=None,
                    help="mimic: the .csv file; soap: the directory with soap_{train,val,test}.jsonl "
                         "(default: the AGENTICSUM_DATA environment variable, else ./data)")
    ap.add_argument("--split", choices=list(SPLITS), required=True)
    ap.add_argument("--n-notes", type=int, default=200)
    ap.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--layer", type=int, default=-1, help="-1 = last layer")
    ap.add_argument("--modes", nargs="+",
                    default=["original", "horizon", "harmonic", "harmonic:cross", "conditional", "conditional:cross"])
    ap.add_argument("--budgets", nargs="+", type=float, default=[0.3, 0.7])
    ap.add_argument("--shuffles", type=int, default=2)
    ap.add_argument("--table", default=None, help="BaselineTable .pt (needed for conditional modes)")
    ap.add_argument("--calibrate", action="store_true", help="build the table from this split and save to --table")
    ap.add_argument("--min-count", type=int, default=200)
    ap.add_argument("--last-policy", choices=["none", "keep"], default="none")
    ap.add_argument("--max-tokens", type=int, default=12000)
    ap.add_argument("--min-sentences", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--final", action="store_true", help="required to touch the test split")
    args = ap.parse_args()

    if args.split == "test" and not (args.final and len(args.modes) == 1):
        sys.exit("Refusing: the test split needs --final and exactly one entry in --modes.")
    if args.calibrate and args.split != "cal":
        sys.exit("Calibrate on the 'cal' split only.")

    hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
    assert hf_token, "HUGGINGFACE_HUB_TOKEN not set"
    assert torch.cuda.is_available(), "CUDA required"
    os.makedirs(args.out, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto", token=hf_token
    )
    model.eval()
    layer = args.layer % model.config.num_hidden_layers

    table = None
    if args.calibrate:
        table = BaselineTable(model.config.num_attention_heads)
    elif any(m.startswith("conditional") for m in args.modes):
        assert args.table, "conditional modes need --table (run --calibrate on the cal split first)"
        table = BaselineTable.load(args.table, args.min_count)

    data_dir = os.environ.get("AGENTICSUM_DATA", "data")
    data = args.data or (os.path.join(data_dir, "mimic-iv-bhc.csv") if args.dataset == "mimic" else data_dir)
    notes = load_split(args.dataset, data, args.split, args.n_notes, args.seed)
    print(f"[{args.dataset}/{args.split}] {len(notes)} notes, layer {layer}, modes {args.modes}", flush=True)

    rows, skipped = [], {"too_long": 0, "too_few_sentences": 0}
    fb_el, fb_n = 0, 0
    t0 = time.time()
    for ni, (note_id, text) in enumerate(notes):
        sentences = sent_tokenize(text.strip())
        if len(sentences) < args.min_sentences:
            skipped["too_few_sentences"] += 1
            continue

        if args.calibrate:
            r = score_document(model, tok, sentences, layer, [], max_tokens=args.max_tokens,
                               calibrate_into=table)
            if r is None:
                skipped["too_long"] += 1
            continue

        res = score_document(model, tok, sentences, layer, args.modes, table, max_tokens=args.max_tokens)
        if res is None:
            skipped["too_long"] += 1
            continue
        tc, m = res["tok_counts"], len(sentences)
        pos = relative_positions(tc)
        st = res["baseline_stats"]
        fb_el += st.get("elements", 0)
        fb_n += st.get("fallback", 0)
        base = {"note_id": note_id, "n_tokens": res["n_tokens"], "n_sentences": m,
                "bos_mass": res["sink"]["bos_mass"], "first5_mass": res["sink"]["first_tokens_mass"]}

        rng = np.random.default_rng(args.seed + ni)
        score_sets = dict(res["scores"])
        for b in BASELINES:
            score_sets[b] = baseline_scores(b, sentences, rng)

        for name, sc in score_sets.items():
            rows.append({**base, "kind": "score_vs_position", "method": name, "budget": np.nan,
                         "spearman_pos": spearman(sc, pos)})
            for B in args.budgets:
                chosen = select_by_token_budget(sc, tc, B, args.last_policy)
                rows.append({**base, "kind": "selection", "method": name, "budget": B,
                             **selection_metrics(chosen, tc, pos)})

        # Shuffle sensitivity (model modes only): content stability vs position artifact
        for r in range(args.shuffles):
            perm = np.random.default_rng(1000 * (ni + 1) + r).permutation(m)
            res_p = score_document(model, tok, [sentences[p] for p in perm], layer, args.modes, table,
                                   max_tokens=args.max_tokens)
            if res_p is None:
                continue
            pos_p = relative_positions(res_p["tok_counts"])
            for spec in args.modes:
                back = np.empty(m)
                back[perm] = res_p["scores"][spec]            # score of original sentence j
                B = args.budgets[0]
                a = set(select_by_token_budget(res["scores"][spec], tc, B, args.last_policy))
                b = set(perm[select_by_token_budget(res_p["scores"][spec], res_p["tok_counts"], B,
                                                    args.last_policy)])
                rows.append({**base, "kind": "shuffle", "method": spec, "budget": B, "shuffle": r,
                             "content_stability": spearman(res["scores"][spec], back),
                             "position_artifact": spearman(res_p["scores"][spec], pos_p),
                             "selection_jaccard": len(a & b) / max(len(a | b), 1)})

        if (ni + 1) % 10 == 0:
            print(f"  {ni + 1}/{len(notes)} notes, {time.time() - t0:.0f}s", flush=True)
        torch.cuda.empty_cache()

    if args.calibrate:
        assert args.table, "--calibrate needs --table <output path>"
        table.save(args.table)
        table.finalize(args.min_count)
        frac = table.is_fallback.float().mean().item()
        print(f"Saved {args.table}; fallback cells {frac:.3f}; skipped {skipped}")
        return

    df = pd.DataFrame(rows)
    tag = f"{args.dataset}_{args.split}_L{layer}"
    df.to_csv(os.path.join(args.out, f"per_note_{tag}.csv"), index=False)

    summary = {"dataset": args.dataset, "split": args.split, "layer": layer, "n_notes_used": int(df["note_id"].nunique()),
               "skipped": skipped,
               "conditional_fallback_fraction": (fb_n / fb_el) if fb_el else None,
               "mean_bos_mass": float(df["bos_mass"].mean()),
               "mean_first5_mass": float(df["first5_mass"].mean())}
    sel = df[df.kind == "selection"]
    cols = ["actual_frac", "share_first30"] + [f"dec{d}" for d in range(10)]
    sel.groupby(["method", "budget"])[cols].mean().to_csv(os.path.join(args.out, f"selection_{tag}.csv"))
    df[df.kind == "score_vs_position"].groupby("method")["spearman_pos"].agg(["mean", "std", "count"]) \
        .to_csv(os.path.join(args.out, f"score_vs_position_{tag}.csv"))
    if args.shuffles:
        df[df.kind == "shuffle"].groupby("method")[
            ["content_stability", "position_artifact", "selection_jaccard"]
        ].mean().to_csv(os.path.join(args.out, f"shuffle_{tag}.csv"))
    with open(os.path.join(args.out, f"summary_{tag}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
