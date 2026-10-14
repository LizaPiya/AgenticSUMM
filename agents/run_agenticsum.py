"""
Full AgenticSum pipeline on the held-out notes of one dataset.

  python run_agenticsum.py --dataset mimic --split test --final --layer L --rho R --tau T --out DIR

Resumable (skips note ids already in the output CSV). --shard I --n-shards K runs notes I, I+K, ... so several
jobs can share the work. Per-note stage times, token counts, termination reasons and the summary after every
repair round are saved for the efficiency table and the module ablation.
"""
import argparse
import csv
import json
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "focus"))

from data_utils import load_split  # noqa: E402
from FixAgent import FixAgent  # noqa: E402
from focus_agent import FocusAgent  # noqa: E402
from hallucination_detector import HallucinationDetectorAgent, ShortAnswerEntailmentJudge  # noqa: E402
from draft_agent import DraftAgent  # noqa: E402
from clinical_supervisor import ClinicalSupervisorAgent  # noqa: E402

FIELDS = ["note_id", "input", "target", "draft_summary", "fixed_summary", "summaries_by_round", "num_iterations",
          "termination_reason", "n_spans", "n_flagged_first", "n_flagged_final", "aura_skipped", "focus_fallback",
          "total_tokens", "retained_tokens", "detect_log", "t_focus", "t_draft", "t_detect", "t_fix", "t_total", "error"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mimic", "soap"], required=True)
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--n-notes", type=int, default=500)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--layer", type=int, required=True, help="FOCUS attention layer")
    ap.add_argument("--rho", type=float, required=True, help="FOCUS token-budget target")
    ap.add_argument("--tau", type=float, required=True, help="AURA threshold")
    ap.add_argument("--entail-cut", type=float, required=True, help="flag a sentence when P(not supported) exceeds this")
    ap.add_argument("--aura-layer", type=int, default=-1)
    ap.add_argument("--detector", choices=["both", "aura", "semantic"], default="both")
    ap.add_argument("--min-sentences", type=int, default=8)
    ap.add_argument("--max-iter", type=int, default=3)
    ap.add_argument("--tables", default=None)
    ap.add_argument("--out", required=True, help="output CSV")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-free-gb", type=float, default=16.0)
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--retry-errors", action="store_true", help="drop rows that failed (e.g. GPU out of memory) and rerun them")
    args = ap.parse_args()
    if args.split == "test" and not args.final:
        sys.exit("Refusing the test split without --final.")

    tables = args.tables or os.path.join(HERE, "..", "..", "outputs", "focus", f"layer_tables_{args.dataset}.pt")
    notes = load_split(args.dataset, args.split, args.n_notes, args.seed)[args.shard::args.n_shards]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    done = set()
    if args.retry_errors and os.path.exists(args.out):
        import pandas as pd
        prev = pd.read_csv(args.out, keep_default_na=False)
        kept = prev[prev["error"] == ""]
        print(f"retrying {len(prev) - len(kept)} failed notes; keeping {len(kept)}", flush=True)
        kept.to_csv(args.out, index=False)
    if os.path.exists(args.out):
        with open(args.out, newline="") as f:
            done = {r["note_id"] for r in csv.DictReader(f)}
    todo = [r for r in notes if r["note_id"] not in done]
    print(f"{args.dataset}/{args.split} shard {args.shard}/{args.n_shards}: {len(notes)} notes, {len(done)} done", flush=True)
    if not todo:
        return

    hf = os.environ.get("HUGGINGFACE_HUB_TOKEN")
    assert hf and torch.cuda.is_available()
    free = torch.cuda.mem_get_info()[0] / 1e9
    if free < args.min_free_gb:
        sys.exit(f"Only {free:.1f} GB of GPU memory free (need {args.min_free_gb}); resubmit later.")
    tok = AutoTokenizer.from_pretrained(args.model, token=hf)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map={"": 0}, token=hf)
    model.eval()

    judge = ShortAnswerEntailmentJudge(model, tok)
    sup = ClinicalSupervisorAgent(
        focus_agent=FocusAgent(model, tok, args.layer, args.rho, tables, min_sentences=args.min_sentences),
        draft_agent=DraftAgent(model, tok, args.dataset),
        hallucination_detector_agent=HallucinationDetectorAgent(
            model, tok, judge, args.dataset, aura_layer=args.aura_layer, aura_threshold=args.tau, entail_cut=args.entail_cut,
            use_aura=args.detector in ("both", "aura"), use_semantic=args.detector in ("both", "semantic")),
        fix_agent=FixAgent(model, tok, max_new_tokens=100),
        max_iterations=args.max_iter,
    )

    new_file = not os.path.exists(args.out) or not done
    with open(args.out, "w" if new_file else "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        t_all = time.time()
        for i, rec in enumerate(todo):
            row = {k: "" for k in FIELDS}
            row.update(note_id=rec["note_id"], input=rec["text"], target=rec["reference"])
            t0 = time.perf_counter()
            try:
                o = sup.run(rec["text"].strip())
                row.update(draft_summary=o["draft_summary"], fixed_summary=o["fixed_summary"],
                           summaries_by_round=json.dumps(o["summaries_by_round"]), num_iterations=o["num_iterations"],
                           termination_reason=o["termination_reason"], n_spans=o["n_spans"],
                           n_flagged_first=o["n_flagged_first"], n_flagged_final=o["n_flagged_final"],
                           aura_skipped=int(o["aura_skipped"]), focus_fallback=o["focus"]["fallback_reason"],
                           total_tokens=o["focus"]["total_tokens"], retained_tokens=o["focus"]["retained_tokens"],
                           detect_log=json.dumps(o["detect_log"]),
                           t_focus=round(o["timings_s"]["focus"], 3), t_draft=round(o["timings_s"]["draft"], 3),
                           t_detect=round(o["timings_s"]["detect"], 3), t_fix=round(o["timings_s"]["fix"], 3))
            except Exception as e:  # keep the run going; failures are counted in the run totals
                row["error"] = str(e)[:300]
                torch.cuda.empty_cache()
            row["t_total"] = round(time.perf_counter() - t0, 3)
            w.writerow(row)
            f.flush()
            if (i + 1) % 5 == 0:
                print(f"  {i + 1}/{len(todo)} notes, {time.time() - t_all:.0f}s", flush=True)
    print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)


if __name__ == "__main__":
    main()
