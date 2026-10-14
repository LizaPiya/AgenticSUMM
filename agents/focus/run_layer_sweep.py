"""
All-layer FOCUS sweep (no text generation).

--calibrate : build one conditional-baseline table per layer on the 'cal' split.
(default)   : score every sentence of every note, at every layer, for every mode; save the scores
              so later steps (position diagnostics, budget selection, generation) read files and
              never recompute attention.
--verify V  : for the first V notes also run the loop reference on layers {first, middle, last}
              and abort if the vectorized scores disagree.

The test split is refused here; final test numbers come from a separate, single-purpose run.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import nltk
from nltk.tokenize import sent_tokenize
from transformers import AutoModelForCausalLM, AutoTokenizer

from data_utils import load_split
from focus_scoring import (
    BaselineTable, build_text_and_spans, layer_attention_chunks, sentence_scores_chunk,
    sink_diagnostics, parse_mode,
)
from focus_fast import sentence_index, bin_index_matrix, fast_sentence_scores, capture_all_layers

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

DEFAULT_MODES = ["original", "horizon", "harmonic", "harmonic:cross", "conditional", "conditional:cross"]


def head_chunk_for(n: int) -> int:
    return 4 if n <= 4096 else (2 if n <= 8192 else 1)


def same(a: torch.Tensor, b: torch.Tensor, rtol=2e-3, atol=1e-6) -> bool:
    if not torch.equal(torch.isnan(a), torch.isnan(b)):
        return False
    m = ~torch.isnan(a)
    return bool(torch.allclose(a[m], b[m], rtol=rtol, atol=atol))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mimic", "soap"], required=True)
    ap.add_argument("--split", choices=["cal", "dev"], required=True)
    ap.add_argument("--n-notes", type=int, default=200)
    ap.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--layers", type=int, nargs="+", default=None, help="default: all layers")
    ap.add_argument("--modes", nargs="+", default=DEFAULT_MODES)
    ap.add_argument("--tables", required=True, help=".pt with per-layer baseline tables")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--min-count", type=int, default=200)
    ap.add_argument("--min-sentences", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verify", type=int, default=0)
    ap.add_argument("--min-free-gb", type=float, default=20.0)
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    if args.calibrate and args.split != "cal":
        sys.exit("Calibrate on the 'cal' split only.")
    if not args.calibrate and args.split == "cal":
        sys.exit("Scoring runs on the 'dev' split (calibration notes are used only to build tables).")

    hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
    assert hf_token, "HUGGINGFACE_HUB_TOKEN not set"
    assert torch.cuda.is_available(), "CUDA required"
    os.makedirs(args.out, exist_ok=True)

    # The GPU is shared. Fail fast instead of letting the loader silently offload layers to CPU.
    free_gb = torch.cuda.mem_get_info()[0] / 1e9
    if free_gb < args.min_free_gb:
        sys.exit(f"Only {free_gb:.1f} GB of GPU memory free (need {args.min_free_gb} GB). "
                 "Other jobs are using the GPU; resubmit later.")

    tok = AutoTokenizer.from_pretrained(args.model, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map={"": 0}, token=hf_token
    )
    model.eval()
    device = next(model.parameters()).device
    H = model.config.num_attention_heads
    L = model.config.num_hidden_layers
    layers = args.layers if args.layers is not None else list(range(L))
    specs = args.modes
    need_cond = any(parse_mode(s)[0] == "conditional" for s in specs)

    notes = load_split(args.dataset, args.split, args.n_notes, args.seed)
    print(f"[{args.dataset}/{args.split}] {len(notes)} notes, {len(layers)} layers, modes {specs}", flush=True)

    tables = {}
    if args.calibrate:
        tables = {li: BaselineTable(H) for li in layers}
    elif need_cond:
        blob = torch.load(args.tables)
        for li in layers:
            t = BaselineTable(H)
            t.total, t.count = blob[li]["total"], blob[li]["count"]
            t.finalize(args.min_count)
            tables[li] = t

    verify_layers = sorted({layers[0], layers[len(layers) // 2], layers[-1]})
    skipped = {"too_long": 0, "too_few_sentences": 0}
    kept_records, score_arrays, bos_arrays = [], {}, {}
    fb_samples = []
    fb_total = {li: 0 for li in layers}
    t0 = time.time()

    for ni, rec in enumerate(notes):
        sentences = sent_tokenize(rec["text"].strip())
        if len(sentences) < args.min_sentences:
            skipped["too_few_sentences"] += 1
            continue
        info = build_text_and_spans(sentences, tok, device)
        n, m = info["n"], len(sentences)
        if n > args.max_tokens:
            skipped["too_long"] += 1
            continue

        caps = capture_all_layers(model, info["ids"])
        hc = head_chunk_for(n)

        if args.calibrate:
            for li in layers:
                cap_l = {"h": caps[li], "pe": caps["pe"]}
                for h0, A in layer_attention_chunks(model, cap_l, li, hc):
                    tables[li].update(A, h0, info["n_sink"])
                    del A
            del caps
        else:
            sid, endk = sentence_index(n, info["spans"], device)
            idx = bin_index_matrix(n, device) if need_cond else None
            out = np.zeros((len(layers), len(specs), m), np.float32)
            bos = np.zeros(len(layers), np.float32)
            do_verify = ni < args.verify
            sample_fb = need_cond and (ni % 10 == 0)
            if sample_fb:
                ar = torch.arange(n, device=device)
                idx_e = idx[(ar[:, None] >= ar[None, :]) & (ar[None, :] >= info["n_sink"])]
                fbacc = np.zeros(len(layers))
            for a_i, li in enumerate(layers):
                cap_l = {"h": caps[li], "pe": caps["pe"]}
                acc = torch.zeros(len(specs), m, device=device)
                for h0, A in layer_attention_chunks(model, cap_l, li, hc):
                    Hc = A.shape[0]
                    tb = tables[li].b.to(device) if need_cond else None
                    res = fast_sentence_scores(A, h0, sid, endk, m, specs, tb, idx)
                    for s_i, s in enumerate(specs):
                        acc[s_i] += res[s].sum(0)
                    bos[a_i] += sink_diagnostics(A, info["n_sink"])["bos_mass"] * Hc / H
                    if sample_fb:
                        fbacc[a_i] += tables[li].is_fallback[h0:h0 + Hc].to(device)[:, idx_e].float().mean().item() * Hc / H
                    if do_verify and li in verify_layers:
                        for s in specs:
                            ref = sentence_scores_chunk(A, h0, info["spans"], s, tables.get(li))
                            if not same(res[s], ref):
                                d = (res[s] - ref).abs().nan_to_num().max().item()
                                raise SystemExit(f"VERIFY FAILED note {ni} layer {li} mode {s}: max diff {d:.3e}")
                    del A
                out[a_i] = (acc / H).cpu().numpy()
            if do_verify:
                print(f"  verified note {ni} at layers {verify_layers} for {len(specs)} modes: OK", flush=True)
            del caps
            if sample_fb:
                fb_samples.append(fbacc)
            score_arrays[f"n{len(kept_records)}"] = out
            bos_arrays[f"b{len(kept_records)}"] = bos
            kept_records.append({
                "note_id": rec["note_id"], "sentences": sentences, "reference": rec["reference"],
                "n_tokens": n, "tok_counts": [int(e - s) for s, e in info["spans"]],
            })

        if (ni + 1) % 10 == 0:
            print(f"  {ni + 1}/{len(notes)} notes, {time.time() - t0:.0f}s", flush=True)
        torch.cuda.empty_cache()

    print(f"Peak GPU memory allocated by this run: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)
    if args.calibrate:
        torch.save({li: {"total": t.total, "count": t.count, "n_heads": H} for li, t in tables.items()}, args.tables)
        fb = [(li, (t.count < args.min_count).float().mean().item()) for li, t in tables.items()]
        print(f"Saved {args.tables}; skipped {skipped}")
        print("fraction of (head,bin) cells below min_count, by layer:", {li: round(f, 3) for li, f in fb})
        return

    tag = f"{args.dataset}_{args.split}"
    np.savez_compressed(os.path.join(args.out, f"layer_scores_{tag}.npz"), **score_arrays, **bos_arrays)
    with open(os.path.join(args.out, f"layer_notes_{tag}.jsonl"), "w") as f:
        for r in kept_records:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(args.out, f"layer_meta_{tag}.json"), "w") as f:
        json.dump({"dataset": args.dataset, "split": args.split, "layers": layers, "modes": specs,
                   "n_notes": len(kept_records), "skipped": skipped, "model": args.model,
                   "seed": args.seed, "min_count": args.min_count,
                   "fallback_fraction_by_layer": (np.mean(fb_samples, axis=0).round(5).tolist() if fb_samples else None)}, f, indent=2)
    print(f"Saved scores for {len(kept_records)} notes; skipped {skipped}; {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
