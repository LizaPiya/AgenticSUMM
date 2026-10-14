"""
One script to (re)generate summaries for every plain baseline model under a single protocol.

Same notes (data_utils.load_split), the shared draft prompt for the dataset (prompts.build_draft_prompt),
greedy decoding, 256 new tokens, only the new tokens decoded. Instruct models get the raw prompt (no chat
template), exactly like the AgenticSum DraftAgent, so the comparison stays like-for-like.
Input limits are explicit: when a note is longer than the model's usable input, the END of the note is cut
(the instruction and the "Summary:" cue are kept) and `note_truncated` is recorded..
The test split is refused unless --final is given. Output is resumable.
"""
import argparse
import csv
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig

from data_utils import load_split
from prompts import DRAFT_MAX_NEW_TOKENS, build_draft_prompt

# name -> Hugging Face id, architecture, lineage, input limit in tokens (None = use the model's own)
REGISTRY = {
    "llama32_3b":   dict(hf="meta-llama/Llama-3.2-3B-Instruct", kind="causal", family="llama", max_in=None),
    "llama31_70b":  dict(hf="meta-llama/Llama-3.1-70B-Instruct", kind="causal", family="llama", max_in=None, nf4=True),
    "qwen25_3b":    dict(hf="Qwen/Qwen2.5-3B-Instruct", kind="causal", family="qwen", max_in=None),
    "qwen25_1p5b":  dict(hf="Qwen/Qwen2.5-1.5B-Instruct", kind="causal", family="qwen", max_in=None),
    "mistral_7b":   dict(hf="mistralai/Mistral-7B-Instruct-v0.3", kind="causal", family="mistral", max_in=None),
    "medalpaca_7b": dict(hf="medalpaca/medalpaca-7b", kind="causal", family="llama", max_in=None),
    "gemma3_1b":    dict(hf="google/gemma-3-1b-it", kind="causal", family="gemma", max_in=None),
    "flan_t5_base": dict(hf="google/flan-t5-base", kind="seq2seq", family="t5", max_in=512),
    "t5_large":     dict(hf="google-t5/t5-large", kind="seq2seq", family="t5", max_in=512, prefix="summarize: "),
    "biobart_large": dict(hf="GanjinZero/biobart-v2-large", kind="seq2seq", family="bart", max_in=1024, raw_note=True),
}
PROMPT_OVERHEAD = 400   # tokens kept free for the instruction text and the generation cue


def truncate_note(tok, note, limit):
    ids = tok(note, add_special_tokens=False).input_ids
    if len(ids) <= limit:
        return note, 0
    return tok.decode(ids[:limit], skip_special_tokens=True), 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(REGISTRY), required=True)
    ap.add_argument("--dataset", choices=["mimic", "soap"], required=True)
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--n-notes", type=int, default=500)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-free-gb", type=float, default=None)
    ap.add_argument("--final", action="store_true")
    args = ap.parse_args()
    if args.split == "test" and not args.final:
        sys.exit("Refusing the test split without --final.")

    spec = REGISTRY[args.model]
    notes = load_split(args.dataset, args.split, args.n_notes, args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    done = 0
    if os.path.exists(args.out):
        with open(args.out, newline="") as f:
            done = sum(1 for _ in csv.DictReader(f))
    if done >= len(notes):
        print("nothing to do")
        return

    token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
    assert token and torch.cuda.is_available()
    need = args.min_free_gb or (45.0 if spec.get("nf4") else 16.0)
    free_gb = torch.cuda.mem_get_info()[0] / 1e9
    if free_gb < need:
        sys.exit(f"Only {free_gb:.1f} GB of GPU memory free (need {need} GB); resubmit later.")

    tok = AutoTokenizer.from_pretrained(spec["hf"], token=token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cls = AutoModelForCausalLM if spec["kind"] == "causal" else AutoModelForSeq2SeqLM
    kw = dict(token=token, torch_dtype=torch.bfloat16)
    if spec.get("nf4"):
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
        kw["device_map"] = {"": 0}
    else:
        kw["device_map"] = {"": 0}
    model = cls.from_pretrained(spec["hf"], **kw).eval()

    # Input limit: fixed for the seq2seq models (BART's 1,024 is hard; T5's 512 is its training length), otherwise
    # the model's own configured context, capped at 32k tokens.
    cfg_max = getattr(model.config, "max_position_embeddings", None)
    max_in = spec["max_in"] or (min(int(cfg_max), 32768) if cfg_max else 100000)
    print(f"effective input limit for {args.model}: {max_in} tokens", flush=True)
    new_file = done == 0
    fields = ["note_id", "input", "target", "fixed_summary", "note_truncated", "new_tokens", "latency_s"]
    with open(args.out, "w" if new_file else "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            w.writeheader()
        for i in range(done, len(notes)):
            rec = notes[i]
            note = rec["text"].strip()
            note_cut, truncated = truncate_note(tok, note, max_in - PROMPT_OVERHEAD)
            if spec["kind"] == "causal" or not spec.get("raw_note"):
                if spec.get("prefix"):
                    prompt = spec["prefix"] + note_cut
                else:
                    prompt = build_draft_prompt(args.dataset, note_cut)
            else:
                prompt = note_cut
            inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=max_in).to(model.device)
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=DRAFT_MAX_NEW_TOKENS, do_sample=False,
                                     pad_token_id=tok.pad_token_id)
            torch.cuda.synchronize()
            lat = round(time.perf_counter() - t0, 2)
            new = out[0] if spec["kind"] == "seq2seq" else out[0][inputs.input_ids.shape[-1]:]
            w.writerow({"note_id": rec["note_id"], "input": note, "target": rec["reference"],
                        "fixed_summary": tok.decode(new, skip_special_tokens=True).strip(),
                        "note_truncated": truncated, "new_tokens": int(new.shape[0]), "latency_s": lat})
            f.flush()
            if (i + 1) % 25 == 0:
                print(f"{args.model}: {i + 1}/{len(notes)}", flush=True)
    print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
