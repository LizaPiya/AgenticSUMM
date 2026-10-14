"""
Shared LLM-as-a-judge utilities: deterministic scoring of a summary against its full source note, with
probability-weighted ratings, missing ratings kept as missing, and over-long notes skipped and counted.
"""
import re

import numpy as np
import torch


def load_judge(model_name, hf_token, min_free_gb=24.0):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    assert torch.cuda.is_available(), "CUDA required"
    free_gb = torch.cuda.mem_get_info()[0] / 1e9
    if free_gb < min_free_gb:
        raise SystemExit(f"Only {free_gb:.1f} GB of GPU memory free (need {min_free_gb} GB); resubmit later.")
    tok = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map={"": 0},
                                                 token=hf_token)
    model.eval()
    return model, tok

def free_model(model):
    """Release a judge's GPU memory before the next judge is loaded."""
    import gc
    del model
    gc.collect()
    torch.cuda.empty_cache()


# Consistent scales and probability-weighted ratings (expected value over the digits 1-5)
SCORED_LABELS = ["Faithfulness", "Accuracy", "Completeness", "Coherence"]


def _digit_token_ids(tok):
    """For d in 1..5: the vocabulary ids that spell the digit, with and without a leading-space marker."""
    ids = {}
    for d in range(1, 6):
        cand = set()
        for form in (str(d), "▁" + str(d), "Ġ" + str(d)):
            i = tok.convert_tokens_to_ids(form)
            if i is not None and i != tok.unk_token_id:
                cand.add(i)
        ids[d] = sorted(cand)
    return ids


def judge_one_scored(model, tok, source, summary, max_source_tokens=7000, max_new_tokens=48):
    """Ratings for the four scales. Returns discrete ratings, probability-weighted ratings (suffix _exp),
    the share of probability mass that fell on the digits 1-5 (digit_mass_min) and the raw reply."""
    from prompts import build_scored_judge_prompt
    rec = {}
    for k in SCORED_LABELS:
        rec[k.lower()] = np.nan
        rec[k.lower() + "_exp"] = np.nan
    rec.update(parsed_all=0, skipped_long=0, digit_mass_min=np.nan, raw="")
    if len(tok(source, add_special_tokens=False).input_ids) > max_source_tokens:
        rec["skipped_long"] = 1
        return rec
    if not str(summary).strip():
        rec["raw"] = "EMPTY_SUMMARY"
        return rec
    ids = tok.apply_chat_template([{"role": "user", "content": build_scored_judge_prompt(source, str(summary))}],
                                  add_generation_prompt=True, return_tensors="pt", return_dict=True)
    inputs = {k: v.to(model.device) for k, v in ids.items()}
    n_in = inputs["input_ids"].shape[-1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.eos_token_id,
                             return_dict_in_generate=True, output_scores=True)
    gen = out.sequences[0][n_in:]
    rec["raw"] = tok.decode(gen, skip_special_tokens=True).strip().replace("\n", " | ")[:200]
    digit_ids = getattr(tok, "_judge_digit_ids", None) or _digit_token_ids(tok)
    tok._judge_digit_ids = digit_ids
    got, masses = 0, []
    for t in range(len(gen)):
        piece = tok.convert_ids_to_tokens(int(gen[t])).replace("▁", "").replace("Ġ", "").strip()
        if piece not in ("1", "2", "3", "4", "5"):
            continue
        before = tok.decode(gen[:t], skip_special_tokens=True)
        m = re.search(r"(Faithfulness|Accuracy|Completeness|Coherence):\s*$", before)
        if not m or rec[m.group(1).lower()] == rec[m.group(1).lower()]:     # unknown label, or already filled (not NaN)
            continue
        probs = torch.softmax(out.scores[t][0].float(), dim=-1)
        mass = torch.stack([probs[digit_ids[d]].sum() if digit_ids[d] else probs.new_zeros(()) for d in range(1, 6)])
        total = float(mass.sum())
        name = m.group(1).lower()
        rec[name] = float(piece)
        if total > 0:
            rec[name + "_exp"] = float((mass / total * torch.arange(1, 6, device=mass.device)).sum())
            masses.append(total)
        got += 1
    rec["parsed_all"] = int(got == len(SCORED_LABELS))
    rec["digit_mass_min"] = min(masses) if masses else np.nan
    return rec


def to_rating_dims(df):
    """Add the four dimensions to a frame of scored ratings (probability-weighted, 1-5):
    hallucination = 6 - faithfulness (lower is better), factual = accuracy, complete, coherent."""
    df = df.copy()
    df["hallucination"] = 6 - df["faithfulness_exp"]
    df["factual"] = df["accuracy_exp"]
    df["complete"] = df["completeness_exp"]
    df["coherent"] = df["coherence_exp"]
    return df
