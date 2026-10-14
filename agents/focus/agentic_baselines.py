"""
Agent-style, training-free baselines on the same frozen backbone as AgenticSum (Llama-3.2-3B-Instruct):

  selfrefine : Self-Refine (Madaan et al., NeurIPS 2023). The same model writes feedback on its own
               summary and revises it, up to T iterations; stops early if the feedback reports no issues.
  cove       : Chain-of-Verification (Dhuliawala et al., ACL Findings 2024). Draft, plan verification
               questions from the draft, answer each question separately, write the final summary.
               Adaptation: questions are answered from the SOURCE note (summarization is grounded in a
               document), and the final step sees the draft and the question/answer pairs.
  reflexion  : Reflexion (Shinn et al., NeurIPS 2023). Trials of draft, self-evaluation, verbal
               reflection stored in memory and fed to the next trial. Adaptation: summarization has no
               environment signal, so the evaluator is the same model rating how well the summary is
               supported by the source (1-5); a trial succeeds at a score >= --success-score.

All methods use the full note, greedy decoding, the shared draft prompt (prompts.build_draft_prompt),
and the same iteration budget as AgenticSum (T = 3). Per-note latency, number of model calls and
generated tokens are saved, to report computational overhead. Output is resumable.
The test split is refused unless --final is given.
"""
import argparse
import csv
import os
import re
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data_utils import load_split
from prompts import DRAFT_MAX_NEW_TOKENS, build_draft_prompt


class Gen:
    """Greedy generation with call and token accounting."""

    def __init__(self, model, tok):
        self.model, self.tok = model, tok
        self.calls = 0
        self.new_tokens = 0
        self.trace = []                # evaluator verdicts of the current note (Reflexion), for inspection

    def __call__(self, prompt, max_new_tokens):
        inputs = self.tok(prompt, return_tensors="pt", truncation=True).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                      eos_token_id=self.tok.eos_token_id, pad_token_id=self.tok.eos_token_id)
        new = out[0][inputs.input_ids.shape[-1]:]
        self.calls += 1
        self.new_tokens += int(new.shape[0])
        return self.tok.decode(new, skip_special_tokens=True).strip()


def task_line(dataset):
    return ("a concise clinical summary of the hospital stay" if dataset == "mimic"
            else "a concise SOAP-format summary (Subjective, Objective, Assessment, Plan)")


# ----------------------------------------------------------------------------- Self-Refine
def self_refine(gen, dataset, note, max_iter):
    summary = gen(build_draft_prompt(dataset, note), DRAFT_MAX_NEW_TOKENS)
    iters = 0
    for _ in range(max_iter):
        fb = gen(
            "You are reviewing a draft summary against its source document. List the specific problems: statements "
            "that are not supported by the source, wrong facts, and important omissions. If there are no problems, "
            "reply exactly: NO ISSUES.\n\n"
            f"SOURCE:\n{note}\n\nDRAFT SUMMARY:\n{summary}\n\nFEEDBACK:", 200)
        if "NO ISSUES" in fb.upper():
            break
        summary = gen(
            f"Revise the draft into {task_line(dataset)} using the feedback. Use only information from the source. "
            "Do not add notes or commentary about the summary itself.\n\n"
            f"SOURCE:\n{note}\n\nDRAFT SUMMARY:\n{summary}\n\nFEEDBACK:\n{fb}\n\nRevised summary:", DRAFT_MAX_NEW_TOKENS)
        iters += 1
    return summary, iters


# ----------------------------------------------------------------------------- CoVe
def cove(gen, dataset, note, n_questions=5):
    draft = gen(build_draft_prompt(dataset, note), DRAFT_MAX_NEW_TOKENS)
    plan = gen(
        f"Write up to {n_questions} short fact-checking questions about the specific claims in this summary, one per "
        f"line, numbered.\n\nSUMMARY:\n{draft}\n\nQUESTIONS:", 150)
    questions = [re.sub(r"^\s*\d+[.)]\s*", "", q).strip() for q in plan.splitlines() if re.match(r"^\s*\d+[.)]", q)]
    qa = []
    for q in questions[:n_questions]:
        a = gen("Answer the question using only the SOURCE. If the source does not say, answer 'not stated'.\n\n"
                f"SOURCE:\n{note}\n\nQUESTION: {q}\nANSWER:", 80)
        qa.append((q, a))
    qa_text = "\n".join(f"Q: {q}\nA: {a}" for q, a in qa) or "(no verification questions were produced)"
    final = gen(
        f"Write the final, verified version of the summary as {task_line(dataset)}. Keep statements that the "
        "verification answers support. Correct or remove any statement that conflicts with the answers or is "
        "'not stated'. Do not add notes or commentary about the summary itself.\n\n"
        f"DRAFT SUMMARY:\n{draft}\n\nVERIFICATION:\n{qa_text}\n\nFinal summary:", DRAFT_MAX_NEW_TOKENS)
    return final, len(qa)


# ----------------------------------------------------------------------------- Reflexion
def reflexion(gen, dataset, note, max_trials, success_score=None):
    """success_score is kept for the command line but no longer used: a trial succeeds only when the evaluator
    finds no unsupported statement. The best trial is the one with the fewest unsupported statements."""
    memory, best, best_bad, trials = [], "", None, 0
    for _ in range(max_trials):
        base = build_draft_prompt(dataset, note)
        if memory:
            lessons = "\n".join(f"- {m}" for m in memory)
            base = base.replace(note, f"{note}\n\nLessons from earlier attempts (follow them):\n{lessons}", 1)
        summary = gen(base, DRAFT_MAX_NEW_TOKENS)
        trials += 1
        verdict = gen(
            "List every statement in the summary that is not supported by the source. If every statement is "
            "supported, reply exactly: NONE. Otherwise list the unsupported statements, one per line, each "
            f"starting with '-'.\n\nSOURCE:\n{note}\n\nSUMMARY:\n{summary}\n\nUNSUPPORTED STATEMENTS:", 200)
        if hasattr(gen, "trace"):
            gen.trace.append(verdict[:400])
        if verdict.strip().upper().startswith("NONE"):
            n_bad = 0
        else:
            n_bad = max(1, sum(1 for ln in verdict.splitlines() if ln.strip().startswith(("-", "*", "•"))))
        if best_bad is None or n_bad <= best_bad:
            best, best_bad = summary, n_bad
        if n_bad == 0:
            break
        memory.append(gen(
            "The summary below was judged to contain unsupported statements. In one or two sentences, say what went "
            f"wrong and what to do differently next time.\n\nSUMMARY:\n{summary}\n\nEVALUATION:\n{verdict}\n\n"
            "Reflection:", 100))
    return best, trials


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["selfrefine", "cove", "reflexion"], required=True)
    ap.add_argument("--dataset", choices=["mimic", "soap"], required=True)
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--n-notes", type=int, default=500)
    ap.add_argument("--max-iter", type=int, default=3, help="iterations or trials; same budget as AgenticSum")
    ap.add_argument("--success-score", type=int, default=4)
    ap.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-free-gb", type=float, default=12.0)
    ap.add_argument("--final", action="store_true", help="required for the test split")
    ap.add_argument("--max-tokens", type=int, default=7000, help="skip notes longer than this many tokens")
    ap.add_argument("--cpu", action="store_true", help="run on CPU (float32); for small checks only")
    args = ap.parse_args()
    if args.split == "test" and not args.final:
        sys.exit("Refusing the test split without --final.")

    notes = load_split(args.dataset, args.split, args.n_notes, args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    done = 0
    if os.path.exists(args.out):
        with open(args.out, newline="") as f:
            done = sum(1 for _ in csv.DictReader(f))
    if done >= len(notes):
        print("nothing to do")
        return

    hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
    assert hf_token
    if args.cpu:
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    else:
        assert torch.cuda.is_available()
        free_gb = torch.cuda.mem_get_info()[0] / 1e9
        if free_gb < args.min_free_gb:
            sys.exit(f"Only {free_gb:.1f} GB of GPU memory free (need {args.min_free_gb} GB); resubmit later.")
    tok = AutoTokenizer.from_pretrained(args.model, token=hf_token)
    if args.cpu:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32, token=hf_token)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map={"": 0}, token=hf_token)
    model.eval()
    gen = Gen(model, tok)

    fields = ["note_id", "input", "target", "fixed_summary", "steps", "n_calls", "new_tokens", "latency_s", "skipped_long", "trace"]
    new_file = not os.path.exists(args.out) or done == 0
    with open(args.out, "a" if not new_file else "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            w.writeheader()
        for i in range(done, len(notes)):
            rec = notes[i]
            note = rec["text"].strip()
            row = {"note_id": rec["note_id"], "input": note, "target": rec["reference"], "fixed_summary": "",
                   "steps": 0, "n_calls": 0, "new_tokens": 0, "latency_s": 0.0, "skipped_long": 0, "trace": ""}
            if len(tok(note, add_special_tokens=False).input_ids) > args.max_tokens:
                row["skipped_long"] = 1
            else:
                gen.calls, gen.new_tokens, gen.trace = 0, 0, []
                t0 = time.perf_counter()
                if args.method == "selfrefine":
                    summary, steps = self_refine(gen, args.dataset, note, args.max_iter)
                elif args.method == "cove":
                    summary, steps = cove(gen, args.dataset, note)
                else:
                    summary, steps = reflexion(gen, args.dataset, note, args.max_iter, args.success_score)
                if not args.cpu:
                    torch.cuda.synchronize()
                row.update(fixed_summary=summary, steps=steps, n_calls=gen.calls, new_tokens=gen.new_tokens,
                           latency_s=round(time.perf_counter() - t0, 2), trace=" || ".join(gen.trace))
            w.writerow(row)
            f.flush()
            if (i + 1) % 10 == 0:
                print(f"{args.method}: {i + 1}/{len(notes)}", flush=True)
    if not args.cpu:
        print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
