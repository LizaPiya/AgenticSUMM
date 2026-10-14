"""
LLM-as-a-judge (Meta-Llama-3-8B-Instruct) for the generated summaries,:

  * the judge sees the FULL source note
  * deterministic decoding (no sampling);
  * a rating that cannot be parsed is recorded as missing and counted, never replaced by 3;
  * notes whose source exceeds judge_max_source_tokens are recorded as skipped and counted.

One resumable CSV per generated-summary file.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from judge_core import free_model, judge_one_scored, load_judge
from sweep_io import load_sweep

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mimic", "soap"], required=True)
    ap.add_argument("--sweep-dir", default="outputs")
    ap.add_argument("--gen-dir", default="outputs/gen")
    ap.add_argument("--judge-dir", default="outputs/judge")
    ap.add_argument("--config", default="sensitivity_config.json")
    ap.add_argument("--only", default=None, help="substring filter on generated-file names")
    ap.add_argument("--judge-name", default="phi35_mini", help="a judge from jury_config.json")
    ap.add_argument("--jury-config", default="jury_config.json")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    _, notes, _, _ = load_sweep(args.sweep_dir, args.dataset, "dev")
    src_by_id = {n["note_id"]: " ".join(n["sentences"]) for n in notes}
    os.makedirs(args.judge_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.gen_dir, f"gen_{args.dataset}_*.csv")))
    if args.only:
        files = [f for f in files if args.only in os.path.basename(f)]
    todo = [f for f in files if not os.path.exists(os.path.join(args.judge_dir, "judge_" + os.path.basename(f)[4:]))]
    print(f"{len(files)} generated files, {len(todo)} to judge", flush=True)
    if not todo:
        return

    jcfg = next(j for j in json.load(open(args.jury_config))["judges"] if j["name"] == args.judge_name)
    model, tok = load_judge(jcfg["hf_id"], os.environ["HUGGINGFACE_HUB_TOKEN"], jcfg.get("min_free_gb", 24))
    print(f"judge: {jcfg['name']} ({jcfg['hf_id']})", flush=True)

    for f in todo:
        df = pd.read_csv(f, keep_default_na=False)
        out_rows = []
        for _, row in df.iterrows():
            rec = {"note_id": row["note_id"], "method": row["method"], "layer": row["layer"],
                   "rho": row["rho"], "actual_frac": row["actual_frac"]}
            rec.update(judge_one_scored(model, tok, src_by_id[row["note_id"]], row["summary"],
                                    cfg["judge_max_source_tokens"]))
            out_rows.append(rec)
        res = pd.DataFrame(out_rows)
        path = os.path.join(args.judge_dir, "judge_" + os.path.basename(f)[4:])
        res.to_csv(path + ".tmp", index=False)
        os.replace(path + ".tmp", path)
        print(f"judged {os.path.basename(f)}: faithfulness {res['faithfulness_exp'].mean():.2f}  "
              f"parse failures {int((res['parsed_all'] == 0).sum())}/{len(res)} (skipped long: {int(res['skipped_long'].sum())})",
              flush=True)
        torch.cuda.empty_cache()
    print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
