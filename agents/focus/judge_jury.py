"""
Run the judge jury (leave-own-family-out) over a manifest of systems.

Manifest CSV columns: system, family, csv, source_col, summary_col
  family is the model lineage of the SYSTEM that wrote the summaries (llama, qwen, mistral, gemma, t5, bart, ...).
A judge never rates a system of its own family. Judges run one at a time (each model is loaded, used for every
system it is allowed to rate, then freed). Output is one resumable CSV per (judge, system):
  <out>/<judge>__<system>.csv   with the four ratings per row (missing kept as NaN) and parse/skip flags.

Aggregate afterwards with aggregate_jury.py.
"""
import argparse
import json
import os
import sys

import pandas as pd

from judge_core import free_model, judge_one_scored, load_judge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--config", default="jury_config.json")
    ap.add_argument("--out", default="outputs/jury")
    ap.add_argument("--only-judge", default=None, help="run a single judge by name (so each run is one job)")
    ap.add_argument("--n", type=int, default=None, help="rate only the first n rows of every system")
    ap.add_argument("--min-rows", type=int, default=500, help="skip systems whose output is still incomplete")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    man = pd.read_csv(args.manifest)
    os.makedirs(args.out, exist_ok=True)
    token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
    assert token, "HUGGINGFACE_HUB_TOKEN not set"

    judges = [j for j in cfg["judges"] if args.only_judge in (None, j["name"])]
    if not judges:
        sys.exit("no such judge")
    for j in judges:
        todo = []
        for _, s in man.iterrows():
            if str(s["family"]).lower() == j["family"]:
                continue                                     # a judge never rates its own family
            path = os.path.join(args.out, f"{j['name']}__{s['system']}.csv")
            if not os.path.exists(s["csv"]):
                print(f"  skip {s['system']}: {s['csv']} not found yet", flush=True)
                continue
            if not os.path.exists(path):
                todo.append((s, path))
        print(f"[{j['name']}] {len(todo)} systems to rate", flush=True)
        if not todo:
            continue
        model, tok = load_judge(j["hf_id"], token, j.get("min_free_gb", 24))
        for s, path in todo:
            df = pd.read_csv(s["csv"], keep_default_na=False)
            if len(df) < args.min_rows:
                print(f"  skip {s['system']}: only {len(df)} of {args.min_rows} notes written so far", flush=True)
                continue
            if args.n:
                df = df.iloc[:args.n]
            rows = []
            for i in range(len(df)):
                rec = judge_one_scored(model, tok, str(df.iloc[i][s["source_col"]]), df.iloc[i][s["summary_col"]],
                                   cfg["max_source_tokens"])
                rec.update(row=i, system=s["system"], judge=j["name"])
                rows.append(rec)
            res = pd.DataFrame(rows)
            res.to_csv(path + ".tmp", index=False)
            os.replace(path + ".tmp", path)
            print(f"  {s['system']}: faithfulness {res['faithfulness_exp'].mean():.2f} | unparsed {int((res['parsed_all'] == 0).sum())}"
                  f"/{len(res)} | skipped long {int(res['skipped_long'].sum())}", flush=True)
        free_model(model)


if __name__ == "__main__":
    main()
