"""
Merge pipeline shards and expose the summary after each stage as its own column, so Table I, the module ablation
and the jury can read them with the existing manifest format:

  draft_summary    draft from the FOCUS-reduced note            (ablation C2)
  repair1_summary  summary after the first repair round         (ablation C3; equals the draft if nothing was flagged)
  fixed_summary    final summary after the supervised loop       (ablation C4, "AgenticSum")

Also prints the run statistics: failures, FOCUS fallbacks, retained-token share, rounds, and
per-stage time. CPU only.
"""
import argparse
import glob
import json

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True, help="shard CSVs, e.g. outputs/pipeline/both_mimic_test_s*.csv")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    files = sorted(glob.glob(args.glob))
    df = pd.concat([pd.read_csv(f, keep_default_na=False) for f in files], ignore_index=True).drop_duplicates("note_id")
    ok = df[df["error"] == ""].copy()

    def pick(row, k):
        rounds = json.loads(row["summaries_by_round"]) if row["summaries_by_round"] else []
        return rounds[min(k, len(rounds) - 1)] if rounds else ""

    ok["repair1_summary"] = ok.apply(lambda r: pick(r, 1), axis=1)
    ok.to_csv(args.out, index=False)

    for c in ("t_focus", "t_draft", "t_detect", "t_fix", "t_total", "total_tokens", "retained_tokens",
              "num_iterations", "n_spans", "n_flagged_first", "n_flagged_final"):
        ok[c] = pd.to_numeric(ok[c], errors="coerce")
    print(f"{len(files)} shard files, {len(df)} notes, {len(ok)} without error, {len(df) - len(ok)} errors")
    print("empty final summaries:", int((ok["fixed_summary"].str.strip() == "").sum()))
    print("FOCUS fallbacks:", ok["focus_fallback"].value_counts().to_dict())
    comp = ok[ok["focus_fallback"] == ""]
    if len(comp):
        frac = comp["retained_tokens"] / comp["total_tokens"]
        print(f"retained token share where FOCUS ran: mean {frac.mean():.3f}  std {frac.std():.3f}  (n={len(comp)})")
    print("termination:", ok["termination_reason"].value_counts().to_dict())
    print(f"repair rounds: mean {ok['num_iterations'].mean():.2f}  std {ok['num_iterations'].std():.2f}")
    print(f"sentences flagged in round 0: mean {ok['n_flagged_first'].mean():.2f} of {ok['n_spans'].mean():.2f} (final spans)")
    print("seconds per note: " + "  ".join(f"{c[2:]} {ok[c].mean():.1f}±{ok[c].std():.1f}"
                                         for c in ("t_focus", "t_draft", "t_detect", "t_fix", "t_total")))
    print("AURA skipped (too long):", int(pd.to_numeric(ok["aura_skipped"], errors="coerce").sum()))


if __name__ == "__main__":
    main()
