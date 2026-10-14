"""
Computational overhead table (CPU only): seconds per note, generated tokens, model calls and peak GPU memory for
every system, per dataset, from the per-note timings the runs saved. Mean +- std across the 500 test notes.

Peak GPU memory can be supplied through PEAK_GB (optional). Latency depends on the hardware and its load.
"""
import argparse
import os

import numpy as np
import pandas as pd

# optional: model -> peak GPU memory in GB (from the run logs); systems not listed leave the column empty
PEAK_GB = {}


def ms(x):
    x = pd.to_numeric(x, errors="coerce").dropna()
    return f"{x.mean():.1f} ± {x.std():.1f}" if len(x) else "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--display", default="display_names.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    man = pd.read_csv(args.manifest)
    names = dict(pd.read_csv(args.display).values)
    rows = []
    for _, s in man.iterrows():
        sysname = s["system"]
        if sysname.startswith("agenticsum_c"):
            continue                                   # ablation rows share the AgenticSum run
        if not os.path.exists(s["csv"]):
            continue
        df = pd.read_csv(s["csv"], keep_default_na=False)
        rec = {"system": sysname, "name": names.get(sysname, sysname), "n_notes": len(df)}
        if sysname == "agenticsum":
            rec["sec_per_note"] = ms(df["t_total"])
            for k in ("focus", "draft", "detect", "fix"):
                rec[f"sec_{k}"] = ms(df[f"t_{k}"])
            kept = pd.to_numeric(df["retained_tokens"], errors="coerce") / pd.to_numeric(df["total_tokens"], errors="coerce").replace(0, np.nan) * 100
            rec["tokens_kept_pct"] = ms(kept)
            rec["repair_rounds"] = ms(df["num_iterations"])
        else:
            if len(df) < 500:
                continue
            rec["sec_per_note"] = ms(df["latency_s"]) if "latency_s" in df else "-"
            rec["new_tokens"] = ms(df["new_tokens"]) if "new_tokens" in df else "-"
            rec["calls"] = ms(df["n_calls"]) if "n_calls" in df else "1"
        rec["peak_gpu_gb"] = PEAK_GB.get(sysname, np.nan)
        rows.append(rec)
    out = pd.DataFrame(rows)
    os.makedirs(args.out, exist_ok=True)
    out.to_csv(os.path.join(args.out, f"efficiency_{args.tag}.csv"), index=False)
    with open(os.path.join(args.out, f"efficiency_{args.tag}.tex"), "w") as f:
        for _, r in out.iterrows():
            f.write(f"\\texttt{{{r['name']}}} & {r['sec_per_note']} & {r.get('calls', '-')} & {r.get('new_tokens', '-')} & {r['peak_gpu_gb']} \\\\\n")
    pd.set_option("display.width", 220, "display.max_columns", 20)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
