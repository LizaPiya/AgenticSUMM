"""
Apply SELECTION_RULES.md step 2 (token budget rho) at a chosen layer, with paired bootstrap over notes.

For each rho in the grid, versus the full note (rho = 1.0) and versus random selection at the same rho:
  * lower end of the 95% interval of (FOCUS - full) is above -1.0 for ROUGE-L and above -0.5 for BERTScore F1;
  * judge hallucination (6 - faithfulness) is not worse than the full note by more than 0.10;
  * FOCUS beats random selection at that rho on both ROUGE-L and BERTScore F1 (mean paired difference > 0).
The chosen rho is the smallest that passes all three; if none passes, 0.9. Cells are mean +- std over notes.
"""
import argparse
import os

import numpy as np
import pandas as pd

from plot_sensitivity import load_frames
from select_layer import per_note

ROUGE_MARGIN, BERT_MARGIN, JUDGE_MARGIN = -1.0, -0.5, 0.10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mimic")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--met-dir", default="outputs/metrics")
    ap.add_argument("--judge-dir", default="outputs/judge/phi35_mini")
    ap.add_argument("--out", default="outputs/figures")
    ap.add_argument("--n-boot", type=int, default=5000)
    args = ap.parse_args()

    df = load_frames(args.met_dir, args.judge_dir, args.dataset)
    note_ids = sorted(df["note_id"].unique())
    focus = df[(df["method"] == "focus") & (df["layer"].astype(str) == str(args.layer))]
    rhos = sorted(r for r in focus["rho"].astype(float).unique() if r < 1.0)
    have_judge = df["hallucination"].notna().any()
    metrics = ["rouge_l", "bert_f1"] + (["hallucination"] if have_judge else [])

    rng = np.random.default_rng(0)
    n = len(note_ids)
    idx = rng.integers(0, n, size=(args.n_boot, n))

    def ci(d):
        d = np.asarray(d, float)
        b = np.array([np.nanmean(d[i]) for i in idx])
        return float(np.nanmean(d)), float(np.nanstd(d, ddof=1)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))

    rows = []
    for r in rhos:
        rec = {"rho": r}
        for m in metrics:
            f = per_note(df, note_ids, m, r, "focus", args.layer)
            full = per_note(df, note_ids, m, 1.0, "full")
            rnd = per_note(df, note_ids, m, r, "random")
            mean_f, sd_f = np.nanmean(f), np.nanstd(f, ddof=1)
            rec[f"{m}_focus"] = f"{mean_f:.2f} ± {sd_f:.2f}"
            for tag, ref in (("vs_full", full), ("vs_random", rnd)):
                mu, sd, lo, hi = ci(f - ref)
                rec[f"{m}_{tag}_mean"], rec[f"{m}_{tag}_std"], rec[f"{m}_{tag}_ci_lo"], rec[f"{m}_{tag}_ci_hi"] = mu, sd, lo, hi
        ok_quality = (rec["rouge_l_vs_full_ci_lo"] > ROUGE_MARGIN) and (rec["bert_f1_vs_full_ci_lo"] > BERT_MARGIN)
        ok_judge = (rec["hallucination_vs_full_mean"] <= JUDGE_MARGIN) if have_judge else None
        ok_random = (rec["rouge_l_vs_random_mean"] > 0) and (rec["bert_f1_vs_random_mean"] > 0)
        rec.update(ok_quality=ok_quality, ok_judge=ok_judge, ok_random=ok_random,
                   passes=bool(ok_quality and ok_random and (ok_judge in (True, None))))
        rows.append(rec)

    out = pd.DataFrame(rows)
    os.makedirs(args.out, exist_ok=True)
    out.to_csv(os.path.join(args.out, f"rho_selection_{args.dataset}_L{args.layer}.csv"), index=False)
    show = ["rho", "rouge_l_focus", "bert_f1_focus", "rouge_l_vs_full_mean", "rouge_l_vs_full_ci_lo", "bert_f1_vs_full_mean",
            "bert_f1_vs_full_ci_lo", "rouge_l_vs_random_mean", "bert_f1_vs_random_mean"]
    if have_judge:
        show += ["hallucination_focus", "hallucination_vs_full_mean"]
    show += ["ok_quality", "ok_judge", "ok_random", "passes"]
    print(out[show].round(3).to_string(index=False))
    ok = [r["rho"] for r in rows if r["passes"]]
    chosen = min(ok) if ok else 0.9
    print(f"\nrho chosen by the rule at layer {args.layer}: {chosen}" + ("" if ok else "  (no rho passed; fallback 0.9)")
          + ("" if have_judge else "  [judge check PENDING]"))


if __name__ == "__main__":
    main()
