"""
Aggregate the jury (CPU only).

  * Each judge's ratings are z-scored per rating dimension (over all rows that judge rated), so judges with
    different leniency are comparable.
  * Extended jury: mean of the z-scores over the judges allowed for that system (leave-own-family-out).
  * Core jury: mean over the judges whose family is in config["core_families"] (families absent from the
    tables); applied identically to every system, so it is the cleanest comparison.
  * Rows skipped by ANY judge (source too long) are dropped for every judge, so all judges see the same notes.
  * Agreement between judges: Spearman correlation per dimension on rows both judges rated.
Outputs a per-system table, the agreement table, and the per-judge raw means.
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from tables import cell, mean_std, plain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--config", default="jury_config.json")
    ap.add_argument("--jury-dir", default="outputs/jury")
    ap.add_argument("--out", default="outputs/jury_summary")
    ap.add_argument("--scale-judge", default="phi35_mini",
                    help="jury cells are z-scores mapped back to this judge's 1-5 scale (its mean and std on the reference systems)")
    ap.add_argument("--z-reference", choices=["all", "common"], default="common",
                    help="common: each judge is z-scored on the systems that EVERY judge rated (same yardstick for all rows); "
                         "all: on every row that judge rated (statistics depend on which systems a judge rated)")
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    dims = cfg["dimensions"]
    fam = {j["name"]: j["family"] for j in cfg["judges"]}
    man = pd.read_csv(args.manifest)
    frames = [pd.read_csv(f) for f in sorted(glob.glob(os.path.join(args.jury_dir, "*__*.csv")))]
    if not frames:
        raise SystemExit("no jury files found")
    df = pd.concat(frames, ignore_index=True)
    from judge_core import to_rating_dims
    df = to_rating_dims(df)           # hallucination = 6 - faithfulness, factual = accuracy, complete, coherent

    skipped = df[df["skipped_long"] == 1][["system", "row"]].drop_duplicates()
    df = df.merge(skipped.assign(_drop=1), on=["system", "row"], how="left")
    df = df[df["_drop"].isna()].drop(columns="_drop")

    if args.z_reference == "common":
        by_judge = df.groupby("judge")["system"].agg(set)
        ref_systems = sorted(set.intersection(*by_judge.tolist()))
        if not ref_systems:
            raise SystemExit("no system was rated by every judge; use --z-reference all")
        print("z-score reference systems (rated by every judge):", ref_systems)
        ref = df[df["system"].isin(ref_systems)]
    else:
        ref = df
    for d in dims:                                              # z-score per judge and dimension
        mu = ref.groupby("judge")[d].mean()
        sd = ref.groupby("judge")[d].std().replace(0, np.nan)
        df[d + "_z"] = (df[d] - df["judge"].map(mu)) / df["judge"].map(sd)
        shifted = df[d] - df["judge"].map(mu) + mu[args.scale_judge]      # judge leniency removed by a shift only; spread untouched
        print(f"{d}: {(shifted > 5).mean():.3%} of shifted ratings above 5, {(shifted < 1).mean():.3%} below 1 (clipped)")
        df[d + "_s15"] = shifted.clip(1, 5)
    core = [j for j, f in fam.items() if f in cfg["core_families"]]

    rows = []
    for system, sub in df.groupby("system"):
        rec = {"system": system, "n_notes": sub["row"].nunique(), "judges_extended": ",".join(sorted(sub["judge"].unique())),
               "judges_core": ",".join(j for j in core if j in set(sub["judge"]))}
        for d in dims:
            per_note = sub.groupby("row")[d + "_z"].mean()
            rec[f"{d}_extended_z"], rec[f"{d}_extended_z_std"] = mean_std(per_note)
            s15 = sub.groupby("row")[d + "_s15"].mean()
            rec[f"{d}_jury15_cell"], rec[f"{d}_jury15_latex"] = plain(s15), cell(s15)
            c = sub[sub["judge"].isin(core)].groupby("row")[d + "_z"].mean()
            rec[f"{d}_core_z"], rec[f"{d}_core_z_std"] = mean_std(c) if len(c) else (np.nan, np.nan)
            # Table cells: raw 1-5 ratings averaged over the allowed judges per note, then mean +- std over notes.
            raw_ext = sub.groupby("row")[d].mean()
            raw_core = sub[sub["judge"].isin(core)].groupby("row")[d].mean()
            rec[f"{d}_extended_cell"], rec[f"{d}_extended_latex"] = plain(raw_ext), cell(raw_ext)
            rec[f"{d}_core_cell"] = plain(raw_core) if len(raw_core) else ""
            rec[f"{d}_core_latex"] = cell(raw_core) if len(raw_core) else ""
            for j in sorted(sub["judge"].unique()):
                rec[f"{d}_raw_{j}"] = plain(sub[sub["judge"] == j][d].dropna()) if sub[sub["judge"] == j][d].notna().any() else ""
        rows.append(rec)
    out = pd.DataFrame(rows)
    os.makedirs(args.out, exist_ok=True)
    out.to_csv(os.path.join(args.out, "jury_by_system.csv"), index=False)

    agree = []
    judges = sorted(df["judge"].unique())
    for i, a in enumerate(judges):
        for b in judges[i + 1:]:
            m = df[df["judge"] == a].merge(df[df["judge"] == b], on=["system", "row"], suffixes=("_a", "_b"))
            for d in dims:
                x, y = m[d + "_a"], m[d + "_b"]
                ok = x.notna() & y.notna()
                if ok.sum() >= 10:
                    agree.append({"judge_a": a, "judge_b": b, "dimension": d, "n": int(ok.sum()),
                                  "spearman": x[ok].rank().corr(y[ok].rank()),      # rank correlation, no scipy needed
                                  "within_one_point": float((x[ok] - y[ok]).abs().le(1).mean())})
    pd.DataFrame(agree).to_csv(os.path.join(args.out, "jury_agreement.csv"), index=False)
    print(out[["system", "n_notes", "judges_extended"] + [f"{d}_extended_cell" for d in dims]].to_string(index=False))
    # Krippendorff's alpha (interval level: ratings are probability-weighted values on 1-5) per dimension. Units are
    # (system, note) pairs; raters are the judges allowed to rate that system, so missing cells are expected.
    def kripp_interval(sub):
        sub = sub.dropna(subset=["value"])
        sub = sub[sub.groupby("unit")["value"].transform("size") >= 2]
        if sub.empty:
            return np.nan, 0, 0
        n, v = len(sub), sub["value"].values
        de = (2 * n * (v ** 2).sum() - 2 * v.sum() ** 2) / (n * (n - 1))
        per = sub.assign(v2=sub["value"] ** 2).groupby("unit").agg(m=("value", "size"), s=("value", "sum"), q=("v2", "sum"))
        do = ((2 * per["m"] * per["q"] - 2 * per["s"] ** 2) / (per["m"] - 1)).sum() / n
        return 1 - do / de, n, per.shape[0]

    alpha_rows = []
    for d in dims:
        u = df[["system", "row", "judge", d]].rename(columns={d: "value"}).copy()
        u["unit"] = u["system"] + "|" + u["row"].astype(str)
        a, n_r, n_u = kripp_interval(u)
        uz = u.assign(value=df[d + "_z"].values)                       # per-judge z-score removes leniency differences
        az, _, _ = kripp_interval(uz)
        alpha_rows.append({"dimension": d, "alpha_raw_1to5": a, "alpha_zscored": az, "n_ratings": n_r, "n_units": n_u})
    pd.DataFrame(alpha_rows).to_csv(os.path.join(args.out, "jury_krippendorff_alpha.csv"), index=False)

    # Do judges rank the systems the same way? Spearman between per-system mean ratings on the systems both rated.
    rank_rows = []
    for d in dims:
        means = df.groupby(["judge", "system"])[d].mean().unstack("judge")
        for i, a in enumerate(judges):
            for b in judges[i + 1:]:
                sh = means[[a, b]].dropna()
                if len(sh) >= 3:
                    rank_rows.append({"dimension": d, "judge_a": a, "judge_b": b, "n_systems": len(sh),
                                      "spearman_of_system_means": sh[a].rank().corr(sh[b].rank())})
    pd.DataFrame(rank_rows).to_csv(os.path.join(args.out, "jury_system_rank_agreement.csv"), index=False)
    print("\nKrippendorff alpha (interval) per dimension, over all (system, note) units with >= 2 judges:")
    print(pd.DataFrame(alpha_rows).to_string(index=False))
    print("\nJudge-to-judge agreement on how they rank systems (Spearman of per-system mean ratings):")
    if rank_rows:
        print(pd.DataFrame(rank_rows).pivot_table(index=["judge_a", "judge_b"], columns="dimension",
                                                  values="spearman_of_system_means").round(2).to_string())
    print("\nPer-note judge agreement (Spearman):")
    ag = pd.DataFrame(agree)
    if len(ag):
        print(ag.pivot_table(index=["judge_a", "judge_b"], columns="dimension", values="spearman").round(2).to_string())

    print("\nCORE JURY (same judge(s) for every system: " + ",".join(core) + "), raw 1-5 ratings")
    print(out[["system", "n_notes"] + [f"{d}_core_cell" for d in dims]].to_string(index=False))
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
