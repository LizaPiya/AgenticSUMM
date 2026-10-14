"""
Standalone code for the FOCUS sensitivity figures (pandas + matplotlib only; reads the saved plot data).

  python make_sensitivity_figure.py --values sensitivity_rho_mimic_light_values.csv  --kind rho   --title MIMIC-IV --out rho_mimic
  python make_sensitivity_figure.py --values sensitivity_layer_soap_light_values.csv --kind layer --title SOAP    --out layer_soap --layout grid

Sized for the IEEE two-column template (7.16 in full width, 3.5 in single column, 7 pt text so it stays legible at print size):
  --layout row  : four panels in one row, full width (7.16 x 2.15 in)
  --layout grid : 2 x 2 panels, one column (3.5 x 3.6 in)
Writes <out>.pdf (vector, for LaTeX) and <out>.png.

rho figure   : metric versus the fraction of tokens kept; FOCUS against column-sum scoring, TF-IDF, first sentences and random
               sentences; the full note is a dashed reference; the chosen budget is marked. Bands are 95% bootstrap intervals.
layer figure : metric versus the attention layer at two budgets, random selection at the same budget (dotted), full note (dashed).
Palette: Okabe-Ito blue and vermillion for the two accented series, grays for the controls; every series also differs in line
style and marker, so the figure survives grayscale printing and colour-vision deficiency.
"""
import argparse
import string

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

PANELS = [("rouge_l", "ROUGE-L", "higher is better"), ("bert_f1", "BERTScore F1", "higher is better"),
          ("factual", "Factual (judge)", "1-5, higher is better"), ("hallucination", "Hallucination (judge)", "1-5, lower is better")]
INK, INK2, GRID = "#111111", "#4a4a47", "#e4e3df"
BLUE, VERMILLION, GRAY = "#0072B2", "#D55E00", "#7f7e79"          # Okabe-Ito blue and vermillion
RHO_STYLE = {"FOCUS": (BLUE, "-", "o"), "Column-sum scoring": (VERMILLION, "-", "s"), "TF-IDF": (GRAY, "-.", "^"),
             "First sentences": (GRAY, ":", "v"), "Random": (GRAY, "--", "x")}
LAYER_STYLE = {"FOCUS rho=0.3": (BLUE, "-", "o"), "FOCUS rho=0.7": (VERMILLION, "-", "s")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--values", required=True)
    ap.add_argument("--kind", choices=["rho", "layer"], required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--out", required=True, help="output path without extension")
    ap.add_argument("--layout", choices=["row", "grid"], default="row")
    ap.add_argument("--chosen-rho", type=float, default=0.9, help="budget to mark on the rho figure")
    args = ap.parse_args()

    fs = 7
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
                         "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "font.size": fs, "axes.titlesize": fs + 0.5, "axes.labelsize": fs, "xtick.labelsize": fs - 0.5,
                         "ytick.labelsize": fs - 0.5, "legend.fontsize": fs - 0.5, "legend.frameon": False,
                         "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    df = pd.read_csv(args.values)
    df = df[df["metric"] != "metric"]
    for col in ("mean", "ci_lo", "ci_hi", "std"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if args.layout == "row":
        fig, axes = plt.subplots(1, 4, figsize=(7.16, 2.15))
        rect, ms, lw = (0, 0.1, 1, 0.9), 3.4, 1.1
    else:
        fig, axes = plt.subplots(2, 2, figsize=(3.5, 3.6))
        axes = axes.ravel()
        rect, ms, lw = (0, 0.11, 1, 0.95), 3.2, 1.0

    for k, (ax, (metric, title, direction)) in enumerate(zip(axes, PANELS)):
        d = df[df["metric"] == metric]
        ax.grid(True, color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        full = d[d["series"] == "Full note"]
        if len(full):
            ax.axhline(full["mean"].iloc[0], color=INK, linestyle="--", linewidth=0.8, alpha=0.85, zorder=1)
        if args.kind == "rho":
            d = d.assign(x=pd.to_numeric(d["x_actual_frac"], errors="coerce"))
            for name, (color, ls, marker) in RHO_STYLE.items():
                s = d[d["series"] == name].sort_values("x")
                if s.empty:
                    continue
                accent = name in ("FOCUS", "Column-sum scoring")
                ax.plot(s["x"], s["mean"], color=color, linestyle=ls, marker=marker, markersize=ms, linewidth=lw,
                        markeredgecolor="white", markeredgewidth=0.4, zorder=3 if accent else 2, label=name)
                if accent:
                    ax.fill_between(s["x"], s["ci_lo"], s["ci_hi"], color=color, alpha=0.13, linewidth=0)
            ax.axvline(args.chosen_rho, color=INK2, linestyle=":", linewidth=0.7, zorder=1)
            ax.set_xlabel("Retained tokens (fraction)")
            ax.set_xlim(0, 1.04)
        else:
            d = d.assign(x=pd.to_numeric(d["layer"], errors="coerce"))
            for name, (color, ls, marker) in LAYER_STYLE.items():
                s = d[d["series"] == name].sort_values("x")
                ax.plot(s["x"], s["mean"], color=color, linestyle=ls, marker=marker, markersize=ms, linewidth=lw,
                        markeredgecolor="white", markeredgewidth=0.4, label=name.replace("rho=", r"$\rho$="), zorder=3)
                ax.fill_between(s["x"], s["ci_lo"], s["ci_hi"], color=color, alpha=0.13, linewidth=0)
            for name, color in (("Random rho=0.3", BLUE), ("Random rho=0.7", VERMILLION)):
                s = d[d["series"] == name]
                if len(s):
                    ax.axhline(s["mean"].mean(), color=color, linestyle=":", linewidth=1.0, zorder=2)
            ax.set_xlabel("Attention layer")
            ticks = [t for t in sorted(d["x"].dropna().unique()) if t >= 0]     # reference rows carry layer -1
            ax.set_xticks(ticks[::2] if args.layout == "grid" else ticks)
            ax.set_xlim(min(ticks) - 1.5, max(ticks) + 1.5)
        ax.set_title(f"({string.ascii_lowercase[k]}) {title}", loc="left", pad=3)
        ax.set_ylabel(direction)
        ax.tick_params(length=2.5, pad=1.5)

    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(Line2D([0], [0], color=INK, linestyle="--", linewidth=0.8))
    labels.append("Full note")
    if args.kind == "layer":
        handles.append(Line2D([0], [0], color=GRAY, linestyle=":", linewidth=1.0))
        labels.append(r"Random (same $\rho$)")
    ncol = len(labels) if args.layout == "row" else 3
    fig.legend(handles, labels, loc="lower center", ncol=ncol, bbox_to_anchor=(0.5, 0.0), columnspacing=1.0,
               handlelength=2.0, borderaxespad=0.2)
    fig.tight_layout(rect=rect, pad=0.4, w_pad=0.6, h_pad=0.8)
    fig.savefig(args.out + ".pdf")
    fig.savefig(args.out + ".png", dpi=300)
    print("saved", args.out + ".pdf / .png")


if __name__ == "__main__":
    main()
