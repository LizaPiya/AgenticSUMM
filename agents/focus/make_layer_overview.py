"""
Overview figure for the choice of the attention layer (pandas + matplotlib only).
Reads layer_selection_mimic.csv and layer_selection_soap.csv (gain over random selection, with 95% intervals).

Left   : the note as a stack of sentences.
Middle : the frozen decoder drawn as 28 layer columns; the 8 evaluated layers are coloured by the lower of the two
         datasets' gains (the quantity the selection rule maximises); the chosen layer is outlined.
Right  : gain over random selection for the 8 evaluated layers, one panel per dataset, with 95% paired-bootstrap intervals.
Bottom : scoring from the chosen layer, then token-budget selection.
Gain = average of the ROUGE-L and BERTScore-F1 differences (FOCUS minus random selection of the same size), averaged over
rho = 0.3 and 0.7; positive means FOCUS selects better sentences than chance.
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

INK, INK2 = "#111111", "#4a4a47"
HIGH, MID, LOW, OFF = "#0072B2", "#8FBCDD", "#D9D9D4", "#EFEFEC"     # Okabe-Ito blue, tint, grays
VERMILLION = "#D55E00"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fig-dir", required=True)
    ap.add_argument("--chosen", type=int, default=24)
    ap.add_argument("--n-layers", type=int, default=28)
    ap.add_argument("--arrow", action="store_true", help="draw an arrow from the chosen layer to its bar")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    m = pd.read_csv(f"{args.fig_dir}/layer_selection_mimic.csv").sort_values("layer")
    s = pd.read_csv(f"{args.fig_dir}/layer_selection_soap.csv").sort_values("layer")
    ev = m["layer"].astype(int).tolist()
    worst = np.minimum(m["gain_over_random"].values, s["gain_over_random"].values)
    cls = {l: (HIGH if w >= 0.30 else (MID if w >= 0.10 else LOW)) for l, w in zip(ev, worst)}
    n = args.n_layers

    plt.rcParams.update({"font.size": 7, "pdf.fonttype": 42, "text.color": INK, "axes.linewidth": 0.6})
    fig = plt.figure(figsize=(7.16, 3.3))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 716)
    ax.set_ylim(0, 330)
    ax.axis("off")
    ax.set_zorder(10)                      # drawing layer above the bar charts, so the arrow is not hidden by their background
    ax.patch.set_alpha(0)

    # left: the note as sentences
    ax.text(58, 268, "Input note $D$", ha="center", fontsize=8, fontweight="bold")
    ax.add_patch(FancyBboxPatch((8, 126), 100, 130, boxstyle="round,pad=0,rounding_size=4", fc="white", ec=INK2, lw=0.7))
    for i, label in enumerate(["$u_1$", "$u_2$", "$u_3$", "$u_4$", "$\\vdots$", "$u_m$"]):
        y = 237 - i * 19
        if label != "$\\vdots$":
            ax.add_patch(Rectangle((16, y - 6), 84, 12, fc="#EFEFEC", ec="none"))
        ax.text(22, y, label, va="center", fontsize=7)
    ax.annotate("", xy=(134, 191), xytext=(112, 191), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.8))
    ax.text(58, 110, "no prompt, no query", ha="center", fontsize=6, color=INK2)

    # middle: layer stack
    x0, w, gap = 144, 8.0, 2.2
    box_w = n * (w + gap) + 12
    ax.add_patch(FancyBboxPatch((136, 118), box_w, 142, boxstyle="round,pad=0,rounding_size=4", fc="none", ec=INK, lw=0.8,
                                linestyle=(0, (3, 2))))
    ax.text(136 + box_w / 2, 270, f"Frozen decoder, {n} layers", ha="center", fontsize=8, fontweight="bold")
    xc = {}
    for layer in range(n):
        x = x0 + layer * (w + gap)
        xc[layer] = x + w / 2
        colour = cls.get(layer, OFF)
        sel = layer == args.chosen
        ax.add_patch(FancyBboxPatch((x, 138), w, 104, boxstyle="round,pad=0,rounding_size=2", fc=colour,
                                    ec=INK if sel else "white", lw=1.5 if sel else 0.4, zorder=3 if sel else 2))
        if layer in cls:
            ax.text(x + w / 2, 128, str(layer), ha="center", va="center", fontsize=5.8, color=INK, zorder=4)
    ax.text(xc[args.chosen], 250, "$\\lambda$", ha="center", fontsize=8.5, fontweight="bold")

    # below: scoring and selection
    ax.add_patch(FancyBboxPatch((150, 20), 250, 66, boxstyle="round,pad=0,rounding_size=4", fc="white", ec=INK2, lw=0.7))
    ax.text(275, 72, "attention $A$ of layer $\\lambda$ (all heads)", ha="center", fontsize=6.5)
    ax.text(275, 56, "score $s_{h,k}=\\frac{1}{n-k+1}\\sum_i A_{h,i,k}/b_h(i,k)$", ha="center", fontsize=6.5)
    ax.text(275, 38, "sentence score $\\beta_j$ = mean of token scores", ha="center", fontsize=6.5)
    ax.text(275, 25, "keep top sentences until $\\rho N$ tokens $\\rightarrow D_{\\mathrm{reduced}}$", ha="center", fontsize=6.5)
    ax.annotate("", xy=(xc[args.chosen], 88), xytext=(xc[args.chosen], 118), arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.9))

    # right: gains with intervals
    left = 0.67
    axes = []
    for k, (name, d) in enumerate((("MIMIC-IV", m), ("SOAP", s))):
        a = fig.add_axes([left, 0.47 - k * 0.32, 0.315, 0.23])
        axes.append(a)
        g, lo, hi = d["gain_over_random"].values, d["ci_lo"].values, d["ci_hi"].values
        pos = np.arange(len(ev))
        a.bar(pos, g, color=[cls[l] for l in ev], width=0.72,
              edgecolor=[INK if l == args.chosen else "none" for l in ev], linewidth=1.0, zorder=2)
        a.errorbar(pos, g, yerr=[g - lo, hi - g], fmt="none", ecolor=INK2, elinewidth=0.7, capsize=1.6, capthick=0.7, zorder=3)
        a.axhline(0, color=INK2, lw=0.6)
        a.set_xlim(-0.6, len(ev) - 0.4)
        a.set_ylim(*((-0.2, 1.15) if k == 0 else (-0.7, 0.85)))
        a.set_xticks(pos)
        a.set_xticklabels([str(l) for l in ev], fontsize=6)
        a.tick_params(length=2, pad=1.5, labelsize=6)
        a.text(0.0, 1.03, name, transform=a.transAxes, ha="left", va="bottom", fontsize=7, fontweight="bold")
        a.set_ylabel("$\\Delta$ score", fontsize=6.5, labelpad=3)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        a.grid(axis="y", color="#e4e3df", lw=0.5, zorder=0)
        a.set_axisbelow(True)
    axes[1].set_xlabel("attention layer", fontsize=7, labelpad=1)
    fig.text(left + 0.158, 0.965, "Improvement over baseline", ha="center", fontsize=8, fontweight="bold")
    fig.text(left + 0.158, 0.925, "mean $\\Delta$ROUGE-L, $\\Delta$BERT-F vs. random baseline (95% CI)", ha="center", fontsize=6, color=INK2)

    # arrow from the chosen column to its bar in the first panel
    if args.arrow:                                    # optional: the outline and colour already link the chosen layer to its bars
        j = ev.index(args.chosen)
        (x_lo, x_hi), (y_lo, y_hi) = axes[0].get_xlim(), axes[0].get_ylim()       # fixed above, so the arithmetic is exact
        box = axes[0].get_position()
        tip_x = (box.x0 + box.width * (j - x_lo) / (x_hi - x_lo)) * 716
        tip_y = (box.y0 + box.height * ((m["ci_hi"].values[j] + 0.05) - y_lo) / (y_hi - y_lo)) * 330
        ax.add_patch(FancyArrowPatch((xc[args.chosen] + 5, 246), (tip_x, tip_y), arrowstyle="-|>", mutation_scale=7,
                                     color=VERMILLION, lw=1.0, connectionstyle="arc3,rad=-0.25", zorder=6))

    # legend
    ly = 305
    for j2, (c, text) in enumerate(((HIGH, "high"), (MID, "middle"), (LOW, "low"), (OFF, "not evaluated"))):
        xx = 150 + j2 * 62
        ax.add_patch(Rectangle((xx, ly - 4), 9, 9, fc=c, ec=INK2 if c == OFF else "none", lw=0.4))
        ax.text(xx + 13, ly, text, va="center", fontsize=6.5)
    ax.text(150, ly - 16, "colour: smaller of the two datasets' improvements (the quantity the rule maximises); outline: chosen layer", fontsize=5.8, color=INK2)

    fig.savefig(args.out + ".pdf")
    fig.savefig(args.out + ".png", dpi=300)
    print("saved", args.out)


if __name__ == "__main__":
    main()
