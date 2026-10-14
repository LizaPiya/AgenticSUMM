"""
Shared matplotlib style, following the dataviz skill's mark specs.

Colors are the skill's reference palette: categorical slot 1 blue, slot 2 orange, slot 3 aqua
(slots 1-3 are documented as validating all-pairs in both modes); the validator could not be run in
this environment (no node), so the documented validation of that ordering is relied on.
Controls (random / first / TF-IDF) are neutral grays told apart by line style and marker, so no
figure depends on color alone. Text uses ink tokens, never series colors.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", grid="#e7e6e2", axis="#bdbcb6",
                  blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", gray="#8a8984",
                  div_low="#e34948", div_mid="#f0efec", div_high="#256abf"),
    "dark": dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", grid="#2e2e2c", axis="#55544f",
                 blue="#3987e5", orange="#d95926", aqua="#199e70", gray="#a9a8a0",
                 div_low="#e66767", div_mid="#383835", div_high="#3987e5"),
}

# Series styling: (color key, linestyle, marker). Accented series carry color; controls are gray.
STYLE = {
    "focus":    ("blue",   "-",  "o"),
    "original": ("orange", "-",  "s"),
    "aqua":     ("aqua",   "-",  "D"),
    "tfidf":    ("gray",   "-.", "^"),
    "first":    ("gray",   ":",  "v"),
    "random":   ("gray",   "--", "x"),
}

LINE_W = 1.6
MARKER_S = 6


def setup(theme="light"):
    t = THEMES[theme]
    plt.rcParams.update({
        "figure.facecolor": t["surface"], "axes.facecolor": t["surface"], "savefig.facecolor": t["surface"],
        "text.color": t["ink"], "axes.labelcolor": t["ink2"], "xtick.color": t["ink2"], "ytick.color": t["ink2"],
        "axes.edgecolor": t["axis"], "axes.linewidth": 0.8, "font.size": 9, "axes.titlesize": 10,
        "axes.titlecolor": t["ink"], "legend.fontsize": 8.5, "legend.frameon": False,
    })
    return t


def tidy_axes(ax, t):
    ax.grid(True, color=t["grid"], linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def draw_series(ax, t, x, y, lo, hi, key, label, zorder=3):
    ck, ls, mk = STYLE[key]
    color = t[ck]
    if lo is not None:
        ax.fill_between(x, lo, hi, color=color, alpha=0.13, linewidth=0, zorder=zorder - 1)
    # unfilled markers such as "x" are drawn with the edge color, so keep that the series color
    edge = color if mk in ("x", "+", "1", "2") else t["surface"]
    ax.plot(x, y, color=color, linestyle=ls, linewidth=LINE_W, marker=mk, markersize=MARKER_S + (1 if edge == color else 0),
            markerfacecolor=color, markeredgecolor=edge, markeredgewidth=1.5 if edge == color else 1.2,
            solid_capstyle="round", label=label, zorder=zorder)


def legend_handles(t, items):
    """items: [(key_or_color, linestyle, marker, label)] -> proxy handles with the same marks."""
    hs = []
    for ck, ls, mk, label in items:
        # ck may be a theme color key ("blue"), a series style key ("focus"), or a hex color
        color = t.get(ck) or (t[STYLE[ck][0]] if ck in STYLE else ck)
        edge = color if mk in ("x", "+", "1", "2") else t["surface"]
        hs.append(Line2D([0], [0], color=color, linestyle=ls, linewidth=LINE_W, marker=mk, markersize=MARKER_S,
                         markerfacecolor=color, markeredgecolor=edge, markeredgewidth=1.5 if edge == color else 1.0,
                         label=label))
    return hs


def boot_ci(mat, rng, n_boot=1000):
    """mat: (n_series_points, n_notes) with NaN allowed. Shared resample of notes (paired).
    Returns mean, lo, hi, each (n_series_points,)."""
    n = mat.shape[1]
    idx = rng.integers(0, n, size=(n_boot, n))
    means = np.nanmean(mat[:, idx], axis=2)              # (points, n_boot)
    return np.nanmean(mat, axis=1), np.nanpercentile(means, 2.5, axis=1), np.nanpercentile(means, 97.5, axis=1)
