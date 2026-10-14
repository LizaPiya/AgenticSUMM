"""Cell formatting in a table style: $mean_{\\pm std}$, std across notes (sample std)."""
import numpy as np


def mean_std(values):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return float("nan"), float("nan")
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


def cell(values, digits=2, bold=False):
    m, s = mean_std(values)
    body = f"{m:.{digits}f}_{{\\pm{s:.{digits}f}}}"
    return f"$\\mathbf{{{body}}}$" if bold else f"${body}$"


def plain(values, digits=2):
    m, s = mean_std(values)
    return f"{m:.{digits}f} \u00b1 {s:.{digits}f}"
