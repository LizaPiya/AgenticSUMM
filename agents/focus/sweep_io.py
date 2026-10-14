"""Read the files written by run_layer_sweep.py."""
import json
import os

import numpy as np


def load_sweep(out_dir: str, dataset: str, split: str = "dev"):
    """Returns meta, notes (list of dicts), and a function scores(i) -> (n_layers, n_modes, n_sent)."""
    tag = f"{dataset}_{split}"
    with open(os.path.join(out_dir, f"layer_meta_{tag}.json")) as f:
        meta = json.load(f)
    with open(os.path.join(out_dir, f"layer_notes_{tag}.jsonl")) as f:
        notes = [json.loads(line) for line in f]
    npz = np.load(os.path.join(out_dir, f"layer_scores_{tag}.npz"))
    return meta, notes, (lambda i: npz[f"n{i}"]), (lambda i: npz[f"b{i}"])


def relative_positions(tok_counts: np.ndarray) -> np.ndarray:
    cum = np.cumsum(tok_counts) - tok_counts
    return (cum + tok_counts / 2.0) / max(tok_counts.sum(), 1)
