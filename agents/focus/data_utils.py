"""
Dataset loading shared by the layer/budget sweep scripts.

Uses exactly the same patient-disjoint bucketing and seeded reservoir sampling as
run_focus_diagnostics.py, so the same notes are selected for the same (dataset, split,
n_notes, seed). The only addition is that each record also carries the reference summary.
"""
import csv
import hashlib
import json
import os
import random
import sys

csv.field_size_limit(sys.maxsize)

SPLITS = {"cal": (0, 20), "dev": (20, 50), "test": (50, 100)}
DATA_DIR = os.environ.get("AGENTICSUM_DATA", "data")


def patient_bucket(note_id: str) -> int:
    pid = note_id.split("-")[0]
    return int(hashlib.md5(pid.encode()).hexdigest(), 16) % 100


def soap_bucket(dialogue: str) -> int:
    return int(hashlib.md5(dialogue.strip().encode()).hexdigest(), 16) % 100


def iter_mimic(path: str):
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            yield row["note_id"], row["input"], row["target"], patient_bucket(row["note_id"])


def iter_soap(path: str):
    seen = set()
    for part in ("train", "val", "test"):
        with open(os.path.join(path, f"soap_{part}.jsonl")) as f:
            for line in f:
                r = json.loads(line)
                key = r["dialogue"].strip()
                if key in seen:
                    continue
                seen.add(key)
                yield r["ID"], r["dialogue"], r["note"], soap_bucket(r["dialogue"])


def load_split(dataset: str, split: str, n_notes: int, seed: int, path: str = None):
    """Returns a list of {"note_id", "text", "reference"}."""
    lo, hi = SPLITS[split]
    if path is None:
        path = os.path.join(DATA_DIR, "mimic-iv-bhc.csv") if dataset == "mimic" else DATA_DIR
    rng = random.Random(seed)
    kept, seen = [], 0
    it = iter_mimic(path) if dataset == "mimic" else iter_soap(path)
    for note_id, text, ref, bucket in it:
        if not (lo <= bucket < hi):
            continue
        seen += 1
        item = {"note_id": note_id, "text": text, "reference": ref}
        if len(kept) < n_notes:
            kept.append(item)
        else:
            j = rng.randrange(seen)
            if j < n_notes:
                kept[j] = item
    return kept
