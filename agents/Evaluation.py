from tqdm import tqdm
import pandas as pd
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bert_score


def clean_text(text):
    if pd.isna(text) or not isinstance(text, str):
        return ""
    return " ".join(text.strip().lower().split())


def compute_bleu_scores(reference, candidate):
    try:
        smoothing = SmoothingFunction().method1
        ref_tokens = reference.split()
        cand_tokens = candidate.split()
        bleu1 = sentence_bleu(
            [ref_tokens],
            cand_tokens,
            weights=(1, 0, 0, 0),
            smoothing_function=smoothing
        )
        bleu2 = sentence_bleu(
            [ref_tokens],
            cand_tokens,
            weights=(0.5, 0.5, 0, 0),
            smoothing_function=smoothing
        )
        return bleu1 * 100, bleu2 * 100
    except Exception as e:
        print(f"BLEU Error: {e}")
        return np.nan, np.nan


def _lcs_length(x, y):
    """Compute length of Longest Common Subsequence between two token lists."""
    m, n = len(x), len(y)
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(curr[j - 1], prev[j])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]


def compute_rouge_l(reference, candidate):
    """Compute ROUGE-L F1 using LCS."""
    try:
        ref_tokens = reference.split()
        cand_tokens = candidate.split()
        if not ref_tokens or not cand_tokens:
            return np.nan
        lcs = _lcs_length(ref_tokens, cand_tokens)
        precision = lcs / len(cand_tokens)
        recall = lcs / len(ref_tokens)
        if precision + recall == 0:
            return np.nan
        f1 = (2 * precision * recall) / (precision + recall)
        return f1 * 100
    except Exception as e:
        print(f"ROUGE-L Error: {e}")
        return np.nan


def compute_bert_score_batched(references, candidates, batch_size=32):
    all_P, all_R, all_F1 = [], [], []
    for i in range(0, len(references), batch_size):
        refs = references[i:i + batch_size]
        cands = candidates[i:i + batch_size]
        try:
            P, R, F1 = bert_score(cands, refs, lang="en", verbose=False)
            all_P.extend((P * 100).tolist())
            all_R.extend((R * 100).tolist())
            all_F1.extend((F1 * 100).tolist())
        except Exception as e:
            print(f"BERTScore Error in batch {i}: {e}")
            all_P.extend([np.nan] * len(refs))
            all_R.extend([np.nan] * len(refs))
            all_F1.extend([np.nan] * len(refs))
    return all_P, all_R, all_F1


def evaluate_summaries(df, summary_column="fixed_summary", reference_column="target"):
    """
    Compute ROUGE-L, BLEU-1, BLEU-2, and BERTScore (P/R/F1).
    Print ONLY Mean and Median.
    """
    for col in [summary_column, reference_column]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found. Available: {list(df.columns)}")

    print(f"📊 Evaluating {len(df)} summaries...")
    print(f"   Summary column:   {summary_column}")
    print(f"   Reference column: {reference_column}\n")

    # --- BLEU + ROUGE-L ---
    bleu1_scores, bleu2_scores, rouge_l_scores = [], [], []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="BLEU + ROUGE-L"):
        ref = clean_text(row[reference_column])
        cand = clean_text(row[summary_column])

        if not ref or not cand:
            bleu1_scores.append(np.nan)
            bleu2_scores.append(np.nan)
            rouge_l_scores.append(np.nan)
        else:
            b1, b2 = compute_bleu_scores(ref, cand)
            rl = compute_rouge_l(ref, cand)
            bleu1_scores.append(b1)
            bleu2_scores.append(b2)
            rouge_l_scores.append(rl)

    # --- BERTScore ---
    print("\n🧠 Computing BERTScore...")
    references = [clean_text(t) for t in df[reference_column]]
    candidates = [clean_text(t) for t in df[summary_column]]
    bert_p, bert_r, bert_f1 = compute_bert_score_batched(references, candidates)

    # --- Attach metrics ---
    df = df.copy()
    df["bleu1"] = bleu1_scores
    df["bleu2"] = bleu2_scores
    df["rouge_l"] = rouge_l_scores
    df["bert_p"] = bert_p
    df["bert_r"] = bert_r
    df["bert_f1"] = bert_f1

    # --- Print results (MEAN + MEDIAN ONLY) ---
    metrics = {
        "ROUGE-L":   "rouge_l",
        "BLEU-1":    "bleu1",
        "BLEU-2":    "bleu2",
        "BERTScore": "bert_f1",
    }

    print("\n" + "=" * 50)
    print("📊 EVALUATION RESULTS")
    print("=" * 50)
    print(f"{'Metric':<12} {'Mean':>10} {'Median':>10}")
    print("-" * 50)

    for label, col in metrics.items():
        print(
            f"{label:<12} "
            f"{df[col].mean():>10.2f} "
            f"{df[col].median():>10.2f}"
        )

    print("=" * 50)

    return df
