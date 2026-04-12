from __future__ import annotations

from functools import lru_cache
from typing import Sequence

import nltk
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from nltk.translate.meteor_score import meteor_score


@lru_cache(maxsize=1)
def ensure_nltk_resources() -> None:
    for resource in ("wordnet", "omw-1.4"):
        try:
            nltk.data.find(f"corpora/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)


def rouge_l_score(reference: Sequence[str], prediction: Sequence[str]) -> float:
    if not reference or not prediction:
        return 0.0

    dp = [[0] * (len(prediction) + 1) for _ in range(len(reference) + 1)]
    for i in range(1, len(reference) + 1):
        for j in range(1, len(prediction) + 1):
            if reference[i - 1] == prediction[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs = dp[-1][-1]
    precision = lcs / len(prediction)
    recall = lcs / len(reference)
    if precision + recall == 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)


def safe_meteor_score(reference: Sequence[str], prediction: Sequence[str]) -> float:
    if not reference or not prediction:
        return 0.0
    try:
        ensure_nltk_resources()
        return float(meteor_score([list(reference)], list(prediction)))
    except Exception:
        overlap = len(set(reference) & set(prediction))
        precision = overlap / max(1, len(prediction))
        recall = overlap / max(1, len(reference))
        if precision + recall == 0:
            return 0.0
        return (10 * precision * recall) / (recall + 9 * precision)


def compute_generation_metrics(
    references: Sequence[Sequence[str]],
    predictions: Sequence[Sequence[str]],
) -> dict[str, float]:
    smoothie = SmoothingFunction().method3
    bleu = corpus_bleu(
        list([[list(reference)] for reference in references]),
        list(predictions),
        smoothing_function=smoothie,
        weights=(0.25, 0.25, 0.25, 0.25),
    )
    meteor = sum(
        safe_meteor_score(reference, prediction)
        for reference, prediction in zip(references, predictions)
    ) / max(1, len(references))
    rouge_l = sum(
        rouge_l_score(reference, prediction)
        for reference, prediction in zip(references, predictions)
    ) / max(1, len(references))
    return {
        "bleu4": bleu * 100.0,
        "meteor": meteor * 100.0,
        "rouge_l": rouge_l * 100.0,
    }
