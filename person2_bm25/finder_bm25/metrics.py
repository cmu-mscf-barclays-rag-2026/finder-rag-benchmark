"""Method-independent ranking metrics. Rankings must contain unique document IDs."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from typing import Mapping, Sequence


def validate_ranking(ranking: Sequence[str], qrels: Mapping[str, float], k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    if len(set(ranking)) != len(ranking):
        raise ValueError("Duplicate document IDs would inflate ranking metrics")
    if any(not math.isfinite(grade) or grade < 0 for grade in qrels.values()):
        raise ValueError("Relevance grades must be finite and nonnegative")


def reciprocal_rank(ranking: Sequence[str], qrels: Mapping[str, float], k: int) -> float:
    validate_ranking(ranking, qrels, k)
    return next((1.0 / rank for rank, doc_id in enumerate(ranking[:k], 1)
                 if qrels.get(doc_id, 0) > 0), 0.0)


def mrr_at_k(rankings: Mapping[str, Sequence[str]], qrels: Mapping[str, Mapping[str, float]], k: int) -> float:
    """Average over all judged queries; missing query results count as zero."""
    validate_ranking([], {}, k)
    if not qrels:
        raise ValueError("MRR requires at least one query")
    if set(rankings) - set(qrels):
        raise ValueError("Rankings contain unknown query IDs")
    return sum(reciprocal_rank(rankings.get(qid, []), grades, k)
               for qid, grades in qrels.items()) / len(qrels)


def ndcg_at_k(ranking: Sequence[str], qrels: Mapping[str, float], k: int) -> float:
    validate_ranking(ranking, qrels, k)
    ideal_grades = sorted(qrels.values(), reverse=True)[:k]
    # Scaling by max grade avoids overflow without changing DCG / IDCG.
    max_grade = max(ideal_grades, default=0)
    def gain(grade: float) -> float:
        return 2.0 ** (grade - max_grade) - 2.0 ** (-max_grade)
    ideal = sum(gain(grade) / math.log2(rank + 1) for rank, grade in enumerate(ideal_grades, 1))
    actual = sum(gain(qrels.get(doc_id, 0)) / math.log2(rank + 1)
                 for rank, doc_id in enumerate(ranking[:k], 1))
    return actual / ideal if ideal else 0.0


def precision_at_k(ranking: Sequence[str], qrels: Mapping[str, float], k: int) -> float:
    validate_ranking(ranking, qrels, k)
    return sum(qrels.get(doc_id, 0) > 0 for doc_id in ranking[:k]) / k


def recall_at_k(ranking: Sequence[str], qrels: Mapping[str, float], k: int) -> float:
    validate_ranking(ranking, qrels, k)
    relevant = sum(grade > 0 for grade in qrels.values())
    return sum(qrels.get(doc_id, 0) > 0 for doc_id in ranking[:k]) / relevant if relevant else 0.0


def normalize_answer(text: str) -> str:
    """Conservative lexical normalization retains signs, decimals and percentages."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def answer_exact_match(prediction: str, reference: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(reference))


def answer_token_f1(prediction: str, reference: str) -> float:
    def tokens(text: str) -> Counter:
        return Counter(re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?%?|[^\W\d_]+", normalize_answer(text)))
    predicted, gold = tokens(prediction), tokens(reference)
    if not predicted or not gold:
        return float(predicted == gold)
    overlap = sum((predicted & gold).values())
    return 2 * overlap / (sum(predicted.values()) + sum(gold.values()))
