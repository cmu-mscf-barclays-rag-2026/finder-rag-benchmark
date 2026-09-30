"""An inverted-index BM25 implementation using only Python's standard library."""

from __future__ import annotations

import heapq
import math
import re
import unicodedata
from collections import Counter, defaultdict

TOKENIZER_VERSION = "unicode_casefold_numbers_possessives_v1"


def tokenize(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).casefold().replace("’", "'")
    text = re.sub(r"'s\b", "", text)
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    return re.findall(r"[^\W_]+(?:\.\d+)?", text)


class BM25Retriever:
    """Positive Robertson IDF; unique query terms; deterministic ID tie breaking.

    score = sum(log(1 + (N-df+0.5)/(df+0.5)) *
                tf*(k1+1)/(tf + k1*(1-b+b*length/average_length)))
    """

    def __init__(self, corpus: list[dict], k1: float = 1.2, b: float = 0.75):
        if not math.isfinite(k1) or k1 < 0 or not math.isfinite(b) or not 0 <= b <= 1:
            raise ValueError("k1 must be finite and >= 0; b must be between 0 and 1")
        if not corpus:
            raise ValueError("Cannot index an empty corpus")
        self.k1, self.b = k1, b
        self.doc_ids = [doc["doc_id"] for doc in corpus]
        if len(set(self.doc_ids)) != len(self.doc_ids):
            raise ValueError("Duplicate corpus document IDs")
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        lengths = []
        for index, doc in enumerate(corpus):
            counts = Counter(tokenize(doc["text"]))
            lengths.append(sum(counts.values()))
            for term, frequency in counts.items():
                self.postings[term].append((index, frequency))
        self.average_length = sum(lengths) / len(lengths)
        if not self.average_length:
            raise ValueError("Corpus contains no searchable tokens")
        self.norms = [k1 * (1 - b + b * length / self.average_length) for length in lengths]
        self.idf = {term: math.log1p((len(corpus) - len(postings) + 0.5) / (len(postings) + 0.5))
                    for term, postings in self.postings.items()}

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        scores: dict[int, float] = defaultdict(float)
        for term in sorted(set(tokenize(query))):
            for index, frequency in self.postings.get(term, ()):
                scores[index] += self.idf[term] * frequency * (self.k1 + 1) / (frequency + self.norms[index])
        best = heapq.nsmallest(top_k, scores, key=lambda index: (-scores[index], self.doc_ids[index]))
        return [{"doc_id": self.doc_ids[index], "score": scores[index]} for index in best]
