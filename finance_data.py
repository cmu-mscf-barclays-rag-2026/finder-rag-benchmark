"""MTEB financial retrieval sets: loading, grouped dev/test split, metrics.

- ``finqa``: ``mteb/FinQARetrieval``, 380 earnings-report passages and 1,138
  questions.
- ``financebench``: ``mteb/FinanceBenchRetrieval``, 145 10-K pages (mostly
  tables) and 150 questions.

Each question has one gold passage, and several questions can share one. The
sets ship a single split, so questions are grouped by gold passage, the groups
are sorted and shuffled with seed 42, and development is filled to 20% (the
team Legal RAG rule). Configurations are chosen on development; test is
reported.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from datasets import load_dataset


DATASETS = {"finqa": "mteb/FinQARetrieval", "financebench": "mteb/FinanceBenchRetrieval"}
SPLIT_SEED = 42
DEV_FRACTION = 0.2


@dataclass(frozen=True)
class FinanceQuestion:
    query_id: str
    text: str
    relevant: frozenset[str]

    @property
    def group(self) -> str:
        return ",".join(sorted(self.relevant))


def load_finance(name: str) -> tuple[list[dict], dict[str, FinanceQuestion]]:
    dataset = DATASETS[name]
    corpus = [{"id": str(r["id"]), "text": r["text"]} for r in load_dataset(dataset, "corpus", split="test")]
    texts = {str(q["id"]): q["text"] for q in load_dataset(dataset, "queries", split="test")}
    relevant: dict[str, set[str]] = {}
    for row in load_dataset(dataset, "qrels", split="test"):
        if row["score"] > 0:
            relevant.setdefault(str(row["query-id"]), set()).add(str(row["corpus-id"]))
    questions = {qid: FinanceQuestion(qid, texts[qid], frozenset(gold)) for qid, gold in relevant.items()}
    return corpus, questions


def split_questions(questions: dict[str, FinanceQuestion]) -> tuple[set[str], set[str]]:
    groups: dict[str, list[str]] = {}
    for qid, question in questions.items():
        groups.setdefault(question.group, []).append(qid)
    keys = sorted(groups)
    random.Random(SPLIT_SEED).shuffle(keys)
    dev: set[str] = set()
    for key in keys:
        if len(dev) >= math.ceil(DEV_FRACTION * len(questions)):
            break
        dev.update(groups[key])
    return dev, set(questions) - dev


def metrics(ranking: list[str], question: FinanceQuestion) -> dict[str, float]:
    rank = next((i for i, pid in enumerate(ranking, 1) if pid in question.relevant), None)
    out = {"gold_rank": rank if rank else ""}
    for k in (1, 5, 10, 100):
        out[f"recall@{k}"] = float(rank is not None and rank <= k)
    out["mrr@10"] = 1 / rank if rank and rank <= 10 else 0.0
    out["ndcg@10"] = 1 / math.log2(rank + 1) if rank and rank <= 10 else 0.0
    return out
