"""Legal RAG Bench loading and schema conversion helpers."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Sequence

from datasets import load_dataset
from langchain_core.documents import Document


LEGAL_DATASET_ID = "isaacus/legal-rag-bench"
LEGAL_DATASET_URL = "https://huggingface.co/datasets/isaacus/legal-rag-bench"


@dataclass(frozen=True)
class LegalQuestion:
    question_id: str
    question: str
    answer: str
    relevant_passage_id: str


def load_legal_corpus() -> list[dict[str, Any]]:
    """Load all 4,876 searchable passages from the official test corpus."""

    return load_dataset(LEGAL_DATASET_ID, "corpus", split="test").to_list()


def load_legal_questions(
    sample_size: int = 0, seed: int = 42
) -> list[dict[str, Any]]:
    """Load all questions or a deterministic smoke-test subset.

    Subsampling applies only to queries.  Evaluation must always retain the
    complete corpus so every gold passage remains retrievable.
    """

    rows = load_dataset(LEGAL_DATASET_ID, "qa", split="test").to_list()
    if sample_size < 0:
        raise ValueError("sample_size cannot be negative")
    if sample_size and sample_size < len(rows):
        indices = list(range(len(rows)))
        random.Random(seed).shuffle(indices)
        return [rows[index] for index in indices[:sample_size]]
    return rows


def legal_records_to_documents(records: Sequence[dict[str, Any]]) -> list[Document]:
    """Convert benchmark passages without rechunking or answer leakage."""

    documents: list[Document] = []
    seen: set[str] = set()
    for row in records:
        passage_id = str(row.get("id", "")).strip()
        text = str(row.get("text", "")).strip()
        if not passage_id or not text or passage_id in seen:
            continue
        seen.add(passage_id)
        title = str(row.get("title", "")).strip()
        footnotes = str(row.get("footnotes") or "").strip()
        content = text if not footnotes else f"{text}\n\nFootnotes:\n{footnotes}"
        documents.append(
            Document(
                page_content=content,
                metadata={
                    "passage_id": passage_id,
                    "title": title,
                    "source": LEGAL_DATASET_URL,
                    "dataset": LEGAL_DATASET_ID,
                },
            )
        )
    return documents


def legal_records_to_questions(
    records: Sequence[dict[str, Any]],
) -> list[LegalQuestion]:
    questions = []
    for row in records:
        question = str(row.get("question", "")).strip()
        relevant = str(row.get("relevant_passage_id", "")).strip()
        if not question or not relevant:
            continue
        questions.append(
            LegalQuestion(
                question_id=str(row.get("id", "")),
                question=question,
                answer=str(row.get("answer", "")).strip(),
                relevant_passage_id=relevant,
            )
        )
    return questions
