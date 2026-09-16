"""Build a reproducible corpus, queries and query relevance judgments (qrels)."""

from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any, Iterable

DATASET_ID = "Linq-AI-Research/FinDER"
SCHEMA_VERSION = 1


def digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def normalize_passage(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def windows(text: str, size: int, overlap: int) -> list[tuple[int, int]]:
    """Character windows with word boundaries; zero size keeps whole references."""
    if size < 0 or overlap < 0 or (size == 0 and overlap) or (size and overlap >= size):
        raise ValueError("Use size=0, overlap=0, or 0 <= overlap < size")
    if not size:
        return [(0, len(text))]
    result = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start + overlap + 1, end + 1)
            if boundary > start:
                end = boundary
        result.append((start, end))
        if end == len(text):
            break
        start = end - overlap
    return result


def prepare_records(records: Iterable[dict], *, seed: int = 42, dev_fraction: float = 0.2,
                    chunk_size: int = 0, overlap: int = 0, source: dict | None = None) -> dict:
    """Index reference text only, across ALL rows before selecting evaluation queries."""
    if not 0 < dev_fraction < 1:
        raise ValueError("dev_fraction must be between 0 and 1")
    windows("validation", chunk_size, overlap)
    passages: dict[str, str] = {}
    queries: dict[str, dict] = {}
    for row in records:
        query_id, text = row.get("_id"), row.get("text")
        if not isinstance(query_id, str) or not query_id.strip() or query_id in queries:
            raise ValueError(f"Missing or duplicate query ID: {query_id!r}")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Missing query text: {query_id}")
        refs = row.get("references")
        if isinstance(refs, str):
            refs = [refs]
        if not isinstance(refs, list) or any(not isinstance(x, str) for x in refs):
            raise ValueError(f"references must contain strings: {query_id}")
        parent_ids = set()
        for reference in refs:
            passage = normalize_passage(reference)
            if passage:
                parent_id = "ref_" + hashlib.sha256(passage.encode("utf-8")).hexdigest()
                passages[parent_id] = passage
                parent_ids.add(parent_id)
        if not parent_ids:
            raise ValueError(f"No nonempty reference evidence: {query_id}")
        answer = row.get("answer")
        if answer is not None and not isinstance(answer, str):
            raise ValueError(f"answer must be a string or null: {query_id}")
        queries[query_id] = {
            "query_id": query_id, "text": text.strip(), "answer": answer or "",
            "category": row.get("category") or "Uncategorized",
            "reasoning": row.get("reasoning"), "type": row.get("type"),
            "reference_ids": sorted(parent_ids),
        }
    if len(queries) < 2:
        raise ValueError("At least two queries are required for local dev/test partitions")
    shuffled = sorted(queries)
    random.Random(seed).shuffle(shuffled)
    dev_count = max(1, min(len(shuffled) - 1, int(len(shuffled) * dev_fraction)))
    dev_ids = set(shuffled[:dev_count])
    for query_id, query in queries.items():
        query["partition"] = "dev" if query_id in dev_ids else "test"
    corpus = []
    parent_chunks: dict[str, list[str]] = {}
    for parent_id, passage in sorted(passages.items()):
        parent_chunks[parent_id] = []
        for start, end in windows(passage, chunk_size, overlap):
            doc_id = parent_id if not chunk_size else f"{parent_id}:{start}:{end}"
            corpus.append({"doc_id": doc_id, "text": passage[start:end], "reference_id": parent_id,
                           "start": start, "end": end, "reference_length": len(passage)})
            parent_chunks[parent_id].append(doc_id)
    query_list = [queries[qid] for qid in sorted(queries)]
    qrels = {q["query_id"]: {doc_id: 1 for parent_id in q["reference_ids"]
                            for doc_id in parent_chunks[parent_id]} for q in query_list}
    protocol = {
        "schema_version": SCHEMA_VERSION, "corpus_scope": "pooled_reference_passages",
        "normalization": "collapse_whitespace_preserve_case_v1",
        "chunk_size_characters": chunk_size, "overlap_characters": overlap,
        "relevance": "binary_reference_membership" if not chunk_size else "binary_inherited_reference_membership",
        "seed": seed, "dev_fraction": dev_fraction,
    }
    content = {"protocol": protocol, "corpus": corpus, "queries": query_list, "qrels": qrels}
    manifest = {
        **protocol, "fingerprint": digest(content), "source": source or {"kind": "local_records"},
        "query_count": len(queries), "reference_count": len(passages), "document_count": len(corpus),
        "dev_count": dev_count, "test_count": len(queries) - dev_count,
    }
    return {"manifest": manifest, **content}


def save_bundle(bundle: dict, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "manifest.json", bundle["manifest"])
    write_json(directory / "protocol.json", bundle["protocol"])
    write_jsonl(directory / "corpus.jsonl", bundle["corpus"])
    write_jsonl(directory / "queries.jsonl", bundle["queries"])
    write_json(directory / "qrels.json", bundle["qrels"])


def load_bundle(directory: Path) -> dict:
    bundle = {name: json.loads((directory / f"{name}.json").read_text(encoding="utf-8"))
              for name in ("manifest", "protocol", "qrels")}
    bundle.update({name: read_jsonl(directory / f"{name}.jsonl") for name in ("corpus", "queries")})
    fingerprint = digest({key: bundle[key] for key in ("protocol", "corpus", "queries", "qrels")})
    if fingerprint != bundle["manifest"]["fingerprint"]:
        raise ValueError("Prepared data changed: fingerprint mismatch. Prepare and share a fresh bundle.")
    return bundle


def select_queries(bundle: dict, partition: str = "test", limit: int | None = None) -> list[dict]:
    if partition not in {"dev", "test", "all"}:
        raise ValueError("partition must be dev, test or all")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    queries = [q for q in bundle["queries"] if partition == "all" or q["partition"] == partition]
    # Hash ordering avoids taking a company/category-biased prefix of dataset rows.
    queries.sort(key=lambda q: digest([bundle["protocol"]["seed"], q["query_id"]]))
    return queries if limit is None else queries[:limit]
