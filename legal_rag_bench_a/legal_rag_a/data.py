"""Pinned download, schema validation, and canonical passage/query records."""

import hashlib
import json
import re
from pathlib import Path

DATASET = "isaacus/legal-rag-bench"
REVISION = "db0b31dc6d195ce9916897e1ac5e4e6209736c8a"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / ".cache" / REVISION


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def download(data_dir=DEFAULT_DATA):
    """Download an immutable source revision; retain a local checksum manifest."""
    import requests

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = data_dir / "source_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["revision"] != REVISION:
            raise ValueError("Cache revision differs from the pinned benchmark.")
        for name, digest in manifest["sha256"].items():
            if not (data_dir / name).exists() or sha256(data_dir / name) != digest:
                raise ValueError(f"Cache integrity failure: {name}. Use a new cache directory.")
        return manifest
    hashes = {}
    for name in ("corpus.jsonl", "qa.jsonl", "README.md"):
        response = requests.get(
            f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{name}",
            timeout=(15, 120),
        )
        response.raise_for_status()
        content = response.content
        if name.endswith("jsonl"):
            for line in content.decode("utf-8").splitlines():
                json.loads(line)
        temporary = data_dir / (name + ".partial")
        temporary.write_bytes(content)
        temporary.replace(data_dir / name)
        hashes[name] = sha256(data_dir / name)
    manifest = {"dataset": DATASET, "revision": REVISION, "sha256": hashes}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_benchmark(data_dir=DEFAULT_DATA):
    """Load local files only. Gold labels and answers never enter passage text."""
    data_dir = Path(data_dir)
    manifest = json.loads((data_dir / "source_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("revision") != REVISION:
        raise ValueError("Unexpected dataset revision.")
    for filename in ("corpus.jsonl", "qa.jsonl"):
        if sha256(data_dir / filename) != manifest["sha256"][filename]:
            raise ValueError(f"Source checksum mismatch: {filename}")
    corpus = read_jsonl(data_dir / "corpus.jsonl")
    queries = read_jsonl(data_dir / "qa.jsonl")
    ids = set()
    for passage in corpus:
        if not isinstance(passage.get("id"), str) or not passage["id"]:
            raise ValueError("Passage IDs must be nonempty strings.")
        if passage["id"] in ids:
            raise ValueError("Duplicate passage ID.")
        ids.add(passage["id"])
        if not isinstance(passage.get("text"), str) or not passage["text"].strip():
            raise ValueError("Empty or invalid passage text.")
        for key in ("title", "footnotes"):
            if passage.get(key) is not None and not isinstance(passage[key], str):
                raise ValueError(f"Unexpected {key} type.")
    query_ids = set()
    normalized = []
    for row in queries:
        if not isinstance(row.get("id"), (str, int)) or isinstance(row["id"], bool):
            raise ValueError("Invalid query ID.")
        qid = str(row["id"])
        if not qid or qid in query_ids:
            raise ValueError("Empty or duplicate query ID.")
        query_ids.add(qid)
        gold = row.get("relevant_passage_id")
        if not isinstance(gold, str) or gold not in ids:
            raise ValueError(f"Unresolved gold passage for {qid}.")
        for key in ("question", "answer"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"Invalid {key} for {qid}.")
        normalized.append({**row, "id": qid, "gold_ids": [gold]})
    return corpus, normalized


def retrieval_text(passage, mode="text_only"):
    """Use the same text policy across all retrieval methods."""
    if mode == "text_only":
        return passage["text"]
    if mode == "title_text_footnotes":
        return "\n\n".join(passage.get(k) or "" for k in ("title", "text", "footnotes")).strip()
    raise ValueError(f"Unknown text policy: {mode}")


def word_count(text):
    """Whitespace-delimited words, deliberately not called model tokens."""
    return len(text.split())


def parent_section(passage_id):
    """ID-derived section grouping, not a recovered full source document."""
    return re.sub(r"-c\d+-s\d+$", "", passage_id)
