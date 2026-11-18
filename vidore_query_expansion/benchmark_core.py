"""Run all ViDoRe Finance queries against the complete matched page corpus."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
from hashlib import file_digest
import importlib.metadata
import importlib.util
from io import BytesIO
import json
import math
import os
from pathlib import Path
import platform
import re
import statistics
from shutil import copyfileobj
from time import perf_counter, sleep
from urllib.request import Request, urlopen

import numpy as np
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parent
DATASET = "vidore/vidore_v3_finance_en"
DATA_REVISION = "7f432c176d82e27546501ad8064a713ac3071809"
MODEL = "Qwen/Qwen3-VL-Embedding-2B"
MODEL_REVISION = "9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda"
DATA_FILES = [
    "queries/test-00000-of-00001.parquet",
    "qrels/test-00000-of-00001.parquet",
    *(f"corpus/test-{i:05d}-of-00003.parquet" for i in range(3)),
]
MODEL_FILES = [
    "config.json", "preprocessor_config.json", "video_preprocessor_config.json",
    "chat_template.jinja", "tokenizer.json", "tokenizer_config.json", "vocab.json",
    "merges.txt", "added_tokens.json", "special_tokens_map.json",
    "scripts/qwen3_vl_embedding.py", "model.safetensors",
]
LARGE_FILES = {
    "corpus-test-00000-of-00003.parquet": (
        449439390, "be17c9aa26e09d1e81fe6bf749ab220f129a99808d08315480a57c7f2b189e7f",
    ),
    "corpus-test-00001-of-00003.parquet": (
        367340393, "d7c0bc0dd3d8666476dc749d700b491c1b3e6335b4227b8563a75e4fec28b5f7",
    ),
    "corpus-test-00002-of-00003.parquet": (
        448564350, "4ee2a59bbc4d9c81726d6a231173a57022c79717e2690037083a8303ec43518d",
    ),
    "model.safetensors": (
        4255140312, "c73fa9caeddeb3ff831d46c085a7a5708343248ca777e90f2d486964464509c1",
    ),
}
LANGUAGES = ("english", "french", "german", "italian", "portuguese", "spanish")
METHODS = ("bm25_ocr", "dense_ocr", "dense_image", "rrf_ocr_image")
K_VALUES = (1, 5, 10)
RRF_DEPTH = 100
RRF_CONSTANT = 60
TEXT_LIMIT = 4096
IMAGE_PIXELS = 32 * 32 * 768
INSTRUCTION = "Given a financial question, retrieve document pages that contain relevant evidence."


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fetch(path, url):
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.name in LARGE_FILES:
        fetch_large(path, url + "?download=true", *LARGE_FILES[path.name])
        return
    partial = path.with_suffix(path.suffix + ".part")
    with urlopen(Request(url + "?download=true"), timeout=90) as response, partial.open("wb") as out:
        copyfileobj(response, out, 8 * 1024 ** 2)
    if response.headers.get("Content-Length") and partial.stat().st_size != int(response.headers["Content-Length"]):
        raise OSError(f"Incomplete download of {path.name}")
    partial.replace(path)
    print(f"Downloaded {path.name} ({path.stat().st_size:,} bytes).", flush=True)


def fetch_large(path, url, size, sha256):
    """Fetch checked ranges so a stalled large download can resume."""
    block_size = 32 * 1024 ** 2

    def download_block(start):
        stop = min(start + block_size, size) - 1
        part = path.with_name(f"{path.name}.block{start}")
        if part.exists() and part.stat().st_size == stop - start + 1:
            return part
        temp = part.with_suffix(part.suffix + ".part")
        request = Request(url, headers={"Range": f"bytes={start}-{stop}"})
        for attempt in range(3):
            try:
                with urlopen(request, timeout=45) as response, temp.open("wb") as out:
                    if response.headers.get("Content-Range") != f"bytes {start}-{stop}/{size}":
                        raise OSError(f"Unexpected byte range for {path.name}")
                    copyfileobj(response, out, 8 * 1024 ** 2)
                if temp.stat().st_size != stop - start + 1:
                    raise OSError(f"Incomplete range for {path.name}")
                temp.replace(part)
                return part
            except OSError:
                if attempt == 2:
                    raise
                sleep(attempt + 1)

    partial = path.with_suffix(path.suffix + ".part")
    with ThreadPoolExecutor(max_workers=6) as pool, partial.open("wb") as out:
        parts = []
        for part in pool.map(download_block, range(0, size, block_size)):
            with part.open("rb") as source:
                copyfileobj(source, out, 8 * 1024 ** 2)
            parts.append(part)
            if len(parts) % 10 == 0:
                print(f"{path.name}: {out.tell():,}/{size:,} bytes", flush=True)
    with partial.open("rb") as source:
        actual = file_digest(source, "sha256").hexdigest()
    if actual != sha256:
        raise OSError(f"Checksum mismatch for {path.name}; clear the downloaded blocks and retry.")
    partial.replace(path)
    for part in parts:
        part.unlink()
    print(f"Downloaded and verified {path.name} ({size:,} bytes).", flush=True)


def download(data, model):
    for name in DATA_FILES:
        fetch(data / name.replace("/", "-"),
              f"https://huggingface.co/datasets/{DATASET}/resolve/{DATA_REVISION}/{name}")
    for name in MODEL_FILES:
        fetch(model / name, f"https://huggingface.co/{MODEL}/resolve/{MODEL_REVISION}/{name}")


def inputs(data):
    paths = [data / f.replace("/", "-") for f in DATA_FILES]
    queries = sorted(pq.read_table(paths[0]).to_pylist(), key=lambda q: q["query_id"])
    qrels = defaultdict(dict)
    for row in pq.read_table(paths[1]).to_pylist():
        qid, pid, score = (row[k] for k in ("query_id", "corpus_id", "score"))
        if pid in qrels[qid] or score not in (1, 2):
            raise ValueError(f"Duplicate or unexpected qrel: {qid}, {pid}, {score}")
        qrels[qid][pid] = score
    pages = []
    for path in paths[2:]:
        pages.extend(pq.read_table(path, columns=[
            "corpus_id", "doc_id", "page_number_in_doc", "markdown"
        ]).to_pylist())
    page_ids = {p["corpus_id"] for p in pages}
    query_ids = {q["query_id"] for q in queries}
    if (
        len(pages) != 2942 or len(page_ids) != len(pages)
        or len({p["doc_id"] for p in pages}) != 6
        or len(queries) != 1854 or len(query_ids) != len(queries)
        or Counter(q["language"] for q in queries) != dict.fromkeys(LANGUAGES, 309)
        or qrels.keys() != query_ids
        or sum(map(len, qrels.values())) != 8766
        or any(not pages_for_query.keys() <= page_ids for pages_for_query in qrels.values())
    ):
        raise ValueError("Dataset membership does not match the pinned full Finance–EN release.")
    return queries, pages, qrels, paths[2:]


def image_inputs(paths, page_ids):
    from PIL import Image

    count = 0
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=8, columns=["corpus_id", "image"]):
            for row in batch.to_pylist():
                if row["corpus_id"] != page_ids[count]:
                    raise ValueError("Image row order does not match OCR row order.")
                with Image.open(BytesIO(row["image"]["bytes"])) as source:
                    yield {"image": source.convert("RGB")}
                count += 1
    if count != len(page_ids):
        raise ValueError("Some pages are missing images.")


def encode(args, queries, pages, paths):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this benchmark configuration.")
    torch.set_num_threads(4)
    spec = importlib.util.spec_from_file_location(
        "qwen_vl_pinned", args.model / "scripts/qwen3_vl_embedding.py"
    )
    wrapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wrapper)

    class CheckedEmbedder(wrapper.Qwen3VLEmbedder):
        def _preprocess_inputs(self, conversations):
            batch = super()._preprocess_inputs(conversations)
            expected_images = sum(
                entry["type"] == "image"
                for example in conversations for message in example for entry in message["content"]
            )
            grid = batch.get("image_grid_thw")
            if expected_images != (len(grid) if grid is not None else 0):
                raise ValueError("Image preprocessing failed or silently discarded an image.")
            if len(batch["attention_mask"]) != len(conversations):
                raise ValueError("Preprocessing changed the number of samples.")
            self.last_lengths = batch["attention_mask"].sum(dim=1).tolist()
            return batch

    start = perf_counter()
    embedder = CheckedEmbedder(
        str(args.model), max_length=TEXT_LIMIT, max_pixels=IMAGE_PIXELS,
        torch_dtype=torch.float16, attn_implementation="sdpa",
    )
    torch.cuda.synchronize()
    model_load_seconds = perf_counter() - start
    model_info = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "transformers": importlib.metadata.version("transformers"),
        "qwen_vl_utils": importlib.metadata.version("qwen-vl-utils"),
        "model_load_seconds": model_load_seconds,
    }
    page_ids = [p["corpus_id"] for p in pages]
    representations = {
        "ocr": (page_ids, lambda: ({"text": p["markdown"] or ""} for p in pages), args.batch_text),
        "image": (page_ids, lambda: image_inputs(paths, page_ids), args.batch_image),
        "query": ([q["query_id"] for q in queries],
                  lambda: ({"text": q["query"], "instruction": INSTRUCTION} for q in queries), 1),
    }
    needed = [args.kind] if args.kind else list(representations)
    for kind in needed:
        ids, records, batch_size = representations[kind]
        if args.smoke:
            from itertools import islice

            sample = list(islice(records(), min(2, batch_size + 1)))
            with torch.inference_mode():
                vectors = embedder.process(sample)
            torch.cuda.synchronize()
            print(f"Smoke {kind}: {tuple(vectors.shape)}, {embedder.last_lengths}", flush=True)
            continue
        checkpoint_dir = args.cache / kind
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        settings = {
            "ids": ids, "model_revision": MODEL_REVISION, "data_revision": DATA_REVISION,
            "text_limit": TEXT_LIMIT, "image_pixels": IMAGE_PIXELS, "batch_size": batch_size,
            "instruction": INSTRUCTION, "gpu": model_info["gpu"], "torch": model_info["torch"],
            "transformers": model_info["transformers"], "qwen_vl_utils": model_info["qwen_vl_utils"],
        }
        identity_path = checkpoint_dir / "identity.json"
        if identity_path.exists() and json.loads(identity_path.read_text(encoding="utf-8")) != settings:
            raise ValueError(f"Incompatible cache at {checkpoint_dir}. Use a fresh --cache directory.")
        save_json(identity_path, settings)

        # Rebuild a completed modality directly from saved blocks.
        complete = all((checkpoint_dir / f"{i:05d}.npz").exists()
                       for i in range(0, len(ids), batch_size))
        if not complete:
            iterator = iter(records())
            for offset in range(0, len(ids), batch_size):
                from itertools import islice

                sample = list(islice(iterator, batch_size))
                target = checkpoint_dir / f"{offset:05d}.npz"
                if not target.exists():
                    torch.cuda.synchronize()
                    start = perf_counter()
                    with torch.inference_mode():
                        vectors = embedder.process(sample).float().cpu().numpy()
                    torch.cuda.synchronize()
                    seconds = perf_counter() - start
                    if vectors.shape != (len(sample), 2048) or not np.isfinite(vectors).all():
                        raise ValueError(f"Invalid {kind} embedding block at {offset}.")
                    temp = target.with_suffix(".part")
                    with temp.open("wb") as out:
                        np.savez(out, vectors=vectors, seconds=seconds, lengths=embedder.last_lengths)
                    temp.replace(target)
                if offset % (batch_size * 20) == 0 or offset + batch_size >= len(ids):
                    print(f"{kind}: {min(offset + batch_size, len(ids))}/{len(ids)}", flush=True)
            # Resume partial runs in source order and validate final image count.
            if next(iterator, None) is not None:
                raise ValueError("Unexpected extra rows after encoding.")
        arrays, lengths, times = [], [], []
        start = perf_counter()
        for offset in range(0, len(ids), batch_size):
            with np.load(checkpoint_dir / f"{offset:05d}.npz") as block:
                arrays.append(block["vectors"])
                lengths.extend(block["lengths"].tolist())
                times.append(float(block["seconds"]))
        matrix = np.concatenate(arrays).astype(np.float32, copy=False)
        norms = np.linalg.norm(matrix, axis=1)
        if matrix.shape != (len(ids), 2048) or not np.allclose(norms, 1, atol=0.002):
            raise ValueError(f"{kind} embeddings are incomplete or not unit length.")
        matrix /= norms[:, None]
        matrix_seconds = perf_counter() - start
        target = args.cache / f"{kind}.npy"
        temp = target.with_suffix(".part")
        with temp.open("wb") as out:
            np.save(out, matrix)
        temp.replace(target)
        save_json(args.cache / f"{kind}.json", {
            **settings, **model_info, "encode_seconds": sum(times), "matrix_seconds": matrix_seconds,
            "sample_lengths": lengths, "batch_seconds": times,
        })


def words(text):
    return re.findall(r"\w+", text.casefold())


class BM25:
    """Positive-IDF BM25, adapted from the prior FiQA implementation."""

    def __init__(self, texts, k1=1.5, b=0.75):
        counts = [Counter(words(text)) for text in texts]
        self.lengths = np.array([sum(count.values()) for count in counts])
        avg = max(self.lengths.mean(), 1)
        self.norm = k1 * (1 - b + b * self.lengths / avg)
        postings = defaultdict(list)
        for page, count in enumerate(counts):
            for term, tf in count.items():
                postings[term].append((page, tf))
        self.postings = {}
        for term, pairs in postings.items():
            ids, tf = np.asarray(pairs).T
            idf = math.log1p((len(texts) - len(ids) + 0.5) / (len(ids) + 0.5))
            self.postings[term] = ids, idf * tf * (k1 + 1) / (tf + self.norm[ids])

    def scores(self, query):
        scores = np.zeros(len(self.lengths))
        for term, freq in Counter(words(query)).items():
            if term in self.postings:
                ids, weights = self.postings[term]
                scores[ids] += freq * weights
        return scores


def fuse(text, image):
    scores = defaultdict(float)
    for ranking in (text, image):
        for rank, pid in enumerate(ranking, 1):
            scores[pid] += 0.5 / (RRF_CONSTANT + rank)
    return sorted(scores, key=lambda pid: (-scores[pid], pid))


def metric_values(ranking, gold, k):
    grades = [gold.get(pid, 0) for pid in ranking[:k]]
    ideal = sorted(gold.values(), reverse=True)[:k]
    dcg = sum(grade / math.log2(rank + 1) for rank, grade in enumerate(grades, 1))
    idcg = sum(grade / math.log2(rank + 1) for rank, grade in enumerate(ideal, 1))
    hits = sum(grade > 0 for grade in grades)
    return {
        "hit": int(hits > 0),
        "recall": hits / len(gold),
        "precision": hits / k,
        "mrr": next((1 / rank for rank, grade in enumerate(grades, 1) if grade), 0),
        "ndcg": dcg / idcg,
        "all_judged_hit": int(hits == len(gold)),
        "complete_page_hit": int(2 in grades) if 2 in gold.values() else None,
    }


def subgroup(tags):
    categories = set(tags or [])
    if categories & {"Chart", "Image", "Infographic"}:
        return "chart_or_other_visual"
    if "Table" in categories:
        return "table"
    return "text_or_other"


def evaluate(args, queries, pages, qrels):
    args.results.mkdir(parents=True, exist_ok=True)
    page_ids = [p["corpus_id"] for p in pages]
    query_ids = [q["query_id"] for q in queries]
    matrices, manifests = {}, {}
    for key, ids in (("ocr", page_ids), ("image", page_ids), ("query", query_ids)):
        meta = json.loads((args.cache / f"{key}.json").read_text(encoding="utf-8"))
        if meta["ids"] != ids or meta["data_revision"] != DATA_REVISION or meta["model_revision"] != MODEL_REVISION:
            raise ValueError(f"Cached {key} matrix does not match dataset order or revisions.")
        matrices[key] = np.load(args.cache / f"{key}.npy")
        manifests[key] = meta
    start = perf_counter()
    bm25 = BM25([p["markdown"] or "" for p in pages])
    bm25_seconds = perf_counter() - start
    # Break all dense and sparse score ties by ascending numeric page ID.
    pid_array = np.array(page_ids)

    def sort_scores(scores):
        return pid_array[np.lexsort((pid_array, -scores))[:RRF_DEPTH]].tolist()

    bm25.scores(queries[0]["query"])
    matrices["ocr"] @ matrices["query"][0]
    matrices["image"] @ matrices["query"][0]
    details, timings, rankings = [], [], {}
    for q, qvec, encode_seconds in zip(queries, matrices["query"], manifests["query"]["batch_seconds"], strict=True):
        qid, language = q["query_id"], q["language"]
        gold = qrels[qid]
        start = perf_counter()
        sparse = sort_scores(bm25.scores(q["query"]))
        sparse_ms = (perf_counter() - start) * 1000
        start = perf_counter()
        ocr = sort_scores(matrices["ocr"] @ qvec)
        ocr_ms = (perf_counter() - start) * 1000
        start = perf_counter()
        image = sort_scores(matrices["image"] @ qvec)
        image_ms = (perf_counter() - start) * 1000
        start = perf_counter()
        combined = fuse(ocr, image)
        fusion_ms = (perf_counter() - start) * 1000
        rankings[qid] = dict(zip(METHODS, (sparse, ocr, image, combined)))
        encode_ms = encode_seconds * 1000
        timing = (
            (0, sparse_ms), (encode_ms, ocr_ms), (encode_ms, image_ms),
            (encode_ms, ocr_ms + image_ms + fusion_ms),
        )
        for method, (encoder_ms, rank_ms) in zip(METHODS, timing):
            timings.append({
                "query_id": qid, "language": language, "method": method,
                "encoder_ms": encoder_ms, "rank_ms": rank_ms, "estimated_total_ms": encoder_ms + rank_ms,
            })
            for k in K_VALUES:
                details.append({
                    "query_id": qid, "language": language, "content_group": subgroup(q["content_type"]),
                    "method": method, "k": k, "positive_pages": len(gold),
                    "full_answer_pages": sum(grade == 2 for grade in gold.values()),
                    **metric_values(rankings[qid][method], gold, k),
                })

    grouped = defaultdict(list)
    for row in details:
        groups = [("all_variants", "all"), ("language", row["language"])]
        if row["language"] == "english":
            groups.append(("english_content", row["content_group"]))
        for label, group in groups:
            grouped[(label, group, row["method"], row["k"])].append(row)
    summary = []
    for (split, group, method, k), rows in grouped.items():
        measures = {}
        for field in metric_values(list(qrels[query_ids[0]]), qrels[query_ids[0]], 1):
            values = [row[field] for row in rows if row[field] is not None]
            measures[field] = statistics.mean(values) if values else None
        summary.append({
            "scope": split, "group": group, "method": method, "k": k, "queries": len(rows),
            "complete_page_eligible_queries": sum(row["complete_page_hit"] is not None for row in rows),
            **measures,
        })
    timing_groups = defaultdict(list)
    for row in timings:
        timing_groups[("all_variants", row["method"])].append(row)
        if row["language"] == "english":
            timing_groups[("english", row["method"])].append(row)
    timing_summary = [
        {"scope": scope, "method": method, "queries": len(rows),
         "median_encode_ms": statistics.median(row["encoder_ms"] for row in rows),
         "median_rank_ms": statistics.median(row["rank_ms"] for row in rows),
         "median_estimated_total_ms": statistics.median(row["estimated_total_ms"] for row in rows),
         "p95_estimated_total_ms": float(np.percentile([row["estimated_total_ms"] for row in rows], 95))}
        for (scope, method), rows in timing_groups.items()
    ]
    write_csv(args.results / "per_query.csv", details)
    write_csv(args.results / "summary.csv", summary)
    write_csv(args.results / "query_times.csv", timings)
    write_csv(args.results / "timing_summary.csv", timing_summary)
    save_json(args.results / "rankings.json", {
        qid: {method: ranking[:10] for method, ranking in methods.items()}
        for qid, methods in rankings.items()
    })
    # Export labels without copying the underlying financial documents.
    save_json(args.results / "qrels.json", dict(qrels))
    save_json(args.results / "run_manifest.json", {
        "dataset": DATASET, "data_revision": DATA_REVISION, "model": MODEL,
        "published_asset_sha256": {name: checksum for name, (_, checksum) in LARGE_FILES.items()},
        "model_revision": MODEL_REVISION, "documents": len({p["doc_id"] for p in pages}),
        "pages": len(pages), "queries": len(queries), "qrels": sum(map(len, qrels.values())),
        "languages": dict(Counter(q["language"] for q in queries)),
        "empty_ocr_pages": [p["corpus_id"] for p in pages if not (p["markdown"] or "").strip()],
        "methods": METHODS, "depth": RRF_DEPTH, "rrf_constant": RRF_CONSTANT,
        "rrf_weights": [0.5, 0.5], "bm25": {"k1": 1.5, "b": 0.75},
        "bm25_build_seconds": bm25_seconds, "nDCG_gain": "linear relevance grade",
        "bm25_numeric_postings_bytes": int(sum(ids.nbytes + weights.nbytes
                                              for ids, weights in bm25.postings.values())),
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "platform": platform.platform(),
                    "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
                    "model_dtype": "float16", "attention": "sdpa"},
        "query_timing": "Single query GPU encoding plus CPU search; component times measured separately.",
        "embeddings": {
            key: {k: v for k, v in meta.items() if k not in ("ids", "batch_seconds", "sample_lengths")}
                 | {"rows": len(meta["ids"]), "tokens_median": statistics.median(meta["sample_lengths"]),
                    "tokens_max": max(meta["sample_lengths"]),
                    "at_token_cap": sum(x == TEXT_LIMIT for x in meta["sample_lengths"]),
                    "matrix_bytes": matrices[key].nbytes}
            for key, meta in manifests.items()
        },
    })
    print(f"Scored {len(queries)} queries on {len(pages)} pages; {len(details)} metric rows.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("download", "validate", "encode", "evaluate"))
    parser.add_argument("--data", type=Path, default=ROOT / ".cache/data")
    parser.add_argument("--model", type=Path, default=ROOT / ".cache/model")
    parser.add_argument("--cache", type=Path, default=ROOT / ".cache/embeddings")
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--batch-text", type=int, default=4)
    parser.add_argument("--batch-image", type=int, default=2)
    parser.add_argument("--kind", choices=("ocr", "image", "query"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.stage == "download":
        download(args.data, args.model)
        return
    queries, pages, qrels, paths = inputs(args.data)
    print(f"Validated {len(pages)} pages, {len(queries)} questions and {sum(map(len, qrels.values()))} qrels.", flush=True)
    if args.stage == "encode":
        encode(args, queries, pages, paths)
    elif args.stage == "evaluate":
        evaluate(args, queries, pages, qrels)


if __name__ == "__main__":
    main()
