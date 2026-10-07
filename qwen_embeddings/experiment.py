"""Controlled dense retrieval experiments on Legal RAG Bench.

Run preparation and development before freezing settings and evaluating test.
Heavy ML dependencies are imported only when model execution is requested.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def load_data():
    manifest = read_json(ROOT / "data/manifest.json")
    if digest(ROOT / "assets/context_tokenizer.json") != manifest["context_tokenizer_sha256"]:
        raise ValueError("Context tokenizer checksum mismatch")
    for name, expected in manifest["sha256"].items():
        if digest(ROOT / "data" / name) != expected:
            raise ValueError(f"Data checksum mismatch: {name}")
    records = {}
    for name in ("children", "parents", "queries"):
        records[name] = [json.loads(line) for line in
                         (ROOT / "data" / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()]
    for name in ("children", "parents", "queries"):
        ids = [r["id"] for r in records[name]]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate {name} IDs")
    return records


def ranking_metrics(ranked, gold, k):
    """Binary relevance on ranked units, with fixed-k precision."""
    import math
    if not gold or len(ranked) != len(set(ranked)):
        raise ValueError("Gold must be nonempty and ranking IDs unique")
    rel = [int(pid in set(gold)) for pid in ranked[:k]]
    ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(set(gold)))))
    return {"hit": float(any(rel)), "recall": sum(rel) / len(set(gold)),
            "precision": sum(rel) / k,
            "mrr": next((1 / (i + 1) for i, r in enumerate(rel) if r), 0.0),
            "ndcg": sum(r / math.log2(i + 2) for i, r in enumerate(rel)) / ideal}


def rank(vector, matrix, k=10):
    import numpy as np
    # Exact cosine search of normalized vectors. Stable corpus-order tie break.
    return np.argsort(-(matrix @ vector), kind="stable")[:k].tolist()


def context_tokenizer():
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(ROOT / "assets/context_tokenizer.json"))
    tok.no_truncation()
    tok.no_padding()
    return tok


class Encoder:
    """Official-style Qwen last-token pooling and MiniLM masked mean pooling."""
    def __init__(self, key, config, device="auto"):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch, self.key, self.config = torch, key, config
        torch.set_num_threads(config["cpu_threads"])
        torch.manual_seed(42)
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        spec = config["models"][key]
        self.tokenizer = AutoTokenizer.from_pretrained(spec["id"], revision=spec["revision"],
                                                       cache_dir=str(ROOT / ".cache/models"),
                                                       padding_side="left" if key == "qwen" else "right")
        self.tokenizer.truncation_side = "right"
        self.model = AutoModel.from_pretrained(
            spec["id"], revision=spec["revision"], cache_dir=str(ROOT / ".cache/models"),
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            attn_implementation="sdpa").to(self.device).eval()

    def sync(self):
        if self.device == "cuda":
            self.torch.cuda.synchronize()

    def query_text(self, text):
        if self.key == "qwen":
            return f'Instruct: {self.config["query_instruction"]}\nQuery:{text}'
        return text

    def encode(self, texts, window, query=False):
        torch = self.torch
        if query:
            texts = [self.query_text(t) for t in texts]
        batch = self.tokenizer(texts, padding=True, truncation=True, max_length=window,
                               return_tensors="pt").to(self.device)
        with torch.inference_mode():
            hidden = self.model(**batch).last_hidden_state
            if self.key == "qwen":
                # Left padding places the final non-padding token at the last position.
                pooled = hidden[:, -1]
            else:
                mask = batch["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            pooled = torch.nn.functional.normalize(pooled.float(), p=2, dim=1)
        return pooled.cpu().numpy()


def window_profile(encoder, texts, window):
    lengths = [len(encoder.tokenizer(t, truncation=False, add_special_tokens=True)["input_ids"])
               for t in texts]
    return {"count": len(lengths), "min": min(lengths), "mean": mean(lengths), "max": max(lengths),
            "truncated_count": sum(n > window for n in lengths),
            "truncation_rate": mean(n > window for n in lengths),
            "mean_visible_input_tokens": mean(min(n, window) for n in lengths)}, lengths


def run_id(key, corpus, window):
    return f"{key}_{corpus}_w{window}"


def identity(config):
    return fingerprint({"config": config, "data": read_json(ROOT / "data/manifest.json"),
                        "code": {p.name: digest(p) for p in [ROOT / "experiment.py",
                                  ROOT / "_hierarchy/handoff.py", ROOT / "_hierarchy/core.py"]}})


def execute(key, corpus, window, split, config, device, batch_size):
    import numpy as np
    from _hierarchy.handoff import evaluate_run
    data = load_data()
    units = data["children" if corpus == "child" else "parents"]
    queries = [q for q in data["queries"] if q["split"] == split]
    rid, signature = run_id(key, corpus, window), identity(config)
    target = ROOT / "results" / split / rid
    if (target / "run.json").exists():
        saved = read_json(target / "run.json")
        if saved["experiment_signature"] != signature:
            raise ValueError("Existing result belongs to different code/config/data; use a separate folder")
        print(f"Already complete: {split}/{rid}", flush=True)
        return
    encoder = Encoder(key, config, device)
    print(f"Running {rid}: {len(units)} units, {len(queries)} {split} queries, {encoder.device}", flush=True)
    texts = [u["text"] for u in units]
    profile, lengths = window_profile(encoder, texts, window)
    query_window = config["models"][key]["query_window"]
    qprofile, _ = window_profile(encoder, [encoder.query_text(q["question"]) for q in queries], query_window)
    target.mkdir(parents=True, exist_ok=True)
    write_json(target / "lengths.json", {"passages": profile, "queries": qprofile,
                                        "per_passage": dict(zip([u["id"] for u in units], lengths))})
    cache = ROOT / ".cache/embeddings" / signature / rid
    cache.mkdir(parents=True, exist_ok=True)
    # One cache file per fixed batch; interrupted GPU/CPU runs can resume.
    arrays, encoding_seconds = [], 0.0
    for start in range(0, len(units), batch_size):
        stop = min(start + batch_size, len(units))
        stem = cache / f"{start}_{stop}"
        npy, meta = stem.with_suffix(".npy"), stem.with_suffix(".json")
        if npy.exists() and meta.exists():
            vectors = np.load(npy)
            info = read_json(meta)
            if info["device"] != encoder.device or info["hardware"] != hardware(encoder):
                raise ValueError("Cached vectors timed on different hardware; remove this embedding cache first")
        else:
            encoder.sync()
            begin = time.perf_counter()
            vectors = encoder.encode(texts[start:stop], window)
            encoder.sync()
            info = {"seconds": time.perf_counter() - begin, "device": encoder.device,
                    "hardware": hardware(encoder)}
            np.save(npy, vectors)
            write_json(meta, info)
        arrays.append(vectors)
        encoding_seconds += info["seconds"]
        if start % (batch_size * 100) == 0:
            print(f"Embedded {stop}/{len(units)}", flush=True)
    begin = time.perf_counter()
    matrix = np.ascontiguousarray(np.concatenate(arrays), dtype=np.float32)
    index_seconds = time.perf_counter() - begin
    if not np.isfinite(matrix).all() or not np.allclose(np.linalg.norm(matrix, axis=1), 1, atol=1e-3):
        raise ValueError("Embeddings must be finite normalized vectors")
    # Warmup excluded. Latency includes single-query encoding and exact CPU search,
    # but not context assembly, model load, corpus encoding, or disk output.
    rank(encoder.encode([queries[0]["question"]], query_window, query=True)[0], matrix)
    predictions, raw_latency = {}, []
    for repeat in range(config["latency_repeats"]):
        for q in queries:
            encoder.sync()
            begin = time.perf_counter()
            vector = encoder.encode([q["question"]], query_window, query=True)[0]
            encoder.sync()
            encoded_at = time.perf_counter()
            positions = rank(vector, matrix, max(config["ks"]))
            finished = time.perf_counter()
            ids = [units[i]["id"] for i in positions]
            if repeat == 0:
                predictions[q["id"]] = ids
            raw_latency.append({"query_id": q["id"], "repeat": repeat,
                                "query_encode_ms": (encoded_at - begin) * 1000,
                                "search_ms": (finished - encoded_at) * 1000,
                                "total_ms": (finished - begin) * 1000})
    detail, summaries = evaluate_run(queries, predictions, {x["id"]: x for x in data["children"]},
                                     {x["id"]: x for x in data["parents"]}, context_tokenizer(),
                                     mode=corpus, ks=config["ks"], token_budget=config["context_budget"])
    for row in detail:
        q = next(q for q in queries if q["id"] == row["query_id"])
        gold = q["gold_parent_ids"] if corpus == "parent" else q["gold_child_ids"]
        row.update({f"unit_{k}": v for k, v in ranking_metrics(predictions[q["id"]], gold, row["k"]).items()})
    for row in summaries:
        for metric in ("hit", "recall", "precision", "mrr", "ndcg"):
            row[f"unit_{metric}"] = mean(d[f"unit_{metric}"] for d in detail if d["k"] == row["k"])
    write_json(target / "predictions.json", predictions)
    write_json(target / "per_query.json", detail)
    write_json(target / "summary.json", summaries)
    write_json(target / "latency_raw.json", raw_latency)
    write_json(target / "run.json", {
        "experiment_signature": signature, "run_id": rid, "model": key, "corpus": corpus,
        "window": window, "query_window": query_window, "split": split, "n_queries": len(queries),
        "config": config, "passage_profile": profile, "query_profile": qprofile,
        "corpus_encode_seconds": encoding_seconds, "index_assembly_seconds": index_seconds,
        "encoding_time_note": "Sum of batch encoding times, including original times for resumed cached batches",
        "latency_mean_ms": mean(r["total_ms"] for r in raw_latency),
        "latency_p95_ms": float(np.percentile([r["total_ms"] for r in raw_latency], 95)),
        "hardware": hardware(encoder), "batch_size": batch_size,
        "versions": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "numpy", "tokenizers")}})


def hardware(encoder):
    return {"platform": platform.platform(), "processor": platform.processor(),
            "device": encoder.device, "gpu": encoder.torch.cuda.get_device_name() if encoder.device == "cuda" else None}


def select(config):
    selected = []
    for corpus in ("child", "parent"):
        candidates = []
        for key, windows in (("minilm", [256]), ("qwen", config["qwen_windows"])):
            for window in windows:
                path = ROOT / "results/dev" / run_id(key, corpus, window)
                run = read_json(path / "run.json")
                if run["experiment_signature"] != identity(config):
                    raise ValueError("Stale development run")
                row = next(r for r in read_json(path / "summary.json") if r["k"] == 5)
                if key == "qwen":
                    candidates.append((row["gold_body_token_coverage"], row["unit_hit"], -window, window))
        winner = max(candidates)[-1]
        # Keep a preregistered short-window anchor alongside the dev-selected setting.
        selected.extend({"model": key, "corpus": corpus, "window": w}
                        for key, w in dict.fromkeys([("minilm", 256), ("qwen", 512), ("qwen", winner)]))
    output = ROOT / "results/selection.json"
    value = {"experiment_signature": identity(config), "runs": selected,
             "rule": "Per corpus: maximize dev gold-body-token-coverage@5 at fixed budget; tie unit Hit@5 then shorter window",
             "dev_hashes": {p.relative_to(ROOT).as_posix(): digest(p) for p in (ROOT / "results/dev").glob("*/summary.json")}}
    if output.exists() and read_json(output) != value:
        raise ValueError("Selection is already frozen; use a new experiment folder to change it")
    write_json(output, value)
    print(json.dumps(value, indent=2))


def report():
    """Generate real-result tables and paired error examples only from completed runs."""
    import pandas as pd
    rows, comparisons = [], []
    data = load_data()
    questions = {q["id"]: q for q in data["queries"]}
    for split in ("dev", "test"):
        for path in (ROOT / "results" / split).glob("*/run.json"):
            run = read_json(path)
            for metric in read_json(path.parent / "summary.json"):
                rows.append({"split": split, "run_id": run["run_id"], "model": run["model"],
                             "corpus": run["corpus"], "window": run["window"], **metric,
                             "device": run["hardware"]["device"], "gpu": run["hardware"].get("gpu"),
                             "truncation_rate": run["passage_profile"]["truncation_rate"],
                             "corpus_encode_seconds": run["corpus_encode_seconds"],
                             "index_assembly_seconds": run["index_assembly_seconds"],
                             "latency_mean_ms": run["latency_mean_ms"], "latency_p95_ms": run["latency_p95_ms"]})
            if run["model"] != "qwen" or run["window"] == 512:
                continue
            anchor = path.parent.parent / run_id("qwen", run["corpus"], 512)
            if not (anchor / "run.json").exists():
                continue
            short = {r["query_id"]: r for r in read_json(anchor / "per_query.json") if r["k"] == 5}
            for r in read_json(path.parent / "per_query.json"):
                if r["k"] != 5:
                    continue
                delta = r["gold_body_token_coverage"] - short[r["query_id"]]["gold_body_token_coverage"]
                if delta:
                    q = questions[r["query_id"]]
                    comparisons.append({"split": split, "run_id": run["run_id"], "query_id": q["id"],
                                        "question": q["question"], "gold_child_ids": q["gold_child_ids"],
                                        "coverage_delta_vs_qwen512": delta, "effect": "helped" if delta > 0 else "hurt",
                                        "short_ranking": read_json(anchor / "predictions.json")[q["id"]],
                                        "long_ranking": read_json(path.parent / "predictions.json")[q["id"]]})
    if rows:
        pd.DataFrame(rows).to_csv(ROOT / "results/comparison.csv", index=False)
    write_json(ROOT / "results/paired_examples.json", comparisons)
    print(f"{len(rows)} result rows; {len(comparisons)} helped/hurt examples. No synthetic scores added.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["audit", "smoke", "dev", "select", "test", "report"])
    parser.add_argument("--model", choices=["minilm", "qwen"], default="minilm")
    parser.add_argument("--corpus", choices=["child", "parent"], default="child")
    parser.add_argument("--window", type=int, default=256)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    config = read_json(ROOT / "config.json")
    if args.batch_size < 1:
        parser.error("Batch size must be positive")
    if args.command == "audit":
        data = load_data()
        children = {r["id"]: r for r in data["children"]}
        for parent in data["parents"]:
            for span in parent["child_spans"]:
                assert parent["text"][span["start_char"]:span["end_char"]] == children[span["child_id"]]["text"]
        for q in data["queries"]:
            assert q["gold_parent_ids"] == list(dict.fromkeys(children[c]["parent_id"] for c in q["gold_child_ids"]))
        result = {"counts": {k: len(v) for k, v in data.items()},
                  "splits": {s: sum(q["split"] == s for q in data["queries"]) for s in ("dev", "test")},
                  "all_parent_spans_and_gold_mappings_valid": True, "scope": "Data validation, not model performance"}
        write_json(ROOT / "results/data_audit.json", result)
        print(json.dumps(result, indent=2))
    elif args.command == "smoke":
        import numpy as np
        enc = Encoder(args.model, config, args.device)
        doc = enc.encode(["A court may excuse a juror.", "The ocean contains salt water."], 256)
        query = enc.encode(["When may a juror be excused?"], config["models"][args.model]["query_window"], True)
        assert np.isfinite(doc).all() and np.allclose(np.linalg.norm(doc, axis=1), 1, atol=1e-3)
        single = enc.encode(["A court may excuse a juror."], 256)
        assert np.allclose(single[0], doc[0], atol=2e-3), "Pooling/padding batch consistency failed"
        write_json(ROOT / f"results/smoke_{args.model}.json", {"shape": list(doc.shape),
                   "ranking": rank(query[0], doc, 2), "hardware": hardware(enc),
                   "batch_padding_consistency_passed": True,
                   "scope": "Two synthetic documents; execution check only, not benchmark results"})
    elif args.command == "dev":
        allowed = [256] if args.model == "minilm" else config["qwen_windows"]
        if args.window not in allowed:
            parser.error(f"Use registered windows: {allowed}")
        execute(args.model, args.corpus, args.window, "dev", config, args.device, args.batch_size)
    elif args.command == "select":
        select(config)
    elif args.command == "test":
        selection = read_json(ROOT / "results/selection.json")
        if selection["experiment_signature"] != identity(config):
            raise ValueError("Frozen configuration no longer matches")
        for path, sha in selection["dev_hashes"].items():
            if digest(ROOT / path) != sha:
                raise ValueError("Development results changed after selection")
        for run in selection["runs"]:
            execute(run["model"], run["corpus"], run["window"], "test", config, args.device, args.batch_size)
    else:
        report()


if __name__ == "__main__":
    main()
