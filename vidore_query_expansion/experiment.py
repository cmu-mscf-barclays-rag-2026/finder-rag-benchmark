"""Controlled full-corpus MultiQueryRetriever and DSPy retrieval experiments."""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import importlib.util
import itertools
import json
import os
from pathlib import Path
import re
import statistics
from time import perf_counter

import numpy as np

from api_runtime import API
from benchmark_core import BM25, INSTRUCTION, inputs, metric_values, save_json, subgroup, write_csv, DATA_REVISION, MODEL_REVISION


ROOT = Path(__file__).resolve().parent
BACKENDS = ("bm25", "ocr", "image", "combined")
STYLES = {
    "general": (
        "Write exactly five different search queries expressing the information need in the question. "
        "Use alternative terminology while preserving the meaning. Keep each query short and "
        "write it in the same language as the question. Output one query per line, without numbering."
    ),
    "financial": (
        "Write exactly five concise alternative financial-document search queries for the question. "
        "Preserve the requested companies, reporting periods, units, securities and comparisons. "
        "Keep necessary conditions and do not introduce guessed figures or answer the question. "
        "Use varied financial terminology and emphasize the evidence sought. Keep the same language "
        "as the question. Output one search query per line without numbering."
    ),
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def clean_queries(values):
    cleaned = []
    for text in values:
        text = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", text).strip().strip('"')
        if text and text.casefold() not in {v.casefold() for v in cleaned}:
            cleaned.append(text)
    if len(cleaned) != 5:
        raise ValueError(f"Expected five unique generated queries, received {len(cleaned)}.")
    return cleaned


def rrf(rankings, weights, constant, depth):
    scores = defaultdict(float)
    for ranking, weight in zip(rankings, weights, strict=True):
        for rank, pid in enumerate(ranking[:depth], 1):
            scores[pid] += weight / (constant + rank)
    return sorted(scores, key=lambda pid: (-scores[pid], pid))


class Experiment:
    def __init__(self, args):
        self.args = args
        self.config = read(ROOT / "protocol.json")
        self.queries, self.pages, self.gold, _ = inputs(args.data)
        self.by_id = {q["query_id"]: q for q in self.queries}
        self.page_ids = [p["corpus_id"] for p in self.pages]
        self.original_vectors = np.load(args.vectors / "query.npy")
        self.matrices = {kind: np.load(args.vectors / f"{kind}.npy") for kind in ("ocr", "image")}
        for kind in ("ocr", "image", "query"):
            meta = read(args.vectors / f"{kind}.json")
            expected = [q["query_id"] for q in self.queries] if kind == "query" else self.page_ids
            if meta["ids"] != expected:
                raise ValueError(f"Cached {kind} vectors are in a different source order.")
            if meta["data_revision"] != DATA_REVISION or meta["model_revision"] != MODEL_REVISION:
                raise ValueError("The embedding cache uses a different dataset/model revision.")
            if meta["text_limit"] != 4096 or meta["instruction"] != INSTRUCTION:
                raise ValueError("The embedding cache uses a different window/instruction.")
        self.query_vector_by_text = {
            q["query"]: v for q, v in zip(self.queries, self.original_vectors, strict=True)
        }
        self.cache = args.cache
        self.results = ROOT / "results"
        self.results.mkdir(exist_ok=True)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.models = {}
        self.rank_cache = {}
        self.api = None

    def prepare(self):
        groups = defaultdict(list)
        for q in self.queries:
            signature = tuple(sorted(self.gold[q["query_id"]].items()))
            groups[signature].append(q)
        ordered = sorted(groups, key=lambda sig: digest([self.config["split_seed"], sig]))
        allocation = {}
        remaining = {name: self.config[f"english_{name}"] for name in ("train", "validation")}
        for name in ("train", "validation"):
            for sig in ordered:
                size = sum(q["language"] == "english" for q in groups[sig])
                if sig not in allocation and 0 < size <= remaining[name]:
                    allocation[sig] = name
                    remaining[name] -= size
            if remaining[name]:
                raise ValueError("Could not allocate the exact requested group-held-out split.")
        splits = []
        for sig, qs in groups.items():
            if {q["language"] for q in qs} != {
                "english", "french", "german", "italian", "portuguese", "spanish"
            }:
                raise ValueError("Translation group is missing a language.")
            for q in qs:
                splits.append({
                    "query_id": q["query_id"], "language": q["language"],
                    "split": allocation.get(sig, "test"), "group": digest(sig),
                })
        write_csv(self.results / "splits.csv", sorted(splits, key=lambda r: r["query_id"]))
        save_json(self.results / "protocol.json", self.config)
        assert len(splits) == 1854
        counts = {}
        for name in ("train", "validation", "test"):
            counts[name] = {
                "english": sum(r["split"] == name and r["language"] == "english" for r in splits),
                "all_variants": sum(r["split"] == name for r in splits),
            }
        save_json(self.results / "data_audit.json", {
            "pages": len(self.pages), "queries": len(self.queries),
            "relevance_groups": len(groups), "split_counts": counts,
            "grouping": "Identical full relevance signatures conservatively group translations and some distinct queries.",
            "historical_scope": "Optimizer-held-out test; these queries were previously examined in the earlier benchmark.",
        })
        print("Full-data split:", counts, flush=True)

    def partition(self, name, english=False):
        import csv

        with (self.results / "splits.csv").open(encoding="utf-8") as source:
            ids = {int(row["query_id"]) for row in csv.DictReader(source)
                   if row["split"] == name and (not english or row["language"] == "english")}
        return [q for q in self.queries if q["query_id"] in ids]

    def api_client(self):
        if self.api is None:
            self.api = API(self.cache / "api", self.config["generator_model"])
        return self.api

    def sparse_model(self, k1, b):
        key = k1, b
        if key not in self.models:
            self.models[key] = BM25([p["markdown"] or "" for p in self.pages], k1=k1, b=b)
        return self.models[key]

    def vector(self, text):
        if text in self.query_vector_by_text:
            return self.query_vector_by_text[text]
        path = self.cache / "query_vectors" / f"{digest(text)}.npy"
        if not path.exists():
            raise ValueError("Generated query vectors are missing. Run the embed stage.")
        vector = np.load(path)
        if vector.shape != (2048,) or not np.isfinite(vector).all() or abs(np.linalg.norm(vector) - 1) > 0.002:
            raise ValueError("A generated-query embedding is invalid.")
        self.query_vector_by_text[text] = vector
        return vector

    def rank(self, text, backend, baseline):
        key = backend, digest(text), baseline.get("k1"), baseline.get("b"), baseline.get("ocr_weight")
        if key in self.rank_cache:
            return self.rank_cache[key]
        if backend == "combined":
            weight = baseline["ocr_weight"]
            ranking = rrf(
                [self.rank(text, "ocr", baseline), self.rank(text, "image", baseline)],
                [weight, 1 - weight], 60, 100,
            )
        else:
            if backend == "bm25":
                scores = self.sparse_model(baseline["k1"], baseline["b"]).scores(text)
            else:
                scores = self.matrices[backend] @ self.vector(text)
            ranking = np.asarray(self.page_ids)[np.lexsort((self.page_ids, -scores))[:100]].tolist()
        self.rank_cache[key] = ranking
        return ranking

    def expanded_rank(self, q, generated, backend, baseline, params):
        queries = generated[:params["count"]]
        ranking = [self.rank(text, backend, baseline) for text in queries]
        weights = [1.0] * len(queries)
        if params["original_weight"]:
            ranking.insert(0, self.rank(q["query"], backend, baseline))
            weights.insert(0, params["original_weight"])
        return rrf(ranking, weights, params["rrf_constant"], params["depth"])

    def score(self, ranking, q):
        values = metric_values(ranking, self.gold[q["query_id"]], 10)
        return values["ndcg"], values["recall"]

    def tune_baselines(self):
        validation = self.partition("validation", english=True)
        rows = []
        for k1, b in itertools.product(self.config["bm25_k1"], self.config["bm25_b"]):
            scores = [self.score(self.rank(q["query"], "bm25", {"k1": k1, "b": b}), q) for q in validation]
            rows.append({
                "family": "bm25", "k1": k1, "b": b, "ocr_weight": "",
                "ndcg_at_10": statistics.mean(s[0] for s in scores),
                "recall_at_10": statistics.mean(s[1] for s in scores),
            })
        best = max(rows, key=lambda r: (
            r["ndcg_at_10"], r["recall_at_10"], -abs(r["k1"] - 1.5), -abs(r["b"] - 0.75),
        ))
        baseline = {"k1": best["k1"], "b": best["b"]}
        # Release alternative BM25 postings once the selected configuration is known.
        self.models = {key: value for key, value in self.models.items() if key == (best["k1"], best["b"])}
        dense_rows = []
        for weight in self.config["ocr_fusion_weights"]:
            params = {**baseline, "ocr_weight": weight}
            scores = [self.score(self.rank(q["query"], "combined", params), q) for q in validation]
            dense_rows.append({
                "family": "text_image", "k1": "", "b": "", "ocr_weight": weight,
                "ndcg_at_10": statistics.mean(s[0] for s in scores),
                "recall_at_10": statistics.mean(s[1] for s in scores),
            })
        chosen = max(dense_rows, key=lambda r: (
            r["ndcg_at_10"], r["recall_at_10"], -abs(r["ocr_weight"] - 0.5),
        ))
        baseline["ocr_weight"] = chosen["ocr_weight"]
        write_csv(self.results / "baseline_tuning.csv", rows + dense_rows)
        save_json(self.results / "baseline_selection.json", baseline)
        print("Frozen baseline settings", baseline, flush=True)

    def dspy_components(self):
        import dspy
        from bridges import DSPyLM

        api = self.api_client()

        class SearchQueries(dspy.Signature):
            """Write exactly five concise alternative search queries for the financial question.
            Preserve companies, dates, units, scope and necessary conditions. Use varied phrasing
            without inventing answers. Write all five in the language of the question."""

            question: str = dspy.InputField()
            queries: list[str] = dspy.OutputField()

        class QueryProgram(dspy.Module):
            def __init__(self):
                super().__init__()
                self.generate = dspy.Predict(SearchQueries)

            def forward(self, question):
                return self.generate(question=question)

        lm = DSPyLM(api)
        return QueryProgram, lm

    def generate_one(self, q, strategy):
        path = self.cache / "generated" / strategy / f"{q['query_id']}.json"
        if path.exists():
            return read(path)["queries"]
        if strategy.startswith("mq_"):
            from bridges import multiquery

            generated = multiquery(self.api_client(), q["query"], STYLES[strategy[3:]])
        else:
            import dspy

            Program, lm = self.dspy_components()
            program = Program()
            if strategy.startswith("dspy_seed_"):
                program.load(str(self.results / f"{strategy}.json"))
            with dspy.context(lm=lm):
                generated = program(question=q["query"]).queries
        repaired = False
        try:
            generated = clean_queries(generated)
        except ValueError:
            # Repair only the required output format; do not use relevance labels.
            response, _ = self.api_client().complete([
                {"role": "system", "content": STYLES["financial"]},
                {"role": "user", "content": q["query"] + "\nReturn exactly five distinct queries, one per line."},
            ])
            generated = clean_queries(response["choices"][0]["message"]["content"].splitlines())
            repaired = True
        save_json(path, {"query_id": q["query_id"], "strategy": strategy,
                         "queries": generated, "format_repaired": repaired})
        return generated

    def generate(self, strategies, questions):
        self.api_client()
        for strategy in strategies:
            with ThreadPoolExecutor(max_workers=16) as pool:
                futures = [pool.submit(self.generate_one, q, strategy) for q in questions]
                for i, future in enumerate(futures, 1):
                    future.result()
                    if i % 25 == 0 or i == len(futures):
                        print(f"Generated {strategy}: {i}/{len(futures)}", flush=True)

    def optimize(self):
        import dspy

        Program, lm = self.dspy_components()
        baseline = read(self.results / "baseline_selection.json")
        trainset = [dspy.Example(question=q["query"], qid=q["query_id"]).with_inputs("question")
                    for q in self.partition("train", english=True)]
        valset = [dspy.Example(question=q["query"], qid=q["query_id"]).with_inputs("question")
                  for q in self.partition("validation", english=True)]
        params = {"count": 3, "original_weight": 1.0, "depth": 100, "rrf_constant": 60}

        def metric(example, prediction, trace=None):
            alternatives = list(prediction.queries)
            if not alternatives or any(not isinstance(q, str) for q in alternatives):
                return 0.0
            q = self.by_id[example.qid]
            ranking = self.expanded_rank(q, alternatives, "bm25", baseline, params)
            return self.score(ranking, q)[0]

        options = self.config["dspy_optimizer"]
        for seed in options["seeds"]:
            target = self.results / f"dspy_seed_{seed}.json"
            if target.exists():
                continue
            with dspy.context(lm=lm):
                optimizer = dspy.MIPROv2(
                    metric=metric, auto=None, num_candidates=options["num_candidates"],
                    num_threads=6, seed=seed, max_bootstrapped_demos=0, max_labeled_demos=0,
                )
                result = optimizer.compile(
                    Program(), trainset=trainset, valset=valset,
                    num_trials=options["num_trials"], minibatch=False,
                    max_bootstrapped_demos=0, max_labeled_demos=0,
                )
            result.set_lm(None)
            result.save(str(target))
            history = [{
                "trial": int(index), "ndcg_at_10": float(info["full_eval_score"]) / 100,
                "instruction_index": info.get("0_predictor_instruction", "default"),
            } for index, info in getattr(result, "trial_logs", {}).items()
                if isinstance(info, dict) and "full_eval_score" in info]
            save_json(self.results / f"optimizer_trials_{seed}.json", history)
            print(f"DSPy optimizer seed {seed} saved.", flush=True)

    def embed(self):
        import torch

        files = sorted((self.cache / "generated").glob("*/*.json"))
        texts = sorted({text for path in files for text in read(path)["queries"]})
        target = self.cache / "query_vectors"
        target.mkdir(exist_ok=True)
        missing = [text for text in texts if not (target / f"{digest(text)}.npy").exists()
                   and text not in self.query_vector_by_text]
        if not missing:
            print("Generated query vectors are complete.", flush=True)
            return
        spec = importlib.util.spec_from_file_location(
            "pinned_qwen_vl", self.args.model / "scripts/qwen3_vl_embedding.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        torch.set_num_threads(4)
        model = module.Qwen3VLEmbedder(
            str(self.args.model), max_length=4096, torch_dtype=torch.float16, attn_implementation="sdpa",
        )
        tokenizer = model.processor.tokenizer
        missing.sort(key=lambda text: len(tokenizer.encode(text)))
        timings = []
        for offset in range(0, len(missing), 16):
            batch = missing[offset:offset + 16]
            torch.cuda.synchronize()
            begin = perf_counter()
            with torch.inference_mode():
                vectors = model.process([{"text": text, "instruction": INSTRUCTION} for text in batch])
            torch.cuda.synchronize()
            seconds = perf_counter() - begin
            vectors = vectors.float().cpu().numpy()
            vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
            for text, vector in zip(batch, vectors, strict=True):
                np.save(target / f"{digest(text)}.npy", vector)
            timings.append({"count": len(batch), "seconds": seconds})
            if offset % 320 == 0:
                print(f"Query embeddings: {offset + len(batch)}/{len(missing)}", flush=True)
        save_json(target / f"batch_times_{len(list(target.glob('*.json')))}.json", timings)

    def tune_expansion(self):
        baseline = read(self.results / "baseline_selection.json")
        validation = self.partition("validation", english=True)
        strategies = ["mq_general", "mq_financial", "dspy_initial"] + [
            f"dspy_seed_{seed}" for seed in self.config["dspy_optimizer"]["seeds"]
        ]
        rows, selected = [], {}
        for strategy in strategies:
            generated = {
                q["query_id"]: read(self.cache / "generated" / strategy / f"{q['query_id']}.json")["queries"]
                for q in validation
            }
            for backend in BACKENDS:
                choices = []
                # Reuse the candidate rankings throughout the grid; no extra API calls.
                grid = itertools.product(
                    self.config["expansion_counts"], self.config["candidate_depths"],
                    self.config["rrf_constants"], self.config["original_query_weights"],
                )
                for count, depth, constant, weight in grid:
                    params = {"count": count, "depth": depth, "rrf_constant": constant,
                              "original_weight": weight}
                    values = [self.score(self.expanded_rank(q, generated[q["query_id"]], backend,
                                                           baseline, params), q)
                              for q in validation]
                    row = {
                        "strategy": strategy, "backend": backend, **params,
                        "ndcg_at_10": statistics.mean(s[0] for s in values),
                        "recall_at_10": statistics.mean(s[1] for s in values),
                    }
                    choices.append(row)
                    rows.append(row)
                chosen = max(choices, key=lambda r: (
                    r["ndcg_at_10"], r["recall_at_10"], -r["count"], -r["depth"],
                    r["original_weight"] > 0, -abs(r["rrf_constant"] - 60),
                    -abs(r["original_weight"] - 1),
                ))
                selected[strategy, backend] = chosen
            print(f"Parameter grid complete: {strategy}", flush=True)
        write_csv(self.results / "expansion_tuning.csv", rows)
        final = {}
        for backend in BACKENDS:
            def choose(candidates):
                return max(candidates, key=lambda r: (
                    r["ndcg_at_10"], r["recall_at_10"], -r["count"], -r["depth"],
                    r["original_weight"] > 0, -abs(r["rrf_constant"] - 60),
                ))
            final[backend] = {
                "multiquery": choose([selected[s, backend] for s in ("mq_general", "mq_financial")]),
                "dspy_initial": selected["dspy_initial", backend],
                "dspy_optimized": choose([selected[f"dspy_seed_{seed}", backend]
                                         for seed in self.config["dspy_optimizer"]["seeds"]]),
            }
        selection = {"baseline": baseline, "expansion": final, "protocol_sha256": digest(self.config)}
        save_json(self.results / "selection.json", selection)
        print("Frozen all development-selected settings.", flush=True)

    def final_generation(self):
        selected = read(self.results / "selection.json")
        strategies = sorted({
            value["strategy"] for config in selected["expansion"].values() for value in config.values()
        })
        self.generate(strategies, self.queries)

    def evaluate(self):
        import csv

        selected = read(self.results / "selection.json")
        if selected["protocol_sha256"] != digest(self.config):
            raise ValueError("Protocol changed after selection.")
        with (self.results / "splits.csv").open(encoding="utf-8") as source:
            splits = {int(row["query_id"]): row for row in csv.DictReader(source)}
        rankings, details, timings = {}, [], []
        for index, q in enumerate(self.queries):
            qid = q["query_id"]
            by_method = {}
            for backend in BACKENDS:
                methods = {"original": None, **selected["expansion"][backend]}
                for family, params in methods.items():
                    start = perf_counter()
                    if params is None:
                        ranking = self.rank(q["query"], backend, selected["baseline"])
                    else:
                        generated = read(
                            self.cache / "generated" / params["strategy"] / f"{qid}.json"
                        )["queries"]
                        ranking = self.expanded_rank(q, generated, backend, selected["baseline"], params)
                    seconds = perf_counter() - start
                    method = f"{backend}/{family}"
                    by_method[method] = ranking[:20]
                    timings.append({
                        "query_id": qid, "method": method, "lookup_and_fusion_ms": seconds * 1000,
                        "note": "Uses cached vectors and ranking lists; excludes query generation and embedding.",
                    })
                    for k in self.config["evaluation_ks"]:
                        details.append({
                            "query_id": qid, "language": q["language"], "split": splits[qid]["split"],
                            "group": splits[qid]["group"], "content_group": subgroup(q["content_type"]),
                            "method": method, "k": k, **metric_values(ranking, self.gold[qid], k),
                        })
            rankings[qid] = by_method
            if index % 100 == 0:
                print(f"Evaluated {index + 1}/{len(self.queries)}", flush=True)
        write_csv(self.results / "per_query.csv", details)
        # Keep the full metric export below GitHub's browser-upload file limit.
        import gzip
        import shutil
        with (self.results / "per_query.csv").open("rb") as source:
            with gzip.open(self.results / "per_query.csv.gz", "wb") as target:
                shutil.copyfileobj(source, target)
        (self.results / "per_query.csv").unlink()
        write_csv(self.results / "ranking_times.csv", timings)
        save_json(self.results / "rankings.json", rankings)
        save_json(self.results / "qrels.json", dict(self.gold))
        metrics = list(metric_values([0], {0: 1}, 1))
        groups = defaultdict(list)
        for row in details:
            scopes = [
                ("all_queries", "all"), ("split", row["split"]),
                ("language", row["language"]), ("split_language", row["split"] + "/" + row["language"]),
            ]
            if row["language"] == "english" and row["split"] == "test":
                scopes.append(("test_english_content", row["content_group"]))
            for scope, value in scopes:
                groups[scope, value, row["method"], row["k"]].append(row)
        summary = []
        for (scope, value, method, k), rows in groups.items():
            result = {"scope": scope, "group": value, "method": method, "k": k,
                      "queries": len(rows)}
            for metric in metrics:
                values = [r[metric] for r in rows if r[metric] is not None]
                result[metric] = statistics.mean(values) if values else None
            result["complete_page_eligible_queries"] = sum(r["complete_page_hit"] is not None for r in rows)
            summary.append(result)
        write_csv(self.results / "summary.csv", summary)
        usage = self.api_client().usage()
        write_csv(self.results / "api_usage.csv", usage)
        save_json(self.results / "manifest.json", {
            "pages": len(self.pages), "queries": len(self.queries), "protocol_sha256": digest(self.config),
            "selection_sha256": digest(selected), "api_requests": len(usage),
            "prompt_tokens": sum(r["prompt_tokens"] for r in usage),
            "completion_tokens": sum(r["completion_tokens"] for r in usage),
            "estimated_api_usd": sum(r["estimated_usd"] for r in usage),
            "cost_note": "Estimated from recorded usage at USD 0.15/M input and 0.60/M output, without cache discounts.",
            "timing_note": "Separate generation API latency, batched embedding time, and cached rank/fusion time.",
        })
        print(f"Full evaluation complete: {len(details)} rows.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=(
        "prepare", "baselines", "generate_dev", "optimize", "embed", "tune",
        "generate_all", "evaluate",
    ))
    parser.add_argument("--data", type=Path, default=ROOT / ".cache/data")
    parser.add_argument("--model", type=Path, default=ROOT / ".cache/model")
    parser.add_argument("--vectors", type=Path, default=ROOT / ".cache/page_vectors")
    parser.add_argument("--cache", type=Path, default=ROOT / ".cache/experiment")
    args = parser.parse_args()
    experiment = Experiment(args)
    actions = {
        "prepare": experiment.prepare,
        "baselines": experiment.tune_baselines,
        "generate_dev": lambda: experiment.generate(
            ["mq_general", "mq_financial", "dspy_initial"] +
            [f"dspy_seed_{s}" for s in experiment.config["dspy_optimizer"]["seeds"]
             if (experiment.results / f"dspy_seed_{s}.json").exists()],
            experiment.partition("validation", english=True),
        ),
        "optimize": experiment.optimize, "embed": experiment.embed,
        "tune": experiment.tune_expansion, "generate_all": experiment.final_generation,
        "evaluate": experiment.evaluate,
    }
    actions[args.stage]()


if __name__ == "__main__":
    main()
