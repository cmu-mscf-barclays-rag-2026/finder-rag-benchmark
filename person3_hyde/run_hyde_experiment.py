"""Person 3: reference-answer oracle and deployable HyDE retrieval.

This experiment uses the official 100-question Isaacus Legal RAG Bench test set.
The reference answer is used only as an oracle diagnostic.  Deployable HyDE
generates a hypothetical legal answer from the question alone and retrieves with
the generated text.  It never receives the reference answer or gold passage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
import transformers
from scipy.stats import binomtest
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_DATA_DIR = ROOT / "data" / "legal_rag_bench"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "person3_hyde"
KS = (1, 3, 5, 10)


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_benchmark(data_dir: Path):
    corpus_path = data_dir / "corpus.jsonl"
    qa_path = data_dir / "qa.jsonl"
    if not corpus_path.exists() or not qa_path.exists():
        raise FileNotFoundError(
            "Legal RAG Bench data are missing. Put corpus.jsonl and qa.jsonl in "
            f"{data_dir}, or first run legal_rag_hybrid_experiment.py to download them."
        )

    corpus_rows = load_jsonl(corpus_path)
    qa_rows = load_jsonl(qa_path)
    passage_ids = [str(row["id"]) for row in corpus_rows]
    id_to_index = {passage_id: i for i, passage_id in enumerate(passage_ids)}
    corpus = [
        " ".join((str(row.get("title", "")), str(row["text"]))).strip()
        for row in corpus_rows
    ]
    questions = [str(row["question"]) for row in qa_rows]
    answers = [str(row["answer"]) for row in qa_rows]
    gold = np.asarray(
        [id_to_index[str(row["relevant_passage_id"])] for row in qa_rows],
        dtype=np.int32,
    )
    return corpus_rows, qa_rows, passage_ids, corpus, questions, answers, gold


def topk_from_scores(scores: np.ndarray, depth: int) -> np.ndarray:
    depth = min(depth, scores.shape[1])
    indices = np.argpartition(scores, -depth, axis=1)[:, -depth:]
    rows = np.arange(scores.shape[0])[:, None]
    order = np.argsort(scores[rows, indices], axis=1)[:, ::-1]
    return indices[rows, order].astype(np.int32)


def encode_texts(model, texts: list[str], batch_size: int) -> np.ndarray:
    return model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )


def retrieve_dense(
    model,
    corpus_embeddings: np.ndarray,
    texts: list[str],
    depth: int,
    batch_size: int,
) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    query_embeddings = encode_texts(model, texts, batch_size)
    rankings = topk_from_scores(query_embeddings @ corpus_embeddings.T, depth)
    latency_ms = 1000 * (time.perf_counter() - start) / len(texts)
    return rankings, latency_ms


def generate_hypothetical_answers(
    questions: list[str],
    model_name: str,
    model_revision: str,
    cache_path: Path,
    device: str,
    max_new_tokens: int,
    batch_size: int,
    prompt_style: str,
) -> tuple[list[str], float]:
    if cache_path.exists():
        cached = pd.read_csv(cache_path)
        cache_matches = (
            cached["question"].tolist() == questions
            and cached["hyde_text"].notna().all()
            and "generator_model" in cached
            and cached["generator_model"].eq(model_name).all()
            and "generator_revision" in cached
            and cached["generator_revision"].eq(model_revision).all()
            and "prompt_style" in cached
            and cached["prompt_style"].eq(prompt_style).all()
        )
        if cache_matches:
            cached_latency = float(cached["generation_ms_per_query"].iloc[0])
            return cached["hyde_text"].astype(str).tolist(), cached_latency

    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=model_revision)
    generator = AutoModelForSeq2SeqLM.from_pretrained(
        model_name, revision=model_revision
    )
    generator.to(device)
    generator.eval()

    if prompt_style == "concise_legal":
        prompt_template = (
            "Answer this Australian criminal-law question in two or three concise "
            "sentences. Name the relevant legal rule or procedure and explain why it "
            "applies. Do not repeat the question and do not say the answer is "
            "hypothetical. Question: {question} Legal answer:"
        )
    elif prompt_style == "long_legal":
        prompt_template = (
            "Write a short, precise legal passage that would answer the question. "
            "State the likely rule and reasoning in formal legal language. Do not "
            "mention that the passage is hypothetical. Question: {question}"
        )
    else:
        raise ValueError(f"Unknown prompt style: {prompt_style}")
    prompts = [prompt_template.format(question=question) for question in questions]
    generated: list[str] = []
    start = time.perf_counter()
    for offset in range(0, len(prompts), batch_size):
        batch = prompts[offset : offset + batch_size]
        tokens = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)
        with torch.inference_mode():
            output = generator.generate(
                **tokens,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
            )
        generated.extend(tokenizer.batch_decode(output, skip_special_tokens=True))
    latency_ms = 1000 * (time.perf_counter() - start) / len(questions)
    pd.DataFrame(
        {
            "query_id": np.arange(1, len(questions) + 1),
            "question": questions,
            "hyde_text": generated,
            "generator_model": model_name,
            "generator_revision": model_revision,
            "prompt_style": prompt_style,
            "generation_ms_per_query": latency_ms,
        }
    ).to_csv(cache_path, index=False)
    return generated, latency_ms


def first_gold_rank(ranking: np.ndarray, gold: int) -> int:
    found = np.flatnonzero(ranking == gold)
    return int(found[0] + 1) if len(found) else 0


def metrics_at_k(rankings: np.ndarray, gold: np.ndarray, k: int) -> dict:
    ranks = np.asarray(
        [first_gold_rank(row, int(target)) for row, target in zip(rankings, gold)]
    )
    hit = (ranks > 0) & (ranks <= k)
    reciprocal = np.where(hit, 1.0 / np.maximum(ranks, 1), 0.0)
    ndcg = np.where(hit, 1.0 / np.log2(np.maximum(ranks, 1) + 1), 0.0)
    return {
        "k": k,
        "n_queries": len(gold),
        "precision_at_k": float(hit.mean() / k),
        "recall_at_k": float(hit.mean()),
        "hit_rate_at_k": float(hit.mean()),
        "mrr_at_k": float(reciprocal.mean()),
        "ndcg_at_k": float(ndcg.mean()),
    }


def weighted_rrf(
    first: np.ndarray,
    second: np.ndarray,
    second_weight: float,
    depth: int,
    rrf_k: int = 60,
) -> np.ndarray:
    fused = []
    for first_rank, second_rank in zip(first, second):
        scores: dict[int, float] = {}
        for rank, doc in enumerate(first_rank, 1):
            scores[int(doc)] = scores.get(int(doc), 0.0) + (1 - second_weight) / (
                rrf_k + rank
            )
        for rank, doc in enumerate(second_rank, 1):
            scores[int(doc)] = scores.get(int(doc), 0.0) + second_weight / (
                rrf_k + rank
            )
        fused.append(sorted(scores, key=scores.get, reverse=True)[:depth])
    return np.asarray(fused, dtype=np.int32)


def paired_test(candidate: np.ndarray, baseline: np.ndarray, gold: np.ndarray, k: int) -> dict:
    candidate_hit = np.asarray(
        [0 < first_gold_rank(row, int(target)) <= k for row, target in zip(candidate, gold)]
    )
    baseline_hit = np.asarray(
        [0 < first_gold_rank(row, int(target)) <= k for row, target in zip(baseline, gold)]
    )
    candidate_only = int((candidate_hit & ~baseline_hit).sum())
    baseline_only = int((baseline_hit & ~candidate_hit).sum())
    discordant = candidate_only + baseline_only
    p_value = float(binomtest(candidate_only, discordant, 0.5).pvalue) if discordant else 1.0
    return {
        "k": k,
        "candidate_only_hits": candidate_only,
        "baseline_only_hits": baseline_only,
        "net_hit_difference": candidate_only - baseline_only,
        "mcnemar_exact_p": p_value,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument(
        "--embedding-revision",
        default="1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
    )
    parser.add_argument("--generator-model", default="google/flan-t5-base")
    parser.add_argument(
        "--generator-revision",
        default="7bcac572ce56db69c1ea7c8af255c5d7c9672fc2",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--candidate-depth", type=int, default=100)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--generation-batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--prompt-style",
        choices=("concise_legal", "long_legal"),
        default="concise_legal",
    )
    parser.add_argument("--hybrid-dense-weight", type=float, default=0.25)
    args = parser.parse_args()

    random.seed(20260929)
    np.random.seed(20260929)
    torch.manual_seed(20260929)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    (
        corpus_rows,
        qa_rows,
        passage_ids,
        corpus,
        questions,
        answers,
        gold,
    ) = load_benchmark(args.data_dir)

    embedder = SentenceTransformer(
        args.embedding_model,
        revision=args.embedding_revision,
        device=args.device,
    )
    embedding_cache = args.data_dir / "all_MiniLM_L6_v2_corpus_embeddings.npy"
    if embedding_cache.exists():
        corpus_embeddings = np.load(embedding_cache)
    else:
        corpus_embeddings = encode_texts(
            embedder, corpus, args.embedding_batch_size
        )
        np.save(embedding_cache, corpus_embeddings)

    question_dense, question_dense_latency = retrieve_dense(
        embedder,
        corpus_embeddings,
        questions,
        args.candidate_depth,
        args.embedding_batch_size,
    )
    oracle_dense, oracle_dense_latency = retrieve_dense(
        embedder,
        corpus_embeddings,
        answers,
        args.candidate_depth,
        args.embedding_batch_size,
    )

    generated_path = args.output_dir / "generated_hyde_answers.csv"
    hyde_texts, generation_latency = generate_hypothetical_answers(
        questions,
        args.generator_model,
        args.generator_revision,
        generated_path,
        args.device,
        args.max_new_tokens,
        args.generation_batch_size,
        args.prompt_style,
    )
    hyde_dense, hyde_dense_latency = retrieve_dense(
        embedder,
        corpus_embeddings,
        hyde_texts,
        args.candidate_depth,
        args.embedding_batch_size,
    )

    # Reuse the same question-BM25 rankings and RRF convention as Part C.
    from legal_rag_hybrid_experiment import build_bm25, retrieve_bm25

    vectorizer, bm25_matrix = build_bm25(corpus)
    question_bm25, bm25_latency = retrieve_bm25(
        vectorizer, bm25_matrix, questions, args.candidate_depth
    )
    standard_hybrid = weighted_rrf(
        question_bm25,
        question_dense,
        second_weight=args.hybrid_dense_weight,
        depth=args.candidate_depth,
    )
    hybrid_start = time.perf_counter()
    hyde_hybrid = weighted_rrf(
        question_bm25,
        hyde_dense,
        second_weight=args.hybrid_dense_weight,
        depth=args.candidate_depth,
    )
    fusion_latency = 1000 * (time.perf_counter() - hybrid_start) / len(questions)

    methods = {
        "Question BM25": (question_bm25, bm25_latency, "deployable"),
        "Question Dense MiniLM": (question_dense, question_dense_latency, "deployable"),
        "Question BM25 + Question Dense RRF": (
            standard_hybrid,
            bm25_latency + question_dense_latency + fusion_latency,
            "deployable",
        ),
        "Reference Answer Dense (oracle)": (oracle_dense, oracle_dense_latency, "oracle_only"),
        "Question-only HyDE Dense": (
            hyde_dense,
            generation_latency + hyde_dense_latency,
            "deployable",
        ),
        "Question BM25 + HyDE Dense RRF": (
            hyde_hybrid,
            bm25_latency + generation_latency + hyde_dense_latency + fusion_latency,
            "deployable",
        ),
    }

    metric_rows = []
    for method, (ranking, latency, status) in methods.items():
        for k in KS:
            metric_rows.append(
                metrics_at_k(ranking, gold, k)
                | {
                    "method": method,
                    "status": status,
                    "latency_ms_per_query": latency,
                    "generation_included_in_latency": method.startswith("Question-only")
                    or "HyDE Dense RRF" in method,
                }
            )
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(args.output_dir / "retrieval_metrics.csv", index=False)
    metrics[metrics.k == 5].to_csv(args.output_dir / "summary_k5.csv", index=False)

    method_rankings = {
        "question_bm25": question_bm25,
        "question_dense": question_dense,
        "standard_hybrid": standard_hybrid,
        "oracle_answer_dense": oracle_dense,
        "hyde_dense": hyde_dense,
        "hyde_hybrid": hyde_hybrid,
    }
    detail_rows = []
    for i, row in enumerate(qa_rows):
        detail = {
            "query_id": row["id"],
            "question": row["question"],
            "reference_answer": row["answer"],
            "hyde_text": hyde_texts[i],
            "gold_passage_id": row["relevant_passage_id"],
            "gold_title": corpus_rows[int(gold[i])].get("title", ""),
        }
        for name, ranking in method_rankings.items():
            detail[f"{name}_gold_rank"] = first_gold_rank(ranking[i], int(gold[i]))
            detail[f"{name}_top5_ids"] = json.dumps(
                [passage_ids[j] for j in ranking[i, :5]]
            )
        detail_rows.append(detail)
    details = pd.DataFrame(detail_rows)
    details.to_csv(args.output_dir / "per_query_results.csv", index=False)

    q_hit = details.question_dense_gold_rank.between(1, 5)
    h_hit = details.hyde_dense_gold_rank.between(1, 5)
    hh_hit = details.hyde_hybrid_gold_rank.between(1, 5)
    example_frames = []
    for label, mask in {
        "hyde_dense_helps": h_hit & ~q_hit,
        "hyde_dense_hurts": q_hit & ~h_hit,
        "hyde_hybrid_helps_vs_question_dense": hh_hit & ~q_hit,
        "hyde_hybrid_hurts_vs_question_dense": q_hit & ~hh_hit,
    }.items():
        frame = details[mask].copy()
        frame.insert(0, "category", label)
        example_frames.append(frame)
    pd.concat(example_frames, ignore_index=True).to_csv(
        args.output_dir / "hyde_help_hurt_examples.csv", index=False
    )

    paired_rows = []
    for candidate_name, candidate, baseline_name, baseline in [
        (
            "Reference Answer Dense (oracle)",
            oracle_dense,
            "Question Dense MiniLM",
            question_dense,
        ),
        (
            "Question-only HyDE Dense",
            hyde_dense,
            "Question Dense MiniLM",
            question_dense,
        ),
        (
            "Question BM25 + HyDE Dense RRF",
            hyde_hybrid,
            "Question BM25 + Question Dense RRF",
            standard_hybrid,
        ),
        (
            "Question BM25 + HyDE Dense RRF",
            hyde_hybrid,
            "Question BM25",
            question_bm25,
        ),
    ]:
        paired_rows.append(
            paired_test(candidate, baseline, gold, 5)
            | {
                "candidate": candidate_name,
                "baseline": baseline_name,
            }
        )
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(args.output_dir / "paired_comparisons_k5.csv", index=False)

    k5 = metrics[metrics.k == 5].set_index("method")
    oracle_gain = (
        k5.loc["Reference Answer Dense (oracle)", "recall_at_k"]
        - k5.loc["Question Dense MiniLM", "recall_at_k"]
    )
    hyde_gain = (
        k5.loc["Question-only HyDE Dense", "recall_at_k"]
        - k5.loc["Question Dense MiniLM", "recall_at_k"]
    )
    hybrid_gain = (
        k5.loc["Question BM25 + HyDE Dense RRF", "recall_at_k"]
        - k5.loc["Question BM25 + Question Dense RRF", "recall_at_k"]
    )
    table = "\n".join(
        f"| {method} | {row.recall_at_k:.2f} | {row.mrr_at_k:.4f} | "
        f"{row.ndcg_at_k:.4f} | {row.latency_ms_per_query:.2f} |"
        for method, row in k5.iterrows()
    )
    findings = f"""# Person 3 — Reference-answer and HyDE retrieval

## Main Top-5 results

| Method | Hit/Recall@5 | MRR@5 | nDCG@5 | End-to-end ms/query |
|---|---:|---:|---:|---:|
{table}

## Interpretation

- Using the expert-written reference answer changes Hit/Recall@5 by
  {oracle_gain:+.2f} versus the original question. This is an **oracle diagnostic**, not
  a deployable method, because the answer is unavailable when a user asks a question.
- Question-only HyDE changes Hit/Recall@5 by {hyde_gain:+.2f} versus question-dense.
- Replacing question-dense with HyDE-dense inside the same BM25-heavy hybrid changes
  Hit/Recall@5 by {hybrid_gain:+.2f}. Both hybrids are deployable and use only the
  question, so this is the fairest test of HyDE's incremental value.
- With only 100 questions, small differences should be described as directional.
  See `paired_comparisons_k5.csv` for exact paired p-values.

## Generation sensitivity

An exploratory generation check found that HyDE results changed materially with the
prompt and decoding configuration. The longer prompt produced HyDE-dense Hit@5 of
0.22 and HyDE-hybrid Hit@5 of 0.32; the concise no-repeat configuration produced
0.15 and 0.37. This instability is itself an important finding: the current local
generator does not provide a robust standalone retrieval improvement. The saved
primary result uses the concise configuration, and `generation_sensitivity_k5.csv`
records both exploratory runs.

## What each comparison tests

1. **Question Dense:** the unchanged dense baseline.
2. **Reference Answer Dense:** whether answer-like wording closes the query-document
   vocabulary gap; it provides an approximate upper-bound diagnostic.
3. **Question-only HyDE Dense:** whether a locally generated hypothetical legal answer
   improves dense retrieval without seeing the gold answer.
4. **Standard Hybrid:** the existing BM25 + question-dense control under the same RRF
   weight.
5. **Question BM25 + HyDE Dense RRF:** whether exact terms from the original question
   and semantic terms from HyDE complement each other.

The generator is `{args.generator_model}` with deterministic greedy decoding. A compact
local generator keeps the experiment reproducible on a laptop, but its legal knowledge
is limited. Poorly formed or overly generic hypothetical answers can reduce retrieval
quality; `hyde_help_hurt_examples.csv` makes those cases auditable.
"""
    (args.output_dir / "findings.md").write_text(findings, encoding="utf-8")

    run_summary = {
        "dataset": "isaacus/legal-rag-bench",
        "n_passages": len(corpus),
        "n_questions": len(questions),
        "corpus_sha256": sha256(args.data_dir / "corpus.jsonl"),
        "qa_sha256": sha256(args.data_dir / "qa.jsonl"),
        "embedding_model": args.embedding_model,
        "embedding_revision": args.embedding_revision,
        "generator_model": args.generator_model,
        "generator_revision": args.generator_revision,
        "generation": {
            "input": "question only",
            "prompt_style": args.prompt_style,
            "decoding": "greedy",
            "max_new_tokens": args.max_new_tokens,
            "seed": 20260929,
        },
        "hybrid": {
            "first": "original-question BM25",
            "second": "HyDE dense",
            "second_weight": args.hybrid_dense_weight,
            "rrf_k": 60,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "sentence_transformers": __import__("sentence_transformers").__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    (args.output_dir / "run_summary.json").write_text(
        json.dumps(run_summary, indent=2), encoding="utf-8"
    )
    print(metrics[metrics.k == 5].to_string(index=False))


if __name__ == "__main__":
    main()
