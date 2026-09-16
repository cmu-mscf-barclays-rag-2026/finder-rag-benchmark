"""Offline checks against hand-calculated scores and evaluation invariants."""

import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

from finder_bm25.bm25 import BM25Retriever, tokenize
from finder_bm25.data import load_bundle, prepare_records, read_jsonl, save_bundle, select_queries, windows
from finder_bm25.evaluation import collect_run, evaluate_rag
from finder_bm25.metrics import (answer_exact_match, answer_token_f1, mrr_at_k, ndcg_at_k,
                                precision_at_k, recall_at_k, reciprocal_rank)
from finder_bm25.reporting import compare_runs, save_evaluation

DEMO = Path(__file__).resolve().parents[1] / "examples/demo.jsonl"


class RankingMetricsTests(unittest.TestCase):
    def test_mrr_rank_one_five_and_missing(self):
        rankings = {"q1": ["gold"], "q2": ["a", "b", "c", "d", "gold"], "q3": []}
        qrels = {qid: {"gold": 1} for qid in rankings}
        self.assertAlmostEqual(mrr_at_k(rankings, qrels, 5), (1 + 1 / 5 + 0) / 3)
        self.assertAlmostEqual(mrr_at_k(rankings, qrels, 3), 1 / 3)

    def test_mrr_does_not_drop_missing_queries(self):
        self.assertEqual(mrr_at_k({"q1": ["g"]}, {"q1": {"g": 1}, "q2": {"g": 1}}, 5), 0.5)

    def test_ndcg_uses_unretrieved_gold_for_ideal(self):
        self.assertAlmostEqual(ndcg_at_k(["a"], {"a": 1, "b": 1}, 5), 1 / (1 + 1 / math.log2(3)))

    def test_graded_ndcg_hand_calculation(self):
        actual = 1 + 7 / math.log2(3)
        ideal = 7 + 1 / math.log2(3)
        self.assertAlmostEqual(ndcg_at_k(["low", "high"], {"high": 3, "low": 1}, 2), actual / ideal)
        self.assertEqual(ndcg_at_k(["high", "low"], {"high": 3, "low": 1}, 2), 1)

    def test_truncation_and_precision_denominator(self):
        self.assertEqual(reciprocal_rank(["miss", "gold"], {"gold": 1}, 1), 0)
        self.assertEqual(precision_at_k(["gold"], {"gold": 1}, 5), 0.2)
        self.assertEqual(recall_at_k(["gold"], {"gold": 1, "other": 1}, 5), 0.5)

    def test_empty_and_no_relevance(self):
        for metric in (reciprocal_rank, ndcg_at_k, precision_at_k, recall_at_k):
            self.assertEqual(metric([], {"gold": 1}, 5), 0)
            self.assertEqual(metric(["a"], {}, 5), 0)

    def test_bad_inputs_rejected(self):
        for metric in (reciprocal_rank, ndcg_at_k, precision_at_k, recall_at_k):
            for ranking, qrels, k in [(["g", "g"], {"g": 1}, 5), ([], {}, 0),
                                      ([], {"g": -1}, 5), ([], {"g": float("nan")}, 5)]:
                with self.assertRaises(ValueError):
                    metric(ranking, qrels, k)

    def test_answer_numbers_retain_sign_decimal_and_percent(self):
        self.assertEqual(answer_exact_match(" $2.5 MILLION ", "$2.5 million"), 1)
        for gold in ("25", "-2.5", "2.5%"):
            self.assertEqual(answer_exact_match("2.5", gold), 0)
            self.assertEqual(answer_token_f1("2.5", gold), 0)


class BM25Tests(unittest.TestCase):
    def test_score_matches_hand_calculation(self):
        corpus = [{"doc_id": "a", "text": "profit profit"}, {"doc_id": "b", "text": "loss"}]
        retriever = BM25Retriever(corpus, k1=1.2, b=0.75)
        hit = retriever.search("profit")[0]
        expected = math.log(2) * 2 * 2.2 / (2 + 1.2 * (0.25 + 0.75 * 2 / 1.5))
        self.assertEqual(hit["doc_id"], "a")
        self.assertAlmostEqual(hit["score"], expected)

    def test_empty_query_unknown_terms_and_stable_ties(self):
        index = BM25Retriever([{"doc_id": "z", "text": "cash"}, {"doc_id": "a", "text": "cash"}])
        self.assertEqual(index.search(""), [])
        self.assertEqual(index.search("unknown"), [])
        self.assertEqual([hit["doc_id"] for hit in index.search("cash", 10)], ["a", "z"])
        self.assertEqual(index.search("cash cash"), index.search("cash"))

    def test_financial_tokenization(self):
        self.assertEqual(tokenize("CBOE’s $1,200.50 FY2023 10-K"), ["cboe", "1200.50", "fy2023", "10", "k"])

    def test_invalid_bm25_configuration(self):
        for k1, b in [(-1, 0.5), (1.2, 1.1), (float("nan"), 0.5)]:
            with self.assertRaises(ValueError):
                BM25Retriever([{"doc_id": "a", "text": "cash"}], k1, b)
        with self.assertRaises(ValueError):
            BM25Retriever([])


class SharedEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.records = read_jsonl(DEMO)
        self.bundle = prepare_records(self.records)
        self.run = collect_run(self.bundle, BM25Retriever(self.bundle["corpus"]), top_k=5, partition="all")

    def test_reference_deduplication_and_no_label_leakage(self):
        records = copy.deepcopy(self.records)
        records[1]["references"].append(records[0]["references"][0].replace(" ", "  "))
        bundle = prepare_records(records)
        self.assertEqual(len(bundle["corpus"]), len(self.bundle["corpus"]))
        self.assertTrue(set(bundle["qrels"][records[0]["_id"]]) & set(bundle["qrels"][records[1]["_id"]]))
        self.assertTrue(all(set(doc) == {"doc_id", "text", "reference_id", "start", "end", "reference_length"}
                            for doc in bundle["corpus"]))
        self.assertNotIn(records[0]["answer"], " ".join(doc["text"] for doc in bundle["corpus"]))

    def test_reordered_records_have_identical_manifest_and_partition(self):
        reordered = prepare_records(list(reversed(self.records)))
        self.assertEqual(reordered["manifest"]["fingerprint"], self.bundle["manifest"]["fingerprint"])
        dev = {q["query_id"] for q in select_queries(self.bundle, "dev")}
        test = {q["query_id"] for q in select_queries(self.bundle, "test")}
        self.assertFalse(dev & test)
        self.assertEqual(len(dev | test), len(self.records))

    def test_query_limit_does_not_shrink_corpus(self):
        count = len(self.bundle["corpus"])
        self.assertEqual(len(select_queries(self.bundle, "all", 2)), 2)
        self.assertEqual(len(self.bundle["corpus"]), count)

    def test_bundle_roundtrip_detects_mutation(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            save_bundle(self.bundle, directory)
            self.assertEqual(load_bundle(directory), self.bundle)
            (directory / "qrels.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_bundle(directory)

    def test_chunk_offsets_cover_reference_and_overlaps_count_once(self):
        text = "alpha beta gamma delta epsilon zeta"
        covered = set()
        for start, end in windows(text, 15, 4):
            self.assertLessEqual(end - start, 15)
            covered.update(range(start, end))
        self.assertEqual(covered, set(range(len(text))))
        bundle = prepare_records(self.records, chunk_size=30, overlap=10)
        run = collect_run(bundle, BM25Retriever(bundle["corpus"]), top_k=100, partition="all")
        for row in run["results"]:
            row["doc_ids"] = list(bundle["qrels"][row["query_id"]])
            row.pop("scores")
        scores = evaluate_rag(bundle, run)
        self.assertEqual(scores["summary"]["evidence_coverage"], 1)
        self.assertEqual(scores["summary"]["recall"], 1)

    def test_generation_unmeasured_is_null(self):
        summary = evaluate_rag(self.bundle, self.run)["summary"]
        self.assertIsNone(summary["answer_accuracy_exact_match"])
        self.assertIsNone(summary["answer_correctness"])
        self.assertEqual(summary["answer_scored_count"], 0)
        self.assertEqual(summary["query_count"], len(self.records))

    def test_person3_correctness_hook_and_partial_answers(self):
        query = {q["query_id"]: q for q in self.bundle["queries"]}[self.run["results"][0]["query_id"]]
        self.run["results"][0]["answer"] = query["answer"]
        summary = evaluate_rag(self.bundle, self.run, answer_scorer=lambda prediction, gold: 0.75,
                               answer_scorer_name="fixture_v1")["summary"]
        self.assertEqual(summary["answer_accuracy_exact_match"], 1)
        self.assertEqual(summary["answer_correctness"], 0.75)
        self.assertEqual(summary["answer_status"], "partial")
        with self.assertRaises(ValueError):
            evaluate_rag(self.bundle, self.run, answer_scorer=lambda p, g: 1)

    def test_invalid_or_incomplete_runs_rejected(self):
        bad_runs = []
        for key, value in [("bundle_fingerprint", "different"), ("top_k", 0), ("query_ids", [])]:
            run = copy.deepcopy(self.run)
            run[key] = value
            bad_runs.append(run)
        run = copy.deepcopy(self.run)
        run["results"].pop()
        bad_runs.append(run)
        for ids in (["unknown"], [self.run["results"][0]["doc_ids"][0]] * 2):
            run = copy.deepcopy(self.run)
            run["results"][0]["doc_ids"] = ids
            run["results"][0].pop("scores")
            bad_runs.append(run)
        for run in bad_runs:
            with self.assertRaises(ValueError):
                evaluate_rag(self.bundle, run)

    def test_method_independence_and_comparison_export(self):
        other = copy.deepcopy(self.run)
        other["method"] = "teammate_fixture"
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            compare_runs(self.bundle, [self.run, other], output)
            self.assertIn("0 wins, 0 losses", (output / "comparison.md").read_text(encoding="utf-8"))
            save_evaluation(self.bundle, self.run, evaluate_rag(self.bundle, self.run), output / "bm25")
            self.assertTrue((output / "bm25/report.md").exists())
            other["top_k"] = 10
            with self.assertRaises(ValueError):
                compare_runs(self.bundle, [self.run, other], output)


if __name__ == "__main__":
    unittest.main()
