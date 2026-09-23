"""Hand-calculated cases for ranking, coverage, and invalid submissions."""

import math
import unittest

from legal_rag_a.evaluation import evaluate, query_metrics
from legal_rag_a.profile import team_split


class EvaluationTests(unittest.TestCase):
    def test_two_required_passages(self):
        row = query_metrics(["x", "a", "b"], {"a", "b"}, 3)
        self.assertAlmostEqual(row["precision_at_k"], 2 / 3)
        self.assertEqual(row["recall_at_k"], 1)
        self.assertEqual(row["all_gold_included_at_k"], 1)
        self.assertEqual(row["mrr_at_k"], .5)
        expected = (1 / math.log2(3) + .5) / (1 + 1 / math.log2(3))
        self.assertAlmostEqual(row["ndcg_at_k"], expected)

    def test_partial_evidence_is_not_complete(self):
        row = query_metrics(["a", "x"], {"a", "b"}, 2)
        self.assertEqual(row["hit_rate_at_k"], 1)
        self.assertEqual(row["evidence_coverage_at_k"], .5)
        self.assertEqual(row["all_gold_included_at_k"], 0)

    def test_short_ranking_and_empty_output(self):
        row = query_metrics(["a"], {"a"}, 5)
        self.assertEqual(row["precision_at_k"], .2)
        self.assertEqual(row["precision_returned"], 1)
        row = query_metrics([], {"a"}, 5)
        self.assertEqual(row["ndcg_at_k"], 0)
        self.assertEqual(row["empty_rate"], 1)

    def test_duplicates_and_unlabeled_queries_fail(self):
        for ranking, gold, k in [(["a", "a"], {"a"}, 5), ([], set(), 5),
                                 (["a"], {"a"}, 0), ("a", {"a"}, 1)]:
            with self.assertRaises(ValueError):
                query_metrics(ranking, gold, k)

    def test_missing_or_foreign_predictions_fail(self):
        queries = [{"id": "1", "gold_ids": ["a"]}]
        for predictions in ({}, {"1": ["foreign"]}, {"1": [], "2": []}):
            with self.assertRaises(ValueError):
                evaluate(queries, predictions, {"a"})

    def test_macro_average_and_single_gold_identity(self):
        queries = [{"id": "1", "gold_ids": ["a"]}, {"id": "2", "gold_ids": ["b"]}]
        _, rows = evaluate(queries, {"1": ["a"], "2": []}, {"a", "b"}, (5,))
        row = rows[0]
        for name in ("recall_at_k", "evidence_coverage_at_k", "all_gold_included_at_k",
                     "hit_rate_at_k"):
            self.assertEqual(row[name], .5)
        self.assertEqual(row["precision_at_k"], .1)

    def test_rank_cutoff(self):
        self.assertEqual(query_metrics(["x", "a"], {"a"}, 1)["recall_at_k"], 0)

    def test_malformed_gold_and_unordered_ranking_fail(self):
        with self.assertRaises(ValueError):
            evaluate([{'id': '1', 'gold_ids': 'ab'}], {'1': ['a']}, {'a', 'b'})
        with self.assertRaises(ValueError):
            query_metrics({'a', 'b'}, {'a'}, 2)
        with self.assertRaises(ValueError):
            evaluate([{'id': '1', 'gold_ids': [1]}], {'1': [1]}, {1})

    def test_split_keeps_shared_gold_together_and_is_order_invariant(self):
        queries = [{"id": str(i), "gold_ids": [str(i // 2)]} for i in range(20)]
        config = {"split_seed": "test", "team_dev_fraction": .2, "team_split_id": "test"}
        first = {r["query_id"]: r["split"] for r in team_split(queries, config)}
        second = {r["query_id"]: r["split"] for r in team_split(list(reversed(queries)), config)}
        self.assertEqual(first, second)
        for i in range(0, 20, 2):
            self.assertEqual(first[str(i)], first[str(i + 1)])


if __name__ == "__main__":
    unittest.main()
