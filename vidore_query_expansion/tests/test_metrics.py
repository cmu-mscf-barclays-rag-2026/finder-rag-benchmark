"""Behavioral checks for graded, partial, missing and alternative evidence."""

import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark_core import metric_values
from experiment import clean_queries, rrf


class RetrievalChecks(unittest.TestCase):
    def test_partial_evidence_is_not_all_judged_pages(self):
        result = metric_values([9, 2, 10], {2: 1, 3: 2}, 3)
        self.assertEqual(result["hit"], 1)
        self.assertEqual(result["recall"], 0.5)
        self.assertEqual(result["all_judged_hit"], 0)
        self.assertEqual(result["complete_page_hit"], 0)
        self.assertEqual(result["precision"], 1 / 3)
        self.assertEqual(result["mrr"], 0.5)

    def test_graded_ranking_penalizes_weaker_page_first(self):
        ideal = metric_values([3, 2], {2: 1, 3: 2}, 2)
        reverse = metric_values([2, 3], {2: 1, 3: 2}, 2)
        self.assertEqual(ideal["ndcg"], 1)
        self.assertAlmostEqual(reverse["ndcg"], (1 + 2 / math.log2(3)) / (2 + 1 / math.log2(3)))

    def test_top_k_excludes_correct_page_beyond_cutoff(self):
        self.assertEqual(metric_values([8, 7, 2], {2: 2}, 2)["hit"], 0)
        self.assertEqual(metric_values([8, 7, 2], {2: 2}, 3)["hit"], 1)

    def test_missing_complete_grade_is_ineligible(self):
        self.assertIsNone(metric_values([2], {2: 1}, 1)["complete_page_hit"])

    def test_complete_page_hit_is_not_all_page_recall(self):
        result = metric_values([2], {2: 2, 3: 1}, 1)
        self.assertEqual(result["complete_page_hit"], 1)
        self.assertEqual(result["all_judged_hit"], 0)

    def test_rrf_rewards_consensus_and_deduplicates(self):
        self.assertEqual(rrf([[1, 3], [2, 3]], [1, 1], 60, 2), [3, 1, 2])

    def test_rrf_respects_candidate_depth(self):
        self.assertEqual(rrf([[1, 3], [2, 3]], [1, 1], 60, 1), [1, 2])

    def test_output_validation_rejects_duplicates(self):
        with self.assertRaises(ValueError):
            clean_queries(["same"] * 5)


if __name__ == "__main__":
    unittest.main()
