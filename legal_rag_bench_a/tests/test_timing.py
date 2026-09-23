"""Protocol tests with stubs and a controlled clock, not retrieval benchmarks."""

import unittest
from unittest.mock import patch
from legal_rag_a.timing import benchmark_retrievers


class TimingTests(unittest.TestCase):
    def test_warmup_counts_units_and_predictions(self):
        calls = []
        def retrieve(text, k):
            calls.append(text)
            return iter(['A'])
        queries = [{'id': '1', 'question': 'first'}, {'id': '2', 'question': 'second'}]
        with patch('legal_rag_a.timing.time.perf_counter', side_effect=[i * .002 for i in range(16)]):
            raw, summary, predictions = benchmark_retrievers({'stub': retrieve}, queries,
                                                            warmup=2, repeats=3)
        self.assertEqual(len(calls), 8)
        self.assertEqual(len(raw), 6)
        self.assertAlmostEqual(summary[0]['mean_ms'], 2)
        self.assertAlmostEqual(summary[0]['p95_ms'], 2)
        self.assertEqual(predictions['stub'], {'1': ['A'], '2': ['A']})

    def test_gpu_hooks_and_empty_results(self):
        fences = []
        raw, _, predictions = benchmark_retrievers(
            {'stub': lambda text, k: []}, [{'id': '1', 'question': 'q'}],
            warmup=1, repeats=1, synchronize={'stub': lambda: fences.append(1)})
        self.assertEqual(len(fences), 4)
        self.assertEqual(raw[0]['n_returned'], 0)
        self.assertEqual(predictions['stub']['1'], [])

    def test_failures_are_not_dropped(self):
        with self.assertRaises(ValueError):
            benchmark_retrievers({'stub': lambda text, k: ['A', 'A']},
                                  [{'id': '1', 'question': 'q'}], warmup=0, repeats=1)

    def test_nonconstant_durations_and_quantiles(self):
        with patch('legal_rag_a.timing.time.perf_counter', side_effect=[0, .001, 1, 1.002, 2, 2.010]):
            raw, summary, _ = benchmark_retrievers(
                {'stub': lambda text, k: []}, [{'id': '1', 'question': 'q'}],
                warmup=0, repeats=3)
        self.assertAlmostEqual(summary[0]['mean_ms'], 13 / 3)
        self.assertAlmostEqual(summary[0]['median_ms'], 2)
        self.assertAlmostEqual(summary[0]['p95_ms'], 9.2)

    def test_unordered_results_are_rejected(self):
        for result in ({'A', 'B'}, {'A': 1}):
            with self.assertRaises(ValueError):
                benchmark_retrievers({'stub': lambda text, k: result},
                                      [{'id': '1', 'question': 'q'}], warmup=0, repeats=1)


if __name__ == '__main__':
    unittest.main()
