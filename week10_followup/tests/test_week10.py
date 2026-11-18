import unittest

from week10_followup.context_grid import pack_trace
from week10_followup.aggregate import (with_baseline_deltas, committed_team_rows,
                                       committed_vidore_query_rows)
from week10_followup.finder_query_expansion import TwoLineParser, normalize_queries, rrf
from week10_followup.report_finder import paired


class ContextGridTests(unittest.TestCase):
    def test_packing_accounts_for_seeds_neighbors_and_budget(self):
        trace = pack_trace(['seed1', 'neighbor', 'seed2', 'neighbor', 'empty'],
                           ['seed1', 'seed2'],
                           {'seed1': 3, 'neighbor': 4, 'seed2': 3,
                            'empty': 0, 'next': 2}, 8, 'next')
        self.assertEqual(trace['delivered_ids'], ['seed1', 'neighbor'])
        self.assertEqual(trace['delivered_seeds'], 1)
        self.assertEqual(trace['delivered_neighbors'], 1)
        self.assertEqual(trace['excluded_seeds_budget'], 1)
        self.assertEqual(trace['zero_text_pages'], 1)
        self.assertEqual(trace['unused_tokens'], 1)
        self.assertEqual(trace['next_ranked_seed_fits'], 0)

    def test_next_seed_already_added_as_neighbor_is_not_new_capacity(self):
        trace = pack_trace(['seed', 'next'], ['seed'], {'seed': 3, 'next': 2}, 8, 'next')
        self.assertEqual(trace['next_ranked_seed_fits'], 0)


class QueryExpansionTests(unittest.TestCase):
    def test_rrf_deduplicates_and_rewards_shared_hits(self):
        ids, scores = rrf([{'doc_ids': ['a', 'b']}, {'doc_ids': ['b', 'c']}])
        self.assertEqual(ids, ['b', 'a', 'c'])
        self.assertEqual(len(scores), len(set(ids)))
        self.assertGreater(scores[0], scores[1])

    def test_query_parser_limits_and_removes_original_duplicate(self):
        lines = TwoLineParser().parse('1. Revenue growth\n2) Income by year\n3. extra')
        self.assertEqual(normalize_queries(lines, 'Revenue growth'), ['Income by year'])

    def test_paired_report_keeps_question_alignment(self):
        baseline = {'a': {'hit': 0, 'recall': 0.0, 'mrr': 0.0, 'ndcg': 0.0},
                    'b': {'hit': 1, 'recall': 1.0, 'mrr': 1.0, 'ndcg': 1.0}}
        expanded = {'b': baseline['b'],
                    'a': {'hit': 1, 'recall': 1.0, 'mrr': 1.0, 'ndcg': 1.0}}
        row = paired(baseline, expanded, 'expanded minus baseline')
        self.assertEqual((row['hit_gained'], row['hit_lost']), (1, 0))
        self.assertEqual(row['delta_hit'], 0.5)


class AggregationTests(unittest.TestCase):
    def test_baseline_delta_requires_same_query_set(self):
        shared = {'dataset': 'FinDER', 'corpus': 'c', 'split': 'test',
                  'query_count': 2, 'scope': 'ranked', 'k': 5, 'budget': '',
                  'hit': 0.5, 'recall': 0.5, 'mrr': 0.5, 'ndcg': 0.5}
        rows = with_baseline_deltas([
            {**shared, 'method': 'single_query_bm25', 'query_set_fingerprint': 'a'},
            {**shared, 'method': 'multiquery_gpt', 'query_set_fingerprint': 'a', 'hit': 1.0},
            {**shared, 'method': 'multiquery_qwen', 'query_set_fingerprint': 'b', 'hit': 1.0},
        ])
        self.assertEqual(rows[1]['delta_hit'], 0.5)
        self.assertEqual(rows[2]['delta_hit'], '')

    def test_committed_team_rows_have_matching_primary_subset(self):
        primary, all_rows, hashes = committed_team_rows()
        self.assertEqual((len(primary), len(all_rows)), (6, 43))
        self.assertEqual(len(hashes), 2)

    def test_committed_vidore_query_rows_keep_backend_baselines(self):
        rows, source = committed_vidore_query_rows()
        self.assertEqual(len(rows), 64)
        self.assertEqual(len({(r['backend'], r['method'], r['k']) for r in rows}), 64)
        self.assertTrue(all(r['queries'] == 247 for r in rows))
        self.assertTrue(all(r['delta_hit'] == 0 and r['delta_ndcg'] == 0
                            for r in rows if r['method'] == 'original'))
        self.assertEqual(len(source['blob_sha256']), 4)


if __name__ == '__main__':
    unittest.main()
