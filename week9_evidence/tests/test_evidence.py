import copy
import unittest

from week9_evidence.evidence import covered_facts, fact_metrics, marginal_gains, validate_annotations
from week9_evidence.run import expand, neighbors, pack
from week8_retrieval.beir_eval import rank_metrics


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.pages = {cid: {'id': cid, 'text': text, 'document': 'bank', 'position': i}
                      for i, (cid, text) in enumerate([('10', 'Revenue 10'), ('2', 'Revenue 20'),
                                                       ('7', 'Revenue 10'), ('8', 'Other text')])}
        def span(cid):
            text = self.pages[cid]['text']
            return {'page_id': cid, 'start': 0, 'end': len(text), 'quote': text}
        self.annotation = {'query_id': 'q', 'status': 'draft', 'question_type': 'multi_fact',
            'facts': [{'fact_id': 'old', 'requirement': 'Prior revenue',
                       'support_sets': [[span('10')], [span('7')]]},
                      {'fact_id': 'new', 'requirement': 'Current revenue',
                       'support_sets': [[span('2')]]}]}

    def test_alternatives_do_not_replace_complementary_facts(self):
        metric = fact_metrics(self.annotation, {'10': [(0, 10)], '7': [(0, 10)]}, 4)
        self.assertEqual(metric['fact_recall'], 0.5)
        self.assertEqual(metric['sufficient'], 0)
        self.assertEqual(rank_metrics(['10', '7'], {'10': 1, '7': 1, '2': 1}, 2)['hit'], 1)
        complete = fact_metrics(self.annotation, {'7': [(0, 10)], '2': [(0, 10)]}, 4)
        self.assertEqual(complete['sufficient'], 1)
        self.assertEqual(complete['facts_per_1k_tokens'], 500)

    def test_joint_support_requires_both_pages(self):
        annotation = copy.deepcopy(self.annotation)
        annotation['facts'] = [{'fact_id': 'comparison', 'requirement': 'Compare periods',
            'support_sets': [self.annotation['facts'][0]['support_sets'][0] +
                             self.annotation['facts'][1]['support_sets'][0]]}]
        self.assertFalse(covered_facts(annotation, {'10': [(0, 10)]}))
        self.assertEqual(covered_facts(annotation, {'10': [(0, 10)], '2': [(0, 10)]}), {'comparison'})

    def test_partial_span_not_credited_and_adjacent_intervals_merge(self):
        self.assertFalse(covered_facts(self.annotation, {'10': [(0, 9)]}))
        self.assertEqual(covered_facts(self.annotation, {'10': [(0, 5), (5, 10)]}), {'old'})
        self.assertFalse(covered_facts(self.annotation, {'10': [(0, 4), (5, 10)]}))

    def test_reject_drafts_by_default_and_validate_source_offsets(self):
        with self.assertRaisesRegex(ValueError, 'Draft'):
            validate_annotations([self.annotation], self.pages, {'q': {} })
        validate_annotations([self.annotation], self.pages, {'q': {}}, allow_draft=True)
        bad = copy.deepcopy(self.annotation)
        bad['facts'][0]['support_sets'][0][0]['quote'] = 'Revenue 99'
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            validate_annotations([bad], self.pages, {'q': {}}, allow_draft=True)

    def test_empty_support_and_unknown_query_rejected(self):
        bad = copy.deepcopy(self.annotation)
        bad['facts'][0]['support_sets'] = [[]]
        with self.assertRaisesRegex(ValueError, 'Empty support'):
            validate_annotations([bad], self.pages, {'q': {}}, allow_draft=True)
        with self.assertRaisesRegex(ValueError, 'unknown'):
            validate_annotations([self.annotation], self.pages, {}, allow_draft=True)

    def test_physical_neighbors_deduplicate_before_budget_accounting(self):
        candidates = expand(['2', '7'], self.pages, neighbors(self.pages), 2, 1, 1)
        self.assertEqual(candidates, ['2', '10', '7', '8'])
        counter = lambda text: len(text.split())
        costs = {cid: counter(p['text']) for cid, p in self.pages.items()}
        kept, tokens = pack(candidates + ['2'], self.pages, costs, 6, counter)
        self.assertEqual(kept, ['2', '10', '7'])
        self.assertEqual(tokens, 6)
        self.pages['8']['document'] = 'other_bank'
        self.assertNotIn('8', expand(['7'], self.pages, neighbors(self.pages), 1, 0, 1))

    def test_marginal_duplicate_fact_has_zero_gain(self):
        rows = marginal_gains(self.annotation, ['10', '7', '2', '2'], self.pages,
                              lambda text: len(text.split()))
        self.assertEqual([r['new_facts'] for r in rows], [1, 0, 1])
        self.assertEqual([r['sufficient'] for r in rows], [0, 0, 1])


if __name__ == '__main__':
    unittest.main()
