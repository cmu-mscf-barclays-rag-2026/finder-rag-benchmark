"""Hand-calculated hierarchy and context-budget tests; no model retrieval."""
import unittest
from types import SimpleNamespace

from hierarchy.core import build_hierarchy, suffix_prefix_length
from hierarchy.handoff import assemble_context, evaluate_run


class CharacterTokenizer:
    """One character per token, to make budget expectations exact in tests."""
    def encode(self, text, add_special_tokens=False):
        return SimpleNamespace(ids=list(range(len(text))), offsets=[(i, i + 1) for i in range(len(text))])


def row(pid, text, title='Heading'):
    return {'id': pid, 'text': text, 'title': title, 'footnotes': None}


class HierarchyTests(unittest.TestCase):
    def setUp(self):
        children, parents, self.mapping, _, _ = build_hierarchy([
            row('1.1-c1-s2', 'BBB'), row('1.1-c1-s1', 'AAA'), row('1.1-c2-s1', 'CCC')])
        self.children = {r['id']: r for r in children}
        self.parents = {r['id']: r for r in parents}
        self.tokenizer = CharacterTokenizer()

    def test_numeric_order_and_no_cross_parent_merge(self):
        source = [row(f'1.1-c1-s{i}', str(i)) for i in [10, 2, 1, 4, 3, 6, 5, 8, 7, 9]]
        _, parents, _, audit, _ = build_hierarchy(source)
        self.assertEqual(parents[0]['child_ids'][-1], '1.1-c1-s10')
        self.assertFalse(audit[0]['source_order_was_numeric'])
        self.assertEqual(len(self.parents), 2)

    def test_unicode_spans_round_trip(self):
        children, parents, mapping, _, _ = build_hierarchy([
            row('1.1-c1-s1', 'Law ⚖️\n'), row('1.1-c1-s2', 'Café 中文')])
        originals = {r['id']: r['text'] for r in children}
        for span in mapping:
            self.assertEqual(parents[0]['text'][span['start_char']:span['end_char']], originals[span['child_id']])

    def test_gaps_duplicates_and_title_conflicts_fail(self):
        variants = [[row('1.1-c1-s2', 'A')],
                    [row('1.1-c1-s1', 'A'), row('1.1-c1-s1', 'B')],
                    [row('1.1-c1-s1', 'A'), row('1.1-c1-s2', 'B', 'Other')]]
        for records in variants:
            with self.assertRaises(ValueError):
                build_hierarchy(records)

    def test_overlap_audited_without_deletion(self):
        self.assertEqual(suffix_prefix_length('abcxyz', 'xyzdef'), 3)
        _, parents, _, _, boundaries = build_hierarchy([
            row('1.1-c1-s1', 'one two three four five six'),
            row('1.1-c1-s2', 'two three four five six seven')])
        self.assertTrue(boundaries[0]['review_flag'])
        self.assertIn('six\n\ntwo', parents[0]['text'])

    def test_expansion_deduplicates_without_backfill(self):
        ctx = assemble_context(['1.1-c1-s2', '1.1-c1-s1', '1.1-c2-s1'], 'child_to_parent',
                               self.children, self.parents, self.tokenizer, k=2)
        self.assertEqual(ctx['unit_ids'], ['1.1-c1'])
        self.assertEqual(ctx['text'], 'AAA\n\nBBB')

    def test_gold_tail_is_lost_under_budget(self):
        queries = [{'id': 'q', 'gold_child_ids': ['1.1-c1-s2']}]
        rows, _ = evaluate_run(queries, {'q': ['1.1-c1']}, self.children, self.parents,
                              self.tokenizer, mode='parent', ks=(1,), token_budget=6)
        r = rows[0]
        self.assertEqual(r['ranked_unit_hit_at_k'], 1)
        self.assertEqual(r['gold_child_hit_in_context'], 0)
        self.assertAlmostEqual(r['gold_body_token_coverage'], 1 / 3)
        self.assertEqual(r['returned_content_tokens'], 6)

    def test_sibling_discovery_differs_from_direct_gold_hit(self):
        queries = [{'id': 'q', 'gold_child_ids': ['1.1-c1-s2']}]
        for mode, expected in [('child', 0), ('child_to_parent', 1)]:
            rows, _ = evaluate_run(queries, {'q': ['1.1-c1-s1']}, self.children, self.parents,
                                  self.tokenizer, mode=mode, ks=(1,))
            self.assertEqual(rows[0]['ranked_unit_hit_at_k'], 0)
            self.assertEqual(rows[0]['gold_child_hit_in_context'], expected)

    def test_empty_and_zero_budget(self):
        ctx = assemble_context([], 'parent', self.children, self.parents, self.tokenizer, 1)
        self.assertEqual(ctx['text'], '')
        ctx = assemble_context(['1.1-c1'], 'parent', self.children, self.parents, self.tokenizer, 1, 0)
        self.assertEqual(ctx['fully_included_child_ids'], [])

    def test_bad_rankings_are_rejected(self):
        for ranking in (['1.1-c1-s1', '1.1-c1-s1'], ['unknown'], {'1.1-c1-s1'}):
            with self.assertRaises(ValueError):
                assemble_context(ranking, 'child', self.children, self.parents, self.tokenizer, 1)

    def test_multiple_gold_children_are_not_double_counted(self):
        rows, _ = evaluate_run([{'id': 'q', 'gold_child_ids': ['1.1-c1-s1', '1.1-c2-s1']}],
            {'q': ['1.1-c1']}, self.children, self.parents, self.tokenizer, mode='parent', ks=(1,))
        self.assertEqual(rows[0]['gold_child_coverage_in_context'], .5)
        self.assertEqual(rows[0]['all_gold_children_in_context'], 0)


if __name__ == '__main__':
    unittest.main()
