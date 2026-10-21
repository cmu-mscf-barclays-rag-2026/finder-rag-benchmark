"""Deterministic evaluator controls, separate from actual BM25 pilot results."""
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import span_bench as s


class SpanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.docs, cls.annotations, cls.tokenizer = s.load()

    def fixture(self):
        doc = {'id': 'doc', 'text': 'abcdefghij klmnop'}
        docs = {'doc': doc}
        annotation = {'spans': [s.fragment(doc, 2, 8, 'gold')]}
        return doc, docs, annotation

    def test_half_open_adjacent_union(self):
        doc, docs, annotation = self.fixture()
        result = s.evaluate(annotation, [s.fragment(doc, 2, 5, 'a'), s.fragment(doc, 5, 8, 'b')], docs, self.tokenizer)
        self.assertEqual(result['all_spans_included'], 1)
        self.assertEqual(result['evidence_char_coverage'], 1)

    def test_gap_not_counted_as_full(self):
        doc, docs, annotation = self.fixture()
        result = s.evaluate(annotation, [s.fragment(doc, 2, 4, 'a'), s.fragment(doc, 5, 8, 'b')], docs, self.tokenizer)
        self.assertEqual(result['all_spans_included'], 0)
        self.assertAlmostEqual(result['evidence_char_coverage'], 5/6)

    def test_duplicate_does_not_inflate(self):
        doc, docs, annotation = self.fixture()
        f = s.fragment(doc, 2, 5, 'a')
        self.assertEqual(s.evaluate(annotation, [f], docs, self.tokenizer), s.evaluate(annotation, [f, f], docs, self.tokenizer))

    def test_wrong_document_is_not_hit(self):
        doc, docs, annotation = self.fixture()
        other = {'id': 'other', 'text': doc['text']}
        docs['other'] = other
        self.assertEqual(s.evaluate(annotation, [s.fragment(other, 0, 10, 'x')], docs, self.tokenizer)['all_spans_included'], 0)

    def test_mutated_text_rejected(self):
        doc, docs, _ = self.fixture()
        f = s.fragment(doc, 2, 5, 'x')
        f['text'] = 'wrong'
        with self.assertRaises(ValueError):
            s.validate_fragment(f, docs)

    def test_source_version_rejected(self):
        doc, docs, _ = self.fixture()
        f = s.fragment(doc, 2, 5, 'x')
        f['source_sha256'] = 'changed'
        with self.assertRaises(ValueError):
            s.validate_fragment(f, docs)

    def test_invalid_offsets_rejected(self):
        doc, docs, _ = self.fixture()
        for start, end in [(-1, 5), (5, 5), (0, 999), (0.0, 5)]:
            f = {'source_document': 'doc', 'start_char': start, 'end_char': end}
            with self.assertRaises(ValueError):
                s.validate_fragment(f, docs)

    def test_multiple_required_spans(self):
        doc, docs, annotation = self.fixture()
        annotation['spans'].append(s.fragment(doc, 11, 16, 'gold2'))
        result = s.evaluate(annotation, [s.fragment(doc, 2, 8, 'x')], docs, self.tokenizer)
        self.assertEqual(result['complete_span_recall'], .5)
        self.assertEqual(result['all_spans_included'], 0)

    def test_multidocument_support_mechanics(self):
        doc, docs, annotation = self.fixture()
        other = {'id': 'other', 'text': 'second fact'}
        docs['other'] = other
        f = s.fragment(other, 0, 6, 'second')
        annotation['spans'].append(f)
        self.assertEqual(s.evaluate(annotation, [*annotation['spans']], docs, self.tokenizer)['all_spans_included'], 1)

    def test_empty_retrieval(self):
        _, docs, annotation = self.fixture()
        self.assertTrue(all(v == 0 for v in s.evaluate(annotation, [], docs, self.tokenizer).values()))

    def test_empty_gold_rejected(self):
        with self.assertRaises(ValueError):
            s.evaluate({'spans': []}, [], {}, self.tokenizer)

    def test_unicode_offsets(self):
        doc = {'id': 'unicode', 'text': '法律 😀 café evidence'}
        f = s.fragment(doc, 3, 4, 'emoji')
        self.assertEqual(f['text'], '😀')
        s.validate_fragment(f, {'unicode': doc})

    def test_zero_budget(self):
        doc, _, _ = self.fixture()
        self.assertEqual(s.apply_budget([s.fragment(doc, 0, len(doc['text']), 'x')], self.tokenizer, 0), ([], 0))

    def test_counting_tokenizer_never_pads_or_truncates(self):
        self.assertEqual(len(self.tokenizer.encode('one', add_special_tokens=False).ids), 1)
        self.assertEqual(len(self.tokenizer.encode('one ' * 3000, add_special_tokens=False).ids), 3000)

    def test_budget_clips_coordinates(self):
        doc = {'id': 'doc', 'text': 'one two three four five'}
        out, count = s.apply_budget([s.fragment(doc, 0, len(doc['text']), 'x')], self.tokenizer, 2)
        self.assertEqual(count, 2)
        self.assertEqual(out[0]['text'], 'one two')
        s.validate_fragment(out[0], {'doc': doc})

    def test_grandchildren_cover_children(self):
        # All source units are tested, not only the 20 gold-bearing children.
        units = s.make_units(self.docs, 'grandchild', self.tokenizer)
        by_doc = {}
        for u in units:
            s.validate_fragment(u, self.docs)
            by_doc.setdefault(u['source_document'], []).append((u['start_char'], u['end_char']))
            self.assertLessEqual(len(self.tokenizer.encode(u['text'], add_special_tokens=False).ids), 128)
        for doc in self.docs.values():
            for child in doc['child_spans']:
                a, b = child['start_char'], child['end_char']
                self.assertEqual(s.covered_length(a, b, by_doc[doc['id']]), b-a)

    def test_annotations_are_exact_dev_only(self):
        self.assertEqual(len(self.annotations), 20)
        self.assertEqual(sum(len(a['spans']) for a in self.annotations), 27)
        for a in self.annotations:
            self.assertEqual(a['split'], 'dev')
            for f in a['spans']:
                s.validate_fragment(f, self.docs)

    def test_bm25_against_hand_calculation(self):
        b = s.BM25(['a a b', 'b c'])
        expected = math.log(1 + 1.5/1.5) * 2 * 2.5 / (2 + 1.5 * (.25 + .75 * 3/2.5))
        self.assertAlmostEqual(b.scores('a')[0], expected)
        self.assertEqual(b.scores('a')[1], 0)

    def test_bm25_stable_ties(self):
        self.assertEqual(s.BM25(['same', 'same']).rank('same', 2), [0, 1])

    def test_boundary_controls(self):
        controls = {r['case']: r for r in s.boundary_control(self.docs, self.annotations, self.tokenizer)}
        for name in ['one_piece', 'two_pieces', 'duplicate']:
            self.assertEqual(controls[name]['all_spans_included'], 1)
        self.assertEqual(controls['missing_tail']['all_spans_included'], 0)

    def test_fine_span_does_not_require_whole_child(self):
        doc, docs, annotation = self.fixture()
        returned = [s.fragment(doc, 2, 8, 'small')]
        whole = {'spans': [s.fragment(doc, 0, len(doc['text']), 'whole')]}
        self.assertEqual(s.evaluate(annotation, returned, docs, self.tokenizer)['all_spans_included'], 1)
        self.assertEqual(s.evaluate(whole, returned, docs, self.tokenizer)['all_spans_included'], 0)

    def test_no_human_approval_is_invented(self):
        self.assertEqual(sum(a['review_status'] == 'human_approved' for a in self.annotations), 0)


if __name__ == '__main__':
    unittest.main()
