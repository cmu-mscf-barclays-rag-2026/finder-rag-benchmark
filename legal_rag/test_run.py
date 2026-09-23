import math
import unittest
from legal_rag.run import coverage, metrics, split_queries, fuse, make_chunks


class Words:
    def num_special_tokens_to_add(self, pair):
        return 2
    def __call__(self, text, add_special_tokens=True, **kwargs):
        import re
        offsets = [(m.start(), m.end()) for m in re.finditer(r'\S+', text)]
        return {'offset_mapping': offsets, 'input_ids': list(range(len(offsets) + (2 if add_special_tokens else 0)))}


class EvaluationTests(unittest.TestCase):
    def test_union_not_sum(self):
        self.assertEqual(coverage([(0, 5), (3, 8), (3, 8)], 10), .8)
        with self.assertRaises(ValueError):
            coverage([(0, 11)], 10)

    def test_repeated_parent_earns_credit_once(self):
        hits = [{'parent_id': 'wrong', 'start': 0, 'end': 10},
                {'parent_id': 'gold', 'start': 0, 'end': 5},
                {'parent_id': 'gold', 'start': 3, 'end': 8}]
        m = metrics(hits, 'gold', 10, 3)
        self.assertEqual(m['recall'], 1)
        self.assertEqual(m['mrr'], .5)
        self.assertAlmostEqual(m['ndcg'], 1 / math.log2(3))
        self.assertEqual(m['evidence_coverage'], .8)
        self.assertTrue(all(v == 0 for v in metrics(hits, 'gold', 10, 1).values()))

    def test_identical_evidence_cannot_cross_splits(self):
        corpus = [{'id': str(i), 'text': str(i // 2)} for i in range(20)]
        qa = [{'id': str(i), 'relevant_passage_id': str(i)} for i in range(20)]
        split = split_queries(qa, corpus)
        self.assertEqual(split, split_queries(list(reversed(qa)), corpus))
        self.assertFalse(set(split['dev']) & set(split['test']))
        self.assertEqual(set(split['dev'] + split['test']), {q['id'] for q in qa})
        self.assertFalse({int(i)//2 for i in split['dev']} & {int(i)//2 for i in split['test']})

    def test_chunks_preserve_offsets_and_cover_parent(self):
        text = '  one two three four five six seven eight nine  '
        chunks = make_chunks([{'id': 'p', 'text': text}], Words(), 6)
        self.assertTrue(all(c['text'] == text[c['start']:c['end']] for c in chunks))
        self.assertEqual(coverage([(c['start'], c['end']) for c in chunks], len(text)), 1)
        self.assertTrue(all(len(Words()(c['text'])['input_ids']) <= 6 for c in chunks))

    def test_boundary_retokenization_shortens_without_losing_text(self):
        class BoundaryTokenizer(Words):
            def __call__(self, text, add_special_tokens=True, **kwargs):
                result = super().__call__(text, add_special_tokens=add_special_tokens, **kwargs)
                if add_special_tokens and not text.startswith('one'):
                    result['input_ids'].append(99)
                return result
        text = 'one two three four five six seven eight nine'
        tokenizer = BoundaryTokenizer()
        chunks = make_chunks([{'id': 'p', 'text': text}], tokenizer, 6)
        self.assertEqual(coverage([(c['start'], c['end']) for c in chunks], len(text)), 1)
        self.assertTrue(all(len(tokenizer(c['text'])['input_ids']) <= 6 for c in chunks))
        self.assertEqual(len(chunks), 3)

    def test_rrf_uses_rank_not_score_and_zero_endpoint(self):
        bm = [{'doc_id': 'a', 'score': 10000}, {'doc_id': 'b', 'score': 1}]
        dense = [{'doc_id': 'b', 'score': .9}, {'doc_id': 'c', 'score': .8}]
        self.assertEqual(fuse(bm, dense, .5)[0]['doc_id'], 'b')
        self.assertEqual([h['doc_id'] for h in fuse(bm, dense, 1)], ['a', 'b'])


class AnswerTests(unittest.TestCase):
    def test_missing_scores_rejected(self):
        from legal_rag.evaluate_answers import summarize
        row = {'method': 'bm25', 'query_id': '1', 'generated_answer': 'answer',
               'reviewer': 'human', 'correctness': '', 'groundedness': '1', 'abstained': '0'}
        with self.assertRaises(ValueError):
            summarize([row], {('bm25', '1')})
        row['correctness'] = '0'
        result = summarize([row], {('bm25', '1')})[0]
        self.assertEqual(result['correct_and_grounded'], 0)
        self.assertEqual(result['groundedness'], 1)
        with self.assertRaises(ValueError):
            summarize([row, row], {('bm25', '1')})


if __name__ == '__main__':
    unittest.main()
