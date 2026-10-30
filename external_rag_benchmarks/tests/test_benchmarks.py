"""Small checks of scoring, fusion, chunk coordinates and answer handling."""
import math
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retrieval import BM25, ranked, fuse, metrics, read, ROOT
from run_longbench import counter, chunks, pack, parse_letter, public_question

class Tests(unittest.TestCase):
    def test_hit_differs_from_recall(self):
        m = metrics(['a'], {'a':1,'b':1}, 5)
        self.assertEqual(m['hit'], 1)
        self.assertEqual(m['recall'], .5)
        self.assertEqual(m['precision'], .2)
        self.assertEqual(m['all_judged_relevant_retrieved'], 0)

    def test_ndcg_and_mrr(self):
        m = metrics(['x','b'], {'a':1,'b':1}, 2)
        self.assertAlmostEqual(m['ndcg'], (1/math.log2(3))/(1+1/math.log2(3)))
        self.assertEqual(m['mrr'], .5)

    def test_graded_gain(self):
        self.assertAlmostEqual(metrics(['a','b'], {'a':2,'b':1}, 2)['ndcg'],1)

    def test_duplicates_rejected(self):
        with self.assertRaises(ValueError):
            metrics(['a','a'], {'a':1}, 2)

    def test_missing_relevance_rejected(self):
        with self.assertRaises(ValueError):
            metrics(['a'], {}, 5)

    def test_fusion_endpoints(self):
        self.assertEqual(fuse(['a','b'],['c','a'],1), ['a','b'])
        self.assertEqual(fuse(['a','b'],['c','a'],0), ['c','a'])

    def test_fusion_rewards_agreement(self):
        self.assertEqual(fuse(['a','b'],['c','b'])[0],'b')

    def test_bm25_numeric(self):
        b = BM25(['a a b','b c'])
        expected = math.log(2)*2*2.5/(2+1.5*(.25+.75*3/2.5))
        self.assertAlmostEqual(b.scores('a')[0], expected)
        self.assertEqual(b.scores('a')[1], 0)

    def test_ranking_stable(self):
        self.assertEqual(ranked([1,1,0],2),[0,1])

    def test_counting_is_unlimited(self):
        tok = counter()
        self.assertEqual(len(tok.encode('one '*3000,add_special_tokens=False).ids),3000)
        self.assertEqual(len(tok.encode('one',add_special_tokens=False).ids),1)

    def test_chunk_offsets_and_budget(self):
        tok = counter()
        text = ('Text about investments and unlikely terminology. '*100)
        units = chunks(text,tok,64,16)
        covered = set()
        for u in units:
            self.assertEqual(u['text'],text[u['start_char']:u['end_char']])
            self.assertLessEqual(len(tok.encode(u['text'],add_special_tokens=False).ids),64)
            covered.update(range(u['start_char'],u['end_char']))
        self.assertEqual(covered,set(range(len(text))))
        chosen, used = pack(units,tok,100)
        self.assertLessEqual(used,100)
        self.assertEqual(used,sum(len(tok.encode(u['text'],add_special_tokens=False).ids) for u in chosen))
        for u in chosen:
            self.assertEqual(u['text'],text[u['start_char']:u['end_char']])

    def test_strict_answer_parser(self):
        self.assertEqual(parse_letter('B'),'B')
        self.assertEqual(parse_letter('Answer: (C).'),'C')
        self.assertIsNone(parse_letter('A or B'))
        self.assertIsNone(parse_letter('The answer might be C because'))

    def test_no_gold_in_retrieval_query(self):
        row = {'question':'Question?', 'answer':'SECRET', **{'choice_'+a:'Option '+a for a in 'ABCD'}}
        self.assertNotIn('SECRET',public_question(row))

    def test_fiqa_split_and_qrels(self):
        q = read(ROOT/'data/fiqa_queries.json')
        self.assertEqual(len(q['dev']),50)
        self.assertEqual(len(q['test']),100)
        self.assertFalse({r['id'] for r in q['dev']} & {r['id'] for r in q['test']})

if __name__=='__main__':
    unittest.main()
