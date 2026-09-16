"""Small algorithm checks; no model downloads or FinDER execution required."""
import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from d_threshold_mmr import refine, summarize, benchmark_latency


class RefinementTests(unittest.TestCase):
    def setUp(self):
        self.embeddings=np.array([[1.,0.],[.99,.01],[0.,1.]],dtype=np.float32)
        self.embeddings/=np.linalg.norm(self.embeddings,axis=1,keepdims=True)
        self.scores=self.embeddings @ np.array([1.,0.])
        self.candidates=np.array([0,1,2])

    def test_lambda_one_preserves_dense(self):
        s=dict(family='mmr',lambda_mult=1.,fetch_k=3)
        self.assertEqual(refine(self.scores,self.candidates,self.embeddings,s,2),[0,1])

    def test_diversity_changes_second_result(self):
        s=dict(family='mmr',lambda_mult=0.,fetch_k=3)
        self.assertEqual(refine(self.scores,self.candidates,self.embeddings,s,2),[0,2])

    def test_threshold_empty_and_denominators(self):
        s=dict(family='threshold',threshold=1.1)
        self.assertEqual(refine(self.scores,self.candidates,self.embeddings,s),[])
        r=summarize([[0],[]],[{0},{1}],5)
        self.assertEqual(r['precision_at_k'],.1)
        self.assertEqual(r['precision_returned'],.5)
        self.assertEqual(r['empty_rate'],.5)

    def test_timer_repeats_and_warmup(self):
        calls=[]
        def method(q):
            calls.append(q)
            return ['evidence']
        raw,summary=benchmark_latency({'one':method},['q1','q2'],repeats=2,warmup=1)
        self.assertEqual(len(calls),5)
        self.assertEqual(len(raw),4)
        self.assertEqual(summary.loc['one','measurements'],4)


if __name__=='__main__':
    unittest.main()
