"""Hand-calculated protocol edge cases; run with unittest."""
import unittest
import numpy as np
from person3_hyde.run_robustness import fuse,top
from person3_hyde.run_hyde_experiment import metrics_at_k
class ProtocolTests(unittest.TestCase):
    def test_one_gold(self):
        r=np.array([[0,1,2],[2,1,0]]); g=np.array([0,0]); m=metrics_at_k(r,g,2)
        self.assertEqual(m['precision_at_k'],.25)
        self.assertEqual(m['recall_at_k'],.5)
        self.assertEqual(m['mrr_at_k'],.5)
        self.assertEqual(m['ndcg_at_k'],.5)
    def test_zero_weight_cannot_change_results(self):
        a=np.array([[1,2,0]]); b=np.array([[0,2,1]])
        np.testing.assert_array_equal(fuse([a,b],[1.,0.],3),a)
    def test_rrf_tie_and_duplicate(self):
        a=np.array([[0,1,2]]); b=np.array([[1,0,2]])
        np.testing.assert_array_equal(fuse([a,b],[.5,.5],3),np.array([[0,1,2]]))
    def test_stable_score_ties(self):
        np.testing.assert_array_equal(top(np.array([[1.,1.,0.]]),3),np.array([[0,1,2]]))
if __name__=='__main__': unittest.main()
