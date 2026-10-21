"""Small invariant checks for rank fusion and the unchanged binary-gold metrics."""
import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from person3_hyde.run_robustness import fuse,top
from person3_hyde.run_hyde_experiment import metrics_at_k,paired_test
class ProtocolTests(unittest.TestCase):
 def test_binary_gold_metrics(self):
  r=np.array([[2,1,0],[0,1,2]]);g=np.array([1,2]);m=metrics_at_k(r,g,2)
  self.assertEqual(m['hit_rate_at_k'],.5);self.assertEqual(m['precision_at_k'],.25);self.assertEqual(m['mrr_at_k'],.25)
 def test_zero_weight_ignored(self):
  a=np.array([[0,1,2]]);b=np.array([[2,1,0]])
  np.testing.assert_array_equal(fuse([a,b],[1,0],3),a)
 def test_paired_order(self):
  a=np.array([[0,1],[1,0]]);b=np.array([[1,0],[1,0]])
  r=paired_test(a,b,np.array([0,0]),1);self.assertEqual(r['candidate_only_hits'],1);self.assertEqual(r['baseline_only_hits'],0)
 def test_stable_ties(self):np.testing.assert_array_equal(top(np.array([[.5,.5,.1]]),3),[[0,1,2]])
if __name__=='__main__':unittest.main()
