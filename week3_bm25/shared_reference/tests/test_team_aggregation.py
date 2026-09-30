"""Guard common-test comparability without running retrieval models."""
from pathlib import Path
import sys
import unittest
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from aggregate_team_metrics import validate


class SharedSubmissionTests(unittest.TestCase):
    def rows(self):
        return pd.concat([pd.read_csv(ROOT/'team_metrics'/name) for name in
                          ('a_dense.csv','b_bm25.csv','c_hybrid_rrf.csv','d_threshold_mmr.csv')],ignore_index=True)

    def test_all_four_shared_submissions_and_missing_metrics(self):
        frame=self.rows()
        validate(frame)
        self.assertEqual(set(frame.owner),{'A','B','C','D'})
        self.assertEqual(set(frame.n_queries),{4575})
        self.assertTrue(frame.loc[frame.owner=='A','mrr_at_k'].isna().all())
        self.assertTrue(frame.loc[frame.owner=='A','ndcg_at_k'].isna().all())

    def test_rejects_old_b_query_count_even_if_identifier_is_relabeled(self):
        frame=self.rows()
        frame.loc[frame.owner=='B','n_queries']=4563
        with self.assertRaisesRegex(ValueError,'same positive query count'):
            validate(frame)

    def test_rejects_mixed_split(self):
        frame=self.rows()
        frame.loc[frame.owner=='B','split_id']='seeded-random-v1'
        with self.assertRaisesRegex(ValueError,'Mixed benchmark values in split_id'):
            validate(frame)


if __name__=='__main__': unittest.main()
