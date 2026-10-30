import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from week8_retrieval.beir_eval import rank_metrics
from week8_retrieval.longbench_v2 import prepare, score
from week8_retrieval.run import mmr_ranking, sibling_candidates


class Week8Tests(unittest.TestCase):
    def test_mmr_changes_redundant_second_choice(self):
        vectors = {'a': np.array([1.0, 0.0]), 'b': np.array([0.99, 0.141067]),
                   'c': np.array([0.8, 0.6])}
        query = np.array([0.95, 0.31])
        self.assertEqual(mmr_ranking(['a', 'b', 'c'], vectors, query, 1.0)[:2], ['b', 'a'])
        self.assertEqual(mmr_ranking(['a', 'b', 'c'], vectors, query, 0.5)[:2], ['b', 'c'])

    def test_directional_siblings_keep_seed_order_and_deduplicate(self):
        children = {cid: {'parent_id': 'p'} for cid in ('a', 'b', 'c', 'd')}
        parents = {'p': {'child_spans': [{'child_id': cid} for cid in ('a', 'b', 'c', 'd')]}}
        self.assertEqual(sibling_candidates(['b', 'c'], children, parents, 2, 0, 2), ['b', 'c', 'd'])
        self.assertEqual(sibling_candidates(['b'], children, parents, 1, 2, 1), ['b', 'a', 'c'])

    def test_multi_relevance_distinguishes_hit_and_recall(self):
        result = rank_metrics(['a', 'x', 'b'], {'a': 1, 'b': 1}, 1)
        self.assertEqual(result['hit'], 1)
        self.assertEqual(result['recall'], 0.5)
        self.assertEqual(result['all_judged_relevant'], 0)

    def test_longbench_adapter_and_answer_score(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = {'_id': 'q1', 'question': 'Which?', 'context': 'one two three four five',
                   'choice_A': 'a', 'choice_B': 'b', 'choice_C': 'c', 'choice_D': 'd', 'answer': 'B'}
            source = root / 'data.json'
            source.write_text(json.dumps([row]))
            prepared = root / 'prepared'
            prepare(source, prepared, chunk_words=3, overlap=1)
            query = json.loads((prepared / 'queries.jsonl').read_text())
            self.assertNotIn('answer', query)
            self.assertEqual(json.loads((prepared / 'answers.jsonl').read_text())['answer'], 'B')
            chunks = [json.loads(line) for line in (prepared / 'chunks.jsonl').read_text().splitlines()]
            self.assertEqual([r['text'] for r in chunks], ['one two three', 'three four five'])
            predictions = root / 'predictions.jsonl'
            predictions.write_text(json.dumps({'query_id': 'q1', 'answer': 'B'}) + '\n')
            self.assertEqual(score(prepared, predictions)['accuracy'], 1.0)


if __name__ == '__main__':
    unittest.main()
