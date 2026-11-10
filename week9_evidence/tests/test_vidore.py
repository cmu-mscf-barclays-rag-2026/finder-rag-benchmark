import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from week9_evidence.vidore import REVISION, digest, fetch_page, load_dataset, write_jsonl


class Response:
    def __init__(self, payload, revision=REVISION):
        self.headers = {'x-revision': revision}
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def read(self):
        return json.dumps(self.payload).encode()


class ViDoReTests(unittest.TestCase):
    def test_download_rejects_revision_drift_and_truncated_ocr(self):
        payload = {'rows': [{'row_idx': 0, 'row': {}, 'truncated_cells': []}],
                   'num_rows_total': 1, 'partial': False}
        with patch('week9_evidence.vidore.urlopen', return_value=Response(payload, 'different')):
            with self.assertRaisesRegex(ValueError, 'revision'):
                fetch_page('corpus', 0, REVISION)
        payload['rows'][0]['truncated_cells'] = ['markdown']
        with patch('week9_evidence.vidore.urlopen', return_value=Response(payload)):
            with self.assertRaisesRegex(ValueError, 'truncated'):
                fetch_page('corpus', 0, REVISION)

    def test_download_rejects_missing_row(self):
        payload = {'rows': [], 'num_rows_total': 1, 'partial': False}
        with patch('week9_evidence.vidore.urlopen', return_value=Response(payload)):
            with self.assertRaisesRegex(ValueError, 'omitted'):
                fetch_page('corpus', 0, REVISION)

    def test_loader_keeps_full_corpus_but_only_english_queries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_jsonl(root / 'corpus.jsonl', [
                {'corpus_id': 0, 'doc_id': 'bank', 'page_number_in_doc': 0, 'markdown': 'Revenue'},
                {'corpus_id': 1, 'doc_id': 'bank', 'page_number_in_doc': 1, 'markdown': ''}])
            write_jsonl(root / 'queries.jsonl', [
                {'query_id': 0, 'language': 'english', 'query': 'Revenue?'},
                {'query_id': 1, 'language': 'french', 'query': 'Revenus ?'}])
            write_jsonl(root / 'qrels.jsonl', [
                {'query_id': 0, 'corpus_id': 0, 'score': 1},
                {'query_id': 1, 'corpus_id': 0, 'score': 1}])
            (root / 'manifest.json').write_text(json.dumps({'sha256': {
                p.name: digest(p) for p in root.glob('*.jsonl')}}))
            pages, queries, qrels, _ = load_dataset(root)
            self.assertEqual(set(pages), {'0', '1'})
            self.assertEqual(set(queries), {'0'})
            self.assertEqual(qrels, {'0': {'0': 1}})
            with (root / 'corpus.jsonl').open('a') as out:
                out.write('\n')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                load_dataset(root)


if __name__ == '__main__':
    unittest.main()
