"""Download the complete ViDoRe Finance EN text view, without image binaries.

Uses the public dataset viewer; every response must have the pinned x-revision.
Rejects truncated cells, missing rows, and mixed revisions. Images/PDFs can be
retrieved separately for annotation review; expiring image URLs are not stored.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import ssl
import time
from urllib.parse import urlencode
from urllib.request import urlopen

DATASET = 'vidore/vidore_v3_finance_en'
REVISION = '7f432c176d82e27546501ad8064a713ac3071809'


def read_jsonl(path):
    with Path(path).open() as source:
        return [json.loads(line) for line in source if line.strip()]


def write_jsonl(path, rows):
    with Path(path).open('w') as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False) + '\n')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fetch_page(config, offset, revision):
    import certifi
    context = ssl.create_default_context(cafile=certifi.where())
    url = 'https://datasets-server.huggingface.co/rows?' + urlencode({
        'dataset': DATASET, 'config': config, 'split': 'test',
        'offset': offset, 'length': 100})
    for attempt in range(4):
        try:
            with urlopen(url, timeout=60, context=context) as response:
                if response.headers.get('x-revision') != revision:
                    raise ValueError('Dataset viewer revision differs from pinned revision')
                data = json.load(response)
            break
        except (OSError, TimeoutError):
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    if data.get('partial') or any(r['truncated_cells'] for r in data['rows']):
        raise ValueError('Dataset viewer returned partial/truncated data')
    expected = list(range(offset, min(offset + 100, data['num_rows_total'])))
    if [r['row_idx'] for r in data['rows']] != expected:
        raise ValueError('Dataset viewer omitted or reordered rows')
    return data


def download(output, revision=REVISION):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('Output must be new or empty')
    counts = {}
    for config in ('corpus', 'queries', 'qrels', 'documents_metadata'):
        first = fetch_page(config, 0, revision)
        total = first['num_rows_total']
        with ThreadPoolExecutor(max_workers=4) as pool:
            rest = list(pool.map(lambda offset: fetch_page(config, offset, revision),
                                 range(100, total, 100)))
        if any(page['num_rows_total'] != total for page in rest):
            raise ValueError('Dataset changed during download')
        rows = [entry['row'] for page in [first, *rest] for entry in page['rows']]
        if config == 'corpus':
            for row in rows:
                row.pop('image', None)
        write_jsonl(output / f'{config}.jsonl', rows)
        counts[config] = len(rows)
        print(f'{config}: {len(rows)} rows', flush=True)
    manifest = {'dataset': DATASET, 'revision': revision, 'split': 'test',
                'transport': 'datasets-server rows; checked x-revision on every response',
                'representation': 'publisher-provided OCR markdown; no image embeddings',
                'counts': counts, 'sha256': {p.name: digest(p) for p in sorted(output.glob('*.jsonl'))}}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


def load_dataset(folder):
    folder = Path(folder)
    manifest = json.loads((folder / 'manifest.json').read_text())
    for name, checksum in manifest['sha256'].items():
        if digest(folder / name) != checksum:
            raise ValueError(f'Dataset checksum mismatch: {name}')
    pages = {}
    positions = set()
    for r in read_jsonl(folder / 'corpus.jsonl'):
        cid = str(r['corpus_id'])
        position = (r['doc_id'], r['page_number_in_doc'])
        if cid in pages or position in positions:
            raise ValueError('Duplicate page ID or document/page position')
        pages[cid] = {'id': cid, 'text': r['markdown'], 'document': r['doc_id'],
                      'position': r['page_number_in_doc']}
        positions.add(position)
    queries = {}
    all_ids = set()
    for r in read_jsonl(folder / 'queries.jsonl'):
        qid = str(r['query_id'])
        if qid in all_ids:
            raise ValueError('Duplicate query ID')
        all_ids.add(qid)
        if r['language'] == 'english':
            queries[qid] = r
    qrels = {qid: {} for qid in queries}
    for r in read_jsonl(folder / 'qrels.jsonl'):
        qid, cid = str(r['query_id']), str(r['corpus_id'])
        if qid not in all_ids or cid not in pages:
            raise ValueError('Dangling qrel')
        if qid in queries and r['score'] > 0:
            if cid in qrels[qid]:
                raise ValueError('Duplicate qrel')
            qrels[qid][cid] = r['score']
    if not queries or any(not labels for labels in qrels.values()):
        raise ValueError('Every English query needs positive qrels')
    return pages, queries, qrels, manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--revision', default=REVISION)
    args = parser.parse_args()
    download(args.output, args.revision)
