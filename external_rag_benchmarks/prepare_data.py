"""Download pinned public data and select queries before running retrieval."""
from collections import Counter
import csv
import gzip
import hashlib
import json
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parent
REPOS = {
    'fiqa': ('BeIR/fiqa', '979c07a7cb5ccc6ca009792241fa1250b98055dd'),
    'qrels': ('BeIR/fiqa-qrels', '252958f2d646e22cab6d0c72dd3f0d5de6d0655a'),
    'longbench': ('zai-org/LongBench-v2', '2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9'),
}

def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def download(key, filename):
    repo, revision = REPOS[key]
    path = ROOT / '.cache/downloads' / key / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        url = f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{filename}'
        with requests.get(url, stream=True, timeout=(30, 180)) as response:
            response.raise_for_status()
            temp = path.with_suffix(path.suffix + '.part')
            with temp.open('wb') as handle:
                for chunk in response.iter_content(1024 * 1024):
                    handle.write(chunk)
            temp.replace(path)
        print(f'Downloaded {key}/{filename}: {path.stat().st_size:,} bytes', flush=True)
    return path

def write_gzip(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows).encode('utf-8')
    path.write_bytes(gzip.compress(payload, mtime=0))

def stable_id(value):
    return hashlib.sha256(('external-rag-pilot-v1:' + value).encode()).hexdigest()

def prepare():
    import pyarrow.parquet as pq
    import ijson
    corpus = pq.read_table(download('fiqa', 'corpus/corpus-00000-of-00001.parquet')).to_pylist()
    queries = {str(q['_id']): q['text'] for q in pq.read_table(download('fiqa', 'queries/queries-00000-of-00001.parquet')).to_pylist()}
    write_gzip(ROOT / 'data/fiqa_corpus.jsonl.gz', corpus)
    chosen, full_counts = {}, {}
    for split, count in [('dev', 50), ('test', 100)]:
        qrels = {}
        with download('qrels', split + '.tsv').open(encoding='utf-8') as handle:
            for row in csv.DictReader(handle, delimiter='\t'):
                qrels.setdefault(str(row['query-id']), {})[str(row['corpus-id'])] = int(row['score'])
        full_counts[split] = len(qrels)
        chosen[split] = [{'id': q, 'question': queries[q], 'qrels': qrels[q]} for q in sorted(qrels, key=stable_id)[:count]]
    assert not ({q['id'] for q in chosen['dev']} & {q['id'] for q in chosen['test']})
    known = {str(d['_id']) for d in corpus}
    assert all(set(q['qrels']) <= known for rows in chosen.values() for q in rows)
    save(ROOT / 'data/fiqa_queries.json', chosen)
    eligible, profiles = [], []
    with download('longbench', 'data.json').open('rb') as handle:
        for row in ijson.items(handle, 'item'):
            profiles.append({'id': row['_id'], 'domain': row['domain'], 'length': row['length']})
            if row['length'] == 'short' and row['domain'] in ['Single-Document QA', 'Multi-Document QA']:
                eligible.append(row)
    if not eligible:
        raise ValueError('No eligible QA rows; observed domains: ' + str(Counter(p['domain'] for p in profiles)))
    by_domain = {}
    for row in eligible:
        by_domain.setdefault(row['domain'], []).append(row)
    selected = {'dev': [], 'test': []}
    for domain in sorted(by_domain):
        rows = sorted(by_domain[domain], key=lambda r: stable_id(r['_id']))[:10]
        selected['dev'].extend(rows[:3])
        selected['test'].extend(rows[3:])
    write_gzip(ROOT / 'data/longbench_pilot.jsonl.gz', [{**row, 'pilot_split': split} for split, rows in selected.items() for row in rows])
    save(ROOT / 'data/selection_manifest.json', {
        'dataset_revisions': REPOS,
        'selection_rule': 'SHA256 external-rag-pilot-v1 plus ID, fixed before scoring; full FiQA corpus',
        'fiqa': {'corpus': len(corpus), 'available_queries': full_counts, 'selected_queries': {k: len(v) for k,v in chosen.items()},
                 'multi_relevant_queries': {k: sum(len(q['qrels']) > 1 for q in v) for k,v in chosen.items()}},
        'longbench': {'official_rows': len(profiles), 'domain_counts': dict(Counter(p['domain'] for p in profiles)),
                      'eligible_short_qa': len(eligible), 'selected_queries': {k: len(v) for k,v in selected.items()},
                      'warning': 'New project pilot split; official train label is not a provided dev/test partition'},
        'files_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / 'data').glob('*') if p.is_file() and p.name!='selection_manifest.json'}
    })
    print('Prepared FiQA and LongBench selections.', flush=True)

if __name__ == '__main__':
    prepare()
