"""Full-corpus FiQA retrieval with a fixed 50-dev/100-test query pilot."""
import csv
import gzip
import json
from pathlib import Path
import time
import numpy as np
from retrieval import ROOT, BM25, QwenEncoder, ranked, fuse, metrics, read, save, signature

def export(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def run():
    config = read(ROOT / 'config.json')
    with gzip.open(ROOT / 'data/fiqa_corpus.jsonl.gz', 'rt', encoding='utf-8') as f:
        docs = [json.loads(line) for line in f]
    queries = read(ROOT / 'data/fiqa_queries.json')
    ids = [str(d['_id']) for d in docs]
    texts = [(d['title']+'\n'+d['text']).strip() for d in docs]
    start = time.perf_counter()
    sparse = BM25(texts, config['bm25_k1'], config['bm25_b'])
    sparse_index_seconds = time.perf_counter()-start
    encoder = QwenEncoder(config)
    matrix, profile = encoder.matrix(texts, 'fiqa')
    output = ROOT / 'results/fiqa'
    output.mkdir(parents=True, exist_ok=True)
    ranks = {}
    query_times = []
    instruction = 'Given a financial question, retrieve relevant passages that answer the question'
    encoder.encode([queries['dev'][0]['question']], instruction)
    depth = config['candidate_depth']
    for split in ['dev', 'test']:
        ranks[split] = {}
        for q in queries[split]:
            start = time.perf_counter()
            sparse_ids = [ids[i] for i in ranked(sparse.scores(q['question']), depth)]
            sparse_ms = (time.perf_counter()-start)*1000
            encoder.sync()
            start = time.perf_counter()
            vector = encoder.encode([q['question']], instruction)[0]
            encoder.sync()
            dense_ids = [ids[i] for i in ranked(matrix @ vector, depth)]
            dense_ms = (time.perf_counter()-start)*1000
            ranks[split][q['id']] = {'BM25': sparse_ids, 'Qwen dense': dense_ids}
            query_times.append({'split': split, 'id': q['id'], 'bm25_ms': sparse_ms, 'qwen_encode_search_ms': dense_ms})
    tuning = []
    for weight in config['bm25_weight_grid']:
        scores = [metrics(fuse(ranks['dev'][q['id']]['BM25'], ranks['dev'][q['id']]['Qwen dense'], weight, config['rrf_constant']), q['qrels'], 10) for q in queries['dev']]
        tuning.append({'bm25_weight': weight, 'ndcg10': float(np.mean([r['ndcg'] for r in scores])), 'recall10': float(np.mean([r['recall'] for r in scores]))})
    winner = max(tuning, key=lambda r: (r['ndcg10'], r['recall10'], -abs(r['bm25_weight']-.5)))
    frozen = {'bm25_weight': winner['bm25_weight'], 'selection': 'Dev nDCG@10, then Recall@10, then weight closest to 0.5',
              'config_signature': signature(config), 'dev_query_ids': [q['id'] for q in queries['dev']]}
    selection = output / 'selection.json'
    if selection.exists() and read(selection) != frozen:
        raise ValueError('Frozen selection differs; use a new experiment output folder')
    save(selection, frozen)
    details, summaries = [], []
    for split in ['dev', 'test']:
        for q in queries[split]:
            rank = ranks[split][q['id']]
            rank['Hybrid equal'] = fuse(rank['BM25'], rank['Qwen dense'], .5, config['rrf_constant'])
            rank['Hybrid selected'] = fuse(rank['BM25'], rank['Qwen dense'], winner['bm25_weight'], config['rrf_constant'])
            for method, ranking in rank.items():
                for k in config['ks']:
                    details.append({'split': split, 'method': method, 'query_id': q['id'], 'k': k, **metrics(ranking, q['qrels'], k)})
        for method in ['BM25', 'Qwen dense', 'Hybrid equal', 'Hybrid selected']:
            for k in config['ks']:
                group = [r for r in details if r['split']==split and r['method']==method and r['k']==k]
                summaries.append({'split': split, 'method': method, 'k': k, 'queries': len(group),
                                  **{m: float(np.mean([r[m] for r in group])) for m in ['hit','recall','precision','mrr','ndcg','all_judged_relevant_retrieved']}})
    export(output / 'per_query.csv', details)
    export(output / 'comparison.csv', summaries)
    export(output / 'tuning_dev.csv', tuning)
    export(output / 'query_times.csv', query_times)
    save(output / 'rankings.json', ranks)
    save(output / 'run_manifest.json', {'config': config, 'corpus_count': len(docs), 'queries': {k:len(v) for k,v in queries.items()},
         'profile': profile, 'bm25_index_seconds': sparse_index_seconds, 'query_instruction': instruction,
         'selection': frozen, 'gpu': encoder.torch.cuda.get_device_name() if encoder.device=='cuda' else None,
         'torch_version': encoder.torch.__version__, 'code_sha256': signature({p: (ROOT/p).read_text(encoding='utf-8') for p in ['retrieval.py','run_fiqa.py']}),
         'metric_note': 'Document relevance; multiple relevant answers may be alternatives, not jointly required evidence.'})
    print('FiQA complete: selected BM25 weight', winner['bm25_weight'], flush=True)

if __name__ == '__main__':
    run()
