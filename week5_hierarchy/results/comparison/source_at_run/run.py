"""Legal RAG retrieval experiment. Run from repository root; see README.md."""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'person2_bm25'))
from finder_bm25.bm25 import BM25Retriever

DATA_REVISION = 'db0b31dc6d195ce9916897e1ac5e4e6209736c8a'
MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
MODEL_REVISION = '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
HASHES = {'corpus.jsonl': '3a3565bc5429f6cead90548e81f87352b449927e4be1cdd804c28d766bb9c246',
          'qa.jsonl': 'e3b869a4e293d081ec5f5b39c2058c8d27b36f611aa9f8275eb1877a7c8b38b0'}
KS = (1, 3, 5, 10, 20)
METRICS = ('recall', 'hit_rate', 'all_evidence_hit', 'evidence_coverage', 'mrr', 'ndcg')


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')


def write_jsonl(path, rows):
    with Path(path).open('w') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def write_csv(path, rows):
    if rows:
        with Path(path).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def split_queries(qa, corpus):
    """Group identical supporting TEXT as well as identical IDs; no label-driven tuning."""
    text = {p['id']: p['text'] for p in corpus}
    groups = {}
    for q in qa:
        key = hashlib.sha256(text[q['relevant_passage_id']].encode()).hexdigest()
        groups.setdefault(key, []).append(q['id'])
    keys = sorted(groups)
    random.Random(42).shuffle(keys)
    dev = set()
    for key in keys:
        if len(dev) >= math.ceil(len(qa) * .2):
            break
        dev.update(groups[key])
    return {'dev': sorted(dev), 'test': sorted(q['id'] for q in qa if q['id'] not in dev)}


def make_chunks(corpus, tokenizer, size):
    """Exact character slices; size counts model tokens including special tokens."""
    chunks = []
    budget = size - tokenizer.num_special_tokens_to_add(False) if size else 0
    if size and budget < 2:
        raise ValueError('Chunk size too small')
    overlap = int(budget * .2) if size else 0
    for p in sorted(corpus, key=lambda p: p['id']):
        text = p['text']
        offsets = tokenizer(text, add_special_tokens=False, truncation=False,
                            return_offsets_mapping=True)['offset_mapping'] if size else []
        i = 0
        while True:
            start = offsets[i][0] if i else 0
            j = min(i + budget, len(offsets)) if size else len(offsets)
            while True:
                end = offsets[j][0] if j < len(offsets) else len(text)
                piece = text[start:end]
                # Subword boundaries may tokenize differently when sliced. Shorten
                # the span until its actual encoding fits, preserving exact offsets.
                if not size or len(tokenizer(piece, truncation=False)['input_ids']) <= size:
                    break
                j -= 1
                if j <= i:
                    raise ValueError(f'Cannot fit token span: {p["id"]}:{start}')
            chunks.append({'doc_id': f'{p["id"]}:{start}:{end}', 'parent_id': p['id'],
                           'start': start, 'end': end, 'text': piece})
            if end == len(text):
                break
            i = max(i + 1, j - overlap)
    return chunks


def coverage(intervals, length):
    end = total = 0
    for a, b in sorted(intervals):
        if not 0 <= a <= b <= length:
            raise ValueError('Invalid evidence offset')
        total += max(0, b - max(a, end))
        end = max(end, b)
    return total / length if length else 0.0


def metrics(hits, gold, text_length, k):
    """K counts returned chunks; repeated gold children earn rank credit only once."""
    relevant = [(i, h) for i, h in enumerate(hits[:k], 1) if h['parent_id'] == gold]
    rank = relevant[0][0] if relevant else None
    hit = float(rank is not None)
    return {'recall': hit, 'hit_rate': hit, 'all_evidence_hit': hit,
            'evidence_coverage': coverage([(h['start'], h['end']) for _, h in relevant], text_length),
            'mrr': 1 / rank if rank else 0.0,
            'ndcg': 1 / math.log2(rank + 1) if rank else 0.0}


def fuse(bm, dense, weight, c=60):
    scores = {}
    for ranking, w in ((bm, weight), (dense, 1 - weight)):
        if not w:
            continue
        for rank, h in enumerate(ranking, 1):
            scores[h['doc_id']] = scores.get(h['doc_id'], 0) + w / (c + rank)
    return [{'doc_id': i, 'score': s} for i, s in sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:100]]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model-cache', type=Path)
    p.add_argument('--offline', action='store_true')
    p.add_argument('--embedding-cache', type=Path)
    args = p.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        p.error('Output must be empty: preserve prior experiments')
    args.output.mkdir(parents=True, exist_ok=True)
    for filename, digest in HASHES.items():
        if hashlib.sha256((args.data / filename).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Unexpected dataset checksum: {filename}')
    corpus, qa = (read_jsonl(args.data / f) for f in ('corpus.jsonl', 'qa.jsonl'))
    parents = {p['id']: p for p in corpus}
    assert len(parents) == len(corpus) == 4876
    assert len({q['id'] for q in qa}) == len(qa) == 100
    assert all(q['relevant_passage_id'] in parents for q in qa)
    split = split_queries(qa, corpus)
    write_json(args.output / 'split.json', split)
    dev = [q for q in qa if q['id'] in split['dev']]
    test = [q for q in qa if q['id'] in split['test']]

    import numpy as np
    import torch
    from threadpoolctl import threadpool_limits
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    threadpool_limits(limits=4)
    model = SentenceTransformer(MODEL, revision=MODEL_REVISION, device='cpu',
                               cache_folder=str(args.model_cache) if args.model_cache else None,
                               local_files_only=args.offline)
    tokenizer = model.tokenizer
    qvec = model.encode([q['question'] for q in dev], normalize_embeddings=True, show_progress_bar=False)
    qvec = dict(zip([q['id'] for q in dev], qvec))
    indexes = {}
    sweep = []
    configs = []
    dev_predictions = {}
    audit = []

    def expand(ranking, lookup):
        return [{**lookup[h['doc_id']], 'score': float(h['score'])} for h in ranking]

    def dense_search(vec, vectors, chunks):
        scores = vectors @ vec
        order = np.argsort(-scores, kind='stable')[:100]  # chunks have deterministic ID ordering
        return [{'doc_id': chunks[i]['doc_id'], 'score': float(scores[i])} for i in order]

    def assess(config, rankings, lookup):
        values = [metrics(expand(rankings[q['id']], lookup), q['relevant_passage_id'],
                          len(parents[q['relevant_passage_id']]['text']), 5) for q in dev]
        row = {**config, 'n_queries': len(dev), **{m: statistics.mean(v[m] for v in values) for m in METRICS}}
        sweep.append(row)
        configs.append(config)
        dev_predictions[config['config_id']] = rankings

    for size in (0, 128, 256):
        print(f'Building chunk_size={size or "original"}', flush=True)
        chunks = make_chunks(corpus, tokenizer, size)
        lookup = {c['doc_id']: c for c in chunks}
        cache_key = hashlib.sha256(json.dumps({
            'model': MODEL, 'revision': MODEL_REVISION, 'max_length': model.max_seq_length,
            'packages': {n: importlib.metadata.version(n) for n in ('torch', 'sentence-transformers', 'transformers')},
            'texts': [c['text'] for c in chunks]}, sort_keys=True).encode()).hexdigest()
        cache_file = args.embedding_cache / (cache_key + '.npy') if args.embedding_cache else None
        if cache_file and cache_file.exists():
            vectors = np.load(cache_file, allow_pickle=False)
            assert len(vectors) == len(chunks)
        else:
            vectors = model.encode([c['text'] for c in chunks], normalize_embeddings=True,
                                   batch_size=32, show_progress_bar=True)
            if cache_file:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache_file, vectors)
        dense = {q['id']: dense_search(qvec[q['id']], vectors, chunks) for q in dev}
        trunc = sum(len(tokenizer(c['text'], truncation=False)['input_ids']) > model.max_seq_length for c in chunks)
        audit.append({'chunk_size': size, 'chunks': len(chunks), 'dense_truncated_chunks': trunc})
        indexes[size] = (chunks, lookup, vectors)
        base = {'method': 'dense', 'chunk_size': size, 'k1': '', 'b': '', 'bm25_weight': ''}
        assess({'config_id': f'dense_s{size}', **base}, dense, lookup)
        bm_candidates = []
        for k1 in (.8, 1.2, 1.6):
            for b in (.25, .75, 1.0):
                retriever = BM25Retriever(chunks, k1=k1, b=b)
                ranks = {q['id']: retriever.search(q['question'], 100) for q in dev}
                cfg = {'config_id': f'bm25_s{size}_k{k1}_b{b}', 'method': 'bm25',
                       'chunk_size': size, 'k1': k1, 'b': b, 'bm25_weight': ''}
                assess(cfg, ranks, lookup)
                bm_candidates.append(sweep[-1])
        best_bm = max(bm_candidates, key=lambda r: (r['ndcg'], r['mrr'], r['recall']))
        bm = dev_predictions[best_bm['config_id']]
        for weight in (.25, .5, .75):
            ranks = {q['id']: fuse(bm[q['id']], dense[q['id']], weight) for q in dev}
            cfg = {'config_id': f'hybrid_s{size}_w{weight}', 'method': 'hybrid', 'chunk_size': size,
                   'k1': best_bm['k1'], 'b': best_bm['b'], 'bm25_weight': weight}
            assess(cfg, ranks, lookup)
    write_csv(args.output / 'dev_sweep.csv', sweep)
    selected = {}
    for method in ('bm25', 'dense', 'hybrid'):
        best = max((r for r in sweep if r['method'] == method), key=lambda r: (r['ndcg'], r['mrr'], r['recall']))
        selected[method] = next(c for c in configs if c['config_id'] == best['config_id'])
    write_json(args.output / 'selected.json', selected)  # Frozen before test retrieval.
    print('Selected on development:', selected, flush=True)
    query_trunc = sum(len(tokenizer(q['question'], truncation=False)['input_ids']) > model.max_seq_length for q in qa)
    manifest = {'dataset': 'isaacus/legal-rag-bench', 'dataset_revision': DATA_REVISION,
                'sha256': HASHES, 'model': MODEL, 'model_revision': MODEL_REVISION,
                'model_max_seq_length': model.max_seq_length, 'query_truncation_count': query_trunc,
                'corpus_count': len(corpus), 'unique_texts': len({p['text'] for p in corpus}),
                'dev_count': len(dev), 'test_count': len(test), 'split': 'group-gold-text-shuffle42-20pct-v1',
                'selection': 'dev nDCG@5, then MRR@5, then Recall@5, then grid order',
                'chunk_audit': audit, 'chunk_overlap': 'floor(0.2 * content token budget)',
                'retrieved_unit': 'chunk; single gold parent; repeated children get no extra rank credit',
                'rrf_c': 60, 'candidate_depth': 100, 'device': 'cpu', 'threads': 4,
                'platform': platform.platform(), 'python': sys.version,
                'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'bm25_sha256': hashlib.sha256((ROOT / 'person2_bm25/finder_bm25/bm25.py').read_bytes()).hexdigest(),
                'packages': {n: importlib.metadata.version(n) for n in ('torch','numpy','sentence-transformers','transformers','tokenizers')},
                'latency': '3 warmups per method, 3 shuffled interleaved repeats of all test queries; includes query embedding, search and fusion, excludes model/index construction and metrics',
                'answer_correctness': 'not evaluated; no generator selected'}
    write_json(args.output / 'manifest.json', manifest)
    searchers = {}
    for method, cfg in selected.items():
        chunks, lookup, vectors = indexes[cfg['chunk_size']]
        bm = BM25Retriever(chunks, k1=cfg['k1'], b=cfg['b']) if method != 'dense' else None
        def search(question, method=method, cfg=cfg, chunks=chunks, vectors=vectors, bm=bm):
            lexical = bm.search(question, 100) if bm else []
            dense = []
            if method != 'bm25':
                vec = model.encode([question], normalize_embeddings=True, show_progress_bar=False)[0]
                dense = dense_search(vec, vectors, chunks)
            return lexical if method == 'bm25' else dense if method == 'dense' else fuse(lexical, dense, cfg['bm25_weight'])
        searchers[method] = search
        for q in dev[:3]:
            search(q['question'])
    timings, predictions = [], {}
    for repeat in range(3):
        jobs = [(m, q) for m in selected for q in test]
        random.Random(100 + repeat).shuffle(jobs)
        for method, q in jobs:
            start = time.perf_counter()
            ranking = searchers[method](q['question'])
            ms = (time.perf_counter() - start) * 1000
            timings.append({'method': method, 'query_id': q['id'], 'repeat': repeat, 'latency_ms': ms})
            if repeat == 0:
                predictions[(method, q['id'])] = ranking[:20]
    per_query, saved, generation, grading = [], [], [], []
    for method, cfg in selected.items():
        lookup = indexes[cfg['chunk_size']][1]
        for q in test:
            hits = expand(predictions[(method, q['id'])], lookup)
            saved.append({'method': method, 'query_id': q['id'], 'question': q['question'],
                          'gold_parent_id': q['relevant_passage_id'], 'hits': hits})
            for k in KS:
                per_query.append({'method': method, 'query_id': q['id'], 'k': k,
                                  **metrics(hits, q['relevant_passage_id'], len(parents[q['relevant_passage_id']]['text']), k)})
            context = '\n\n'.join(f'[{h["doc_id"]}]\n{h["text"]}' for h in hits[:5])
            prompt = ('Answer the question using only the supplied evidence. Cite passage IDs. '
                      'Include material conditions and exceptions. If evidence is insufficient, say so.\n\n'
                      f'Question: {q["question"]}\n\nEvidence:\n{context}')
            generation.append({'method': method, 'query_id': q['id'], 'prompt': prompt,
                               'context_tokens_minilm': len(tokenizer(context, truncation=False)['input_ids'])})
            grading.append({'method': method, 'query_id': q['id'], 'reference_answer': q['answer'],
                            'generated_answer': '', 'correctness': '', 'groundedness': '', 'judge_notes': '', 'reviewer': '', 'abstained': ''})
    for q in test:
        generation.append({'method': 'oracle', 'query_id': q['id'],
                           'prompt': 'Answer the question using only the supplied evidence. Cite passage IDs. Include material conditions and exceptions. If evidence is insufficient, say so.\n\nQuestion: '
                           + q['question'] + '\n\nEvidence:\n[' + q['relevant_passage_id'] + ']\n' + parents[q['relevant_passage_id']]['text'],
                           'context_tokens_minilm': len(tokenizer(parents[q['relevant_passage_id']]['text'], truncation=False)['input_ids'])})
        grading.append({'method': 'oracle', 'query_id': q['id'], 'reference_answer': q['answer'],
                        'generated_answer': '', 'correctness': '', 'groundedness': '', 'judge_notes': '', 'reviewer': '', 'abstained': ''})
    summaries = []
    for method, cfg in selected.items():
        ts = [t['latency_ms'] for t in timings if t['method'] == method]
        for k in KS:
            rows = [r for r in per_query if r['method'] == method and r['k'] == k]
            summaries.append({'method': method, 'config_id': cfg['config_id'], 'k': k, 'n_queries': len(rows),
                              **{m: statistics.mean(r[m] for r in rows) for m in METRICS},
                              'latency_mean_ms': statistics.mean(ts), 'latency_p95_ms': float(np.percentile(ts, 95))})
    write_csv(args.output / 'test_metrics.csv', summaries)
    write_csv(args.output / 'per_query.csv', per_query)
    write_csv(args.output / 'latency_raw.csv', timings)
    write_jsonl(args.output / 'retrievals.jsonl', saved)
    write_jsonl(args.output / 'generation_inputs.jsonl', generation)
    write_csv(args.output / 'answer_review.csv', grading)
    lines = ['# Legal RAG Bench — held-out retrieval results', '',
             f'{len(dev)} development questions; {len(test)} held-out questions. Configurations selected on development nDCG@5.', '',
             '| Method | Recall@5 | Hit@5 | All evidence hit@5 | Gold-text coverage@5 | MRR@5 | nDCG@5 | Mean ms |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in summaries:
        if r['k'] == 5:
            lines.append('| ' + r['method'] + ' | ' + ' | '.join(f'{r[m]:.4f}' for m in METRICS) + f' | {r["latency_mean_ms"]:.2f} |')
    lines += ['', 'Recall, Hit and All Evidence Hit coincide because there is one gold passage per question.',
              'Coverage is union character coverage of the gold passage, NOT semantic evidence completeness.',
              'K counts chunks, including repeated parents. Rank credit is awarded only at the first gold-parent occurrence.',
              'Each method selects its own chunk size; context lengths differ. See selected.json and generation_inputs.jsonl.',
              'Latency is warm CPU top-100 candidate retrieval, not generation or separate per-K latency.',
              'Original-passage dense embeddings can truncate; inspect manifest.json before interpreting comparisons.',
              'Answer correctness and groundedness: NOT EVALUATED. No answer model has been selected.',
              'This is a small internal split, not the official full-100 benchmark score.', '']
    (args.output / 'report.md').write_text('\n'.join(lines))
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
