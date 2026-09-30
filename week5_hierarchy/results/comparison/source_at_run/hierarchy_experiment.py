"""Matched child / parent / expansion experiment using Kevin's pinned handoff."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import statistics
import sys
from pathlib import Path, PureWindowsPath

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'legal_rag_hierarchy'))
sys.path.insert(0, str(ROOT / 'person2_bm25'))
from finder_bm25.bm25 import BM25Retriever, TOKENIZER_VERSION
from hierarchy.handoff import evaluate_run
from hierarchy.io import digest, read_jsonl, write_csv, write_json, write_jsonl
from legal_rag.run import MODEL, MODEL_REVISION, fuse

KS = (1, 5, 10)
MODES = ('child', 'parent', 'child_to_parent')


def load_data(root):
    manifest = json.loads((root / 'results/build_manifest.json').read_text())
    for relative, expected in manifest['outputs_sha256'].items():
        # The upstream manifest was generated on Windows; do not rewrite it.
        if digest(root / PureWindowsPath(relative)) != expected:
            raise ValueError(f'Processed-data checksum mismatch: {relative}')
    children, parents = [{r['id']: r for r in read_jsonl(root / f'data/{name}.jsonl')}
                         for name in ('children', 'parents')]
    queries = read_jsonl(root / 'data/queries.jsonl')
    for parent in parents.values():
        for span in parent['child_spans']:
            child = children[span['child_id']]
            assert child['parent_id'] == parent['id']
            assert parent['text'][span['start_char']:span['end_char']] == child['text']
    assert len({q['id'] for q in queries}) == len(queries) == 100
    assert all(q['gold_parent_ids'] == [children[g]['parent_id'] for g in q['gold_child_ids']]
               for q in queries)
    return children, parents, queries, manifest


def rank_score(queries, rankings, mode, k=5):
    values = []
    for q in queries:
        gold = set(q['gold_parent_ids'] if mode == 'parent' else q['gold_child_ids'])
        rank = next((i for i, h in enumerate(rankings[q['id']][:k], 1)
                     if h['doc_id'] in gold), None)
        values.append((1 / math.log2(rank + 1), 1 / rank, 1.) if rank else (0., 0., 0.))
    return tuple(statistics.mean(v[i] for v in values) for i in range(3))


def dense_search(vector, vectors, units):
    import numpy as np
    scores = vectors @ vector
    order = np.argsort(-scores, kind='stable')[:100]
    return [{'doc_id': units[i]['id'], 'score': float(scores[i])} for i in order]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model-cache', type=Path, default=ROOT / '.cache/models')
    p.add_argument('--embedding-cache', type=Path, default=ROOT / '.cache/hierarchy_embeddings')
    p.add_argument('--offline', action='store_true')
    args = p.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        p.error('Output must be new or empty; preserve prior runs.')
    args.output.mkdir(parents=True, exist_ok=True)
    children, parents, queries, build = load_data(ROOT / 'legal_rag_hierarchy')
    dev = [q for q in queries if q['split'] == 'dev']
    test = [q for q in queries if q['split'] == 'test']
    assert (len(dev), len(test)) == (20, 80)
    split = {s: [q['id'] for q in queries if q['split'] == s] for s in ('dev', 'test')}
    write_json(args.output / 'split.json', split)

    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    threadpool_limits(limits=4)
    model = SentenceTransformer(MODEL, revision=MODEL_REVISION, device='cpu',
                               cache_folder=str(args.model_cache), local_files_only=args.offline)
    # Separate tokenizer for exact full-text evaluation; never change the encoder.
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_str(model.tokenizer.backend_tokenizer.to_str())
    tokenizer.no_truncation()
    tokenizer.no_padding()
    packages = {n: importlib.metadata.version(n) for n in
                ('torch', 'numpy', 'sentence-transformers', 'transformers', 'tokenizers')}
    units = {'child': sorted(children.values(), key=lambda r: r['id']),
             'parent': sorted(parents.values(), key=lambda r: r['id'])}
    indexes, audit = {}, []
    dev_vectors = dict(zip([q['id'] for q in dev], model.encode(
        [q['question'] for q in dev], normalize_embeddings=True, show_progress_bar=False)))
    dense_dev = {}
    for mode, records in units.items():
        print(f'Indexing {mode}: {len(records)} units', flush=True)
        texts = [r['text'] for r in records]
        key = hashlib.sha256(json.dumps({'model': MODEL, 'revision': MODEL_REVISION,
            'packages': packages, 'texts': texts, 'limit': model.max_seq_length}, sort_keys=True).encode()).hexdigest()
        cache = args.embedding_cache / f'{key}.npy'
        if cache.exists():
            vectors = np.load(cache, allow_pickle=False)
        else:
            vectors = model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=True)
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, vectors)
        assert vectors.shape == (len(records), model.get_sentence_embedding_dimension())
        assert np.isfinite(vectors).all()
        indexes[mode] = vectors
        dense_dev[mode] = {q['id']: dense_search(dev_vectors[q['id']], vectors, records) for q in dev}
        sizes = [len(tokenizer.encode(t, add_special_tokens=False).ids) for t in texts]
        truncated = sum(len(tokenizer.encode(t, add_special_tokens=True).ids) > model.max_seq_length for t in texts)
        audit.append({'mode': mode, 'units': len(records), 'content_tokens_min': min(sizes),
                      'content_tokens_mean': statistics.mean(sizes), 'content_tokens_max': max(sizes),
                      'dense_truncated_units': truncated, 'dense_max_tokens_including_specials': model.max_seq_length})
    write_csv(args.output / 'index_audit.csv', audit)

    # Select ONE shared BM25 setting by the mean of child and parent dev scores.
    # This preserves identical retrieval settings across representations.
    sweep, bm_dev = [], {}
    for k1 in (.8, 1.2, 1.6):
        for b in (.25, .75, 1.0):
            scores = []
            for mode, records in units.items():
                retriever = BM25Retriever([{'doc_id': r['id'], 'text': r['text']} for r in records], k1=k1, b=b)
                ranking = {q['id']: retriever.search(q['question'], 100) for q in dev}
                bm_dev[(mode, k1, b)] = ranking
                score = rank_score(dev, ranking, mode)
                scores.append(score)
                sweep.append({'method': 'bm25', 'mode': mode, 'k1': k1, 'b': b, 'bm25_weight': '',
                              'ndcg5': score[0], 'mrr5': score[1], 'hit5': score[2]})
            score = tuple(statistics.mean(s[i] for s in scores) for i in range(3))
            sweep.append({'method': 'bm25', 'mode': 'joint', 'k1': k1, 'b': b, 'bm25_weight': '',
                          'ndcg5': score[0], 'mrr5': score[1], 'hit5': score[2]})
    ordering = lambda r: (r['ndcg5'], r['mrr5'], r['hit5'])
    best = max((r for r in sweep if r['mode'] == 'joint'), key=ordering)
    k1, b = best['k1'], best['b']
    for weight in (.25, .5, .75):
        scores = []
        for mode in units:
            ranking = {q['id']: fuse(bm_dev[(mode, k1, b)][q['id']], dense_dev[mode][q['id']], weight) for q in dev}
            score = rank_score(dev, ranking, mode)
            scores.append(score)
            sweep.append({'method': 'hybrid', 'mode': mode, 'k1': k1, 'b': b, 'bm25_weight': weight,
                          'ndcg5': score[0], 'mrr5': score[1], 'hit5': score[2]})
        score = tuple(statistics.mean(s[i] for s in scores) for i in range(3))
        sweep.append({'method': 'hybrid', 'mode': 'joint', 'k1': k1, 'b': b, 'bm25_weight': weight,
                      'ndcg5': score[0], 'mrr5': score[1], 'hit5': score[2]})
    best_hybrid = max((r for r in sweep if r['mode'] == 'joint' and r['method'] == 'hybrid'), key=ordering)
    selected = {'k1': k1, 'b': b, 'bm25_weight': best_hybrid['bm25_weight'],
                'selection': 'mean child/parent dev nDCG@5, then MRR@5, then Hit@5, then grid order',
                'candidate_depth': 100, 'rrf_constant': 60}
    write_json(args.output / 'selected.json', selected)  # Freeze BEFORE test retrieval.
    write_csv(args.output / 'dev_sweep.csv', sweep)
    print('Frozen shared settings:', selected, flush=True)

    test_vectors = dict(zip([q['id'] for q in test], model.encode(
        [q['question'] for q in test], normalize_embeddings=True, show_progress_bar=False)))
    all_rankings = {}
    for mode, records in units.items():
        retriever = BM25Retriever([{'doc_id': r['id'], 'text': r['text']} for r in records], k1=k1, b=b)
        bm = {q['id']: retriever.search(q['question'], 100) for q in test}
        dense = {q['id']: dense_search(test_vectors[q['id']], indexes[mode], records) for q in test}
        methods = {'bm25': bm, 'dense': dense}
        for w in (.25, .5, .75):
            methods[f'hybrid_w{w}'] = {q['id']: fuse(bm[q['id']], dense[q['id']], w) for q in test}
        for method, rankings in methods.items():
            all_rankings[(method, mode)] = {qid: [h['doc_id'] for h in hits] for qid, hits in rankings.items()}
            write_jsonl(args.output / 'rankings' / f'{method}_{mode}.jsonl',
                        [{'query_id': qid, 'passage_ids': [h['doc_id'] for h in hits],
                          'scores': [h['score'] for h in hits]} for qid, hits in rankings.items()])

    details, summaries = [], []
    for method in methods:
        for mode in MODES:
            ranking = all_rankings[(method, 'child' if mode == 'child_to_parent' else mode)]
            for budget in (None, 2048):
                print(f'Evaluating {method} {mode} budget={budget}', flush=True)
                rows, aggregate = evaluate_run(test, ranking, children, parents, tokenizer,
                                               mode=mode, ks=KS, token_budget=budget)
                for row in aggregate:
                    contexts = [r['returned_content_tokens'] for r in rows if r['k'] == row['k']]
                    row['context_tokens_p95'] = float(np.percentile(contexts, 95))
                    row['context_tokens_min'] = min(contexts)
                    row['context_tokens_max'] = max(contexts)
                    row['gold_context_hit_count'] = round(row['gold_child_hit_in_context'] * len(test))
                metadata = {'method': method, 'split': 'test', 'split_id': test[0]['split_id'],
                            'primary': method in ('bm25', f'hybrid_w{selected["bm25_weight"]}')}
                details.extend([{**metadata, **r} for r in rows])
                summaries.extend([{**metadata, **r} for r in aggregate])
    write_csv(args.output / 'per_query.csv', details)
    write_csv(args.output / 'summary.csv', summaries)

    old = json.loads((ROOT / 'legal_rag/results/initial/split.json').read_text())
    leakage = sorted(set(g for q in dev for g in q['gold_parent_ids']) &
                     set(g for q in test for g in q['gold_parent_ids']))
    write_json(args.output / 'manifest.json', {
        'hierarchy_source_branch': 'origin/finder-threshold-mmr',
        'hierarchy_source_commit': 'ed72cf8473dea5df38e56ee1f62ece325bebb788',
        'build_manifest_sha256': digest(ROOT / 'legal_rag_hierarchy/results/build_manifest.json'),
        'input_sha256': build['outputs_sha256'], 'model': MODEL, 'model_revision': MODEL_REVISION,
        'packages': packages, 'python': sys.version, 'platform': platform.platform(),
        'device': 'cpu', 'torch_threads': 4, 'bm25_tokenizer': TOKENIZER_VERSION,
        'dense_policy': 'one normalized vector per released child or merged parent; right truncation at 256 tokens',
        'tie_break': 'lexicographic document ID', 'text_policy': 'body only, no appended titles/footnotes',
        'expansion': 'top-k child seeds, parents deduplicated in first-hit order, no backfill',
        'budgets': [None, 2048], 'ks': KS, 'selection': selected,
        'overlapping_dev_test_gold_parents': leakage,
        'previous_test_overlap': len(set(split['test']) & set(map(str, old['test']))),
        'query_truncated_count': sum(len(tokenizer.encode(q['question']).ids) > model.max_seq_length for q in queries),
        'latency': 'not measured; this run evaluates retrieval quality and context size',
        'code_sha256': {str(p.relative_to(ROOT)): digest(p) for p in
                        [Path(__file__), ROOT / 'legal_rag/run.py',
                         ROOT / 'person2_bm25/finder_bm25/bm25.py',
                         ROOT / 'legal_rag_hierarchy/hierarchy/handoff.py']}})
    report(args.output, summaries, selected)


def report(output, summaries, selected):
    lines = ['# Person 2 — Hierarchical retrieval results', '',
             'Same 80 test questions from Kevin’s handoff; all settings selected on its 20 development questions.',
             f'Shared BM25: k1={selected["k1"]}, b={selected["b"]}. Selected hybrid BM25 weight: {selected["bm25_weight"]}.', '',
             'Hit means a complete original gold child is present in the delivered context. Coverage measures gold-body tokens, not answer correctness.', '']
    for budget in (None, 2048):
        lines += [f'## Context budget: {budget or "unbounded"}', '',
                  '| Method | Mode | Hit@1 | Hit@5 | Hit@10 | Coverage@10 | Mean tokens@10 | P95 tokens@10 |',
                  '|---|---|---:|---:|---:|---:|---:|---:|']
        for method in ('bm25', f'hybrid_w{selected["bm25_weight"]}'):
            for mode in MODES:
                rows = {r['k']: r for r in summaries if r['method'] == method and r['mode'] == mode and r['token_budget'] == budget}
                r = rows[10]
                hits = ' | '.join(f'{rows[k]["gold_context_hit_count"]}/80' for k in KS)
                lines.append(f'| {method} | {mode} | {hits} | {r["gold_body_token_coverage"]:.1%} | {r["returned_content_tokens"]:.0f} | {r["context_tokens_p95"]:.0f} |')
        lines.append('')
    lines += ['## Interpretation limits', '',
              '- K counts child seeds for expansion and parent results for direct parent retrieval. Expansion reuses exactly the child ranking and does not backfill.',
              '- Unbounded expansion can recover evidence through a sibling. It cannot improve the original child ranking.',
              '- MiniLM truncates encoder inputs at 256 tokens, separately from the final 2,048-token context cap. See index_audit.csv.',
              '- The split differs from Florence’s initial 20/80 run (63 shared test questions). Three gold parents span Kevin’s dev/test split. This is exploratory query-held-out evaluation, not parent-disjoint validation.',
              '- The original 44 → 56 configuration is unverified; this controlled experiment is not a reproduction claim.',
              '- summary.csv also reports dense and all three hybrid weights. Test scores do not select the weight.',
              '- No answer generation, correctness judging, or latency benchmark was performed.', '']
    (output / 'report.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
