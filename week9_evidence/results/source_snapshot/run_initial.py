"""Controlled whole-page retrieval, context budgets, and annotated fact coverage."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import importlib.metadata
import json
from pathlib import Path
import platform
import statistics
import time

import numpy as np

from week8_retrieval.beir_eval import rank_metrics
from week6_qwen_hierarchy.qwen_hierarchy import QwenEncoder, evaluation_tokenizer
from week3_bm25.finder_bm25.bm25 import BM25Retriever
from week9_evidence.evidence import fact_metrics, marginal_gains, validate_annotations
from week9_evidence.vidore import digest, load_dataset, read_jsonl, write_jsonl

ROOT = Path(__file__).resolve().parents[1]
TOKENIZER = {'model': 'sentence-transformers/all-MiniLM-L6-v2',
             'revision': '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'}
QWEN = {'model': 'Qwen/Qwen3-Embedding-0.6B',
        'revision': '97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3',
        'query_instruction': 'Find financial report passages containing evidence that answers the question.'}
POLICIES = ((0, 0), (1, 1), (2, 1))


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='') as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def neighbors(pages):
    """Adjacency is original PDF page order, never lexicographic corpus ID order."""
    return {(p['document'], p['position']): cid for cid, p in pages.items()}


def expand(ranking, pages, positions, k, before, after):
    candidates, seen = [], set()
    for seed in ranking[:k]:
        page = pages[seed]
        additions = [seed]
        for distance in range(1, max(before, after) + 1):
            for delta, enabled in ((-distance, distance <= before), (distance, distance <= after)):
                cid = positions.get((page['document'], page['position'] + delta))
                if enabled and cid is not None:
                    additions.append(cid)
        for cid in additions:
            if cid not in seen:
                seen.add(cid)
                candidates.append(cid)
    return candidates


def pack(candidates, pages, costs, budget, count_tokens):
    kept, used = [], 0
    for cid in dict.fromkeys(candidates):
        # Whole pages for every method. Skip over-budget pages and continue.
        if costs[cid] and used + costs[cid] <= budget:
            kept.append(cid)
            used += costs[cid]
    actual = count_tokens('\n\n'.join(pages[cid]['text'] for cid in kept))
    if actual != used or actual > budget:
        raise ValueError('Measurement tokenizer additivity or budget invariant failed')
    return kept, actual


def retrieve(method, pages, queries, args):
    ids, qids = sorted(pages), sorted(queries)
    texts = [pages[cid]['text'] for cid in ids]
    questions = [queries[qid]['query'] for qid in qids]
    timings = []
    started = time.perf_counter()
    if method == 'bm25':
        index = BM25Retriever([{'doc_id': cid, 'text': text} for cid, text in zip(ids, texts)])
        build_seconds = time.perf_counter() - started
        rankings = {}
        for qid in qids:
            tick = time.perf_counter()
            rankings[qid] = [hit['doc_id'] for hit in index.search(queries[qid]['query'], 100)]
            timings.append(time.perf_counter() - tick)
        return rankings, {'method': method, 'build_seconds': build_seconds,
                          'query_search_mean_ms': 1000 * statistics.mean(timings),
                          'query_search_p95_ms': 1000 * float(np.percentile(timings, 95)),
                          'k1': 1.2, 'b': 0.75}
    if method == 'minilm':
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(TOKENIZER['model'], revision=TOKENIZER['revision'],
            cache_folder=str(args.model_cache), local_files_only=True, device=args.device)
        model.max_seq_length = 256
        vectors = model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=True)
        build_seconds = time.perf_counter() - started
        tick = time.perf_counter()
        qvectors = model.encode(questions, batch_size=32, normalize_embeddings=True, show_progress_bar=True)
        query_encode_seconds = time.perf_counter() - tick
        info = {'encoder': TOKENIZER, 'embedding_window': 256,
                'pooling': 'mean', 'query_instruction': None,
                'note': 'Different encoder window from Qwen; model plus configuration comparison.'}
    else:
        encoder = QwenEncoder(QWEN, args)
        window = 4096 if method == 'qwen' else 256
        vectors = encoder.encode(texts, window, 'page')
        build_seconds = time.perf_counter() - started
        tick = time.perf_counter()
        qvectors = encoder.encode(questions, 2048, 'query',
            prompt=f'Instruct: {QWEN["query_instruction"]}\nQuery: ')
        query_encode_seconds = time.perf_counter() - tick
        info = {'encoder': encoder.identity, 'embedding_window': window,
                'query_window': 2048, 'query_instruction': QWEN['query_instruction'],
                'input_audit': encoder.audit}
    rankings = {}
    # Single-threaded exact search; deterministic ties use sorted IDs.
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        for qid, vector in zip(qids, qvectors):
            tick = time.perf_counter()
            order = np.argsort(-(vectors @ vector), kind='stable')[:100]
            rankings[qid] = [ids[i] for i in order]
            timings.append(time.perf_counter() - tick)
    np.savez_compressed(args.output / 'vectors.npz', page_ids=np.array(ids), query_ids=np.array(qids),
                        pages=vectors, queries=qvectors)
    return rankings, {'method': method, **info, 'build_seconds_including_model_load_or_cache': build_seconds,
        'query_encode_seconds': query_encode_seconds, 'search': 'exact normalized dot product, 1 BLAS thread',
        'query_search_mean_ms': 1000 * statistics.mean(timings),
        'query_search_p95_ms': 1000 * float(np.percentile(timings, 95)),
        'vector_bytes': vectors.nbytes, 'vector_shape': list(vectors.shape)}


def evaluate(rankings, pages, queries, qrels, count_tokens, ks, budgets, annotations):
    positions = neighbors(pages)
    costs = {cid: count_tokens(page['text']) for cid, page in pages.items()}
    per_query, fact_rows, marginal_rows = [], [], []
    for qid, ranking in rankings.items():
        for k in ks:
            ranked = rank_metrics(ranking, qrels[qid], k)
            for before, after in POLICIES:
                candidates = expand(ranking, pages, positions, k, before, after)
                for budget in budgets:
                    kept, tokens = pack(candidates, pages, costs, budget, count_tokens)
                    delivered = rank_metrics(kept, qrels[qid], len(kept))
                    base = {'query_id': qid, 'k': k, 'before': before, 'after': after, 'budget': budget}
                    per_query.append({**base, 'tokens': tokens, 'pages': len(kept),
                        **{'ranked_' + key: value for key, value in ranked.items()},
                        **{'delivered_' + key: value for key, value in delivered.items()},
                        'returned_ids': json.dumps(kept)})
                    if qid in annotations:
                        annotation = annotations[qid]
                        visible = {cid: [(0, len(pages[cid]['text']))] for cid in kept}
                        metrics = fact_metrics(annotation, visible, tokens)
                        metrics['covered_fact_ids'] = json.dumps(metrics['covered_fact_ids'])
                        fact_rows.append({**base, 'status': annotation['status'],
                            'question_type': annotation['question_type'], 'tokens': tokens,
                            'delivered_hit': delivered['hit'], 'delivered_recall': delivered['recall'], **metrics})
                        marginal_rows.extend({**base, 'step': step, **row,
                            'new_fact_ids': json.dumps(row['new_fact_ids'])}
                            for step, row in enumerate(marginal_gains(annotation, kept, pages, count_tokens), 1))
    summary = summarize(per_query, ['tokens', 'pages'] + [key for key in per_query[0]
        if key.startswith(('ranked_', 'delivered_'))])
    fact_summary = summarize(fact_rows, ['fact_recall', 'sufficient', 'facts_per_1k_tokens',
        'tokens', 'delivered_hit', 'delivered_recall']) if fact_rows else []
    minimum = []
    groups = defaultdict(list)
    for row in fact_rows:
        groups[(row['query_id'], row['k'], row['before'], row['after'])].append(row)
    for (qid, k, before, after), rows in groups.items():
        sufficient = [r for r in rows if r['sufficient']]
        minimum.append({'query_id': qid, 'k': k, 'before': before, 'after': after,
            'minimum_tested_sufficient_budget': min((r['budget'] for r in sufficient), default=None),
            'minimum_sufficient_delivered_tokens_in_grid': min((r['tokens'] for r in sufficient), default=None),
            'sufficient_at_any_tested_budget': int(bool(sufficient))})
    return dict(per_query=per_query, summary=summary, facts=fact_rows,
                fact_summary=fact_summary, marginal=marginal_rows, minimum_budgets=minimum)


def summarize(rows, metrics):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in ('k', 'before', 'after', 'budget'))].append(row)
    return [{**dict(zip(('k', 'before', 'after', 'budget'), key)), 'n': len(group),
             **{metric: statistics.mean(r[metric] for r in group) for metric in metrics}}
            for key, group in sorted(groups.items())]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--method', choices=['bm25', 'minilm', 'qwen', 'qwen256'], default='bm25')
    parser.add_argument('--rankings', type=Path, help='Reuse saved rankings; no encoder inference')
    parser.add_argument('--annotations', type=Path)
    parser.add_argument('--allow-draft', action='store_true')
    parser.add_argument('--ks', type=int, nargs='+', default=[5, 10, 20])
    parser.add_argument('--budgets', type=int, nargs='+', default=[2000, 4000, 6000, 8000])
    parser.add_argument('--device', choices=['cpu', 'mps'], default='cpu')
    parser.add_argument('--dtype', default='float32', choices=['float32'])
    args = parser.parse_args()
    if any(x <= 0 for x in args.ks + args.budgets) or max(args.ks) > 100:
        parser.error('Use positive budgets and k in [1, 100]')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output must be new or empty')
    pages, queries, qrels, dataset_manifest = load_dataset(args.dataset)
    annotations = read_jsonl(args.annotations) if args.annotations else []
    if annotations:
        validate_annotations(annotations, pages, queries, args.allow_draft)
    args.model_cache = ROOT / '.cache/models'
    args.embedding_cache = ROOT / '.cache/week9_embeddings'
    args.offline, args.batch_size, args.batch_tokens = True, 8, 4096
    import torch
    torch.set_num_threads(4)
    tokenizer, tokenizer_sha = evaluation_tokenizer({'evaluation_tokenizer': TOKENIZER}, args.model_cache, True)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.rankings:
        rows = read_jsonl(args.rankings)
        ranks = {r['query_id']: r['doc_ids'] for r in rows}
        if len(rows) != len(ranks) or set(ranks) != set(queries):
            raise ValueError('Ranking/query mismatch')
        if any(len(ids) != len(set(ids)) or not set(ids) <= pages.keys() for ids in ranks.values()):
            raise ValueError('Duplicate or unknown ranking page IDs')
        retrieval = {'method': 'supplied', 'rankings_sha256': digest(args.rankings)}
    else:
        ranks, retrieval = retrieve(args.method, pages, queries, args)
    write_jsonl(args.output / 'rankings.jsonl', [{'query_id': qid, 'doc_ids': ids} for qid, ids in ranks.items()])
    results = evaluate(ranks, pages, queries, qrels, lambda text: len(tokenizer.encode(text).ids),
                       args.ks, args.budgets, {r['query_id']: r for r in annotations})
    for name, rows in results.items():
        write_csv(args.output / f'{name}.csv', rows)
    manifest = {'dataset': dataset_manifest, 'queries': len(queries), 'pages': len(pages),
        'language': 'english', 'retrieval': retrieval, 'ks': args.ks, 'budgets': args.budgets,
        'policies_before_after': POLICIES, 'candidate_depth': 100,
        'packing': 'whole unique pages, seed then nearest siblings; skip pages that do not fit; no backfill seeds',
        'tokenizer': TOKENIZER, 'tokenizer_sha256': tokenizer_sha,
        'annotations_sha256': digest(args.annotations) if args.annotations else None,
        'annotation_status_counts': {s: sum(r['status'] == s for r in annotations) for s in ('draft', 'reviewed')},
        'evaluation_note': 'Page qrels are relevance, not required facts. Draft fact coverage is exploratory only.',
        'source_sha256': {p.name: digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
        'python': platform.python_version(), 'device': args.device,
        'packages': {p: importlib.metadata.version(p) for p in ('torch', 'numpy', 'transformers', 'tokenizers', 'sentence-transformers')},
        'output_sha256': {p.name: digest(p) for p in sorted(args.output.iterdir()) if p.is_file()}}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Finished {len(queries)} questions over {len(pages)} pages -> {args.output}', flush=True)


if __name__ == '__main__':
    main()
