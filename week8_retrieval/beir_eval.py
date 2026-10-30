"""Evaluate a BEIR corpus with multi-relevance qrels and optional saved rankings.

Expected folder: corpus.jsonl, queries.jsonl, qrels/{split}.tsv.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from week5_hierarchy.hierarchy_experiment import ROOT
from week3_bm25.finder_bm25.bm25 import BM25Retriever
from week6_qwen_hierarchy.qwen_hierarchy import evaluation_tokenizer
from hierarchy.io import read_jsonl


def load_beir(folder, split):
    corpus_rows = read_jsonl(folder / 'corpus.jsonl')
    query_rows = read_jsonl(folder / 'queries.jsonl')
    corpus = {str(r['_id']): ((r.get('title') or '') + '\n' + r['text']).strip() for r in corpus_rows}
    queries = {str(r['_id']): r['text'] for r in query_rows}
    if len(corpus) != len(corpus_rows) or len(queries) != len(query_rows):
        raise ValueError('Duplicate document or query ID')
    qrels = defaultdict(dict)
    with (folder / 'qrels' / f'{split}.tsv').open(newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        if not {'query-id', 'corpus-id', 'score'} <= set(reader.fieldnames or []):
            raise ValueError('Invalid BEIR qrels header')
        for row in reader:
            qid, docid, score = str(row['query-id']), str(row['corpus-id']), int(row['score'])
            if qid not in queries or docid not in corpus:
                raise ValueError('Qrel references absent query or document')
            if docid in qrels[qid]:
                raise ValueError('Duplicate qrel')
            if score > 0:
                qrels[qid][docid] = score
    if not qrels:
        raise ValueError('No positive relevance judgments')
    return corpus, {qid: queries[qid] for qid in qrels}, dict(qrels)


def load_rankings(path, queries, corpus):
    rows = read_jsonl(path)
    ranks = {str(r['query_id']): [str(x) for x in r['doc_ids']] for r in rows}
    if len(ranks) != len(rows) or set(ranks) != set(queries):
        raise ValueError('Ranking/query IDs differ')
    if any(len(ids) != len(set(ids)) or not set(ids) <= corpus.keys() for ids in ranks.values()):
        raise ValueError('Invalid ranking IDs')
    return ranks


def rank_metrics(ids, relevance, k):
    chosen = ids[:k]
    found = set(chosen) & relevance.keys()
    first = next((i for i, cid in enumerate(chosen, 1) if cid in relevance), None)
    dcg = sum((2**relevance.get(cid, 0)-1) / math.log2(i+1) for i, cid in enumerate(chosen, 1))
    ideal = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum((2**score-1) / math.log2(i+1) for i, score in enumerate(ideal, 1))
    return {'hit': int(bool(found)), 'recall': len(found)/len(relevance),
            'all_judged_relevant': int(len(found) == len(relevance)),
            'mrr': 1/first if first else 0, 'ndcg': dcg/idcg if idcg else 0}


def budgeted_ids(ids, corpus, tokenizer, budget):
    kept, used = [], 0
    for cid in ids:
        cost = len(tokenizer.encode(corpus[cid]).ids)
        separator = 0 if not kept else len(tokenizer.encode('\n\n').ids)
        if used + separator + cost <= budget:
            kept.append(cid)
            used += separator + cost
    return kept, used


def evaluate(rankings, qrels, corpus, tokenizer, ks=(5, 10, 20), budgets=(4000, 6000, 8000)):
    rows = []
    for qid, relevant in qrels.items():
        ids = rankings[qid]
        for k in ks:
            ranked = rank_metrics(ids, relevant, k)
            for budget in budgets:
                kept, tokens = budgeted_ids(ids[:k], corpus, tokenizer, budget)
                delivered = rank_metrics(kept, relevant, k)
                rows.append({'query_id': qid, 'k': k, 'budget': budget,
                    'n_relevant': len(relevant), 'tokens': tokens, 'returned_docs': len(kept),
                    **{'ranked_'+key: value for key, value in ranked.items()},
                    **{'delivered_'+key: value for key, value in delivered.items()}})
    summary = []
    for k in ks:
        for budget in budgets:
            group = [r for r in rows if r['k'] == k and r['budget'] == budget]
            summary.append({'k': k, 'budget': budget, 'n_queries': len(group),
                'multi_relevant_queries': sum(r['n_relevant'] > 1 for r in group),
                **{key: statistics.mean(r[key] for r in group) for key in group[0]
                   if key not in ('query_id', 'k', 'budget', 'n_relevant')}})
    return rows, summary


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--split', default='test')
    parser.add_argument('--rankings', type=Path, help='JSONL with query_id and ordered doc_ids; otherwise run BM25')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output must be new or empty')
    corpus, queries, qrels = load_beir(args.dataset, args.split)
    config = json.loads((ROOT / 'week6_qwen_hierarchy/results/qwen/manifest.json').read_text())['config']
    tokenizer, _ = evaluation_tokenizer(config, ROOT / '.cache/models', True)
    if args.rankings:
        ranks = load_rankings(args.rankings, queries, corpus)
        method = 'supplied'
    else:
        print(f'Indexing {len(corpus)} documents for BM25', flush=True)
        bm25 = BM25Retriever([{'doc_id': cid, 'text': text} for cid, text in sorted(corpus.items())])
        ranks = {qid: [hit['doc_id'] for hit in bm25.search(question, 20)] for qid, question in queries.items()}
        method = 'BM25 k1=1.2 b=0.75'
    print(f'Evaluating {len(queries)} {args.split} queries', flush=True)
    rows, summary = evaluate(ranks, qrels, corpus, tokenizer)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / 'per_query.csv', rows)
    write_csv(args.output / 'summary.csv', summary)
    with (args.output / 'rankings.jsonl').open('w') as f:
        for qid, ids in ranks.items():
            f.write(json.dumps({'query_id': qid, 'doc_ids': ids}) + '\n')
    (args.output / 'manifest.json').write_text(json.dumps({'dataset': str(args.dataset),
        'split': args.split, 'method': method, 'queries': len(queries), 'documents': len(corpus),
        'relevance_note': 'Multiple relevant documents can be alternative answers, not jointly required evidence.'}, indent=2) + '\n')


if __name__ == '__main__':
    main()
