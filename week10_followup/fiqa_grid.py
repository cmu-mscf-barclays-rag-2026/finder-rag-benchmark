"""Repeat the Meeting 10 K/budget grid on the saved BEIR FiQA BM25 ranking."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from week8_retrieval.beir_eval import evaluate, load_beir, load_rankings, write_csv
from week6_qwen_hierarchy.qwen_hierarchy import evaluation_tokenizer
from week9_evidence.vidore import digest

ROOT = Path(__file__).resolve().parents[1]
KS = (5, 8, 10, 20)
BUDGETS = (6000, 10000, 12000)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT / '.cache/fiqa')
    parser.add_argument('--rankings', type=Path,
                        default=ROOT / 'week8_retrieval/fiqa_results/rankings.jsonl')
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'week10_followup/results/fiqa_grid')
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output must be new or empty')
    corpus, queries, qrels = load_beir(args.dataset, 'test')
    rankings = load_rankings(args.rankings, queries, corpus)
    if any(len(ranking) < max(KS) for ranking in rankings.values()):
        raise ValueError('Saved FiQA ranking shorter than K=20')
    config = json.loads((ROOT / 'week6_qwen_hierarchy/results/qwen/manifest.json').read_text())['config']
    tokenizer, tokenizer_sha = evaluation_tokenizer(config, ROOT / '.cache/models', True)
    rows, summary = evaluate(rankings, qrels, corpus, tokenizer, KS, BUDGETS)
    args.output.mkdir(parents=True)
    write_csv(args.output / 'per_query.csv', rows)
    write_csv(args.output / 'summary.csv', summary)
    (args.output / 'manifest.json').write_text(json.dumps({
        'dataset': 'BEIR FiQA', 'split': 'test', 'queries': len(queries),
        'passages': len(corpus), 'ks': KS, 'budgets': BUDGETS,
        'retrieval': 'saved BM25 k1=1.2 b=0.75, top-20',
        'ranking_sha256': digest(args.rankings), 'tokenizer_sha256': tokenizer_sha,
        'source_sha256': digest(Path(__file__)),
        'output_sha256': {p.name: digest(p) for p in args.output.glob('*.csv')},
        'relevance_note': 'Relevant answers can be alternatives rather than jointly required facts.',
    }, indent=2) + '\n')
    print(f'Re-scored {len(queries)} FiQA questions at {len(KS)*len(BUDGETS)} settings -> {args.output}')


if __name__ == '__main__':
    main()
