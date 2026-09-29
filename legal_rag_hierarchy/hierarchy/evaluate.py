"""Evaluate saved Part 2 rankings with consistent hierarchy and context budgets."""
import argparse
import json
from pathlib import Path

from .handoff import evaluate_run
from .io import ROOT, digest, read_jsonl, load_tokenizer, write_json, write_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--method', required=True)
    parser.add_argument('--mode', choices=['child', 'parent', 'child_to_parent'], required=True)
    parser.add_argument('--split', choices=['dev', 'test', 'official_test'], default='test')
    parser.add_argument('--token-budget', type=int, help='Content-token budget; omit for unbounded context')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'results/build_manifest.json').read_text(encoding='utf-8'))
    for relative, expected in manifest['outputs_sha256'].items():
        if digest(ROOT / relative) != expected:
            raise ValueError(f'Processed-data integrity failure: {relative}')
    children = {r['id']: r for r in read_jsonl(ROOT / 'data/children.jsonl')}
    parents = {r['id']: r for r in read_jsonl(ROOT / 'data/parents.jsonl')}
    queries = read_jsonl(ROOT / 'data/queries.jsonl')
    if args.split != 'official_test':
        queries = [q for q in queries if q['split'] == args.split]
    predictions = {}
    for row in read_jsonl(args.predictions):
        qid = str(row['query_id'])
        if qid in predictions:
            raise ValueError('Duplicate prediction query ID.')
        predictions[qid] = row['passage_ids']
    tokenizer, metadata = load_tokenizer()
    detail, summary = evaluate_run(queries, predictions, children, parents, tokenizer,
                                    mode=args.mode, token_budget=args.token_budget)
    summary = [{'method': args.method, 'split': args.split, **row} for row in summary]
    write_csv(args.out / 'per_query.csv', detail)
    write_csv(args.out / 'summary.csv', summary)
    write_json(args.out / 'run_manifest.json', {'method': args.method, 'mode': args.mode,
               'split': args.split, 'token_budget': args.token_budget,
               'prediction_sha256': digest(args.predictions),
               'build_manifest_sha256': digest(ROOT / 'results/build_manifest.json'),
               'tokenizer': metadata,
               'scope': 'Saved-ranking evaluation; retrieval model settings and timing supplied by Part 2'})
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
