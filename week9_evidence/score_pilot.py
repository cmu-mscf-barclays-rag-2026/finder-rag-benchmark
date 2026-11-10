"""Score draft/reviewed facts on an existing run without rerunning retrieval."""
import argparse
import csv
import json
from pathlib import Path

from week6_qwen_hierarchy.qwen_hierarchy import evaluation_tokenizer
from week9_evidence.evidence import validate_annotations
from week9_evidence.run import ROOT, TOKENIZER, evaluate, write_csv
from week9_evidence.vidore import digest, load_dataset, read_jsonl


def score(dataset, run, annotations_path, output, allow_draft=False):
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output must be new or empty')
    pages, queries, qrels, dataset_manifest = load_dataset(dataset)
    manifest = json.loads((run / 'manifest.json').read_text())
    if manifest['dataset'] != dataset_manifest:
        raise ValueError('Scoring dataset differs from retrieval dataset')
    for name in ('rankings.jsonl', 'per_query.csv'):
        if digest(run / name) != manifest['output_sha256'][name]:
            raise ValueError('Retrieval output checksum mismatch')
    annotations = read_jsonl(annotations_path)
    validate_annotations(annotations, pages, queries, allow_draft)
    by_id = {r['query_id']: r for r in annotations}
    ranks = {r['query_id']: r['doc_ids'] for r in read_jsonl(run / 'rankings.jsonl') if r['query_id'] in by_id}
    tokenizer, tokenizer_sha = evaluation_tokenizer({'evaluation_tokenizer': TOKENIZER}, ROOT / '.cache/models', True)
    if tokenizer_sha != manifest['tokenizer_sha256']:
        raise ValueError('Scoring tokenizer differs from retrieval tokenizer')
    results = evaluate(ranks, pages, queries, qrels, lambda text: len(tokenizer.encode(text).ids),
                       manifest['ks'], manifest['budgets'], by_id)
    def key(row):
        return tuple(str(row[k]) for k in ('query_id', 'k', 'before', 'after', 'budget'))
    with (run / 'per_query.csv').open() as source:
        original = {key(r): r for r in csv.DictReader(source) if r['query_id'] in by_id}
    if len(original) != len(results['per_query']):
        raise ValueError('Delivery trace count differs')
    for row in results['per_query']:
        old = original[key(row)]
        if old['returned_ids'] != row['returned_ids'] or int(old['tokens']) != row['tokens']:
            raise ValueError('Recomputed delivery differs from frozen retrieval run')
    output.mkdir(parents=True, exist_ok=True)
    for name in ('facts', 'fact_summary', 'marginal', 'minimum_budgets'):
        write_csv(output / f'{name}.csv', results[name])
    (output / 'manifest.json').write_text(json.dumps({
        'source_run': str(run), 'source_manifest_sha256': digest(run / 'manifest.json'),
        'annotations_sha256': digest(annotations_path), 'n': len(annotations),
        'status': 'exploratory_draft' if any(r['status'] == 'draft' for r in annotations) else 'reviewed',
        'method': manifest['retrieval']['method'],
        'note': 'Known annotated fact coverage; not a validated answer-accuracy measurement.',
        'source_sha256': {p.name: digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
        'output_sha256': {p.name: digest(p) for p in sorted(output.glob('*.csv'))}}, indent=2) + '\n')
    print(f'Scored {len(annotations)} annotations against frozen delivered contexts -> {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-draft', action='store_true')
    args = parser.parse_args()
    score(args.dataset, args.run, args.annotations, args.output, args.allow_draft)
