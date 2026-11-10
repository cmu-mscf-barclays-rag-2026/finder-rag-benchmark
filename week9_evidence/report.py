"""Summarize completed finance runs; never manufacture missing experiment results."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics

from week9_evidence.run import write_csv
from week9_evidence.vidore import digest, load_dataset


def rows(path):
    with path.open() as source:
        return list(csv.DictReader(source))


def select(data, k=10, before=0, after=0, budget=6000):
    return [r for r in data if all(int(r[key]) == value for key, value in
        [('k', k), ('before', before), ('after', after), ('budget', budget)])]


def paired(left, right, label):
    a, b = {r['query_id']: r for r in left}, {r['query_id']: r for r in right}
    if set(a) != set(b) or not a:
        raise ValueError('Paired comparison query IDs differ or are empty')
    result = []
    for metric in ('delivered_hit', 'delivered_recall', 'tokens'):
        deltas = [float(a[q][metric]) - float(b[q][metric]) for q in a]
        result.append({'comparison_left_minus_right': label, 'metric': metric, 'n': len(a),
            'mean_difference': statistics.mean(deltas), 'left_greater': sum(x > 0 for x in deltas),
            'right_greater': sum(x < 0 for x in deltas), 'equal': sum(x == 0 for x in deltas)})
    return result


def report(dataset, root, methods):
    _, queries, _, dataset_manifest = load_dataset(dataset)
    all_rows, manifests, summaries = {}, {}, {}
    pairs, strata, disagreements, pilot_types = [], [], [], []
    annotation_versions = set()
    for method in methods:
        folder = root / method
        manifest = json.loads((folder / 'manifest.json').read_text())
        if manifest['dataset'] != dataset_manifest or manifest['retrieval']['method'] != method:
            raise ValueError('Dataset or method mismatch')
        if method.startswith('qwen') and manifest['retrieval']['encoder'].get('tokenization') != 'tokenizer_truncation_preserve_special_tokens_v1':
            raise ValueError('Refusing Qwen results without the end-token preservation fix')
        if method.startswith('qwen'):
            check = json.loads((root / f'{method}_cpu_check.json').read_text())
            if not check['passed'] or check['vectors_sha256'] != manifest['output_sha256']['vectors.npz']:
                raise ValueError('Qwen results need a passing CPU check of the exact saved vectors')
        for name in ('per_query.csv', 'summary.csv'):
            if digest(folder / name) != manifest['output_sha256'][name]:
                raise ValueError('Result checksum mismatch')
        manifests[method] = manifest
        data = all_rows[method] = rows(folder / 'per_query.csv')
        summaries[method] = rows(folder / 'summary.csv')
        pairs.extend(paired(select(data, k=20), select(data), f'{method}: Top20 - Top10 at 6000'))
        pairs.extend(paired(select(data, before=2, after=1), select(data, k=20),
                            f'{method}: Top10 siblings2/1 - Top20 at 6000'))
        selected = select(data)
        for field in ('query_types', 'content_type'):
            groups = defaultdict(list)
            for r in selected:
                for label in queries[r['query_id']][field]:
                    groups[label].append(r)
            for label, group in sorted(groups.items()):
                strata.append({'method': method, 'field': field, 'label': label, 'n': len(group),
                    'k': 10, 'budget': 6000,
                    **{key: statistics.mean(float(r[key]) for r in group) for key in
                       ('ranked_hit', 'ranked_recall', 'ranked_ndcg', 'delivered_hit', 'delivered_recall', 'tokens')}})
        pilot = root / f'pilot_{method}'
        pilot_manifest = json.loads((pilot / 'manifest.json').read_text())
        if pilot_manifest['source_manifest_sha256'] != digest(folder / 'manifest.json'):
            raise ValueError('Pilot scores belong to a different retrieval run')
        annotation_versions.add(pilot_manifest['annotations_sha256'])
        if len(annotation_versions) != 1:
            raise ValueError('Methods were scored with different fact annotations')
        for name in ('facts.csv', 'fact_summary.csv'):
            if digest(pilot / name) != pilot_manifest['output_sha256'][name]:
                raise ValueError('Pilot output checksum mismatch')
        facts = rows(pilot / 'facts.csv')
        for r in select(facts, budget=4000):
            hit, recall, sufficient = float(r['delivered_hit']), float(r['delivered_recall']), int(r['sufficient'])
            if (hit == 1 and not sufficient) or (recall < 1 and sufficient):
                disagreements.append({'method': method, 'query_id': r['query_id'],
                    'question': queries[r['query_id']]['query'],
                    'case': 'hit_without_all_annotated_facts' if not sufficient else 'sufficient_despite_incomplete_page_recall',
                    'k': 10, 'budget': 4000, 'delivered_hit': hit, 'delivered_recall': recall,
                    'fact_recall': r['fact_recall'], 'sufficient': sufficient, 'status': r['status']})
        grouped = defaultdict(list)
        for r in select(facts, budget=4000):
            grouped[r['question_type']].append(r)
        for label, group in grouped.items():
            pilot_types.append({'method': method, 'question_type': label, 'n': len(group),
                **{key: statistics.mean(float(r[key]) for r in group)
                   for key in ('fact_recall', 'sufficient', 'facts_per_1k_tokens', 'tokens')}})
    if 'qwen256' in methods and 'minilm' in methods:
        pairs.extend(paired(select(all_rows['qwen256']), select(all_rows['minilm']),
                            'Qwen256 - MiniLM256 Top10 at 6000'))
    write_csv(root / 'paired_comparisons.csv', pairs)
    write_csv(root / 'query_and_content_types.csv', strata)
    write_csv(root / 'pilot_question_types.csv', pilot_types)
    write_csv(root / 'metric_disagreements.csv', disagreements)
    lines = ['# Week 9 finance findings — November 9, 2026', '',
        'Completed text-retrieval experiments on the full ViDoRe V3 Finance EN corpus:',
        '**309 English queries, 2,942 pages, six reports.** All methods search the same corpus.',
        'No test-set tuning or claims of statistical significance. The ten-question fact pilot is',
        '**assistant-drafted and unreviewed**, so its scores are exploratory annotation coverage.', '',
        '## Full benchmark: ranked pages', '',
        '| Method | Hit@5 | Hit@10 | Recall@10 | MRR@10 | nDCG@10 |',
        '|---|---:|---:|---:|---:|---:|']
    for method in methods:
        s = select(summaries[method])[0]
        s5 = select(summaries[method], k=5)[0]
        lines.append(f'| {method} | {float(s5["ranked_hit"]):.4f} | ' +
            ' | '.join(f'{float(s[key]):.4f}' for key in ('ranked_hit', 'ranked_recall', 'ranked_mrr', 'ranked_ndcg')) + ' |')
    lines += ['', '## Equal 6,000-token budgets', '',
        '| Method | Seed depth | Siblings before/after | Delivered Hit | Delivered Recall | Mean tokens |',
        '|---|---:|---|---:|---:|---:|']
    for method in methods:
        for k, before, after in ((10, 0, 0), (20, 0, 0), (10, 1, 1), (10, 2, 1)):
            s = select(summaries[method], k, before, after)[0]
            lines.append(f'| {method} | {k} | {before}/{after} | {float(s["delivered_hit"]):.4f} | '
                         f'{float(s["delivered_recall"]):.4f} | {float(s["tokens"]):.0f} |')
    lines += ['', 'All policies use whole-page skip-to-fit packing. Siblings are immediate PDF neighbors',
              'in the same document, deduplicated before token accounting. Ranked metrics describe',
              'seed retrieval; delivered metrics describe the entire context after expansion/packing.', '',
              '## Ten-question draft fact pilot: Top10, no siblings, 4,000 tokens', '',
              '| Method | Page Hit | Fact recall | All required facts covered | Facts / 1k tokens |',
              '|---|---:|---:|---:|---:|']
    for method in methods:
        s = select(rows(root / f'pilot_{method}' / 'fact_summary.csv'), budget=4000)[0]
        lines.append(f'| {method} | ' + ' | '.join(f'{float(s[key]):.4f}' for key in
            ('delivered_hit', 'fact_recall', 'sufficient', 'facts_per_1k_tokens')) + ' |')
    lines += ['', 'These are rubric-dependent scores, not validated answerability or answer accuracy.',
        'The pilot was selected after examining questions and sources; it is not a held-out evaluation.',
        'Unannotated alternative evidence can cause false negatives. See the',
        '[annotation protocol and issues](../annotations/PROTOCOL.md) before interpreting the numbers.', '',
        '## Cases where relevance and fact coverage disagree', '']
    examples = [r for method in methods for r in [d for d in disagreements if d['method'] == method][:4]]
    for r in examples:
        lines.append(f'- {r["method"]}, query {r["query_id"]}: {r["case"].replace("_", " ")}; '
                     f'page recall {float(r["delivered_recall"]):.3f}, fact recall {float(r["fact_recall"]):.3f}.')
    lines += ['', 'Full cases: [metric_disagreements.csv](metric_disagreements.csv).', '',
        '## Cost and measurement limits', '',
        '| Method | Local build / document encoding seconds | Query encoding seconds | Mean search ms |',
        '|---|---:|---:|---:|']
    for method in methods:
        info = manifests[method]['retrieval']
        build = info.get('build_seconds', info.get('build_seconds_including_model_load_or_cache'))
        lines.append(f'| {method} | {build:.2f} | {info.get("query_encode_seconds", 0):.2f} | '
                     f'{info["query_search_mean_ms"]:.3f} |')
    lines += ['', '- Timings are local observations, affected by cache state and concurrent work; they are not a controlled hardware speed ranking.',
        '- Qwen timing excludes the discarded invalid run and earlier interrupted work. The manifest records the corrected run and its cache hits; it is not the total development cost.',
        '- BM25 indexes full OCR pages. The dense models encode only their first 256 tokens: MiniLM truncates 2,764 of 2,942 pages (93.9%). A sparse-versus-dense difference therefore mixes retrieval method with visible input length. Do not generalize it to full-page dense retrieval.',
        '- Search timing excludes query embedding, model load, context packing and answer generation. Dense search uses one BLAS thread.',
        '- MiniLM and Qwen256 both use 256-token document encoder windows. Their tokenizers, pooling and query instructions differ, so 256 tokens need not cover identical characters. The 4,096-token Qwen option was not completed or scored.',
        '- The delivery ruler is the same pinned MiniLM tokenizer for every method and includes no reader prompt overhead.',
        '- OCR can lose table/layout details. This run does not measure visual or combined retrieval.',
        '- No HNSW, external embedding API, answer model, paid API usage, or human validation result is claimed.',
        '- Published relevance labels and questions are retained for the full run, including flagged entity/answer-leakage issues. See the protocol for examples.', '',
        '## Qwen preprocessing validation', '',
        'An initial Qwen run was invalid: slicing a tokenized page removed its final end-of-text',
        'token when truncation occurred, while short queries retained it. Week 9 now delegates',
        'truncation to the tokenizer, asserts preservation of the actual special-token suffix,',
        'and uses a distinct cache identity. The corrected six-input MPS smoke test agreed with',
        'independent single-input CPU encodings above cosine 0.9999999. The invalid run is isolated',
        'under `invalid_qwen256/` and excluded from every comparison above.', '',
        'See [the failed diagnostic](invalid_qwen_cpu_check.json) and',
        '[the corrected smoke check](qwen_smoke_check.json). Each reported Qwen run also has',
        'a `<method>_cpu_check.json` tied to its exact saved vector file.', '',
        '## Next experiment', '',
        'Have a second annotator review the ten-question pilot against PDFs, expand alternative support,',
        'and adjudicate the scope of conditional/rate questions. Freeze that version before comparing',
        'sufficiency on new questions. Visual retrieval and HNSW remain separately scoped follow-ups.', '']
    (root / 'findings.md').write_text('\n'.join(lines))
    (root / 'report_manifest.json').write_text(json.dumps({'methods': methods,
        'source_manifests': {m: digest(root / m / 'manifest.json') for m in methods},
        'pilot_manifests': {m: digest(root / f'pilot_{m}' / 'manifest.json') for m in methods},
        'report_source_sha256': digest(Path(__file__))}, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--methods', nargs='+', default=['bm25', 'minilm', 'qwen256'])
    args = parser.parse_args()
    report(args.dataset, args.results, args.methods)
