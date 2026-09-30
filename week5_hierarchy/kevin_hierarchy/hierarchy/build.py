"""Build both corpora, audit hierarchy boundaries, and measure exact model tokens."""
import argparse
import hashlib
import json
import math
import platform
from collections import Counter
from statistics import mean

from .core import build_hierarchy
from .io import ROOT, DATA_REVISION, SOURCE_HASHES, digest, prepare_source, load_tokenizer, write_json, write_jsonl, write_csv


def stats(values):
    values = sorted(values)
    def quantile(q):
        position = (len(values) - 1) * q
        a, b = math.floor(position), math.ceil(position)
        return values[a] + (values[b] - values[a]) * (position - a)
    return {'count': len(values), 'min': min(values), 'mean': mean(values),
            'median': quantile(.5), 'p95': quantile(.95), 'max': max(values)}


def run(source_dir=None):
    source, questions = prepare_source(source_dir)
    children, parents, mapping, audit, boundaries = build_hierarchy(source)
    child_lookup = {row['id']: row for row in children}
    parent_lookup = {row['id']: row for row in parents}
    for row in mapping:
        parent = parent_lookup[row['parent_id']]
        assert parent['text'][row['start_char']:row['end_char']] == child_lookup[row['child_id']]['text']
    config = json.loads((ROOT / 'config.json').read_text())
    # Match the prior Part A gold-passage-group split exactly.
    gold_groups = sorted({q['relevant_passage_id'] for q in questions}, key=lambda pid:
                         hashlib.sha256((config['split_seed'] + '|' + pid).encode()).hexdigest())
    dev_gold = set(gold_groups[:max(1, round(len(gold_groups) * config['dev_group_fraction']))])
    mapped = []
    for q in questions:
        gold = q['relevant_passage_id']
        if gold not in child_lookup:
            raise ValueError(f'Missing gold child {gold}')
        mapped.append({'id': str(q['id']), 'question': q['question'], 'answer': q['answer'],
                       'relevant_passage_id': gold, 'gold_ids': [gold],
                       'gold_child_ids': [gold], 'gold_parent_ids': [child_lookup[gold]['parent_id']],
                       'split': 'dev' if gold in dev_gold else 'test', 'split_id': config['team_split_id']})
    if len({q['id'] for q in mapped}) != len(mapped):
        raise ValueError('Duplicate query IDs.')
    tokenizer, tokenizer_meta = load_tokenizer()
    limit = tokenizer_meta['sentence_transformer_config']['max_seq_length']
    lengths = []
    for variant, records in (('child', children), ('parent', parents)):
        for record in records:
            content = len(tokenizer.encode(record['text'], add_special_tokens=False).ids)
            with_specials = len(tokenizer.encode(record['text'], add_special_tokens=True).ids)
            lengths.append({'variant': variant, 'id': record['id'], 'section_id': record['section_id'],
                            'words': len(record['text'].split()), 'characters': len(record['text']),
                            'content_tokens': content, 'tokens_with_specials': with_specials,
                            'over_model_limit': with_specials > limit,
                            'n_children': 1 if variant == 'child' else len(record['child_ids'])})
    token_summary = []
    for variant in ('child', 'parent'):
        selected = [row for row in lengths if row['variant'] == variant]
        token_summary.append({'variant': variant, **stats([row['content_tokens'] for row in selected]),
                              'model_limit_including_specials': limit,
                              'over_model_limit_count': sum(row['over_model_limit'] for row in selected),
                              'over_model_limit_fraction': mean(row['over_model_limit'] for row in selected)})
    summary = {'source_revision': DATA_REVISION, 'n_children': len(children), 'n_parents': len(parents),
               'n_sections': len({row['section_id'] for row in children}), 'n_queries': len(mapped),
               'children_per_parent': stats([len(row['child_ids']) for row in parents]),
               'singleton_parents': sum(len(row['child_ids']) == 1 for row in parents),
               'adjacent_boundaries_checked': len(boundaries),
               'overlap_review_flags': sum(row['review_flag'] for row in boundaries),
               'boundaries_with_any_normalized_word_overlap': sum(row['normalized_overlap_words'] > 0 for row in boundaries),
               'identical_adjacent_bodies': sum(row['identical_text'] for row in boundaries),
               'repeated_adjacent_footnotes': sum(row['same_nonempty_footnotes'] for row in boundaries),
               'parents_reordered': sum(not row['source_order_was_numeric'] for row in audit),
               'all_spans_round_trip': True, 'split_counts': dict(Counter(q['split'] for q in mapped)),
               'shared_gold_parent_ids_between_dev_test': sorted(
                   {pid for q in mapped if q['split'] == 'dev' for pid in q['gold_parent_ids']} &
                   {pid for q in mapped if q['split'] == 'test' for pid in q['gold_parent_ids']}),
               'tokenizer': tokenizer_meta, 'tokens': token_summary,
               'method': 'ID-derived c-parent reconstruction; exact child text joined with two newlines; no deduplication'}
    data, results = ROOT / 'data', ROOT / 'results'
    write_jsonl(data / 'children.jsonl', children)
    write_jsonl(data / 'parents.jsonl', parents)
    write_jsonl(data / 'queries.jsonl', mapped)
    write_json(data / 'child_to_parent.json', {row['child_id']: row['parent_id'] for row in mapping})
    write_json(data / 'parent_to_children.json', {row['id']: row['child_ids'] for row in parents})
    write_csv(data / 'child_spans.csv', mapping)
    write_csv(results / 'parent_audit.csv', audit)
    write_csv(results / 'boundary_overlap.csv', boundaries)
    write_csv(results / 'unit_lengths.csv', lengths)
    write_csv(results / 'token_summary.csv', token_summary)
    write_json(results / 'summary.json', summary)
    examples = ['1.1-c1', '1.2-c2', max(parents, key=lambda row: len(row['child_ids']))['id']]
    write_json(results / 'hierarchy_examples.json', [parent_lookup[pid] for pid in examples])
    write_json(results / 'build_manifest.json', {'source_sha256': SOURCE_HASHES, 'config': config,
               'tokenizer': tokenizer_meta, 'python': platform.python_version(),
               'outputs_sha256': {str(p.relative_to(ROOT)): digest(p) for p in sorted(data.glob('*'))},
               'code_sha256': {p.name: digest(p) for p in sorted((ROOT / 'hierarchy').glob('*.py'))}})
    print(json.dumps({key: summary[key] for key in ('n_children', 'n_parents', 'n_sections',
                      'overlap_review_flags', 'parents_reordered', 'tokens',
                      'shared_gold_parent_ids_between_dev_test')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', help='Optional existing pinned corpus.jsonl and qa.jsonl directory')
    args = parser.parse_args()
    run(args.source_dir)
