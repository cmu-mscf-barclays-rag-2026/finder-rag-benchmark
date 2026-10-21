"""Development-only Qwen/BM25 tuning and frozen context-budget evaluation.

Run from the repository root with .venv-week6/bin/python week7_qwen_tuning.py.
Only saved Qwen rankings are used; no encoder weights are loaded.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from week6_qwen_hierarchy.qwen_hierarchy import (
    evaluation_tokenizer, evaluate,
)
from week5_hierarchy.hierarchy_experiment import ROOT, load_data
from week4_legal_rag.run import fuse
from finder_bm25.bm25 import BM25Retriever
from hierarchy.handoff import assemble_context
from hierarchy.core import SEPARATOR
from hierarchy.io import digest, read_jsonl, write_csv, write_json

SOURCE = ROOT / 'week6_qwen_hierarchy/results/qwen'
OUTPUT = ROOT / 'week7_qwen_tuning/results'
WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
BUDGETS = (2048, 4000, 6000, 8000)
MODES = ('child', 'parent', 'child_to_parent', 'child_to_selected_siblings')


def saved_rankings(split, queries, window):
    rows = read_jsonl(SOURCE / 'rankings' / f'{split}_qwen_{window}_dense.jsonl')
    rankings = {r['query_id']: {'child': r['child'], 'parent': r['parent']} for r in rows}
    if set(rankings) != {q['id'] for q in queries}:
        raise ValueError(f'{split} dense ranking IDs differ from the frozen split')
    return rankings


def bm25_rankings(queries, children, parents, config, split):
    result = {q['id']: {} for q in queries}
    for unit, corpus in (('child', children), ('parent', parents)):
        records = sorted(corpus.values(), key=lambda r: r['id'])
        retriever = BM25Retriever([{'doc_id': r['id'], 'text': r['text']} for r in records], **config['bm25'])
        for q in queries:
            result[q['id']][unit] = [h['doc_id'] for h in retriever.search(q['question'], config['candidate_depth'])]
        if split == 'test':
            expected = {r['query_id']: r['passage_ids'] for r in read_jsonl(
                ROOT / 'week5_hierarchy/results/comparison/rankings' / f'bm25_{unit}.jsonl')}
            if any(result[qid][unit] != ids for qid, ids in expected.items()):
                raise ValueError(f'{unit} BM25 ranking differs from the saved test reference')
    return result


def combine(bm25, dense, weight, c):
    if weight == 0:
        return dense
    if weight == 1:
        return bm25
    result = {}
    for qid in dense:
        result[qid] = {}
        for unit in ('child', 'parent'):
            sparse = [{'doc_id': pid} for pid in bm25[qid][unit]]
            embedded = [{'doc_id': pid} for pid in dense[qid][unit]]
            result[qid][unit] = [h['doc_id'] for h in fuse(sparse, embedded, weight, c)]
    return result


def score(queries, rankings, children, parents, tokenizer, weight, modes=MODES, budgets=BUDGETS):
    config = {'modes': list(modes), 'context_budgets': list(budgets), 'ks': [5, 10]}
    return evaluate(queries, rankings, children, parents, tokenizer, config,
                    {'bm25_weight': weight, 'method': 'Qwen dense' if weight == 0 else
                     ('BM25' if weight == 1 else 'BM25/Qwen RRF')})


def select(summary):
    """One BM25 weight per budget and hierarchy mode, using development only."""
    selected = []
    for budget in BUDGETS:
        for mode in MODES:
            candidates = [r for r in summary if r['context_budget'] == budget and
                          r['mode'] == mode and r['k'] == 10]
            winner = max(candidates, key=lambda r: (
                r['hit_count'], r['gold_text_coverage'], r['mean_coverage_efficiency'],
                -r['mean_context_tokens'], -r['bm25_weight']))
            selected.append(winner)
    return selected


def parent_evidence_position(ranking, gold, children, parents, tokenizer):
    """One-indexed token offsets of gold children within ten assembled parents."""
    units = list(dict.fromkeys(ranking[:10]))
    text = SEPARATOR.join(parents[pid]['text'] for pid in units)
    spans = []
    cursor = 0
    for parent_id in units:
        parent = parents[parent_id]
        for span in parent['child_spans']:
            if span['child_id'] in gold:
                start = cursor + span['start_char']
                end = cursor + span['end_char']
                spans.append({'gold_child_id': span['child_id'], 'parent_id': parent_id,
                              'start_token': len(tokenizer.encode(text[:start]).ids) + 1,
                              'end_token': len(tokenizer.encode(text[:end]).ids)})
        cursor += len(parent['text']) + len(SEPARATOR)
    return spans


def analyze_parent_wins(queries, by_weight, selected, children, parents, tokenizer):
    out = []
    for choice in selected:
        if choice['mode'] != 'parent':
            continue
        budget, weight = choice['context_budget'], choice['bm25_weight']
        rankings = by_weight[weight]
        for q in queries:
            ranking = rankings[q['id']]
            parent_ctx = assemble_context(ranking['parent'], 'parent', children, parents, tokenizer, 10, budget)
            expansion_ctx = assemble_context(ranking['child'], 'child_to_parent', children, parents, tokenizer, 10, budget)
            gold = set(q['gold_child_ids'])
            if not (gold & set(parent_ctx['fully_included_child_ids'])) or gold & set(expansion_ctx['fully_included_child_ids']):
                continue
            parent_units = ranking['parent'][:10]
            expansion_units = list(dict.fromkeys(children[cid]['parent_id'] for cid in ranking['child'][:10]))
            gold_parents = set(q['gold_parent_ids'])
            direct_pos = parent_evidence_position(parent_units, gold, children, parents, tokenizer)
            expansion_pos = parent_evidence_position(expansion_units, gold, children, parents, tokenizer)
            reason = ('gold parent not seeded by top-10 children' if not gold_parents.intersection(expansion_units)
                      else 'gold child after expansion context cutoff')
            out.append({'budget': budget, 'bm25_weight': weight, 'query_id': q['id'],
                        'question': q['question'], 'gold_child_ids': json.dumps(q['gold_child_ids']),
                        'gold_parent_ids': json.dumps(q['gold_parent_ids']),
                        'direct_parent_top10': json.dumps(parent_units),
                        'expanded_parent_top10': json.dumps(expansion_units),
                        'direct_gold_parent_rank': min((i for i, pid in enumerate(parent_units, 1) if pid in gold_parents), default=''),
                        'expanded_gold_parent_rank': min((i for i, pid in enumerate(expansion_units, 1) if pid in gold_parents), default=''),
                        'best_gold_child_rank': min((i for i, cid in enumerate(ranking['child'], 1) if cid in gold), default=''),
                        'direct_gold_positions': json.dumps(direct_pos),
                        'expansion_gold_positions': json.dumps(expansion_pos),
                        'direct_returned_tokens': parent_ctx['returned_content_tokens'],
                        'expansion_returned_tokens': expansion_ctx['returned_content_tokens'],
                        'reason': reason})
    return out


def report(dev, frozen, selected, test, errors, window):
    lines = ['# Qwen hybrid tuning and context-budget curve', '',
             f'Qwen document window: {window:,} tokens (frozen Week 6 development choice). '
             'Budgets below use the pinned, untruncated MiniLM measurement tokenizer. '
             'The Qwen and BM25 top-100 rankings were reused or reconstructed; no embeddings were rerun.', '',
             '![Frozen test context-budget curve](context_budget_curve.svg)', '',
             'Selection used only 20 development questions. For each budget and hierarchy mode, '
             'the BM25 weight among 0, .25, .5, .75, and 1 was chosen by Hit@10, then '
             'gold-text coverage, coverage per token, fewer returned tokens, and lower BM25 weight. '
             'The four selected mode configurations were then evaluated once on 80 test questions.', '',
             '## Development-selected strategy by budget', '',
             '| Budget | Mode | BM25 weight | Dev Hit@10 | Test Hit@5 | Test Hit@10 | Test coverage | Mean tokens | Test coverage / 1k |',
             '|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for budget in BUDGETS:
        candidates = [r for r in selected if r['context_budget'] == budget]
        winner = max(candidates, key=lambda r: (
            r['hit_count'], r['gold_text_coverage'], r['mean_coverage_efficiency'],
            -r['mean_context_tokens'], -MODES.index(r['mode'])))
        t5 = next(r for r in frozen if r['context_budget'] == budget and r['mode'] == winner['mode'] and
                  r['bm25_weight'] == winner['bm25_weight'] and r['k'] == 5)
        t10 = next(r for r in frozen if r['context_budget'] == budget and r['mode'] == winner['mode'] and
                   r['bm25_weight'] == winner['bm25_weight'] and r['k'] == 10)
        lines.append(f"| {budget:,} | {winner['mode']} | {winner['bm25_weight']:.2f} | "
                     f"{winner['hit_count']}/20 | {t5['hit_count']}/80 | {t10['hit_count']}/80 | "
                     f"{t10['gold_text_coverage']:.1%} | {t10['mean_context_tokens']:.0f} | "
                     f"{1000*t10['mean_coverage_efficiency']:.3f} |")
    lines += ['', '## All frozen mode configurations on test', '',
              '| Budget | Mode | BM25 weight | Hit@5 | Hit@10 | Coverage | Mean tokens | Coverage / 1k |',
              '|---:|---|---:|---:|---:|---:|---:|---:|']
    for choice in selected:
        budget, mode, weight = choice['context_budget'], choice['mode'], choice['bm25_weight']
        t5, t10 = [next(r for r in frozen if r['context_budget'] == budget and r['mode'] == mode and
                         r['bm25_weight'] == weight and r['k'] == k) for k in (5, 10)]
        lines.append(f"| {budget:,} | {mode} | {weight:.2f} | {t5['hit_count']}/80 | "
                     f"{t10['hit_count']}/80 | {t10['gold_text_coverage']:.1%} | "
                     f"{t10['mean_context_tokens']:.0f} | {1000*t10['mean_coverage_efficiency']:.3f} |")
    lines += ['', '## Qwen dense-only comparator', '',
              '| Budget | Mode | Hit@5 | Hit@10 | Coverage | Mean tokens | Coverage / 1k |',
              '|---:|---|---:|---:|---:|---:|---:|']
    for budget in BUDGETS:
        for mode in MODES:
            a, b = [next(r for r in test if r['bm25_weight'] == 0 and r['context_budget'] == budget and
                         r['mode'] == mode and r['k'] == k) for k in (5, 10)]
            lines.append(f"| {budget:,} | {mode} | {a['hit_count']}/80 | {b['hit_count']}/80 | "
                         f"{b['gold_text_coverage']:.1%} | {b['mean_context_tokens']:.0f} | "
                         f"{1000*b['mean_coverage_efficiency']:.3f} |")
    lines += ['', 'The hybrid weights selected for direct parents at 4,000 and 8,000 tokens '
              'score 44/80 and 49/80 on test, versus 48/80 and 54/80 for Qwen dense alone. '
              'The 0.75 BM25 weight selected for child-to-parent at 2,048 tokens scores '
              '19/80, versus 33/80 for dense. These reversals show how unstable tuning '
              'can be with only 20 development questions; the frozen results do not '
              'support a held-out hybrid advantage.']
    lines += ['', '## Direct-parent wins over child-to-parent expansion', '',
              'This diagnostic compares both modes at the development-selected **direct-parent weight** '
              'for each budget. It uses test questions only after the weight was frozen. '
              'The full per-question parent lists, ranks, gold-child token positions, and cutoff '
              'causes are in `parent_wins.csv`.', '']
    for budget in BUDGETS:
        subset = [r for r in errors if r['budget'] == budget]
        unseeded = sum(r['reason'] == 'gold parent not seeded by top-10 children' for r in subset)
        lines.append(f'- {budget:,}: {len(subset)} direct-parent-only hits; '
                     f'{unseeded} with gold parents absent from expanded top-10 seeds, '
                     f'{len(subset)-unseeded} present but beyond the context cutoff.')
    lines += ['', 'For example, at 4,000 tokens, question 20 puts gold child '
              '`4.6-c4-s1` at tokens 1–397 in the directly retrieved parent. '
              'The expanded ranking places the same child at tokens 5,948–6,344, '
              'past the cutoff. Question 6 shows a ranking failure: direct parent '
              '`1.7-c3` ranks first, but none of the ten child seeds belongs to it. '
              'These examples are diagnostic observations, not additional tuning.']
    lines += ['', 'Coverage per 1k is the mean of per-question coverage fractions divided by '
              'returned tokens, multiplied by 1,000. Hit@k requires a complete gold child in '
              'delivered context; coverage includes partial gold text. The 20-question development '
              'set and 80-question test set are small and drawn from one legal corpus. '
              'Differences among close methods are descriptive.', '']
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=OUTPUT)
    args = p.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        p.error('Output must be new or empty.')
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((SOURCE / 'manifest.json').read_text())
    if manifest['status'] != 'complete' or manifest.get('reference_only'):
        raise ValueError('Saved Qwen run is incomplete.')
    config = manifest['config']
    window = json.loads((SOURCE / 'selected.json').read_text())['embedding_window']
    children, parents, queries, build = load_data(ROOT / 'week5_hierarchy/kevin_hierarchy')
    dev, test = [[q for q in queries if q['split'] == split] for split in ('dev', 'test')]
    if (len(dev), len(test)) != (20, 80):
        raise ValueError('Frozen split changed.')
    tokenizer, tokenizer_hash = evaluation_tokenizer(config, ROOT / '.cache/models', True)
    if tokenizer_hash != manifest['measurement_tokenizer_sha256']:
        raise ValueError('Measurement tokenizer changed.')
    all_rankings = {}
    dense = saved_rankings('dev', dev, window)
    bm25 = bm25_rankings(dev, children, parents, config, 'dev')
    all_rankings['dev'] = {weight: combine(bm25, dense, weight, config['rrf_constant']) for weight in WEIGHTS}
    saved_hybrid = {r['query_id']: {'child': r['child'], 'parent': r['parent']} for r in
                    read_jsonl(SOURCE / 'rankings' / f'dev_qwen_{window}_hybrid.jsonl')}
    if all_rankings['dev'][0.75] != saved_hybrid:
        raise ValueError('Development reconstructed 0.75 fusion differs from saved Qwen hybrid.')
    dev_details, dev_summary = [], []
    for weight in WEIGHTS:
        print(f'Development: BM25 weight {weight:.2f}', flush=True)
        detail, summary = score(dev, all_rankings['dev'][weight], children, parents, tokenizer, weight)
        dev_details.extend(detail)
        dev_summary.extend(summary)
    selected = select(dev_summary)
    write_json(args.output / 'selected.json', [{'budget': r['context_budget'], 'mode': r['mode'],
        'bm25_weight': r['bm25_weight'], 'dev_hit10': r['hit_count'],
        'dev_coverage': r['gold_text_coverage']} for r in selected])
    # Access the saved test rankings only after development selection is recorded.
    dense = saved_rankings('test', test, window)
    bm25 = bm25_rankings(test, children, parents, config, 'test')
    all_rankings['test'] = {weight: combine(bm25, dense, weight, config['rrf_constant']) for weight in WEIGHTS}
    saved_hybrid = {r['query_id']: {'child': r['child'], 'parent': r['parent']} for r in
                    read_jsonl(SOURCE / 'rankings' / f'test_qwen_{window}_hybrid.jsonl')}
    if all_rankings['test'][0.75] != saved_hybrid:
        raise ValueError('Test reconstructed 0.75 fusion differs from saved Qwen hybrid.')
    test_details, test_summary = [], []
    for weight in sorted({0.0, *(r['bm25_weight'] for r in selected if r['bm25_weight'] != 0)}):
        choices = [r for r in selected if r['bm25_weight'] == weight]
        if weight == 0:
            choices = [{'mode': mode, 'context_budget': budget} for mode in MODES for budget in BUDGETS]
        for choice in choices:
            print(f"Test: weight {weight:.2f}, {choice['mode']}, {choice['context_budget']}", flush=True)
            detail, summary = score(test, all_rankings['test'][weight], children, parents, tokenizer,
                                    weight, (choice['mode'],), (choice['context_budget'],))
            test_details.extend(detail)
            test_summary.extend(summary)
    frozen = [r for r in test_summary if any(r['context_budget'] == s['context_budget'] and
              r['mode'] == s['mode'] and r['bm25_weight'] == s['bm25_weight'] for s in selected)]
    errors = analyze_parent_wins(test, all_rankings['test'], selected, children, parents, tokenizer)
    write_csv(args.output / 'dev_sweep.csv', dev_summary)
    write_csv(args.output / 'test_frozen.csv', frozen)
    write_csv(args.output / 'test_dense.csv', [r for r in test_summary if r['bm25_weight'] == 0])
    write_csv(args.output / 'test_per_query.csv', test_details)
    if errors:
        write_csv(args.output / 'parent_wins.csv', errors)
    (args.output / 'findings.md').write_text(report(dev_summary, frozen, selected, test_summary,
                                                   errors, window), encoding='utf-8')
    write_json(args.output / 'manifest.json', {'status': 'complete', 'input_sha256': build['outputs_sha256'],
        'source_manifest_sha256': digest(SOURCE / 'manifest.json'),
        'qwen_dense_dev_sha256': digest(SOURCE / 'rankings' / f'dev_qwen_{window}_dense.jsonl'),
        'qwen_dense_test_sha256': digest(SOURCE / 'rankings' / f'test_qwen_{window}_dense.jsonl'),
        'measurement_tokenizer_sha256': tokenizer_hash, 'embedding_window': window,
        'weights': WEIGHTS, 'budgets': BUDGETS, 'source_code_sha256': digest(Path(__file__))})
    print(f'Wrote {args.output / "findings.md"}', flush=True)


if __name__ == '__main__':
    main()
