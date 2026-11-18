"""Build one cross-study presentation table from checked result artifacts.

The selected rows are headline cutoffs. Scores remain tied to their own
dataset, query subset, metric, and baseline; no cross-dataset average is made.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'week10_followup/results/comparison'
YUCHEN_COMMIT = '8738b543b8770a29acb5d82c481b1a67fe530140'
OUTPUT = ROOT / 'outputs/meeting10_presentation_20261118'


def csv_rows(path):
    return list(csv.DictReader(path.open()))


def exact(rows, **criteria):
    matches = [row for row in rows if all(str(row[key]) == str(value)
                                         for key, value in criteria.items())]
    if len(matches) != 1:
        raise ValueError(f'Expected one row for {criteria}; found {len(matches)}')
    return matches[0]


def fmt_score(value):
    return f'{value:.1%}'


def fmt_delta(value):
    return '—' if value is None else f'{value * 100:+.1f} pp'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--data-json', type=Path,
                        default=Path('/private/tmp/meeting10_presentation_data.json'))
    args = parser.parse_args()
    combined = csv_rows(RESULTS / 'comparison.csv')
    kevin = csv_rows(RESULTS / 'team_vidore_query_expansion.csv')
    team = csv_rows(RESULTS / 'team_method_comparison.csv')
    manifest = json.loads((RESULTS / 'manifest.json').read_text())
    if manifest['committed_vidore_source']['commit'] != '8df85985ea1ea3f3b7e7508d7390f2970f4dab6c':
        raise ValueError('Kevin branch source changed; review the selection')
    if manifest['committed_team_source']['commit'] != '18bf882b612b5fa3c0d83a4b861603fd8b424446':
        raise ValueError('Legal team source changed; review the selection')
    rows = []

    def add(study, n, setting, method, metric, score, control='', delta=None,
            secondary='', secondary_score=None):
        score = float(score)
        if not 0 <= score <= 1 or (secondary_score is not None and
                                    not 0 <= float(secondary_score) <= 1):
            raise ValueError(f'Invalid percentage metric: {study} {method}')
        rows.append({'study': study, 'n': int(n), 'setting': setting,
                     'method': method, 'metric': metric, 'score': score,
                     'control': control, 'delta': None if delta is None else float(delta),
                     'secondary': secondary,
                     'secondary_score': None if secondary_score is None else float(secondary_score)})

    # Florence: matched FinDER query-expansion experiment, Top-10 headline.
    finder = {method: exact(combined, dataset='FinDER', method=method, k=10,
                            query_count=200)
              for method in ('single_query_bm25', 'multiquery_qwen', 'multiquery_gpt',
                             'dspy_qwen', 'dspy_gpt')}
    baseline = float(finder['single_query_bm25']['hit'])
    for method, label in (('single_query_bm25', 'BM25 original'),
                          ('multiquery_qwen', 'MultiQuery · Qwen'),
                          ('multiquery_gpt', 'MultiQuery · GPT-4o mini'),
                          ('dspy_qwen', 'DSPy · Qwen'),
                          ('dspy_gpt', 'DSPy · GPT-4o mini')):
        row = finder[method]
        add('FinDER · Florence', 200, 'Ranked passages · K=10', label,
            'Hit@10', row['hit'], '' if method == 'single_query_bm25' else 'BM25 original',
            None if method == 'single_query_bm25' else float(row['hit']) - baseline,
            'nDCG@10', row['ndcg'])

    # Florence: page delivery under two budgets; window delta only compares
    # the same Qwen encoder, query vectors, and budget.
    for budget in (6000, 12000):
        qwen_short = exact(combined, dataset='ViDoRe_finance_en',
                           method='qwen256_neighbors_0/0', k=10, budget=budget,
                           query_count=309)
        for method, label in (('bm25_neighbors_0/0', 'BM25 OCR'),
                              ('minilm_neighbors_0/0', 'MiniLM 256'),
                              ('qwen256_neighbors_0/0', 'Qwen 256'),
                              ('qwen1024_neighbors_0/0', 'Qwen 1024')):
            row = exact(combined, dataset='ViDoRe_finance_en', method=method,
                        k=10, budget=budget, query_count=309)
            is_window = method == 'qwen1024_neighbors_0/0'
            add('ViDoRe · Florence', 309,
                f'Delivered OCR pages · K=10 · {budget // 1000}k tokens',
                label, 'Hit@10', row['hit'],
                'Qwen 256, same budget' if is_window else '',
                float(row['hit']) - float(qwen_short['hit']) if is_window else None,
                'Recall', row['recall'])

    # FiQA: candidate-depth/budget context, not query-expansion comparisons.
    for k, budget in ((10, 6000), (20, 6000), (20, 12000)):
        row = exact(combined, dataset='FiQA', method='bm25', k=k,
                    budget=budget, query_count=648)
        add('FiQA · Florence', 648, f'Delivered answer passages · K={k} · {budget // 1000}k tokens',
            'BM25 original', f'Hit@{k}', row['hit'], secondary='Recall',
            secondary_score=row['recall'])

    # Kevin: GPT-generated query expansion over four matched page backends.
    for backend, label in (('bm25', 'BM25 OCR'), ('ocr', 'Dense OCR'),
                           ('image', 'Dense image'), ('combined', 'OCR/image fusion')):
        base = exact(kevin, backend=backend, method='original', k=10, queries=247)
        for method, method_label in (('original', 'Original query'),
                                     ('multiquery', 'MultiQuery · GPT'),
                                     ('dspy_initial', 'DSPy initial · GPT'),
                                     ('dspy_optimized', 'DSPy optimized · GPT')):
            row = exact(kevin, backend=backend, method=method, k=10, queries=247)
            delta = None if method == 'original' else float(row['ndcg']) - float(base['ndcg'])
            if delta is not None and abs(delta - float(row['delta_ndcg'])) > 1e-10:
                raise ValueError('Kevin backend delta does not reconcile')
            add('ViDoRe · Kevin', 247, f'Ranked pages · K=10 · {label}',
                method_label, 'nDCG@10', row['ndcg'],
                '' if method == 'original' else 'Original, same backend', delta,
                'Hit@10', row['hit'])

    # Legal: show retrieval and reader outcomes separately.
    legal_hit = {method: exact(team, dataset='Legal RAG Bench', method=method,
                               metric='passage_hit_at_5', n=80)
                 for method in ('BM25 + fixed reader', 'MultiQuery + fixed reader')}
    hit_base = float(legal_hit['BM25 + fixed reader']['score'])
    for method, label in (('BM25 + fixed reader', 'BM25 original'),
                          ('MultiQuery + fixed reader', 'MultiQuery · GPT')):
        row = legal_hit[method]
        add('Legal · team branch', 80, 'Ranked passages · K=5', label,
            'Hit@5', row['score'], '' if label == 'BM25 original' else 'BM25 original',
            None if label == 'BM25 original' else float(row['score']) - hit_base)
    legal_f1 = {method: exact(team, dataset='Legal RAG Bench', method=method,
                              metric='reference_token_f1_lexical_proxy', n=80)
                for method in ('BM25 + fixed reader', 'MultiQuery + fixed reader',
                               'BM25 + DSPy CoT, optimized')}
    f1_base = float(legal_f1['BM25 + fixed reader']['score'])
    for method, label in (('BM25 + fixed reader', 'BM25 + fixed reader'),
                          ('MultiQuery + fixed reader', 'MultiQuery + fixed reader'),
                          ('BM25 + DSPy CoT, optimized', 'BM25 + DSPy reader')):
        row = legal_f1[method]
        add('Legal · team branch', 80, 'Answer · Top-8 · 2,400 words', label,
            'Lexical token F1', row['score'],
            '' if method == 'BM25 + fixed reader' else 'BM25 + fixed reader',
            None if method == 'BM25 + fixed reader' else float(row['score']) - f1_base)

    # Yuchen: include Graph RAG from the branch's original accuracy file.
    blob = subprocess.run(['git', 'show',
                           f'{YUCHEN_COMMIT}:results/longbench_v2_methods/accuracy.csv'],
                          cwd=ROOT, capture_output=True, check=True).stdout
    accuracy = exact(list(csv.DictReader(io.StringIO(blob.decode()))), group='all', n=402)
    longbench_team = {method: exact(team, dataset='LongBench v2', method=method,
                                    metric='multiple_choice_answer_accuracy', n=402)
                      for method in ('BM25 + fixed reader', 'MultiQuery + fixed reader',
                                     'BM25 + DSPy CoT, optimized')}
    for method, source_key in (('BM25 + fixed reader', 'bm25'),
                               ('MultiQuery + fixed reader', 'multiquery'),
                               ('BM25 + DSPy CoT, optimized', 'dspy')):
        if abs(float(longbench_team[method]['score']) - float(accuracy[source_key])) > 1e-10:
            raise ValueError('LongBench branch and team table differ')
    long_base = float(accuracy['bm25'])
    for key, label in (('bm25', 'BM25 + fixed reader'),
                       ('graph', 'Graph RAG + fixed reader'),
                       ('multiquery', 'MultiQuery + fixed reader'),
                       ('dspy', 'BM25 + DSPy reader')):
        add('LongBench v2 · Yuchen', 402, 'Answer · Top-8 · 2,400 words', label,
            'MC accuracy', accuracy[key],
            '' if key == 'bm25' else 'BM25 + fixed reader',
            None if key == 'bm25' else float(accuracy[key]) - long_base)

    if len(rows) != 41:
        raise ValueError(f'Expected 41 presentation rows, found {len(rows)}')
    args.output.mkdir(parents=True, exist_ok=True)
    args.data_json.write_text(json.dumps({'rows': rows, 'source_commits': {
        'kevinzhang': manifest['committed_vidore_source']['commit'],
        'feature/person3-hybrid': manifest['committed_team_source']['commit'],
        'yuchenlu': YUCHEN_COMMIT}}, indent=2) + '\n')
    lines = [
        '# Cross-dataset results for presentation', '',
        'Meeting 10 follow-up · 18 November 2026 · 41 selected headline rows', '',
        'Read each change against the **control in its own row**. The datasets use different '
        'questions, retrieval units, and outcome metrics.', '',
        '| Study | Test n | Setting | Method | Primary metric | Score | Compared with | Δ | Secondary |',
        '|---|---:|---|---|---|---:|---|---:|---|',
    ]
    for row in rows:
        secondary = (f"{row['secondary']} {fmt_score(row['secondary_score'])}"
                     if row['secondary_score'] is not None else '—')
        lines.append(f"| {row['study']} | {row['n']} | {row['setting']} | {row['method']} | "
                     f"{row['metric']} | {fmt_score(row['score'])} | "
                     f"{row['control'] or '—'} | {fmt_delta(row['delta'])} | {secondary} |")
    lines += [
        '', '## Presenter notes', '',
        '- FinDER uses 200 hash-selected test questions from 4,575; its scores are a pilot. '
        'The retrieval corpus is 5,830 pooled reference passages.',
        '- Florence’s ViDoRe rows score **delivered** OCR pages after context packing on 309 '
        'questions. Kevin’s rows score **ranked** pages on a different 247-question '
        'optimizer-held-out English split. Their raw scores should not be compared as a paired test.',
        '- Kevin’s nDCG changes use each backend’s original query as the control. '
        'Florence’s Qwen1024 changes use the same Qwen encoder at 256 page tokens and the same budget.',
        '- Legal DSPy and LongBench DSPy optimize the answer reader. FinDER and Kevin use DSPy '
        'for query rewriting. Legal answer F1 is a lexical proxy; LongBench is multiple-choice accuracy.',
        '- The table selects headline cutoffs. The underlying files retain other K values, '
        'budgets, paired intervals, and per-question detail.', '',
        '## Sources', '',
        '- Florence local: [combined results](../../week10_followup/results/comparison/comparison.csv), '
        '[FinDER details](../../week10_followup/results/finder_comparison/findings.md), and '
        '[Qwen-window details](../../week10_followup/results/qwen_window_compare/findings.md).',
        '- Kevin `kevinzhang@8df8598`: [64 English test rows]'
        '(../../week10_followup/results/comparison/team_vidore_query_expansion.csv), '
        'derived from `vidore_query_expansion/results/summary.csv` in his branch.',
        '- Legal `feature/person3-hybrid@18bf882`: [team metric rows]'
        '(../../week10_followup/results/comparison/team_method_comparison.csv).',
        '- LongBench `yuchenlu@8738b54`: `results/longbench_v2_methods/accuracy.csv` '
        'in that branch; the BM25, MultiQuery, and DSPy scores reconcile to the team metric rows.',
    ]
    (args.output / 'combined_results.md').write_text('\n'.join(lines) + '\n')
    print(f'Wrote {len(rows)} selected rows to {args.output / "combined_results.md"}')


if __name__ == '__main__':
    main()
