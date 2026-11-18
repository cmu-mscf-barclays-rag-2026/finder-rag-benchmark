"""Merge verified Legal and teammate LongBench aggregates without model calls."""
from pathlib import Path
import hashlib
import json
import math
import shutil

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LEGAL = ROOT / 'presentation/meeting10_legal_20261117'
REFERENCE = ROOT.parent / 'work/meeting10_20261117/team_reference'
if not REFERENCE.exists():
    REFERENCE = ROOT / 'presentation/meeting10_team_20261117/sources/longbench'
DOC = ROOT / 'presentation/meeting10_team_20261117'
EXPORT = ROOT.parent / 'outputs/meeting10_20261117/team_comparison'
REPO = 'https://github.com/cmu-mscf-barclays-rag-2026/finder-rag-benchmark'
COMMIT = '8738b543b8770a29acb5d82c481b1a67fe530140'
LABELS = ['BM25 + fixed reader', 'MultiQuery + fixed reader', 'BM25 + DSPy CoT, optimized']


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    verified = read(REFERENCE / 'VERIFIED_LONG_BENCH_ROWS.json')
    receipt = read(REFERENCE / 'SOURCE_RECEIPT.json')
    assert verified['commit'] == receipt['commit'] == COMMIT
    assert verified['dspy_default_instruction_retained']
    for name, metadata in receipt['files'].items():
        data = (REFERENCE / name).read_bytes()
        assert sha(REFERENCE / name) == metadata['sha256']
        assert hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest() == metadata['github_blob_sha']
    manifest = read(LEGAL / 'PUBLICATION_MANIFEST.json')
    for name, metadata in manifest['files'].items():
        assert sha(LEGAL / name) == metadata['sha256'], name
    assert read(LEGAL / 'independent_validation.json')['passed']
    legal = pd.read_csv(LEGAL / 'team_results_long.csv', float_precision='round_trip')
    assert len(legal) == 40 and (legal.n == 80).all()
    legal['source_kind'] = 'Locally independently validated Legal experiment'
    legal['source_url'] = '../meeting10_legal_20261117/README.md'
    legal['source_commit'] = ''
    tests = pd.read_csv(LEGAL / 'reader_paired_tests.csv', float_precision='round_trip')
    pvalues = {LABELS[1]: float(tests[(tests.candidate == 'multiquery_reader') & (tests.control == 'bm25_reader')].p_holm3.iloc[0]),
               LABELS[2]: float(tests[(tests.candidate == 'dspy_optimized') & (tests.control == 'bm25_reader')].p_holm3.iloc[0])}
    legal['reader_p_holm3_vs_bm25'] = [pvalues.get(row.method) if row.metric == 'reference_token_f1_lexical_proxy' else None for row in legal.itertuples()]
    longbench = []
    for index, item in enumerate(verified['rows']):
        ci = item['paired_bootstrap95'] or [None, None]
        longbench.append(dict(dataset='LongBench v2', method=LABELS[index],
            stage_changed=['baseline', 'retrieval', 'reader_prompt_and_CoT_default_instruction_retained'][index],
            n=item['test_n'], metric=item['metric'], score=item['score'], baseline=item['matched_baseline_score'],
            delta=item['delta'], ci_low=ci[0], ci_high=ci[1],
            test_inference_usd_per1000=item['test_inference_usd_per1000'], setup_usd=item['one_time_setup_usd'],
            notes='Top8 300-word chunks,2400 document words; query includes choices. MC accuracy. DSPy optimizer retained default instruction. Marginal paired95% intervals,10000 draws,not simultaneous. Query generation included for MQR; one-time optimization separate.',
            source_kind='Teammate-reported CSV/JSON; Git blob hashes verified, predictions not independently rerun',
            source_url=f'{REPO}/blob/{COMMIT}/GRAPH_RAG_README.md#longbench-v2-with-gpt-4o-mini-multiqueryretriever-and-dspy',
            source_commit=COMMIT, reader_p_holm3_vs_bm25=None))
    combined = pd.concat([legal, pd.DataFrame(longbench)], ignore_index=True)
    primary = pd.concat([legal[(legal.metric == 'reference_token_f1_lexical_proxy') & legal.method.isin(LABELS)], pd.DataFrame(longbench)], ignore_index=True)
    assert len(primary) == 6 and len(combined) == 43
    assert not primary.duplicated(['dataset', 'method', 'metric']).any()
    for row in combined.itertuples():
        assert math.isclose(row.score - row.baseline, row.delta, abs_tol=1e-12)
    sources = {str(p.relative_to(ROOT.parent)): sha(p) for p in
               [LEGAL / 'PUBLICATION_MANIFEST.json', LEGAL / 'team_results_long.csv', LEGAL / 'reader_paired_tests.csv',
                REFERENCE / 'SOURCE_RECEIPT.json', REFERENCE / 'VERIFIED_LONG_BENCH_ROWS.json']}
    provenance = dict(date='2026-11-17', source_sha256=sources, longbench_repository=REPO,
        longbench_branch='yuchenlu', longbench_commit=COMMIT, source_receipt='sources/longbench/SOURCE_RECEIPT.json',
        longbench_source_urls={name: f'{REPO}/blob/{COMMIT}/{name}' for name in receipt['files']},
        legal_scope='Independent local validation; see sibling Legal report',
        longbench_scope=verified['verification_scope'], primary_rows=6, all_metric_rows=43,
        metrics_comparable_across_datasets=False, pooled_mean_calculated=False, api_calls=0,
        generator_sha256=sha(Path(__file__)))
    lines = ['# Legal and LongBench framework comparison', '', 'Meeting 10 · 17 November 2026', '',
        'This table brings the Legal results together with the teammate’s published LongBench results. It compares each method with its own dataset baseline. **Legal token-F1 and LongBench multiple-choice accuracy measure different outcomes; their absolute scores and deltas must not be ranked or averaged across datasets.**', '',
        '| Dataset | Test n | Answer metric | Method | Score | Δ vs BM25 | Marginal 95% interval for Δ | USD / 1,000 | Setup USD |',
        '|---|---:|---|---|---:|---:|---|---:|---:|']
    names = dict(zip(LABELS, ['BM25 + reader', 'MultiQuery + BM25 + reader', 'BM25 + DSPy CoT']))
    for row in primary.itertuples():
        ci = '—' if pd.isna(row.ci_low) else f'[{row.ci_low:+.4f}, {row.ci_high:+.4f}]'
        metric = 'Token-F1, lexical proxy' if row.dataset == 'Legal RAG Bench' else 'MC accuracy'
        lines.append(f'| {row.dataset} | {row.n} | {metric} | {names[row.method]} | {row.score:.4f} | {row.delta:+.4f} | {ci} | {row.test_inference_usd_per1000:.4f} | {row.setup_usd:.4f} |')
    lines += ['', 'Intervals are marginal paired question-bootstrap intervals, not simultaneous or multiplicity-adjusted: 20,000 draws for Legal and 10,000 for LongBench. **None of the three Legal reader comparisons passes Holm3 correction.** Legal MultiQuery token-F1 versus BM25 has Holm3 p=0.09471, despite its unadjusted interval excluding zero. Legal optimized DSPy versus BM25 has p=0.25042; versus unoptimized CoT p=0.60423.', '',
        'Legal has a separate primary retrieval finding: passage Hit@5 rises from 32.5% to 42.5% with MultiQuery, exact paired McNemar p=0.02148, exploratory. This is not an answer-accuracy result.', '',
        '## What changed in each dataset', '',
        '- Both use gpt-4o-mini; readers and MultiQuery generation use temperature 0. Native optimizer proposal sampling may differ. MultiQueryRetriever requests three rewrites plus the original query, with explicit RRF60 to retain a matched evidence allowance. DSPy changes the reader over unchanged BM25 evidence.',
        '- Legal groups the 100 questions by their 95 labeled passages into optimizer train10 / validation10 / test80. It uses the original 4,876 passages, not 300-word rechunking. Readers receive at most Top8 / 2,400 document words. The benchmark was previously inspected, so this split isolates only current optimization.',
        '- LongBench uses seed42 dev101 / test402, 300-word chunks, Top8 / 2,400 document words, and includes the four choices in the retrieval query. It uses official reader prompting except for DSPy. The selected DSPy instruction equals the default; optimization did not produce a better selected instruction.',
        '- Legal F1 measures lexical overlap with a free-text reference; it does not validate legal correctness. LongBench accuracy measures whether the selected choice matches the reference label. Evidence granularity, task format and prompts differ.',
        '- Legal cost estimates model fresh test inference and include test-only MultiQuery generation; setup is the newly incurred optimizer-call estimate. LongBench costs are derived from the teammate’s saved usage, also including rewrites. Development generation and one-time optimization are excluded from per-query inference costs. These are estimates, not billing statements.', '',
        '## Sources and reproduction', '',
        '[Legal report and validation](../meeting10_legal_20261117/README.md) · [Pinned teammate report](' + longbench[0]['source_url'] + ') · [Provenance](provenance.json) · [Six-row answer table](team_primary_comparison.csv) · [All 43 metric rows](team_all_metrics_long.csv)', '',
        'Legal results were independently validated locally. LongBench values were read from the teammate’s published CSV/JSON and checked against the pinned GitHub blob hashes; their predictions were not independently rerun. The source aggregates and download receipt are retained in `sources/longbench/`. No pooled metric, cross-dataset significance test or new model call was produced.', '',
        'Run from the repository root after the two source packages are available:', '',
        '```bash', '../work/meeting10_venv/bin/python person3_meeting10_legal/combine_team.py', '```']
    for destination in [DOC, EXPORT]:
        destination.mkdir(parents=True, exist_ok=True)
        target_all, target_primary = combined.copy(), primary.copy()
        text = '\n'.join(lines) + '\n'
        if destination == EXPORT:
            text = text.replace('../meeting10_legal_20261117/README.md', '../legal/README.md')
            for target in [target_all, target_primary]:
                target.loc[target.dataset == 'Legal RAG Bench', 'source_url'] = '../legal/README.md'
        target_all.to_csv(destination / 'team_all_metrics_long.csv', index=False)
        target_primary.to_csv(destination / 'team_primary_comparison.csv', index=False)
        (destination / 'README.md').write_text(text)
        (destination / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
        for name in ['SOURCE_RECEIPT.json', 'VERIFIED_LONG_BENCH_ROWS.json'] + list(receipt['files']):
            target = destination / 'sources/longbench' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REFERENCE / name, target)
            assert sha(target) == sha(REFERENCE / name)
        assert len(pd.read_csv(destination / 'team_primary_comparison.csv')) == 6
        assert len(pd.read_csv(destination / 'team_all_metrics_long.csv')) == 43
    print('Verified six answer rows and43 total metric rows; pinned teammate sources mirrored. No API calls.')
    print(DOC)
    print(EXPORT)


if __name__ == '__main__':
    main()
