"""Check published Meeting 10 statistics without API calls or execution caches."""
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import binomtest

D = Path(__file__).resolve().parent


def load(name):
    return pd.read_csv(D / name, dtype={'query_id': str})


def ci(delta):
    rng = np.random.default_rng(42)
    values = delta[rng.integers(0, len(delta), (20000, len(delta)))].mean(axis=1)
    return np.quantile(values, [.025, .975])


def main(output=None):
    split = json.loads((D / 'selection.json').read_text())
    ids = split['test_ids']
    assert len(ids) == len(set(ids)) == 80
    retrieval, answers = load('retrieval_per_query.csv'), load('reader_per_query.csv')
    assert len(retrieval) == 1600 and len(answers) == 320
    assert not retrieval.duplicated(['query_id', 'method', 'k']).any()
    assert not answers.duplicated(['query_id', 'method']).any()
    assert set(answers.query_id) == set(ids)
    summary = load('retrieval_summary.csv')
    for row in summary.itertuples(index=False):
        subset = retrieval[(retrieval.method == row.method) & (retrieval.k == row.k)]
        if row.cohort != 'full100':
            subset = subset[subset.cohort == row.cohort]
        assert len(subset) == row.n
        for metric in ['hit', 'recall', 'mrr', 'ndcg']:
            assert abs(subset[metric].mean() - getattr(row, metric)) < 1e-12
    reader_summary = load('reader_summary.csv')
    for row in reader_summary.itertuples(index=False):
        subset = answers[answers.method == row.method]
        assert len(subset) == row.n == 80
        for metric in ['em', 'token_f1', 'used_words', 'full_passages', 'partial_passage', 'gold_full', 'gold_touched']:
            assert abs(subset[metric].mean() - getattr(row, metric)) < 1e-12
    wide = retrieval[(retrieval.cohort == 'test80') & (retrieval.k == 5)].pivot(
        index='query_id', columns='method', values='hit').loc[ids]
    delta = (wide.multiquery - wide.bm25).to_numpy()
    wins, losses = int((delta > 0).sum()), int((delta < 0).sum())
    primary = json.loads((D / 'primary_retrieval_test.json').read_text())
    p = float(binomtest(wins, wins + losses, .5).pvalue) if wins + losses else 1.
    assert (wins, losses) == (primary['wins'], primary['losses'])
    assert abs(delta.mean() - primary['mean_delta']) < 1e-12
    assert abs(p - primary['p_exact_mcnemar']) < 1e-12
    assert np.allclose(ci(delta), primary['paired_bootstrap95'], atol=1e-12, rtol=0)
    scores = answers.pivot(index='query_id', columns='method', values='token_f1').loc[ids]
    tests = load('reader_paired_tests.csv')
    assert len(tests) == 3
    matrix = np.column_stack([(scores[row.candidate] - scores[row.control]).to_numpy()
                              for row in tests.itertuples(index=False)])
    rng = np.random.default_rng(42)
    counts = np.zeros(3, dtype=int)
    observed = abs(matrix.sum(axis=0))
    for _ in range(100):
        signs = rng.integers(0, 2, (1000, 80), dtype=np.int8) * 2 - 1
        counts += (abs(signs @ matrix) >= observed - 1e-12).sum(axis=0)
    raw = (counts + 1) / 100001
    order = np.argsort(raw, kind='stable')
    adjusted = np.empty(3)
    adjusted[order] = np.minimum(1, np.maximum.accumulate((3 - np.arange(3)) * raw[order]))
    assert np.allclose(raw, tests.p_raw, atol=1e-12, rtol=0)
    assert np.allclose(adjusted, tests.p_holm3, atol=1e-12, rtol=0)
    for j, row in enumerate(tests.itertuples(index=False)):
        delta = matrix[:, j]
        assert abs(delta.mean() - row.mean_delta) < 1e-12
        assert (int((delta > 1e-12).sum()), int((delta < -1e-12).sum()), int((abs(delta) <= 1e-12).sum())) == (row.wins, row.losses, row.ties)
        assert np.allclose(ci(delta), json.loads(row.paired_bootstrap95), atol=1e-12, rtol=0)
    receipt = dict(passed=True, retrieval_rows=1600, reader_rows=320,
                   primary_exact_mcnemar_p=p, reader_tests=3, signflip_draws=100000,
                   bootstrap_draws=20000, api_calls=0,
                   scope='Published metrics and statistics; original API prompt/response audit is a separate retained record.')
    if output:
        Path(output).write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    main(parser.parse_args().out)
