"""Recompute held-out metrics from saved rankings independently of run.py."""
import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


def validate(data, result):
    corpus = {r['id']: r['text'] for r in map(json.loads, (data / 'corpus.jsonl').read_text().splitlines())}
    qa = {r['id']: r for r in map(json.loads, (data / 'qa.jsonl').read_text().splitlines())}
    split = json.loads((result / 'split.json').read_text())
    assert set(split['dev']).isdisjoint(split['test'])
    assert set(split['dev'] + split['test']) == set(qa)
    dev_texts = {corpus[qa[q]['relevant_passage_id']] for q in split['dev']}
    test_texts = {corpus[qa[q]['relevant_passage_id']] for q in split['test']}
    assert dev_texts.isdisjoint(test_texts)
    predictions = [json.loads(line) for line in (result / 'retrievals.jsonl').read_text().splitlines()]
    keyset = {(r['method'], r['query_id']) for r in predictions}
    assert len(keyset) == len(predictions) == 3 * len(split['test'])
    assert keyset == {(m,q) for m in ('bm25','dense','hybrid') for q in split['test']}
    computed = {}
    for r in predictions:
        qid, method = r['query_id'], r['method']
        gold = qa[qid]['relevant_passage_id']
        assert len({h['doc_id'] for h in r['hits']}) == len(r['hits'])
        for h in r['hits']:
            assert h['text'] == corpus[h['parent_id']][h['start']:h['end']]
        for k in (1,3,5,10,20):
            relevant = [(i,h) for i,h in enumerate(r['hits'][:k],1) if h['parent_id']==gold]
            positions = set()
            for _,h in relevant:
                positions.update(range(h['start'],h['end']))
            rank = relevant[0][0] if relevant else 0
            computed[(method,str(qid),k)] = {
                'recall': int(bool(rank)), 'hit_rate': int(bool(rank)), 'all_evidence_hit': int(bool(rank)),
                'evidence_coverage': len(positions)/len(corpus[gold]),
                'mrr': 1/rank if rank else 0, 'ndcg': 1/math.log2(1+rank) if rank else 0}
    per_query = list(csv.DictReader((result/'per_query.csv').open()))
    assert len(per_query) == len(computed)
    assert {(r['method'],r['query_id'],int(r['k'])) for r in per_query} == set(computed)
    for r in per_query:
        expected = computed[(r['method'],r['query_id'],int(r['k']))]
        assert all(abs(float(r[m])-v)<1e-12 for m,v in expected.items())
    summaries = list(csv.DictReader((result/'test_metrics.csv').open()))
    assert len(summaries)==15
    for r in summaries:
        group = [v for (m,q,k),v in computed.items() if m==r['method'] and k==int(r['k'])]
        assert int(r['n_queries'])==len(group)
        for metric in group[0]:
            assert abs(float(r[metric])-sum(x[metric] for x in group)/len(group))<1e-12
    selected = json.loads((result/'selected.json').read_text())
    sweep = list(csv.DictReader((result/'dev_sweep.csv').open()))
    for method in selected:
        best = max((r for r in sweep if r['method']==method),key=lambda r: (float(r['ndcg']),float(r['mrr']),float(r['recall'])))
        assert selected[method]['config_id']==best['config_id']
    timings=list(csv.DictReader((result/'latency_raw.csv').open()))
    assert len(timings)==len(split['test'])*3*3
    assert len({(r['method'],r['query_id'],r['repeat']) for r in timings})==len(timings)
    assert all(float(r['latency_ms'])>0 for r in timings)
    manifest=json.loads((result/'manifest.json').read_text())
    assert manifest['code_sha256']==hashlib.sha256(Path(__file__).with_name('run.py').read_bytes()).hexdigest()
    for f,digest in manifest['sha256'].items():
        assert hashlib.sha256((data/f).read_bytes()).hexdigest()==digest
    inputs=[json.loads(line) for line in (result/'generation_inputs.jsonl').read_text().splitlines()]
    assert len(inputs)==4*len(split['test'])
    assert all('reference_answer' not in r and 'answer' not in r for r in inputs)
    report={'status':'passed','held_out_queries':len(split['test']), 'rankings_checked':len(predictions),
            'per_query_metric_rows_checked':len(per_query),'summary_rows_checked':len(summaries),
            'latency_measurements_checked':len(timings), 'checks':['gold-text split isolation','complete query/method coverage',
            'exact source slices','independent character-set coverage','first-hit reciprocal/log rank',
            'aggregate means','development-only selection','input and code fingerprints','answer prompt schema']}
    (result/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--results',type=Path,required=True)
    a=p.parse_args()
    validate(a.data,a.results)
