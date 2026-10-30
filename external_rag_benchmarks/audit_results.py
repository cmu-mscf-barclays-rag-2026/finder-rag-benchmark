"""Audit saved rankings, frozen development selection, answers and source slices."""
from collections import defaultdict
import csv
import gzip
import hashlib
import json
import math
import unittest
import io
import numpy as np
from retrieval import ROOT, read, save, signature
from run_longbench import counter, parse_letter

def csv_rows(path):
    with path.open(encoding='utf-8') as f:
        return list(csv.DictReader(f))

def independent_metrics(ranking, relevance, k):
    gold = {str(key):value for key,value in relevance.items() if value>0}
    positions = [i+1 for i,d in enumerate(ranking[:k]) if str(d) in gold]
    ideal = sorted(gold.values(), reverse=True)[:k]
    actual_dcg = sum((2**gold.get(str(d),0)-1)/math.log2(i+2) for i,d in enumerate(ranking[:k]))
    ideal_dcg = sum((2**rel-1)/math.log2(i+2) for i,rel in enumerate(ideal))
    return {'hit': int(bool(positions)), 'recall':len(positions)/len(gold), 'precision':len(positions)/k,
            'mrr':1/min(positions) if positions else 0, 'ndcg':actual_dcg/ideal_dcg,
            'all_judged_relevant_retrieved':int(len(positions)==len(gold))}

def audit():
    selection_manifest = read(ROOT/'data/selection_manifest.json')
    for name, expected in selection_manifest['files_sha256'].items():
        assert hashlib.sha256((ROOT/'data'/name).read_bytes()).hexdigest()==expected, name
    config = read(ROOT/'config.json')
    fiqa = ROOT/'results/fiqa'
    ranks = read(fiqa/'rankings.json')
    queries = read(ROOT/'data/fiqa_queries.json')
    with gzip.open(ROOT/'data/fiqa_corpus.jsonl.gz','rt',encoding='utf-8') as f:
        corpus_ids = {str(json.loads(line)['_id']) for line in f}
    labels = {split:{q['id']:q for q in rows} for split,rows in queries.items()}
    details = csv_rows(fiqa/'per_query.csv')
    assert len(details)==1800
    for row in details:
        ids = ranks[row['split']][row['query_id']][row['method']]
        assert len(ids)==len(set(ids))
        assert set(ids)<=corpus_ids
        expected = independent_metrics(ids,labels[row['split']][row['query_id']]['qrels'],int(row['k']))
        for key,value in expected.items():
            assert math.isclose(float(row[key]),value,abs_tol=1e-12), (row,key)
    summaries = csv_rows(fiqa/'comparison.csv')
    for summary in summaries:
        group = [r for r in details if all(r[k]==summary[k] for k in ['split','method','k'])]
        assert len(group)==int(summary['queries'])
        for metric in ['hit','recall','precision','mrr','ndcg','all_judged_relevant_retrieved']:
            assert math.isclose(float(summary[metric]),sum(float(r[metric]) for r in group)/len(group),abs_tol=1e-12)
    tuning = csv_rows(fiqa/'tuning_dev.csv')
    best = max(tuning,key=lambda r:(float(r['ndcg10']),float(r['recall10']),-abs(float(r['bm25_weight'])-.5)))
    frozen = read(fiqa/'selection.json')
    assert frozen['bm25_weight']==float(best['bm25_weight'])
    assert frozen['config_signature']==signature(config)
    assert set(frozen['dev_query_ids'])==set(labels['dev'])
    assert not set(labels['dev']) & set(labels['test'])
    assert read(fiqa/'run_manifest.json')['config']==config
    assert read(fiqa/'run_manifest.json')['corpus_count']==57638
    assert read(fiqa/'run_manifest.json')['code_sha256']==signature({p:(ROOT/p).read_text(encoding='utf-8') for p in ['retrieval.py','run_fiqa.py']})
    longbench = ROOT/'results/longbench'
    with gzip.open(ROOT/'data/longbench_pilot.jsonl.gz','rt',encoding='utf-8') as f:
        source = {r['_id']:r for r in map(json.loads,f)}
    context = read(longbench/'contexts.json')
    assert read(longbench/'retrieval_manifest.json')['config_signature']==signature(config)
    assert read(longbench/'retrieval_manifest.json')['code_sha256']==signature({p:(ROOT/p).read_text(encoding='utf-8') for p in ['retrieval.py','run_longbench.py']})
    identity = read(longbench/'answer_identity.json')
    assert identity['config']==signature(config)
    assert identity['contexts_sha256']==hashlib.sha256((longbench/'contexts.json').read_bytes()).hexdigest()
    assert hashlib.sha256((ROOT/'assets/context_tokenizer.json').read_bytes()).hexdigest()=='be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037'
    tok = counter()
    for qid, methods in context.items():
        original = source[qid]['context']
        for method, result in methods.items():
            slices = [original[u['start_char']:u['end_char']] for u in result['delivered']]
            assert result['context']=='\n\n'.join(slices)
            tokens = sum(len(tok.encode(t,add_special_tokens=False).ids) for t in slices)
            assert tokens==result['returned_tokens'] and tokens<=config['context_budget']
            assert len(result['delivered'])<=config['longbench_top_k']
    answers = read(longbench/'answers.json')
    assert len(answers)==80 and len({(r['id'],r['method']) for r in answers})==80
    for row in answers:
        q = source[row['id']]
        assert row['correct_answer']==q['answer']
        assert row['split']==q['pilot_split']
        parsed = parse_letter(row['raw_response'])
        assert parsed==row['predicted']
        assert int(parsed==q['answer'])==row['correct']
        assert row['prompt_tokens']<=config['reader_prompt_limit']
    long_summary = csv_rows(longbench/'comparison.csv')
    for row in long_summary:
        group = [r for r in answers if r['split']==row['split'] and r['method']==row['method']]
        assert len(group)==int(row['questions'])
        assert math.isclose(float(row['accuracy']),sum(r['correct'] for r in group)/len(group),abs_tol=1e-12)
    suite = unittest.defaultTestLoader.discover(str(ROOT/'tests'))
    report = io.StringIO()
    tested = unittest.TextTestRunner(stream=report,verbosity=2).run(suite)
    if not tested.wasSuccessful():
        raise ValueError(report.getvalue())
    result = {'fiqa_query_metric_rows_checked':len(details),'fiqa_summary_rows_checked':len(summaries),
        'longbench_predictions_checked':len(answers),'longbench_summary_rows_checked':len(long_summary),
        'software_tests_passed':tested.testsRun,'source_hashes_and_context_slices':'passed','frozen_dev_selection':'passed',
        'limits':'Audit checks arithmetic and protocol, not correctness of publisher relevance judgments or model reasoning'}
    save(ROOT/'results/audit.json',result)
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    audit()
