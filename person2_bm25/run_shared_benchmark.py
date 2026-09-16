"""Run Florence's unchanged BM25 with dev-only tuning on the common team split.

From the repository root: python person2_bm25/run_shared_benchmark.py
"""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from finder_bm25.bm25 import BM25Retriever, TOKENIZER_VERSION
from finder_bm25.data import digest, prepare_records, save_bundle, select_queries, write_json, write_jsonl
from finder_bm25.evaluation import collect_run, evaluate_rag
from finder_bm25.reporting import write_csv, table
ROOT = Path(__file__).resolve().parents[1]


def prepare_shared(records, config, source):
    if config['split_id'] != 'sha1-mod5-dev-v1':
        raise ValueError('Unsupported shared split')
    bundle = prepare_records(records, split_protocol=config['split_id'], source=source)
    assert bundle['manifest']['dev_count'] == config['dev_queries']
    assert bundle['manifest']['test_count'] == config['test_queries']
    expected = {' '.join(str(ref).split()) for row in records for ref in row['references']}
    documents = {doc['doc_id']: doc['text'] for doc in bundle['corpus']}
    assert set(documents.values()) == expected and len(documents) == 5830
    for row in records:
        assert {documents[d] for d in bundle['qrels'][row['_id']]} == {' '.join(str(r).split()) for r in row['references']}
    return bundle


def benchmark(root=ROOT, output=None):
    import pyarrow.parquet as parquet
    root=Path(root); output=Path(output or root/'results/b_bm25_common')
    output.mkdir(parents=True,exist_ok=True)
    config=json.loads((root/'config/benchmark_config.json').read_text())
    data=root/config['dataset_file']
    checksum=hashlib.sha256(data.read_bytes()).hexdigest()
    if checksum != config['dataset_sha256']: raise ValueError('Dataset checksum mismatch')
    bundle=prepare_shared(parquet.read_table(data).to_pylist(),config,{'kind':'shared_parquet','sha256':checksum})
    save_bundle(bundle,root/'person2_bm25/data/team_shared')
    dev_ids=sorted(q['query_id'] for q in select_queries(bundle,'dev'))
    test_ids=sorted(q['query_id'] for q in select_queries(bundle,'test'))
    write_json(output/'query_partitions.json',{'split_id':config['split_id'],'dev':dev_ids,'test':test_ids})
    print(f'Verified 5830 passages; dev {len(dev_ids)}; test {len(test_ids)}',flush=True)
    sweep=[]
    for k1 in (0.8,1.2,1.6):
        for b in (0.25,0.75,1.0):
            print(f'DEV k1={k1}, b={b}',flush=True)
            run=collect_run(bundle,BM25Retriever(bundle['corpus'],k1=k1,b=b),method='bm25_dev',top_k=5,partition='dev',config={'k1':k1,'b':b})
            sweep.append({'k1':k1,'b':b,**evaluate_rag(bundle,run)['summary']})
    write_csv(output/'dev_sweep.csv',sweep)
    best=max(sweep,key=lambda row:(row['ndcg'],row['mrr']))
    selected={'k1':best['k1'],'b':best['b'],'selection':'dev nDCG@5, then MRR@5; ties retain first sorted grid entry',
        'dev_ndcg':best['ndcg'],'dev_mrr':best['mrr'],'split_id':config['split_id'],
        'dev_query_fingerprint':digest(dev_ids),'test_query_fingerprint':digest(test_ids),'bundle_fingerprint':bundle['manifest']['fingerprint']}
    write_json(output/'selected_config.json',selected)
    print(f"Frozen on DEV: k1={best['k1']}, b={best['b']}",flush=True)
    summaries=[]; team=[]
    for method,k1,b,cutoffs in [('b_bm25_default',1.2,0.75,[5]),('b_bm25_tuned',best['k1'],best['b'],config['evaluation_k'])]:
        start=time.perf_counter(); retriever=BM25Retriever(bundle['corpus'],k1=k1,b=b)
        build_ms=(time.perf_counter()-start)*1000
        for k in cutoffs:
            print(f'TEST {method} k={k}',flush=True)
            run=collect_run(bundle,retriever,method=method,top_k=k,partition='test',config={'k1':k1,'b':b,'tokenizer':TOKENIZER_VERSION},build_latency_ms=build_ms)
            evaluation=evaluate_rag(bundle,run); summary=evaluation['summary']; summaries.append(summary)
            if k==5: write_csv(output/f'{method}_per_query_k5.csv',evaluation['per_query'])
            if method=='b_bm25_tuned':
                team.append({**{key:config[key] for key in ['schema_version','dataset_id','corpus_id','split_id']},
                    'owner':'B','method_id':method,'method':f'Florence BM25 k1={k1:g}, b={b:g}',
                    'split':'test','k':k,'n_queries':summary['query_count'],'precision_at_k':summary['precision'],
                    'recall_at_k':summary['recall'],'hit_rate_at_k':statistics.fmean(r['first_relevant_rank'] is not None for r in evaluation['per_query']),
                    'mrr_at_k':summary['mrr'],'ndcg_at_k':summary['ndcg'],'latency_ms_per_query':summary['latency_ms'],
                    'answer_n':0,'answer_exact_match':None,'answer_token_f1':None,'answer_semantic_similarity':None,
                    'generator_model':None,'answer_status':'not_run','answer_protocol_id':None})
                if k==max(config['evaluation_k']): write_jsonl(output/'test_top10.jsonl',run['results'])
    write_csv(output/'test_summary.csv',summaries)
    sys.path.insert(0,str(root/'scripts'))
    from export_method_metrics import METRIC_COLUMNS
    write_csv(root/'team_metrics/b_bm25.csv',[{key:row[key] for key in METRIC_COLUMNS} for row in team])
    write_json(output/'manifest.json',{**selected,'dataset_sha256':checksum,'corpus_id':config['corpus_id'],
        'dev_queries':len(dev_ids),'test_queries':len(test_ids),'corpus_passages':len(bundle['corpus']),
        'corpus_and_qrels_verified':True,'tokenizer':TOKENIZER_VERSION,'python':platform.python_version(),
        'platform':platform.platform(),'machine':platform.machine(),
        'latency_protocol':'One search per query/cutoff; descriptive only. See common_latency for controlled timings.',
        'default_control':'Fixed original k1=1.2, b=0.75; not selected on test','answer_status':'not_run'})
    (output/'report.md').write_text('# Florence / Person 2 — common-split BM25\n\n'
        f'Verified 5,830 passages, {len(dev_ids)} development queries, and {len(test_ids)} test queries.\n'
        'Parameters were selected on development queries and frozen before test evaluation.\n\n'+table(summaries)+'\n\n'
        'The tokenizer and BM25 implementation are unchanged; the partition protocol changes. '
        'Hashed document IDs map to the same normalized passages and exact qrels. '
        'Original seed-42 experiments remain historical, separate results. '
        'The nine-point sweep, split IDs, per-query metrics, and top-10 test rankings are retained here. '
        'No answer generation was run. These single-run timings are not a cross-method speed ranking.\n')
    print(table(summaries),flush=True)
    return selected


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--root',type=Path,default=ROOT); parser.add_argument('--output',type=Path)
    args=parser.parse_args(); benchmark(args.root,args.output)
