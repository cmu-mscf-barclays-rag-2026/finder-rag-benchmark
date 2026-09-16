"""Compare selected A/B/C/D retrieval on one CPU and one timing protocol.

Only imports exact retrieval functions from pinned teammate commits; does not
modify their files or branches. Index/model setup is outside the timed region.
"""
from __future__ import annotations
import argparse
import ast
import __future__
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
A_COMMIT='bd5e7c35443c6eaa5a1f24ada240b2c5ffdb3eb1'
D_COMMIT='4013a25b33a7dc62845359941f062870c822f91a'
MODEL='sentence-transformers/all-MiniLM-L6-v2'
MODEL_REVISION='1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
PROTOCOL='cpu-single-query-top5-warm3-repeat5-sample100-seed42-v1'


def pinned_functions(commit, path, names, namespace):
    source=subprocess.check_output(['git','show',f'{commit}:{path}'],cwd=ROOT,text=True)
    tree=ast.parse(source)
    body=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
    if {node.name for node in body} != set(names): raise ValueError('Missing pinned source function')
    code=compile(ast.Module(body=body,type_ignores=[]),path,'exec',flags=__future__.annotations.compiler_flag)
    exec(code,namespace)
    return hashlib.sha256(source.encode()).hexdigest()


def measure(methods, queries, *, repeats=5, warmup=3, seed=42, timer=time.perf_counter):
    if not methods or not queries or repeats<1 or warmup<0: raise ValueError('Invalid timing protocol')
    for method in methods.values():
        for query in queries[:warmup]: method(query['text'])
    jobs=[(name,index,repeat) for name in methods for index in range(len(queries)) for repeat in range(repeats)]
    random.Random(seed).shuffle(jobs)
    rows=[]
    for name,index,repeat in jobs:
        query=queries[index]
        started=timer(); passages=methods[name](query['text']); elapsed=1000*(timer()-started)
        if not 0<=len(passages)<=5: raise ValueError('Every method must return at most five passages')
        rows.append({'method_id':name,'query_id':query['query_id'],'repeat':repeat,'latency_ms':elapsed,'returned':len(passages)})
    return rows


def write_csv(path, rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n'); writer.writeheader(); writer.writerows(rows)


def run(args):
    import numpy as np
    import pandas as pd
    import torch
    from threadpoolctl import threadpool_limits, threadpool_info
    from sentence_transformers import SentenceTransformer
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'person2_bm25'))
    import finder_hybrid_experiment as common
    from finder_bm25.bm25 import BM25Retriever
    from finder_bm25.data import load_bundle
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    threadpool_limits(limits=args.threads)
    output=ROOT/'results/common_latency'; output.mkdir(parents=True,exist_ok=True)
    cache=Path(args.cache); cache.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((ROOT/'config/benchmark_config.json').read_text())
    data=ROOT/cfg['dataset_file']
    assert hashlib.sha256(data.read_bytes()).hexdigest()==cfg['dataset_sha256']
    records=pd.read_parquet(data)
    corpus,qrels=common.build_reference_corpus(records)
    split=common.deterministic_split(records['_id'].astype(str).tolist())
    assert len(corpus)==5830 and int(sum(split=='test'))==4575
    model=SentenceTransformer(MODEL,revision=MODEL_REVISION,device='cpu',cache_folder=args.model_cache,local_files_only=True)
    model.eval()
    a_namespace={'np':np,'time':time,'Document':Document,'RecursiveCharacterTextSplitter':RecursiveCharacterTextSplitter}
    a_hash=pinned_functions(A_COMMIT,'evaluate_dense.py',['split_source_corpus','rank_sources'],a_namespace)
    d_namespace={'np':np}
    d_hash=pinned_functions(D_COMMIT,'d_threshold_mmr.py',['refine'],d_namespace)
    chunks,source_ids=a_namespace['split_source_corpus'](corpus,SimpleNamespace(chunk_size=500,chunk_overlap=75))
    assert len(chunks)==44679
    def cached_encode(texts, label):
        fingerprint=hashlib.sha256(json.dumps([MODEL_REVISION,model.max_seq_length,texts],ensure_ascii=False).encode()).hexdigest()
        path=cache/f'{label}-{fingerprint}.npy'
        if path.exists():
            value=np.load(path); assert value.shape==(len(texts),384); return value
        print(f'Indexing {label}: {len(texts)} texts (excluded from timing)',flush=True)
        value=model.encode(texts,batch_size=64,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=True)
        np.save(path,value); return value
    whole=cached_encode(corpus,'whole')
    chunk_embeddings=cached_encode(chunks,'chunks500-75')
    if args.prepare_only:
        print('CPU indexes cached; no latency measurements taken.',flush=True); return
    selected=json.loads((ROOT/'results/b_bm25_common/selected_config.json').read_text())
    bundle=load_bundle(ROOT/'person2_bm25/data/team_shared')
    b=BM25Retriever(bundle['corpus'],k1=selected['k1'],b=selected['b'])
    b_text={doc['doc_id']:doc['text'] for doc in bundle['corpus']}
    vectorizer,bm_matrix=common.build_bm25(corpus)
    settings=json.loads(subprocess.check_output(['git','show',f'{D_COMMIT}:results/d_refinement/selected_configs.json'],cwd=ROOT))
    threshold=next(s for s in settings if s['family']=='threshold')
    mmr=next(s for s in settings if s['family']=='mmr')
    alpha=json.loads((ROOT/'results/run_summary.json').read_text())['hybrid']['best_dense_weight']
    def encode(query):
        return model.encode([query],normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False)
    def texts(ids): return [corpus[int(i)] for i in ids[:5]]
    def dense_candidates(query):
        scores=encode(query)@whole.T
        return scores[0],common.topk_from_scores(scores,100)[0]
    def a_search(query):
        rankings,_=a_namespace['rank_sources'](encode(query),chunk_embeddings,source_ids,len(corpus),depth=5,batch_size=1)
        return texts(rankings[0])
    def b_search(query): return [b_text[hit['doc_id']] for hit in b.search(query,top_k=5)]
    def c_search(query):
        _,dense=dense_candidates(query)
        sparse,_=common.retrieve_bm25(vectorizer,bm_matrix,[query],100)
        ranked,_=common.weighted_rrf(sparse,dense[None,:],alpha,100)
        return texts(ranked[0])
    def d_search(setting):
        def retrieve(query):
            scores,candidates=dense_candidates(query)
            return texts(d_namespace['refine'](scores,candidates,whole,setting,depth=5))
        return retrieve
    methods={'a_dense_minilm_500_75':a_search,'b_bm25_tuned':b_search,
             'hybrid_rrf_75bm25_25dense':c_search,'d_threshold_0.3':d_search(threshold),'d_mmr_1.0_10':d_search(mmr)}
    test=sorted([{'query_id':str(row['_id']),'text':str(row['text'])} for (_,row),part in zip(records.iterrows(),split) if part=='test'],key=lambda row:row['query_id'])
    queries=random.Random(42).sample(test,100)
    write_csv(output/'queries.csv',queries)
    print('Timing 5 methods, same 100 queries, 3 warmups, 5 interleaved repetitions on CPU...',flush=True)
    raw=measure(methods,queries)
    write_csv(output/'raw.csv',raw)
    summary=[]
    for name in methods:
        values=[row['latency_ms'] for row in raw if row['method_id']==name]
        assert len(values)==500
        summary.append({'method_id':name,'protocol_id':PROTOCOL,'queries':100,'repeats':5,'measurements':len(values),
            'mean_ms':statistics.fmean(values),'median_ms':statistics.median(values),'p95_ms':float(np.percentile(values,95)),
            'mean_returned':statistics.fmean(row['returned'] for row in raw if row['method_id']==name)})
    write_csv(output/'summary.csv',summary)
    metadata={'protocol_id':PROTOCOL,'model':MODEL,'model_revision':MODEL_REVISION,'max_seq_length':model.max_seq_length,
        'device':'cpu','threads':args.threads,'torch_interop_threads':1,'threadpools':threadpool_info(),
        'platform':platform.platform(),'machine':platform.machine(),'processor':platform.processor(),'logical_cpus':os.cpu_count(),
        'created_at':datetime.now(timezone.utc).isoformat(),'python':platform.python_version(),'dataset_sha256':cfg['dataset_sha256'],'split_id':cfg['split_id'],
        'query_sample':'random.Random(42).sample(sorted test query IDs, 100)','queries_sha256':hashlib.sha256((output/'queries.csv').read_bytes()).hexdigest(),
        'warmup':3,'repeats':5,'interleave_seed':42,'scope':'raw question to up to 5 evidence strings; includes tokenization/encoding/search/pooling/fusion/refinement/lookup; excludes index build, disk IO, generation',
        'a_source_commit':subprocess.check_output(['git','rev-parse',A_COMMIT],cwd=ROOT,text=True).strip(),'a_source_sha256':a_hash,
        'd_source_commit':subprocess.check_output(['git','rev-parse',D_COMMIT],cwd=ROOT,text=True).strip(),'d_source_sha256':d_hash,'c_source_sha256':hashlib.sha256((ROOT/'finder_hybrid_experiment.py').read_bytes()).hexdigest(),
        'b_source_sha256':hashlib.sha256((ROOT/'person2_bm25/finder_bm25/bm25.py').read_bytes()).hexdigest(),'b_selected':selected,'c_dense_weight':alpha,'d_threshold':threshold,'d_mmr':mmr,
        'packages':{p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','torch','sentence-transformers','transformers','langchain-text-splitters','pyarrow','threadpoolctl']}}
    (output/'manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
    lines=['# Controlled retrieval latency — all four workstreams','','One Mac, CPU only, the same 100 held-out questions, 3 warmup calls per method, and 5 randomly interleaved repetitions. Each method has 500 measurements.','',
        '| Method | Mean ms | Median ms | p95 ms |','| --- | ---: | ---: | ---: |']
    lines += [f"| {r['method_id']} | {r['mean_ms']:.3f} | {r['median_ms']:.3f} | {r['p95_ms']:.3f} |" for r in summary]
    lines += ['', 'Measured from raw question to top-five evidence text, including query encoding where needed, search, source pooling, fusion/refinement, and text lookup. Indexing, downloads, and generation are excluded. All model calls use CPU with four intra-op threads and one inter-op thread; BLAS pools are limited to four threads.', '',
        'These measurements support a local comparison of these implementations and selected configurations, not a general claim about hardware or algorithm speed. A indexes 44,679 chunks while C/D index 5,830 whole passages; this is part of each selected method. B is Florence’s implementation, not the BM25 control inside C. Raw repetitions, query IDs, source commits, model revision, and package versions are retained. Retrieval scores in team_metrics remain the owners’ full-test results; this timing run is not a quality rerun.', '',
        'Reproduce after running the shared BM25 benchmark: `python scripts/common_latency.py --model-cache PATH --cache PATH`. Download the pinned model revision first. The original mixed-hardware latencies remain in team_metrics for provenance and must not be used as a speed ranking.']
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--cache',default=str(ROOT/'.cache/common_latency'))
    parser.add_argument('--model-cache',default=str(ROOT/'.cache/models'))
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    if args.threads!=4: parser.error('This named protocol uses exactly four threads')
    run(args)
