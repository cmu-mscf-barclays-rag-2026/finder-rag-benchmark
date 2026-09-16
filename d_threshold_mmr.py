"""Person D refinement, using the team's corpus, split, search and metrics.

Run from any directory: python d_threshold_mmr.py
Outputs are restricted to results/d_refinement/ and team_metrics/d_threshold_mmr.csv.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from finder_hybrid_experiment import (
    build_reference_corpus, deterministic_split, topk_from_scores,
    query_metrics, build_bm25, retrieve_bm25, weighted_rrf,
)
from scripts.export_method_metrics import METRIC_COLUMNS

ROOT = Path(__file__).resolve().parent
MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
NAMES = ['precision_at_k', 'recall_at_k', 'hit_rate_at_k', 'mrr_at_k', 'ndcg_at_k']


def refine(scores, candidates, embeddings, setting, depth=10):
    """Return distinct integer IDs. Empty threshold results remain empty lists."""
    candidates = np.asarray(candidates, dtype=int)
    if setting['family'] == 'dense':
        return candidates[:depth].tolist()
    if setting['family'] == 'threshold':
        return [int(i) for i in candidates if scores[i] >= setting['threshold']][:depth]
    pool = candidates[:setting['fetch_k']]
    lam = setting['lambda_mult']
    selected = [0]
    redundancy = embeddings[pool] @ embeddings[pool[0]]
    while len(selected) < min(depth, len(pool)):
        objective = lam * scores[pool] - (1 - lam) * redundancy
        objective[selected] = -np.inf
        nxt = int(np.argmax(objective))
        selected.append(nxt)
        redundancy = np.maximum(redundancy, embeddings[pool] @ embeddings[pool[nxt]])
    return pool[selected].tolist()


def summarize(rankings, qrels, k):
    values = np.asarray([query_metrics(r, gold, k) for r, gold in zip(rankings, qrels)])
    result = dict(zip(NAMES, values.mean(axis=0).tolist()))
    result['mean_returned'] = float(np.mean([min(len(r), k) for r in rankings]))
    result['empty_rate'] = float(np.mean([not r for r in rankings]))
    result['precision_returned'] = float(np.mean([
        len(set(r[:k]) & gold) / len(r[:k]) if r[:k] else 0
        for r, gold in zip(rankings, qrels)]))
    return result


def benchmark_latency(methods, queries, repeats=3, warmup=3, synchronize=None):
    """Raw query -> final evidence, including query encoding, excluding setup/LLM."""
    sync = synchronize or (lambda: None)
    for method in methods.values():
        for query in queries[:warmup]:
            method(query)
            sync()
    jobs = [(name, i, repeat) for name in methods for i in range(len(queries))
            for repeat in range(repeats)]
    random.Random(42).shuffle(jobs)
    rows = []
    for name, i, repeat in jobs:
        sync()
        start = time.perf_counter()
        passages = methods[name](queries[i])
        sync()
        rows.append(dict(method=name, sample_index=i, repeat=repeat,
                         latency_ms=1000*(time.perf_counter()-start), returned=len(passages)))
    raw = pd.DataFrame(rows)
    summary = raw.groupby('method').agg(mean_ms=('latency_ms','mean'),
        median_ms=('latency_ms','median'), p95_ms=('latency_ms', lambda x:x.quantile(.95)),
        measurements=('latency_ms','size'))
    return raw, summary


def run(model_path=MODEL, threads=4, timing_queries=100):
    torch.set_num_threads(threads)
    cfg = json.loads((ROOT/'config/benchmark_config.json').read_text())
    data = ROOT/cfg['dataset_file']
    if hashlib.sha256(data.read_bytes()).hexdigest() != cfg['dataset_sha256']:
        raise ValueError('Dataset checksum does not match the team configuration')
    df = pd.read_parquet(data).reset_index(drop=True)
    corpus, qrels = build_reference_corpus(df)
    splits = deterministic_split(df['_id'].astype(str).tolist())
    dev, test = np.flatnonzero(splits=='dev'), np.flatnonzero(splits=='test')
    assert len(dev)==cfg['dev_queries'] and len(test)==cfg['test_queries']
    assert len(corpus)==5830
    k, depth = cfg['primary_k'], cfg['candidate_depth']
    out = ROOT/'results/d_refinement'
    out.mkdir(parents=True, exist_ok=True)
    cache = ROOT/'.cache/d_refinement'
    cache.mkdir(parents=True, exist_ok=True)
    queries = df[cfg['query_field']].astype(str).tolist()
    model = SentenceTransformer(model_path, device='cpu')
    fingerprint = hashlib.sha256((cfg['dataset_sha256'] + MODEL + str(model_path) +
        json.dumps(corpus, ensure_ascii=False)).encode()).hexdigest()
    embedding_file = cache/f'{fingerprint}.npy'
    if embedding_file.exists():
        embeddings = np.load(embedding_file)
    else:
        embeddings = model.encode(corpus, batch_size=64, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=True)
        np.save(embedding_file, embeddings)
    print(f'Corpus {len(corpus)}; dev {len(dev)}; test {len(test)}', flush=True)

    # Warm up before the team's batch-throughput measurement. Encoding is included.
    model.encode(queries[:3], normalize_embeddings=True, convert_to_numpy=True)
    score_batches, candidate_batches, query_batches = [], [], []
    start = time.perf_counter()
    for begin in range(0, len(queries), 128):
        q = model.encode(queries[begin:begin+128], batch_size=128,
            normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        scores = q @ embeddings.T
        candidates = topk_from_scores(scores, depth)
        score_batches.append(scores)
        candidate_batches.append(candidates)
        query_batches.append(q)
    dense_batch_ms = 1000*(time.perf_counter()-start)/len(queries)
    scores = np.vstack(score_batches)
    candidates = np.vstack(candidate_batches)
    del score_batches, candidate_batches, query_batches
    settings = [dict(method_id='d_dense_control', family='dense')]
    settings += [dict(method_id=f'd_threshold_{t}', family='threshold', threshold=t)
                 for t in (.3,.5,.7)]
    settings += [dict(method_id=f'd_mmr_{lam}_{fetch}', family='mmr',
                     lambda_mult=lam, fetch_k=fetch)
                 for lam in (.3,.5,.7,1.) for fetch in (10,20,50,100)]
    tuning = []
    for setting in settings:
        ranked = [refine(scores[i], candidates[i], embeddings, setting) for i in dev]
        tuning.append(dict(**setting, **summarize(ranked,[qrels[i] for i in dev],k)))
    tuning_df = pd.DataFrame(tuning)
    tuning_df.to_csv(out/'tuning_dev.csv',index=False)
    # Match C's priority: Recall@5, then nDCG@5. Freeze each family before test.
    best = tuning_df.sort_values(['recall_at_k','ndcg_at_k','method_id'],
        ascending=[False,False,True]).groupby('family').head(1).method_id.tolist()
    chosen = [s for s in settings if s['method_id'] in best]
    (out/'selected_configs.json').write_text(json.dumps(chosen,indent=2))
    print('Frozen configurations:',chosen,flush=True)

    all_rankings, test_rows, team_rows = {}, [], []
    for setting in chosen:
        start = time.perf_counter()
        ranked = [refine(scores[i],candidates[i],embeddings,setting) for i in range(len(df))]
        refinement_ms = 1000*(time.perf_counter()-start)/len(df)
        batch_ms = dense_batch_ms + refinement_ms
        all_rankings[setting['method_id']] = ranked
        for cutoff in cfg['evaluation_k']:
            row = dict(method_id=setting['method_id'],method=setting['method_id'],
                split='test',k=cutoff,n_queries=len(test),latency_ms_per_query=batch_ms,
                **summarize([ranked[i] for i in test],[qrels[i] for i in test],cutoff))
            test_rows.append(row)
            if setting['family'] != 'dense':
                submitted = dict(row,owner='D',answer_status='not_run',answer_n=0,
                    answer_exact_match=None,answer_token_f1=None,
                    answer_semantic_similarity=None,generator_model=None,answer_protocol_id=None)
                for field in ['schema_version','dataset_id','corpus_id','split_id']:
                    submitted[field] = cfg[field]
                team_rows.append(submitted)
    pd.DataFrame(test_rows).to_csv(out/'test_metrics.csv',index=False)
    pd.DataFrame(team_rows)[METRIC_COLUMNS].to_csv(ROOT/'team_metrics/d_threshold_mmr.csv',index=False)

    # Repeat single-query timing on identical inputs and hardware for all methods.
    vectorizer, bm25 = build_bm25(corpus)
    bm_rank, _ = retrieve_bm25(vectorizer,bm25,queries,depth)
    alpha_scores=[]
    for alpha in (.25,.5,.75):
        fused,_=weighted_rrf(bm_rank[dev],candidates[dev],alpha,depth)
        alpha_scores.append((summarize(fused.tolist(),[qrels[i] for i in dev],k),alpha))
    alpha=max(alpha_scores,key=lambda x:(x[0]['recall_at_k'],x[0]['ndcg_at_k']))[1]
    def dense_candidates(query):
        q=model.encode([query],normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False)
        s=q @ embeddings.T
        return s[0],topk_from_scores(s,depth)[0]
    def bm_search(query):
        return retrieve_bm25(vectorizer,bm25,[query],depth)[0][0]
    def get_text(ids):
        return [corpus[int(i)] for i in ids[:k]]
    def hybrid(query):
        _, dr=dense_candidates(query)
        br=bm_search(query)
        rank,_=weighted_rrf(br[None,:],dr[None,:],alpha,depth)
        return get_text(rank[0])
    def pipeline(setting):
        def retrieve(query):
            s,c=dense_candidates(query)
            return get_text(refine(s,c,embeddings,setting,depth=k))
        return retrieve
    methods={'BM25':lambda q:get_text(bm_search(q)), 'Hybrid':hybrid}
    methods.update({s['method_id']:pipeline(s) for s in chosen})
    timing_ids=np.random.default_rng(42).choice(test,min(timing_queries,len(test)),replace=False)
    pd.DataFrame({'query_index':timing_ids,'query_id':df.iloc[timing_ids]['_id'].tolist()}).to_csv(out/'latency_queries.csv',index=False)
    raw,latency=benchmark_latency(methods,[queries[i] for i in timing_ids])
    raw.to_csv(out/'latency_raw.csv',index=False)
    latency.to_csv(out/'latency_single_query.csv')

    details=[]
    baseline=all_rankings['d_dense_control']
    for s in chosen:
        if s['family']=='dense': continue
        rank=all_rankings[s['method_id']]
        for i in test:
            old,new=baseline[i][:k],rank[i][:k]
            lost=(set(old)&qrels[i])-set(new)
            gained=(set(new)&qrels[i])-set(old)
            details.append(dict(method=s['method_id'],query_index=int(i),query_id=df.iloc[i]['_id'],
                question=queries[i],category=df.iloc[i]['category'],reasoning=bool(df.iloc[i]['reasoning']),
                gold_ids=json.dumps(sorted(qrels[i])),baseline_ids=json.dumps(old),refined_ids=json.dumps(new),
                lost_gold=json.dumps(sorted(lost)),gained_gold=json.dumps(sorted(gained)),
                status='improved' if len(gained)>len(lost) else 'worsened' if len(lost)>len(gained) else 'unchanged',
                review_notes=''))
    pd.DataFrame(details).to_csv(out/'paired_errors.csv',index=False)
    # Confirm that this run's dense ranking matches C's saved ranking on the shared split.
    reference=pd.read_csv(ROOT/'results/per_query_rankings.csv')
    agreement=float(np.mean([json.loads(row.dense_minilm_top10)==candidates[int(row.query_index),:10].tolist()
                            for row in reference.itertuples()]))
    manifest=dict(dataset_id=cfg['dataset_id'],corpus_id=cfg['corpus_id'],split_id=cfg['split_id'],
        dataset_sha256=cfg['dataset_sha256'],dev=len(dev),test=len(test),corpus=len(corpus),
        candidate_depth=depth,model=MODEL,device='cpu',threads=threads,python=platform.python_version(),
        model_max_seq_length=model.max_seq_length,selected_configs=chosen,hybrid_dense_weight=alpha,
        dense_top10_exact_agreement_with_C=agreement,
        team_latency_protocol='Batch 128, all queries, encoding + cosine search + top100 + refinement; ms/query',
        interactive_latency_protocol='100 fixed test queries (or requested count), warmup 3, repeats 3, raw query to top5 texts',
        packages={p:importlib.metadata.version(p) for p in ['numpy','pandas','torch','sentence-transformers','transformers']})
    (out/'run_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(pd.DataFrame(test_rows).query('k==5').to_string(index=False),flush=True)
    print(latency.to_string(),flush=True)
    print('Dense top10 agreement with C:',agreement,flush=True)
    return out


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model-path',default=MODEL,help='Model ID or locally cached MiniLM directory')
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--timing-queries',type=int,default=100)
    args=parser.parse_args()
    run(args.model_path,args.threads,args.timing_queries)
