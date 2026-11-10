"""Offline index-only comparison on existing FiQA vectors. No API/encoder calls."""
from pathlib import Path
import hashlib,json,os,platform,time
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'.cache/matplotlib'))
import faiss,numpy as np,pandas as pd
from api import dump
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/person3_meeting9_workflow/hnsw_v1'
EXPORT=ROOT.parent/'outputs/meeting9_20261109/hnsw'

def metrics(rank,rels,ids):
    g=np.array([rels.get(str(ids[i]),0) for i in rank],dtype=float)
    ideal=np.array(sorted(rels.values(),reverse=True)[:10],dtype=float)
    den=np.sum(ideal/np.log2(np.arange(len(ideal))+2))
    return dict(hit10=float(np.any(g>0)),qrels_recall10=float(np.sum(g>0)/sum(v>0 for v in rels.values())),
        ndcg10=float(np.sum(g/np.log2(np.arange(len(g))+2))/den) if den else 0.)

def main():
    OUT.mkdir(parents=True,exist_ok=True);EXPORT.mkdir(parents=True,exist_ok=True)
    prior=ROOT/'results/person3_fiqa_validation/frozen628_v1'
    sel=json.loads((prior/'selection.json').read_text());base=ROOT/'results/person3_cross_dataset/gpt54_focused160_v1'
    z=np.load(base/'fiqa_query_embeddings.npz');q=z['questions'][[x['source_index'] for x in sel]]
    short=np.concatenate([np.load(p) for p in sorted((prior/'vectors/short_30_50').glob('*.npy'))])
    mix=.5*q+.5*short;mix/=np.linalg.norm(mix,axis=1,keepdims=True)
    queries={'question_dense':q,'short_mix50':mix}
    paths=sorted((ROOT/'data/external_benchmarks/pilot_cache_v1/fiqa_corpus').glob('*.npy'))
    docs=np.ascontiguousarray(np.concatenate([np.load(p) for p in paths]),dtype=np.float32)
    assert docs.shape==(57638,384) and q.shape==short.shape==(628,384)
    raw=[json.loads(l) for l in (base/'fiqa_queries.jsonl').read_text().splitlines()]
    rels={str(x['query_id']):x['qrels'] for x in raw};ids=np.load(base/'fiqa_rankings.npz')['passage_ids']
    proto=dict(dataset='FiQA',n=628,corpus=57638,dim=384,representations=list(queries),
        encoder='MiniLM-L6-v2 pinned existing normalized float32 vectors',metric='inner product',
        M=[16,32],efConstruction=100,efSearch=[32,64,128],k=10,threads=1,seed=20261109,repeats=3,
        timing='Warm single-query search only; excludes embedding, disk, network, generation. Shared-machine timings.',
        memory='Serialized index bytes including vectors; not peak RAM.',scope='Descriptive offline efficiency; no prompt/model tuning',
        environment=dict(faiss=faiss.__version__,numpy=np.__version__,system=platform.platform()),
        sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths+[prior/'selection.json',base/'fiqa_query_embeddings.npz']})
    pp=OUT/'protocol.json'
    if pp.exists():assert json.loads(pp.read_text())==proto
    else:dump(pp,proto)
    supplemental=[base/'fiqa_queries.jsonl',base/'fiqa_rankings.npz']+sorted((prior/'vectors/short_30_50').glob('*.npy'))
    dump(OUT/'supplemental_source_hashes.json',{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in supplemental})
    faiss.omp_set_num_threads(1)
    t=time.perf_counter();flat=faiss.IndexFlatIP(384);flat.add(docs);flat_build=time.perf_counter()-t
    exact={name:flat.search(np.ascontiguousarray(v),10)[1] for name,v in queries.items()}
    old=np.load(base/'fiqa_rankings.npz')['question_dense'][[x['source_index'] for x in sel],:10]
    maxdiff=max(float(np.max(np.abs((q[i]@docs[old[i]].T)-(q[i]@docs[exact['question_dense'][i]].T)))) for i in range(628))
    assert maxdiff<=1e-6
    rows=[];summ=[];rankout={}
    configurations=[('exact',flat,0,0,flat_build)]
    for m in [16,32]:
        idx=faiss.IndexHNSWFlat(384,m,faiss.METRIC_INNER_PRODUCT);idx.hnsw.efConstruction=100
        idx.hnsw.rng=faiss.RandomGenerator(20261109)
        t=time.perf_counter();idx.add(docs);build=time.perf_counter()-t
        for ef in [32,64,128]:configurations.append((f'hnsw_m{m}_ef{ef}',idx,m,ef,build))
    evaluate(configurations,queries,exact,sel,rels,ids,docs,maxdiff,rows,summ,rankout)

def evaluate(configurations,queries,exact,sel,rels,ids,docs,maxdiff,rows,summ,rankout):
    rawtimes=[]
    for config,index,m,ef,build in configurations:
        if ef:index.hnsw.efSearch=ef
        size=int(faiss.serialize_index(index).nbytes)
        for name,v in queries.items():
            v=np.ascontiguousarray(v);index.search(v[:10],10)
            lat=[[] for _ in range(628)];ranks=[None]*628
            for trial in range(3):
                for i in np.random.default_rng(20261109+trial).permutation(628):
                    t=time.perf_counter();_,rr=index.search(v[i:i+1],10);elapsed=(time.perf_counter()-t)*1000;lat[i].append(elapsed)
                    rawtimes.append(dict(configuration=config,representation=name,query_id=sel[i]['query_id'],trial=trial,latency_ms=elapsed))
                    if trial==0:ranks[i]=rr[0]
            ranks=np.stack(ranks);rankout[config+'__'+name]=ranks;start=len(rows)
            for i,x in enumerate(sel):
                rows.append(dict(configuration=config,representation=name,query_id=x['query_id'],
                    neighbor_recall10=len(set(ranks[i])&set(exact[name][i]))/10,
                    latency_ms=float(np.median(lat[i])),**metrics(ranks[i],rels[str(x['query_id'])],ids)))
            f=pd.DataFrame(rows[start:]);timed=np.array(lat).ravel()
            summ.append(dict(configuration=config,representation=name,n=628,M=m,efSearch=ef,
                efConstruction=100 if m else 0,index_build_seconds=build,serialized_index_bytes=size,
                vector_bytes=int(docs.nbytes),latency_p50_ms=float(np.median(timed)),latency_p95_ms=float(np.quantile(timed,.95)),
                **{k:float(f[k].mean()) for k in ['neighbor_recall10','hit10','qrels_recall10','ndcg10']}))
            print(config,name,'p50ms',summ[-1]['latency_p50_ms'],'neighbor recall',summ[-1]['neighbor_recall10'],flush=True)
        pd.DataFrame(rows).to_csv(OUT/'per_query.csv',index=False);pd.DataFrame(summ).to_csv(OUT/'summary.csv',index=False)
        np.savez_compressed(OUT/'rankings_top10.npz',**rankout)
        pd.DataFrame(rawtimes).to_csv(OUT/'raw_search_timings.csv',index=False)
    pd.DataFrame(summ).to_csv(EXPORT/'summary.csv',index=False)
    plot(pd.DataFrame(summ))
    dump(OUT/'validation.json',dict(passed=True,queries=628,corpus=57638,configurations=7,
        representations=2,per_query_rows=len(rows),historical_exact_max_score_difference=maxdiff,no_api_calls=True))

def plot(f):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7.6,4.5),layout='constrained')
    for name,marker in [('question_dense','o'),('short_mix50','s')]:
        part=f[(f.representation==name)&(f.M>0)]
        ax.scatter(part.latency_p50_ms,part.neighbor_recall10*100,label=name,marker=marker,s=50)
        for _,r in part.iterrows():ax.annotate(f'M{int(r.M)}/ef{int(r.efSearch)}',(r.latency_p50_ms,r.neighbor_recall10*100),fontsize=7,xytext=(4,-12 if name=='short_mix50' else 5),textcoords='offset points')
    ax.set(xlabel='Warm median search (ms/query; one CPU thread)',ylabel='Exact-neighbor Recall@10 (%)',title='FiQA: HNSW speed versus exact-neighbor recovery')
    ax.margins(x=.14,y=.10);ax.legend();ax.grid(alpha=.2);fig.savefig(EXPORT/'hnsw_tradeoff.png',dpi=180);plt.close(fig)

if __name__=='__main__':main()
