"""Measure warm ranking latency on20 development queries, using cached vectors.

Excludes API generation and MiniLM inference. Recomputes lexical/dense rankings
and both RRF steps each timed invocation; does not reuse stored ranking lists.
"""
from pathlib import Path
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import importlib.util
import time
import json
import numpy as np
import pandas as pd

path=Path(__file__).with_name('run.py');spec=importlib.util.spec_from_file_location('meeting11_profile_runner',path)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def main():
    m.check();qa,pages=m.data();mapping={q['id']:q for q in qa};split=m.read(m.OUT/'selection.json')
    querysets=m.querysets();vecs={}
    for variant in ['default','legal']:
        z=np.load(m.OUT/f'{variant}_query_vectors.npz');vecs[variant]=dict(zip(z['texts'].tolist(),z['vectors']))
    z=np.load(m.ROOT/'results/person3_openai_hyde/embeddings.npz');corpus=z['corpus']
    original={q['question']:z['questions'][i] for i,q in enumerate(qa)}
    bm=m.old.BM25(pages)
    def request(qid,variant,backend):
        q=mapping[qid];queries=[q['question']] if variant=='question' else querysets[variant][qid]['queries'];routes=[]
        for text in queries:
            b=bm.indices(text,100) if backend in ['bm25','hybrid'] else None
            if backend in ['dense','hybrid']:
                v=original[text] if text in original else vecs[variant][text]
                d=m.rank(v@corpus.T)
            route=b if backend=='bm25' else d if backend=='dense' else m.fuse([b,d],[.5,.5])
            routes.append(list(map(int,route)))
        return routes[0][:10] if variant=='question' else m.fuse([r[:10] for r in routes],depth=10)
    expected=m.read(m.OUT/'rankings.json');rows=[]
    configs=[(v,b) for b in m.BACKENDS for v in m.VARIANTS]
    for v,b in configs:request(split['dev_ids'][0],v,b)
    for repeat in range(3):
        for qi,qid in enumerate(split['dev_ids']):
            # Rotate configuration order to avoid always measuring one method first.
            offset=(repeat+qi)%len(configs)
            for v,b in configs[offset:]+configs[:offset]:
                start=time.perf_counter_ns();ids=request(qid,v,b);elapsed=(time.perf_counter_ns()-start)/1e6
                assert ids==expected[qid][v+'_'+b]['10']
                rows.append(dict(query_id=qid,method=v+'_'+b,repeat=repeat,cached_vector_ranking_ms=elapsed))
    frame=pd.DataFrame(rows);frame.to_csv(m.OUT/'ranking_latency_samples.csv',index=False)
    frame.groupby('method',as_index=False).agg(n=('query_id','size'),
        p50_ms=('cached_vector_ranking_ms','median'),p95_ms=('cached_vector_ranking_ms',lambda s:s.quantile(.95)),
        mean_ms=('cached_vector_ranking_ms','mean')).to_csv(m.OUT/'ranking_latency_summary.csv',index=False)
    (m.OUT/'ranking_latency_protocol.json').write_text(json.dumps(dict(
        development_questions=20,repeats=3,configurations=9,samples=540,k=10,candidate_depth=100,
        api_calls=0,corpus_vector_cache=True,query_vector_cache=True,bm25_query_transform_included=True,
        score_sort_and_rrf_included=True,api_generation_excluded=True,encoder_inference_excluded=True,
        cpu_threads=1,not_end_to_end=True),indent=2)+'\n')
    print(frame.groupby('method').cached_vector_ranking_ms.median().to_string())


if __name__=='__main__':main()
