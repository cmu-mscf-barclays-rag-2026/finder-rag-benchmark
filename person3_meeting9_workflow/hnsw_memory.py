"""Isolated process RSS snapshots; supplementary to the fixed HNSW speed grid."""
from pathlib import Path
import argparse,gc,json,os,platform,resource,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1]

def child(name):
    import faiss,numpy as np
    faiss.omp_set_num_threads(1)
    paths=sorted((ROOT/'data/external_benchmarks/pilot_cache_v1/fiqa_corpus').glob('*.npy'))
    v=np.ascontiguousarray(np.concatenate([np.load(p) for p in paths]),dtype=np.float32);gc.collect()
    def rss():return int(subprocess.check_output(['ps','-o','rss=','-p',str(os.getpid())],text=True).strip())*1024
    before=rss()
    if name=='exact':index=faiss.IndexFlatIP(384)
    else:
        index=faiss.IndexHNSWFlat(384,int(name[1:]),faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction=100;index.hnsw.rng=faiss.RandomGenerator(20261109)
    t=time.perf_counter();index.add(v);elapsed=time.perf_counter()-t;after=rss()
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system()!='Darwin':peak*=1024
    print(json.dumps(dict(index=name,n=len(v),dimensions=384,threads=1,vector_bytes=int(v.nbytes),
        rss_before_index_bytes=before,rss_after_index_bytes=after,rss_increment_bytes=after-before,
        peak_process_rss_bytes=peak,serialized_index_bytes=int(faiss.serialize_index(index).nbytes),
        index_build_seconds=elapsed,platform=platform.platform())))

def main():
    import pandas as pd
    rows=[]
    for name in ['exact','m16','m32']:
        p=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--index',name],capture_output=True,text=True,check=True)
        rows.append(json.loads(p.stdout));print('Measured',name,flush=True)
    out=ROOT/'results/person3_meeting9_workflow/hnsw_v1';export=ROOT.parent/'outputs/meeting9_20261109/hnsw'
    f=pd.DataFrame(rows);f.to_csv(out/'isolated_memory.csv',index=False);f.to_csv(export/'isolated_memory.csv',index=False)
    notes=dict(scope='Three independent fresh processes on same machine; one build per index, same cached document vectors and HNSW parameters.',
        rss='OS ps resident snapshots after vector ingestion and after add. Increment includes vector copy, graph and allocation overhead; allocator effects apply.',
        peak='Process high-water RSS includes ingestion/import/transient allocations; not index-only memory.',
        timing='Build times from these isolated memory runs do not replace the fixed-grid timings.',
        no_api_or_encoder_calls=True,rows=3,positive_index_increments=bool((f.rss_increment_bytes>0).all()))
    assert notes['positive_index_increments']
    (out/'memory_validation.json').write_text(json.dumps(notes,indent=2));(export/'memory_validation.json').write_text(json.dumps(notes,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',choices=['exact','m16','m32']);a=p.parse_args()
    child(a.index) if a.index else main()
