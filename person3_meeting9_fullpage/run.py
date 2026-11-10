"""Secondary full-OCR page retrieval control. No API or primary-result writes."""
from pathlib import Path
import argparse, hashlib, importlib.util, json, math, os, time
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/person3_meeting9_fullpage/v1'
EXPORT=ROOT.parent/'outputs/meeting9_20261109/fullpage'
A=ROOT/'results/person3_meeting9_transfer/vidore_finance_v1'
B=ROOT/'results/person3_meeting9_transfer/vidore_finance_remaining249_v1'
CORPUS=A/'data/corpus.jsonl'
KS=[5,10,15,20,25]
METHODS=['question_dense','focused160_pure_dense','focused160_mix_dense','short_30_50_pure_dense','short_30_50_mix_dense']
REV='1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())
    tmp.replace(path)
def frozen(path,value):
    if path.exists():assert read(path)==value,'Frozen artifact changed: '+str(path)
    else:dump(path,value)
def atomicsave(path,v):
    path=Path(path);tmp=path.with_suffix('.tmp.npy');np.save(tmp,v);tmp.replace(path)
def corpus():return [json.loads(x) for x in CORPUS.read_text().splitlines() if x.strip()]
def selections():return read(A/'selection.json')['queries']+read(B/'selection.json')['queries']
def helper():
    s=importlib.util.spec_from_file_location('fullpage_offline_helpers',ROOT/'person3_cross_dataset/run.py')
    h=importlib.util.module_from_spec(s);s.loader.exec_module(h);h.OUT=OUT;return h

def prepare():
    OUT.mkdir(parents=True,exist_ok=True);EXPORT.mkdir(parents=True,exist_ok=True)
    rows=corpus();qs=selections();assert len(rows)==2942 and len(qs)==309 and len({q['query_id'] for q in qs})==309
    protocol={'status':'Post-primary secondary descriptive representation control; never replaces frozen60/249 primary tests',
      'trigger':'93.95% supplied OCR pages exceed the original256-token MiniLM page input allowance',
      'dataset':'vidore/vidore_v3_finance_en','dataset_revision':'7f432c176d82e27546501ad8064a713ac3071809',
      'queries':309,'stages':{'initial60':60,'remaining249':249},'corpus_pages':2942,
      'encoder':'sentence-transformers/all-MiniLM-L6-v2','encoder_revision':REV,'encoder_tokens':256,
      'encoder_backend':'offline CPU float32, masked mean pooling and L2 normalization',
      'chunk_tokens':224,'overlap_tokens':32,'stride_tokens':192,'chunk_source':'existing person3_cross_dataset.run.chunks_for offset-preserving helper; no label-dependent choices',
      'empty_page':'one empty chunk retained; do not remove corpus pages','document_representation':'all supplied page OCR, including every source token; no image encoder',
      'page_score':'maximum cosine similarity over all chunks belonging to the page; stable page-index ties',
      'query_representations':METHODS,'mix_question_weight':.5,'depth':100,'top_k':KS,
      'bm25':'Reuse original-query full-OCR BM25 rankings unchanged',
      'comparison':'Within each query representation, compare original first256 page representation versus full-page chunk maximum; fixed same query IDs and all2942pages',
      'metric':'Hit, Recall, MRR, nDCG with linear graded relevance gains','inference':'All new results descriptive; zero new p-values; do not supersede original primary experiments',
      'generation':'Reuse exact saved questions and original Focused/Short generation vectors. No new API calls.',
      'limitations':['Page maximum introduces multiple opportunities for longer pages; this changes document representation, not the prompt alone.','Provided OCR can omit visual/table structure. This is text-only, not a visual-model evaluation.','Full309 summaries combine two frozen stages and are descriptive only.','Query vectors and generated paragraphs retain original MiniLM256-token truncation; document OCR coverage is the intervention.']}
    frozen(OUT/'protocol.json',protocol)
    frozen(OUT/'selection.json',{'queries':qs,'initial60_ids':[q['query_id'] for q in read(A/'selection.json')['queries']],
        'remaining249_ids':[q['query_id'] for q in read(B/'selection.json')['queries']]})
    sourcefiles=[CORPUS,A/'data/manifest.json',A/'selection.json',B/'selection.json',A/'vectors/questions.npy',B/'vectors/questions.npy',ROOT/'person3_cross_dataset/run.py']
    frozen(OUT/'source_hashes.json',{str(p.relative_to(ROOT)):sha(p) for p in sourcefiles})
    print('Frozen secondary control: all309queries/all2942pages; no new tests or API calls',flush=True)

def build_index():
    h=helper();e=h.encoder();e.tokenizer.model_max_length=10**9;rows=corpus()
    path=OUT/'chunks.json'
    if path.exists():manifest=read(path)
    else:
        pages=[];chunks=[];starts=[0];lengths=[];source_counts=[]
        for pi,page in enumerate(rows):
            text=page['markdown'];texts,spans=h.chunks_for(e,text)
            tok=e.tokenizer(text,add_special_tokens=False,truncation=False,return_offsets_mapping=True,verbose=False)
            offsets=tok['offset_mapping'];source_counts.append(len(offsets))
            actual_lengths=e.tokenizer(texts,add_special_tokens=True,truncation=False,return_length=True,verbose=False)['length']
            assert all(n<=256 for n in actual_lengths),'A chunk would be truncated; do not silently claim full coverage'
            assert len(texts)==len(spans)==len(actual_lengths)
            records=[]
            for ci,(t,span,n) in enumerate(zip(texts,spans,actual_lengths)):
                assert text[span[0]:span[1]]==t
                records.append({'chunk_index':len(chunks),'char_start':span[0],'char_end':span[1],
                  'source_token_start':ci*192 if offsets else 0,'source_token_end':min(ci*192+224,len(offsets)),
                  'encoder_tokens_including_special':n,'text_sha256':hashlib.sha256(t.encode()).hexdigest()})
                chunks.append(t);lengths.append(n)
            assert records and all(records[i]['source_token_end']>=records[i+1]['source_token_start'] for i in range(len(records)-1))
            assert records[0]['source_token_start']==0 and records[-1]['source_token_end']==len(offsets)
            # Verify actual chunk character spans enclose every source token.
            for j,(a,b) in enumerate(offsets):
                assert any(c['source_token_start']<=j<c['source_token_end'] and c['char_start']<=a<=b<=c['char_end'] for c in records)
            pages.append({'page_index':pi,'page_id':str(page['corpus_id']),'ocr_sha256':hashlib.sha256(text.encode()).hexdigest(),
              'source_tokens':len(offsets),'chunk_start':starts[-1],'chunk_end':len(chunks),'chunks':records})
            starts.append(len(chunks))
        manifest={'pages':pages,'chunk_texts':chunks,'page_chunk_starts':starts,'coverage':{'pages':len(rows),'chunks':len(chunks),
          'empty_token_pages':sum(n==0 for n in source_counts),'all_source_tokens_covered':True,'all_chunk_source_spans_verified':True,
          'encoder_token_min_including_special':min(lengths),'encoder_token_max_including_special':max(lengths),
          'source_tokens_total':sum(source_counts),'chunks_have_no_encoder_truncation':True}}
        frozen(path,manifest);print(json.dumps(manifest['coverage']),flush=True)
    folder=OUT/'chunk_vectors';folder.mkdir(exist_ok=True);timings=[]
    for start in range(0,len(manifest['chunk_texts']),512):
        p=folder/f'{start:07d}.npy';batch=manifest['chunk_texts'][start:start+512]
        if not p.exists():
            t=time.perf_counter();v=e.encode(batch);elapsed=time.perf_counter()-t
            atomicsave(p,v);timings.append({'chunk_start':start,'n':len(batch),'seconds':elapsed})
            print('Encoded fullpage chunks',start+len(batch),'/',len(manifest['chunk_texts']),flush=True)
        v=np.load(p);assert v.shape==(len(batch),384) and v.dtype==np.float32 and np.isfinite(v).all() and np.allclose(np.linalg.norm(v,axis=1),1,atol=1e-5)
    previous=read(OUT/'encoding_timings.json') if (OUT/'encoding_timings.json').exists() else []
    dump(OUT/'encoding_timings.json',previous+timings)
    dump(OUT/'index_validation.json',{'complete':True,**manifest['coverage'],'normalized_float32_vectors':True,
      'vectors':len(manifest['chunk_texts']),'dim':384,'api_calls':0,'vector_bytes':len(manifest['chunk_texts'])*384*4,
      'chunk_manifest_sha256':sha(path),'vector_hashes':{p.name:sha(p) for p in sorted(folder.glob('*.npy'))}})

def resources(timing_file=None):
    index=read(OUT/'index_validation.json');timings=read(OUT/'encoding_timings.json')
    result={'corpus_pages':2942,'fullpage_chunks':index['vectors'],'float32_dim':384,'fullpage_vector_bytes':index['vector_bytes'],
      'first256_page_vector_bytes':2942*384*4,'vector_count_storage_ratio':index['vectors']/2942,
      'encoded_chunks_with_retained_timing':sum(x['n'] for x in timings),'summed_batch_encode_seconds':sum(x['seconds'] for x in timings),
      'encoding_timing_excludes':'model load, token chunk preparation, disk writes','api_calls':0,
      'warning':'Shared-machine local timings are descriptive, not a fixed SLA or end-to-end latency.'}
    if timing_file is not None:
        f=pd.read_csv(timing_file);result['search_timing_summary']=[{'method':m,'n':len(g),'p50_seconds':float(g.seconds.median()),'p95_seconds':float(g.seconds.quantile(.95))} for m,g in f.groupby('method',sort=False)]
        result['search_timing_excludes']='disk/model/embedding/generation; includes chunk dot products, page max aggregation, and top100 sort'
    dump(OUT/'resources.json',result)

def score(rank,rels,k):
    grades=[rels.get(str(p),0) for p in rank[:k]];found=[i+1 for i,v in enumerate(grades) if v>0]
    ideal=sorted(rels.values(),reverse=True)[:k];den=sum(v/math.log2(i+2) for i,v in enumerate(ideal))
    return {'hit':float(bool(found)),'recall':len(found)/sum(v>0 for v in rels.values()),
            'mrr':1/found[0] if found else 0.,'ndcg':sum(v/math.log2(i+2) for i,v in enumerate(grades))/den if den else 0.}

def evaluate(baseline_only=False):
    h=helper();manifest=read(OUT/'chunks.json');assert read(OUT/'index_validation.json')['complete']
    ce=np.concatenate([np.load(p) for p in sorted((OUT/'chunk_vectors').glob('*.npy'))]);starts=np.array(manifest['page_chunk_starts'])
    qs=selections();qids=[q['query_id'] for q in qs];pids=[p['page_id'] for p in manifest['pages']]
    qe=np.concatenate([np.load(p/'vectors/questions.npy') for p in [A,B]]);assert qe.shape==(309,384) and qe.dtype==np.float32 and np.isfinite(qe).all() and np.allclose(np.linalg.norm(qe,axis=1),1,atol=1e-5)
    vectors={'question_dense':qe};sourcevec=[A/'vectors/questions.npy',B/'vectors/questions.npy']
    if not baseline_only:
        for style in ['focused160','short_30_50']:
            needed=[p/'vectors'/f'{style}.npy' for p in [A,B]]
            assert all(p.exists() for p in needed),'Missing remaining249 HyDE vectors; do not impute or call API'
            he=np.concatenate([np.load(p) for p in needed]);assert he.shape==(309,384)
            vectors[style+'_pure_dense']=he;vectors[style+'_mix_dense']=h.normalize(.5*qe+.5*he);sourcevec+=needed
    ranks={};rows=[];timing=[]
    originals=[]
    for folder in [A,B]:
        original=np.load(folder/('baseline_rankings.npz' if baseline_only else 'rankings.npz'));originals.append(original)
    for z,folder in zip(originals,[A,B]):
        assert z['query_ids'].tolist()==[q['query_id'] for q in read(folder/'selection.json')['queries']]
        assert z['passage_ids'].tolist()==pids
    old={m:np.concatenate([z[m] for z in originals]) for m in list(vectors)+['bm25']}
    for method,v in vectors.items():
        assert v.shape==(309,384) and v.dtype==np.float32 and np.isfinite(v).all() and np.allclose(np.linalg.norm(v,axis=1),1,atol=1e-5)
        current=[]
        for q,x in zip(qs,v):
            t=time.perf_counter();chunk_scores=ce@x;page_scores=np.maximum.reduceat(chunk_scores,starts[:-1]);rank=np.argsort(-page_scores,kind='stable')[:100];elapsed=time.perf_counter()-t
            assert page_scores.shape==(2942,) and np.isfinite(page_scores).all();current.append(rank)
            timing.append({'query_id':q['query_id'],'method':method,'seconds':elapsed})
        ranks[method]=np.stack(current)
    np.savez_compressed(OUT/('baseline_rankings.npz' if baseline_only else 'rankings.npz'),query_ids=np.array(qids),passage_ids=np.array(pids),**ranks)
    rel={**read(A/'qrels.json'),**read(B/'qrels.json')}
    for representation,rr in [('page_first256',old),('fullpage_chunkmax',ranks)]:
        for method,arrays in rr.items():
            if method=='bm25':continue
            for qi,(q,rank) in enumerate(zip(qs,arrays)):
                for k in KS:rows.append({'query_id':q['query_id'],'cohort':'initial60' if qi<60 else 'remaining249','method':method,'representation':representation,'k':k,**score([pids[i] for i in rank],rel[q['query_id']],k)})
    for qi,(q,rank) in enumerate(zip(qs,old['bm25'])):
        for k in KS:rows.append({'query_id':q['query_id'],'cohort':'initial60' if qi<60 else 'remaining249','method':'bm25','representation':'full_ocr_bm25','k':k,**score([pids[i] for i in rank],rel[q['query_id']],k)})
    f=pd.DataFrame(rows);prefix='baseline_' if baseline_only else ''
    f.to_csv(OUT/(prefix+'per_query_metrics.csv'),index=False)
    frames=[f,f.assign(cohort='full309')];summary=pd.concat(frames).groupby(['cohort','method','representation','k'],sort=False).agg(n=('query_id','size'),**{m:(m,'mean') for m in ['hit','recall','mrr','ndcg']}).reset_index();summary.to_csv(OUT/(prefix+'summary.csv'),index=False)
    pd.DataFrame(timing).to_csv(OUT/(prefix+'retrieval_timings.csv'),index=False)
    resources(OUT/(prefix+'retrieval_timings.csv'))
    frozen(OUT/('baseline_vector_source_hashes.json' if baseline_only else 'query_vector_source_hashes.json'),{str(p.relative_to(ROOT)):sha(p) for p in sourcevec})
    if baseline_only:
        dump(OUT/'baseline_completion.json',{'complete':True,'queries':309,'dense_methods':1,'secondary_descriptive_only':True,'api_calls':0});print('Fullpage question baseline complete for309 queries',flush=True)
    else:
        validate();report();dump(OUT/'completion_status.json',{'complete':True,'queries':309,'corpus_pages':2942,'dense_query_representations':5,'api_calls':0,'new_pvalues':0,'metric_rows':len(f),'secondary_descriptive_only':True});print('Fullpage complete:309queries,5dense representations plus fullOCR BM25; no new p-values',flush=True)

def validate():
    for rel,dig in read(OUT/'source_hashes.json').items():assert sha(ROOT/rel)==dig
    for rel,dig in read(OUT/'query_vector_source_hashes.json').items():assert sha(ROOT/rel)==dig
    m=read(OUT/'chunks.json');v=read(OUT/'index_validation.json');assert sha(OUT/'chunks.json')==v['chunk_manifest_sha256']
    for name,dig in v['vector_hashes'].items():assert sha(OUT/'chunk_vectors'/name)==dig
    ce=np.concatenate([np.load(p) for p in sorted((OUT/'chunk_vectors').glob('*.npy'))]);ranks=np.load(OUT/'rankings.npz');qs=selections();ids=ranks['passage_ids'].tolist();rel={**read(A/'qrels.json'),**read(B/'qrels.json')}
    metric=pd.read_csv(OUT/'per_query_metrics.csv',dtype={'query_id':str});assert len(metric)==309*11*5 and not metric.duplicated(['query_id','method','representation','k']).any()
    # Original first256 and BM25 rows must exactly reproduce both frozen primary stages.
    for folder in [A,B]:
        original=pd.read_csv(folder/'per_query_metrics.csv',dtype={'query_id':str})
        for x in original.to_dict('records'):
            rep='full_ocr_bm25' if x['method']=='bm25' else 'page_first256'
            got=metric[(metric.query_id==x['query_id'])&(metric.method==x['method'])&(metric.representation==rep)&(metric.k==x['k'])]
            assert len(got)==1
            for col in ['hit','recall','mrr','ndcg']:assert abs(got.iloc[0][col]-x[col])<1e-12
    for method in METHODS:
        arr=ranks[method];assert arr.shape==(309,100) and all(len(set(x))==100 for x in arr)
        for q,rank in zip(qs,arr):
            for k in KS:
                row=metric[(metric.query_id==q['query_id'])&(metric.method==method)&(metric.representation=='fullpage_chunkmax')&(metric.k==k)].iloc[0]
                computed=score([ids[i] for i in rank],rel[q['query_id']],k)
                assert all(abs(row[c]-x)<1e-12 for c,x in computed.items())
    # Independent page-aggregation equality on a deterministic query subset, every page.
    qvec=np.concatenate([np.load(p/'vectors/questions.npy') for p in [A,B]])
    for qi in [0,59,60,308]:
        chunk_scores=ce@qvec[qi];page_scores=np.array([max(chunk_scores[p['chunk_start']:p['chunk_end']]) for p in m['pages']]);independent=sorted(range(2942),key=lambda i:(-float(page_scores[i]),i))[:100]
        assert np.array_equal(independent,ranks['question_dense'][qi])
    summary=pd.read_csv(OUT/'summary.csv');assert len(summary)==165
    for row in summary.to_dict('records'):
        f=metric[(metric.method==row['method'])&(metric.representation==row['representation'])&(metric.k==row['k'])]
        if row['cohort']!='full309':f=f[f.cohort==row['cohort']]
        assert len(f)==row['n']=={'initial60':60,'remaining249':249,'full309':309}[row['cohort']]
        for c in ['hit','recall','mrr','ndcg']:assert abs(row[c]-f[c].mean())<1e-12
    dump(OUT/'validation.json',{'passed':True,'all_source_token_coverage':True,'no_document_chunk_truncation':True,'normalized_float32_vectors':True,'queries':309,'corpus_pages':2942,'metric_rows':len(metric),'summary_rows':len(summary),'max_chunk_page_aggregation_independent_checks':4*2942,'all_summary_denominators_verified':True,'original_first256_and_bm25_metrics_reproduced':True,'query_vector_source_hashes_verified':True,'api_calls':0,'new_pvalues':0,'status':'secondary descriptive control; original primary untouched'})

def report():
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'));import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    summary=pd.read_csv(OUT/'summary.csv');m=read(OUT/'chunks.json');labels={'question_dense':'Question','focused160_pure_dense':'Focused pure','focused160_mix_dense':'Focused mix','short_30_50_pure_dense':'Short pure','short_30_50_mix_dense':'Short mix'}
    fig,ax=plt.subplots(figsize=(8.6,4.8));x=np.arange(5);width=.36;data=summary[(summary.cohort=='full309')&(summary.k==10)]
    for i,(rep,label,color) in enumerate([('page_first256','First 256 tokens/page','#9ba6b0'),('fullpage_chunkmax','Full OCR chunk maximum','#347da5')]):
        vals=data[data.representation==rep].set_index('method').loc[METHODS,'ndcg'];bars=ax.bar(x+(i-.5)*width,vals,width,label=label,color=color)
        for bar,val in zip(bars,vals):ax.text(bar.get_x()+bar.get_width()/2,val+.006,f'{val:.3f}',ha='center',fontsize=8)
    bm=data[data.method=='bm25'].ndcg.iloc[0];ax.axhline(bm,color='#cc783d',linestyle='--',label=f'Full OCR BM25 ({bm:.3f})');ax.set_xticks(x,[labels[m] for m in METHODS]);ax.set_ylabel('nDCG@10');ax.set_ylim(0,max(data.ndcg)*1.24);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);ax.legend(frameon=False,fontsize=9,loc='upper left');fig.suptitle('ViDoRe Finance: full-page OCR coverage control (309 queries)');fig.subplots_adjust(left=.09,right=.98,bottom=.16,top=.86);fig.text(.5,.035,'Secondary descriptive control. All 2,942 pages; no new p-values; text only.',ha='center',fontsize=9);fig.savefig(OUT/'fullpage_control.png',dpi=180);plt.close(fig)
    lines=['# Full-page OCR coverage control','','Secondary descriptive analysis triggered by the original page truncation audit. Original frozen 60/249-query prompt-transfer protocols, scores, and primary tests remain unchanged. Same 309 English questions and all 2,942 page OCR documents. No API calls, visual model, or new p-values.','','Each page is represented by all 224-token chunks with 32-token overlap. Page score is the maximum normalized MiniLM chunk cosine. Original question and saved Focused/Short query vectors are reused. BM25 still uses full supplied OCR.','','![Page coverage control](fullpage_control.png)','']
    cohort_labels={'initial60':'Initial 60 queries','remaining249':'Remaining 249 queries','full309':'All 309 queries'}
    for cohort in ['initial60','remaining249','full309']:
        d=summary[(summary.cohort==cohort)&(summary.k==10)];lines += [f"## {cohort_labels[cohort]} (descriptive)",'','| Query representation | First256 nDCG@10 | Fullpage nDCG@10 | Difference | Fullpage Hit@10 | Fullpage Recall@10 |','|---|---:|---:|---:|---:|---:|']
        for method in METHODS:
            rows=d[d.method==method].set_index('representation');a=rows.loc['page_first256'];b=rows.loc['fullpage_chunkmax'];lines.append(f'| {labels[method]} | {a.ndcg:.4f} | {b.ndcg:.4f} | {b.ndcg-a.ndcg:+.4f} | {b.hit:.1%} | {b.recall:.4f} |')
        bm=d[d.method=='bm25'].iloc[0];lines.append(f'| Full OCR BM25 (same baseline) | {bm.ndcg:.4f} | {bm.ndcg:.4f} | 0 | {bm.hit:.1%} | {bm.recall:.4f} |')
    c=m['coverage'];lines += ['',f"Coverage: {c['pages']:,} pages, {c['chunks']:,} chunks, {c['source_tokens_total']:,} source tokens; every source token covered. Actual re-tokenized chunk inputs including special tokens range {c['encoder_token_min_including_special']}–{c['encoder_token_max_including_special']}; none exceeds 256. Empty-token pages retained: {c['empty_token_pages']}.",'',f"The full-page index stores {c['chunks']*384*4:,} float32 vector bytes versus {2942*384*4:,} bytes for one vector per page ({c['chunks']/2942:.2f}x). Encoding and local search timing records are in resources.json; these are not end-to-end latency.",'','Max-chunk scoring changes document representation and gives longer pages more chances to match. Differences from first256 do not isolate prompt effects or establish improved final-answer accuracy. OCR may omit visual information. No visual retrieval claim is made.','','## Cached reproduction','','```bash','.venv/bin/python person3_meeting9_fullpage/run.py prepare','.venv/bin/python person3_meeting9_fullpage/run.py index','.venv/bin/python person3_meeting9_fullpage/run.py evaluate','.venv/bin/python person3_meeting9_fullpage/run.py validate','```','','Successful chunk embedding batches are reused. `baseline` evaluates question-only plus unchanged BM25 before all generation vectors become available. No command calls an API.']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n');EXPORT.mkdir(parents=True,exist_ok=True)
    for name in ['README.md','fullpage_control.png','summary.csv','per_query_metrics.csv','protocol.json','validation.json','index_validation.json','query_vector_source_hashes.json','retrieval_timings.csv','resources.json']:(EXPORT/name).write_bytes((OUT/name).read_bytes())

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','index','baseline','evaluate','validate','report']);args=parser.parse_args()
    if args.stage=='prepare':prepare()
    elif args.stage=='index':build_index()
    elif args.stage=='baseline':evaluate(baseline_only=True)
    elif args.stage=='evaluate':evaluate()
    elif args.stage=='validate':validate()
    elif args.stage=='report':report()
