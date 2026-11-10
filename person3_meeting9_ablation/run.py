"""Cached, dense-only 2x2 prompt ablation. This program never calls an API."""
from pathlib import Path
import argparse, hashlib, importlib.util, json, os, sys, time, math
import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/person3_meeting9_ablation/factorial20_v1'
EXPORT = ROOT.parent / 'outputs/meeting9_20261109/ablation'
PRIOR = ROOT / 'results/person3_hyde_pilot/focused_weight_v1'
LEGAL = ROOT / 'results/person3_openai_hyde'
CROSS = ROOT / 'results/person3_cross_dataset/gpt54_focused160_v1'
CACHE = ROOT / 'data/external_benchmarks/pilot_cache_v1'
MODEL = 'gpt-5.4-2026-03-05'
EMBED = 'sentence-transformers/all-MiniLM-L6-v2'
REV = '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
KS = [5, 10, 15, 20, 25]
STYLES = {'short_base': (30, 50, False), 'long_base': (70, 100, False),
          'short_constraints': (30, 50, True), 'long_constraints': (70, 100, True)}
CONSTRAINT = " Preserve all explicitly stated entities, quantities, time horizons, jurisdictions, negations, comparison criteria, and relationships; do not fill in unstated conditions."
TEMPLATE = "Write one hypothetical evidence paragraph of {low} to {high} words to help retrieve passages for the question below about {domain}. Do not invent names, numerical values, dates, citations, or facts about the questioner. Output only the paragraph with no heading.{constraint}\nQuestion: {question}\nHypothetical evidence paragraph:"
EDGES = [('short_base', 'long_base', 'length_without_constraints'),
         ('short_constraints', 'long_constraints', 'length_with_constraints'),
         ('short_constraints', 'short_base', 'constraints_at_short_length'),
         ('long_constraints', 'long_base', 'constraints_at_long_length')]

def digest(b): return hashlib.sha256(b).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, ensure_ascii=False, indent=2); f.flush(); os.fsync(f.fileno())
    tmp.replace(path)
def frozen(path, value):
    if path.exists(): assert read(path) == value, f'Frozen artifact changed: {path}'
    else: dump(path, value)
def lines(path): return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
def write_lines(path, rows):
    path = Path(path); tmp = path.with_suffix('.tmp')
    with tmp.open('w') as f:
        for row in rows: f.write(json.dumps(row, ensure_ascii=False) + '\n')
        f.flush(); os.fsync(f.fileno())
    tmp.replace(path)
def call_id(prompt):
    return 'meeting9_ablation::' + digest(json.dumps({'model':MODEL,'prompt':prompt,'max_output_tokens':160,'temperature':0,'reasoning_effort':'none','store':False,'tools':[]},sort_keys=True).encode())

def prepare():
    OUT.mkdir(parents=True, exist_ok=True); EXPORT.mkdir(parents=True, exist_ok=True)
    sel = read(PRIOR/'selection.json')
    datasets = {ds: sel['datasets'][ds] for ds in ['legal', 'fiqa']}
    frozen(OUT/'selection.json', {'seed':sel['seed'],'datasets':datasets,'status':'Previously inspected exploratory subset; not a new holdout'})
    protocol = {'experiment':'2x2 requested length x explicit constraint preservation','n_per_dataset':20,
      'model':MODEL,'settings':{'temperature':0,'reasoning_effort':'none','max_output_tokens':160,'store':False,'tools':[],'sdk_retries':0},
      'encoder':EMBED,'encoder_revision':REV,'encoder_tokens':256,'backend':'offline CPU float32; masked mean pooling; L2 normalization',
      'styles':STYLES,'base_template':TEMPLATE,'constraint_sentence':CONSTRAINT,'mix_question_weight':0.5,
      'routes':['pure_dense','mix_dense'],'top_k':KS,'primary_metrics':{'legal':'Hit@5','fiqa':'nDCG@10'},
      'primary_edges':EDGES,'primary_route':'mix_dense','primary_average_effects':['short versus long, averaged over constraints','constraints versus base, averaged over length'],'secondary_route':'pure_dense descriptive only','holm_family':'12 two-sided tests = (4 edges + 2 averaged factorial main effects) x mixed50 only x 2 datasets',
      'test_legal':'exact paired McNemar for binary edge effects; exact sign-flip for averaged effects','test_fiqa':'exact sign-flip randomization of nonzero paired differences (at most 2^20 permutations)',
      'interpretation':'Exploratory reused 20-query samples; no post-hoc selection or additional generation. Manipulates requested length, not enforced actual length. Equal 160-token cap can censor long outputs; retain and report.',
      'new_generations_expected':160,'new_api_calls_by_this_program':0,
      'baseline_files':{str(p.relative_to(ROOT)):digest(p.read_bytes()) for p in [LEGAL/'saved_rankings.npz',CROSS/'fiqa_rankings.npz',PRIOR/'selection.json']}}
    # JSON canonicalization preserves tuples as arrays on subsequent runs.
    protocol=json.loads(json.dumps(protocol)); frozen(OUT/'protocol.json',protocol)
    jobs=[]
    for ds, qs in datasets.items():
        domain={'legal':'Victorian criminal law','fiqa':'finance'}[ds]
        for q in qs:
            for style,(low,high,has_constraint) in STYLES.items():
                prompt=TEMPLATE.format(low=low,high=high,domain=domain,question=q['question'],constraint=CONSTRAINT if has_constraint else '')
                jobs.append({'call_id':call_id(prompt),'dataset':ds,'query_id':q['query_id'],'style':style,'prompt':prompt,
                             'model':MODEL,'max_output_tokens':160,'temperature':0,'reasoning_effort':'none','store':False,'tools':[]})
    frozen(OUT/'jobs.json', jobs)
    completed=generated(required=False)
    pending=[j for j in jobs if j['call_id'] not in completed]
    write_lines(OUT/'pending_calls.jsonl', pending)
    estimate=sum(((len(j['prompt'].encode())+256)*2.5+160*15)/1e6 for j in pending)
    dump(OUT/'preparation.json',{'frozen_queries':{k:len(v) for k,v in datasets.items()},'jobs':len(jobs),'pending':len(pending),
        'conservative_new_call_reservation_usd':estimate,'pricing':'historical runner rates; parent must verify before paid calls',
        'injection':'Shared workflow writes successful_api_outputs.json list; each row keeps job call_id,text,model_returned,status,usage,latency_seconds.'})
    print(json.dumps(read(OUT/'preparation.json'),indent=2))

def generated(required=True):
    path=OUT/'successful_api_outputs.json'
    if not path.exists():
        if required: raise RuntimeError('Missing successful_api_outputs.json; no results are imputed')
        return {}
    rows=read(path); assert len(rows)==len({r['call_id'] for r in rows})
    jobs={j['call_id']:j for j in read(OUT/'jobs.json')}
    result={}
    for row in rows:
        assert row['call_id'] in jobs
        assert row['text'].strip() and row.get('model_returned',row.get('model'))==MODEL
        assert row.get('status') in ['completed','incomplete']
        if 'prompt' in row: assert row['prompt']==jobs[row['call_id']]['prompt']
        assert row.get('usage') and row['usage'].get('output_tokens',0)>0
        result[row['call_id']]=row
    if required: assert len(result)==160, f'Only {len(result)}/160 generations; no partial average'
    return result

def helper():
    spec=importlib.util.spec_from_file_location('ablation_encoder',ROOT/'person3_cross_dataset/run.py')
    h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
    h.OUT=OUT; return h

def corpus(ds):
    if ds=='legal':
        z=np.load(LEGAL/'embeddings.npz')
        ids=pd.read_csv(LEGAL/'passage_index.csv',dtype=str).passage_id.tolist()
        return z['corpus'],ids
    ids=np.load(CROSS/'fiqa_rankings.npz')['passage_ids'].tolist()
    manifest=read(CACHE/'fiqa_corpus/manifest.json');assert manifest['count']==57638 and manifest['embedding_revision']==REV
    blocks=[]
    for start in range(0,57638,512):
        v=np.load(CACHE/'fiqa_corpus'/f'{start:07d}.npy')
        assert v.shape==(min(512,57638-start),384) and v.dtype==np.float32 and np.isfinite(v).all()
        blocks.append(v)
    return np.concatenate(blocks),ids

def metrics(docids,rels,k):
    grades=np.array([rels.get(str(d),0) for d in docids[:k]],dtype=float)
    relevant=np.flatnonzero(grades>0);ideal=sorted(rels.values(),reverse=True)[:k]
    dcg=float(np.sum(grades/np.log2(np.arange(len(grades))+2)))
    idcg=float(np.sum(np.array(ideal)/np.log2(np.arange(len(ideal))+2)))
    return {'hit':float(len(relevant)>0),'recall':len(relevant)/sum(v>0 for v in rels.values()),
            'mrr':1/(int(relevant[0])+1) if len(relevant) else 0.,'ndcg':dcg/idcg if idcg else 0.}

def evaluate():
    gs=generated(); jobs=read(OUT/'jobs.json');selection=read(OUT/'selection.json')['datasets']
    h=helper();e=h.encoder(); rows=[];audit=[];genaudit=[];timings=[]
    for ds,qs in selection.items():
        c,ids=corpus(ds);indices=[q['source_index'] for q in qs]
        old=np.load((LEGAL/'saved_rankings.npz') if ds=='legal' else (CROSS/'fiqa_rankings.npz'))
        qvec=np.load(LEGAL/'embeddings.npz')['questions'][indices] if ds=='legal' else np.load(CROSS/'fiqa_query_embeddings.npz')['questions'][indices]
        started=time.perf_counter();fresh=e.encode([q['question'] for q in qs]);elapsed=time.perf_counter()-started
        timings.append({'stage':'encode_question_batch','dataset':ds,'style':'question','route':'question_dense','batch_size':20,'query_id':None,'latency_seconds':elapsed,'cache_reused':False})
        delta=float(np.max(np.abs(fresh-qvec)));assert delta<=1e-5
        audit.append({'dataset':ds,'question_coordinate_max_difference':delta})
        rankdict={m:old[m][indices] for m in ['question_dense','focused160_pure_dense','focused160_mix_dense']}
        for i,q in enumerate(qs):
            scores=qvec[i]@c.T; ranks=h.top(scores);saved=rankdict['question_dense'][i]
            diff=float(np.max(np.abs(scores[ranks]-scores[saved])))
            assert diff<=1e-6
            audit.append({'dataset':ds,'query_id':q['query_id'],'question_rank_exact':bool(np.array_equal(ranks,saved)),'max_rank_score_difference':diff})
        for style in STYLES:
            dsjobs=[next(j for j in jobs if j['dataset']==ds and j['query_id']==q['query_id'] and j['style']==style) for q in qs]
            texts=[gs[j['call_id']]['text'] for j in dsjobs]
            vectorfile=OUT/f'{ds}_{style}_vectors.npz'
            textsha=digest(json.dumps(texts,ensure_ascii=False).encode())
            if vectorfile.exists():
                z=np.load(vectorfile);assert str(z['text_sha256'])==textsha;v=z['vectors']
            else:
                started=time.perf_counter();v=e.encode(texts);elapsed=time.perf_counter()-started
                timings.append({'stage':'encode_hyde_batch','dataset':ds,'style':style,'route':'both','batch_size':20,'query_id':None,'latency_seconds':elapsed,'cache_reused':False})
                np.savez_compressed(vectorfile,vectors=v,text_sha256=np.array(textsha))
            assert v.shape==(20,384) and v.dtype==np.float32 and np.isfinite(v).all() and np.allclose(np.linalg.norm(v,axis=1),1,atol=1e-5)
            for route in ['pure_dense','mix_dense']:
                vectors=v if route=='pure_dense' else h.normalize(.5*qvec+.5*v)
                ranked=[]
                for q,x in zip(qs,vectors):
                    started=time.perf_counter();ranked.append(h.top(x@c.T));elapsed=time.perf_counter()-started
                    timings.append({'stage':'exact_search','dataset':ds,'style':style,'route':route,'batch_size':1,'query_id':q['query_id'],'latency_seconds':elapsed,'cache_reused':False})
                rankdict[f'{style}_{route}']=np.stack(ranked)
            for j,text in zip(dsjobs,texts):
                o=gs[j['call_id']];lo,hi,_=STYLES[style];words=len(text.split())
                genaudit.append({'dataset':ds,'query_id':j['query_id'],'style':style,'call_id':j['call_id'],'word_count':words,
                  'target_low':lo,'target_high':hi,'outside_target':not lo<=words<=hi,'status':o['status'],
                  'input_tokens':o['usage'].get('input_tokens',0),'output_tokens':o['usage'].get('output_tokens',0),
                  'length_capped':o['status']=='incomplete','latency_seconds':o.get('latency_seconds',None)})
        np.savez_compressed(OUT/f'{ds}_rankings.npz',query_ids=np.array([q['query_id'] for q in qs]),passage_ids=np.array(ids),**rankdict)
        for method,ranks in rankdict.items():
            for q,rank in zip(qs,ranks):
                rels={q['relevant_passage_id']:1} if ds=='legal' else q['qrels']
                for k in KS:
                    rows.append({'dataset':ds,'query_id':q['query_id'],'method':method,'k':k,**metrics([ids[i] for i in rank],rels,k)})
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'per_query_metrics.csv',index=False)
    frame.groupby(['dataset','method','k'],sort=False).agg(n=('query_id','size'),**{m:(m,'mean') for m in ['hit','recall','mrr','ndcg']}).reset_index().to_csv(OUT/'summary.csv',index=False)
    pd.DataFrame(genaudit).to_csv(OUT/'generation_audit.csv',index=False)
    dump(OUT/'baseline_replay.json',audit)
    if (OUT/'timing_rows.csv').exists():
        previous=pd.read_csv(OUT/'timing_rows.csv').to_dict('records')
        # Keep first observed encoding runs on cached-only reruns, but fresh search times.
        timings=[x for x in previous if x['stage']!='exact_search']+timings
    tf=pd.DataFrame(timings)
    tf=pd.concat([tf[tf.stage!='exact_search'].drop_duplicates(['stage','dataset','style','route'],keep='first'),tf[tf.stage=='exact_search']],ignore_index=True)
    tf.to_csv(OUT/'timing_rows.csv',index=False)
    tf.groupby(['stage','dataset','style','route'],sort=False).agg(observations=('latency_seconds','size'),p50_seconds=('latency_seconds','median'),p95_seconds=('latency_seconds',lambda x:x.quantile(.95))).reset_index().to_csv(OUT/'timing_summary.csv',index=False)
    analyze();validate();report()
    dump(OUT/'completion_status.json',{'complete':True,'generations':160,'queries':40,'datasets':2,'methods':11,'metric_rows':len(rows),'primary_tests':12,'api_calls_by_runner':0,'human_validation':'not applicable; official benchmark qrels reused; exploratory subset'})
    print('Ablation complete: 160 generations; 40 questions; 11 methods; 12 primary tests')

def exact_signflip(delta):
    delta=np.asarray(delta,dtype=float);delta=delta[np.abs(delta)>1e-15]
    if not len(delta): return 1.
    observed=abs(delta.sum());n=len(delta);assert n<=20
    extreme=0;total=1<<n
    for start in range(0,total,65536):
        values=np.arange(start,min(total,start+65536),dtype=np.uint32)[:,None]
        signs=1-2*((values>>np.arange(n,dtype=np.uint32))&1).astype(np.int8)
        sums=signs@delta
        extreme+=int(np.count_nonzero(np.abs(sums)>=observed-1e-12))
    return extreme/total

def holm(ps):
    ps=np.asarray(ps);order=np.argsort(ps,kind='stable');result=np.empty(len(ps))
    result[order]=np.minimum(1,np.maximum.accumulate(ps[order]*(len(ps)-np.arange(len(ps)))))
    return result

def analyze():
    frame=pd.read_csv(OUT/'per_query_metrics.csv',dtype={'query_id':str});tests=[]
    for ds in ['legal','fiqa']:
        metric,k=('hit',5) if ds=='legal' else ('ndcg',10)
        wide=frame[(frame.dataset==ds)&(frame.k==k)].pivot(index='query_id',columns='method',values=metric)
        route='mix_dense'
        comparisons=[(effect,a,b,wide[a+'_'+route],wide[b+'_'+route],False) for a,b,effect in EDGES]
        comparisons += [
            ('average_length_effect','short_average','long_average',(wide['short_base_'+route]+wide['short_constraints_'+route])/2,(wide['long_base_'+route]+wide['long_constraints_'+route])/2,True),
            ('average_constraint_effect','constraints_average','base_average',(wide['short_constraints_'+route]+wide['long_constraints_'+route])/2,(wide['short_base_'+route]+wide['long_base_'+route])/2,True)]
        for effect,a,b,candidate,control,is_average in comparisons:
            delta=(candidate-control).to_numpy();wins=int(np.count_nonzero(delta>1e-12));losses=int(np.count_nonzero(delta< -1e-12))
            binary=ds=='legal' and not is_average
            raw=(binomtest(wins,wins+losses,.5).pvalue if wins+losses else 1.) if binary else exact_signflip(delta)
            tests.append({'dataset':ds,'route':route,'effect':effect,'candidate':a,'control':b,'metric':metric,'k':k,'n':20,
                'candidate_mean':candidate.mean(),'control_mean':control.mean(),'delta':delta.mean(),
                'wins':wins,'losses':losses,'ties':20-wins-losses,'test':'exact_McNemar' if binary else 'exact_sign_flip','p_raw':raw})
    tests=pd.DataFrame(tests);tests['p_holm12']=holm(tests.p_raw);tests.to_csv(OUT/'primary_tests.csv',index=False)

def validate():
    protocol=read(OUT/'protocol.json')
    for rel,dig in protocol['baseline_files'].items():assert digest((ROOT/rel).read_bytes())==dig
    assert len(generated())==160
    assert exact_signflip([1,1,1])==.25 and exact_signflip([1,-1,0])==1. and exact_signflip([0,0])==1.
    assert np.allclose(holm([.001,.04,.05]),[.003,.08,.08])
    frame=pd.read_csv(OUT/'per_query_metrics.csv',dtype={'query_id':str});assert len(frame)==2200 and not frame.isna().any().any()
    checks=0
    for ds,qs in read(OUT/'selection.json')['datasets'].items():
        ranks=np.load(OUT/f'{ds}_rankings.npz');ids=ranks['passage_ids'].tolist()
        assert ranks['query_ids'].tolist()==[q['query_id'] for q in qs]
        for method in [x for x in ranks.files if x not in ['query_ids','passage_ids']]:
            arr=ranks[method];assert arr.shape==(20,100)
            for i,q in enumerate(qs):
                rank=arr[i];assert len(set(rank))==100 and min(rank)>=0 and max(rank)<len(ids)
                rels={q['relevant_passage_id']:1} if ds=='legal' else q['qrels']
                retrieved=[ids[x] for x in rank]
                for k in KS:
                    row=frame[(frame.dataset==ds)&(frame.query_id==q['query_id'])&(frame.method==method)&(frame.k==k)].iloc[0]
                    positives=[j+1 for j,d in enumerate(retrieved[:k]) if rels.get(d,0)>0]
                    assert abs(row.hit-float(bool(positives)))<1e-12
                    assert abs(row.recall-len(positives)/sum(v>0 for v in rels.values()))<1e-12
                    assert abs(row.mrr-(1/positives[0] if positives else 0))<1e-12
                    numerator=sum(rels.get(d,0)/math.log2(j+2) for j,d in enumerate(retrieved[:k]))
                    denominator=sum(g/math.log2(j+2) for j,g in enumerate(sorted(rels.values(),reverse=True)[:k]))
                    assert abs(row.ndcg-numerator/denominator)<1e-12
                    checks+=1
    for _,g in frame.groupby(['dataset','query_id','method']):
        g=g.sort_values('k');assert (np.diff(g.hit)>=-1e-12).all() and (np.diff(g.recall)>=-1e-12).all()
    tests=pd.read_csv(OUT/'primary_tests.csv');assert len(tests)==12 and np.allclose(holm(tests.p_raw),tests.p_holm12,atol=1e-12)
    # Confirm factorial design differs only by the predeclared factors.
    for ds,qs in read(OUT/'selection.json')['datasets'].items():
        jobs=[x for x in read(OUT/'jobs.json') if x['dataset']==ds]
        for q in qs:
            prompts={x['style']:x['prompt'] for x in jobs if x['query_id']==q['query_id']}
            assert prompts['short_base'].replace('30 to 50 words','70 to 100 words',1)==prompts['long_base']
            assert prompts['short_base'].replace('\nQuestion:',CONSTRAINT+'\nQuestion:',1)==prompts['short_constraints']
            assert prompts['long_base'].replace('\nQuestion:',CONSTRAINT+'\nQuestion:',1)==prompts['long_constraints']
    dump(OUT/'validation.json',{'status':'passed','metric_rows_checked':checks,'primary_tests':12,'checks':['frozen inputs unchanged','factorial prompt equality','complete generations','rank uniqueness and bounds','independent Hit/Recall/MRR/nDCG','monotonic Hit/Recall','Holm correction'],'scope':'exploratory 20 reused queries per dataset; no claim of independent replication'})

def report():
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary=pd.read_csv(OUT/'summary.csv');tests=pd.read_csv(OUT/'primary_tests.csv');audit=pd.read_csv(OUT/'generation_audit.csv')
    colors={'short_base':'#2964a8','long_base':'#b7c9df','short_constraints':'#d16a2b','long_constraints':'#edc7ae'}
    labels={'short_base':'Short / base','long_base':'Long / base','short_constraints':'Short / constraints','long_constraints':'Long / constraints'}
    fig,axes=plt.subplots(1,2,figsize=(10,4.7));x=np.arange(2);width=.18
    for ax,ds,metric,k in zip(axes,['legal','fiqa'],['hit','ndcg'],[5,10]):
        for i,style in enumerate(STYLES):
            vals=[summary[(summary.dataset==ds)&(summary.method==style+'_'+route)&(summary.k==k)][metric].iloc[0] for route in ['pure_dense','mix_dense']]
            bars=ax.bar(x+(i-1.5)*width,vals,width,color=colors[style],label=labels[style])
            for b,v in zip(bars,vals):ax.text(b.get_x()+b.get_width()/2,b.get_height()+.012,f'{v:.2f}',ha='center',fontsize=8)
        ax.set_xticks(x,['Pure HyDE','Question + HyDE']);ax.set_title(('Legal Hit@5' if ds=='legal' else 'FiQA nDCG@10')+' (n=20)');ax.set_ylim(0, min(1,max(summary[(summary.dataset==ds)&(summary.k==k)][metric])+.15));ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=4,frameon=False,fontsize=9)
    fig.suptitle('Requested length x explicit constraint preservation',fontsize=13)
    fig.subplots_adjust(left=.07,right=.98,top=.84,bottom=.20,wspace=.25)
    fig.savefig(OUT/'factorial_primary.png',dpi=180);plt.close(fig)
    lines_=['# HyDE prompt factorial ablation','','Exploratory: the same 20 previously inspected Legal and 20 FiQA queries. No new holdout. Four prompts differ only in requested length and one explicit constraint-preservation sentence. All use the same 160-token output cap, fixed GPT-5.4 snapshot, MiniLM encoder, corpus, and 50/50 mixing.','','![Factorial prompt comparison](factorial_primary.png)','']
    for ds in ['legal','fiqa']:
        lines_ += [f'## {ds}: matched top-k results','','| Representation | Prompt | Hit@5 | Hit@10 | Hit@15 | Hit@20 | Hit@25 | nDCG@10 |','|---|---|---:|---:|---:|---:|---:|---:|']
        methods=[('Question only','question_dense')]+[(('Pure HyDE' if route=='pure_dense' else 'Question + HyDE'),s+'_'+route) for route in ['pure_dense','mix_dense'] for s in STYLES]
        for rep,method in methods:
            data=summary[(summary.dataset==ds)&(summary.method==method)].set_index('k')
            lab=labels.get(method.rsplit('_pure_dense',1)[0].rsplit('_mix_dense',1)[0],'None')
            lines_.append('| '+rep+' | '+lab+' | '+' | '.join(f'{100*data.loc[k,"hit"]:.0f}%' for k in KS)+f' | {data.loc[10,"ndcg"]:.4f} |')
    lines_+=['','## Predeclared statistical comparisons','','Exact paired McNemar for Legal binary edge effects; exact sign-flip randomization for FiQA and averaged factorial effects. Mixed50 is primary; pure HyDE is descriptive. One Holm family of 12 tests. Small reused samples support exploratory comparisons only.','','| Dataset | Route | Effect | Difference | Raw p | Holm p (12) |','|---|---|---|---:|---:|---:|']
    for r in tests.itertuples():lines_.append(f'| {r.dataset} | {r.route} | {r.effect} | {r.delta:+.4f} | {r.p_raw:.4f} | {r.p_holm12:.4f} |')
    lines_+=['','## Realized length audit','','| Dataset | Prompt | Mean words | Outside requested range | Length capped |','|---|---|---:|---:|---:|']
    for (ds,style),g in audit.groupby(['dataset','style'],sort=False):lines_.append(f'| {ds} | {labels[style]} | {g.word_count.mean():.1f} | {int(g.outside_target.sum())}/20 | {int(g.length_capped.sum())}/20 |')
    lines_+=['','Changing requested length does not guarantee compliance. Length-capped and out-of-range outputs are retained without regeneration. An instruction effect is an effect of adding this sentence, not proof of improved factual quality. API latency and cost are summarized by the shared workflow. timing_rows.csv and timing_summary.csv report local encoding batches and exact search p50/p95; model/disk loading is excluded. Encoding is batch timing, not per-query latency. These separate measurements are not end-to-end latency.','','Reproduce without API calls from the repository root:','','```bash','.venv/bin/python person3_meeting9_ablation/run.py evaluate','```','']
    (OUT/'README.md').write_text('\n'.join(lines_))
    EXPORT.mkdir(parents=True,exist_ok=True)
    for name in ['README.md','summary.csv','primary_tests.csv','generation_audit.csv','factorial_primary.png','protocol.json','selection.json','validation.json','timing_rows.csv','timing_summary.csv']:
        (EXPORT/name).write_bytes((OUT/name).read_bytes())

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','evaluate','validate','report']);stage=parser.parse_args().stage
    if stage=='prepare':prepare()
    elif stage=='evaluate':evaluate()
    elif stage=='validate':validate()
    elif stage=='report':report()
