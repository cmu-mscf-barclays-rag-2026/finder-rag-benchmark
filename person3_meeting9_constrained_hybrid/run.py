"""Six fixed offline RRF hybrids of cached bank-constrained ranks."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json, math, os, shutil, time
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/person3_meeting9_constrained_hybrid/v1'
EXPORT=ROOT.parent/'outputs/meeting9_constrained_hybrid_20261109'
C=ROOT/'results/person3_meeting9_constraints/v1'
S=ROOT/'results/person3_meeting9_transfer/vidore_finance_full309_summary_v1'
CORPUS=ROOT/'results/person3_meeting9_transfer/vidore_finance_v1/data/corpus.jsonl'
BASES=['bm25','question_dense','focused160_mix_dense','short_30_50_mix_dense']
DENSES=[('question','question_dense'),('focusedmix','focused160_mix_dense'),('shortmix','short_30_50_mix_dense')]
GRID=[{'name':f'{label}_bank_rrf_w{int(w*100):03d}','dense':dense,'dense_weight':w} for label,dense in DENSES for w in [.25,.50]]
KS=[5,10,15,20,25]; METRICS=['hit','recall','mrr','ndcg']
LABELS={'bm25':'BM25','question_dense':'Question dense','focused160_mix_dense':'Focused mix','short_30_50_mix_dense':'Short mix'}
def read(p):return json.loads(Path(p).read_text())
def csv(p,**kwargs):return pd.read_csv(p,float_precision='round_trip',**kwargs)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def frozen(p,v):
    if p.exists():assert read(p)==v,f'Frozen artifact changed: {p}'
    else:dump(p,v)
def export():
    EXPORT.mkdir(parents=True,exist_ok=True)
    for p in OUT.iterdir():
        if p.is_file():shutil.copy2(p,EXPORT/p.name)
def inputs():
    z=np.load(C/'filtered_rankings.npz');qids=z['query_ids'].tolist();pids=z['passage_ids'].tolist();routes=read(C/'routes.json')
    assert qids==[r['query_id'] for r in routes]==[q['query_id'] for q in read(S/'selection.json')['queries']]
    assert len(qids)==len(set(qids))==309 and len(pids)==len(set(pids))==2942
    assert all(r['cohort']==('initial60' if i<60 else 'remaining249') for i,r in enumerate(routes))
    for m in BASES:
        assert z[m].shape==(309,100) and all(len(set(row))==100 for row in z[m]) and ((z[m]>=0)&(z[m]<2942)).all()
    previous=read(C/'validation.json');assert previous['passed'] and previous['original_complete_qrels_IDCG_and_recall_denominators_verified']
    return z,qids,pids,routes
def check():
    for name in ['source_hashes.json','implementation_hashes.json']:
        for rel,h in read(OUT/name).items():assert sha(ROOT/rel)==h,f'Changed: {rel}'
    receipt=read(OUT/'freeze_receipt.json')
    assert receipt['implementation_hashes_sha256']==sha(OUT/'implementation_hashes.json')
    for name in ['protocol','selection','routes']:assert receipt[name+'_sha256']==sha(OUT/(name+'.json'))

def prepare():
    OUT.mkdir(parents=True,exist_ok=True);EXPORT.mkdir(parents=True,exist_ok=True)
    _,qids,pids,routes=inputs()
    protocol={'status':'One fixed exploratory hybrid round on repeatedly inspected ViDoRe Finance English309; no new heldout confirmation',
      'candidate_grid':GRID,'reporting':'ALL6 settings, no development/winner selection, no weight/alias tuning, no new p-values',
      'primary_descriptive_cohort':'All remaining249 including fallback; repeatedly inspected',
      'secondary_descriptive_cohorts':['initial60','full309','routed/fallback within each cohort'],
      'branches':'Three retained bank-constrained dense top100 lists, each fused with the SAME retained bank-constrained BM25 top100 list',
      'mix_question_weight':.5,'metadata_routes':'Exact preceding frozen309 question-only routes, unchanged',
      'rrf':{'k':60,'branch_depth':100,'output_depth':100,'rank':'one-based',
             'score':'w_dense/(60+dense_rank)+(1-w_dense)/(60+bm25_rank)',
             'absent':0,'pages':'union of branch top100 lists','ties':'ascending original corpus index'},
      'standalone_baselines':BASES,'baseline_policy':'Reuse preceding constrained ranks/metrics unchanged; no retrieval recomputation',
      'primary_endpoint':'Mean nDCG@10 with linear graded gains','metrics':METRICS,'k':KS,
      'contrasts':'Each hybrid vs same-mask BM25 and matching dense; each HyDE hybrid also vs same-weight question hybrid; report every comparison descriptively',
      'denominators':'Complete original qrels, original IDCG, original positive-page Recall denominator; never trim by route',
      'limitations':['Top100 truncated-rank RRF, not full-score fusion.','Both cohorts repeatedly inspected; six shared annual-report sources.','Original bank constraint can exclude positive labels; no route changes.','OCR retrieval only, no final-answer reader or image encoder.'],
      'stop_rule':'Exactly these6 configurations; report and stop, including no improvement; no expanded search',
      'resources':{'api_calls':0,'new_embeddings':0,'new_generations':0,'new_models':0,'paid_cost_usd':0},
      'original_routes_sha256':sha(C/'routes.json'),'original_selection_sha256':sha(S/'selection.json')}
    frozen(OUT/'protocol.json',protocol)
    frozen(OUT/'selection.json',{'status':'Fixed cohorts and corpus order; no method selection','initial60_ids':qids[:60],'remaining249_ids':qids[60:],'full309_ids':qids,'passage_ids':pids})
    frozen(OUT/'routes.json',routes)
    paths=[C/n for n in ['filtered_rankings.npz','per_query_metrics.csv','routes.json','protocol.json','validation.json','freeze_receipt.json','implementation_hashes.json','qrel_retention_summary.csv']]+[S/'qrels.json',S/'selection.json',CORPUS]
    frozen(OUT/'source_hashes.json',{str(p.relative_to(ROOT)):sha(p) for p in paths})
    frozen(OUT/'implementation_hashes.json',{str(Path(__file__).resolve().relative_to(ROOT)):sha(__file__)})
    if not (OUT/'freeze_receipt.json').exists():
        assert not any((OUT/n).exists() for n in ['rankings.npz','per_query_metrics.csv','summary.csv'])
        dump(OUT/'freeze_receipt.json',{'utc':datetime.now(timezone.utc).isoformat(),**{n+'_sha256':sha(OUT/(n+'.json')) for n in ['protocol','selection','routes']},
          'implementation_hashes_sha256':sha(OUT/'implementation_hashes.json'),'no_fused_outcomes_existed':True,'all6_fixed_before_outcomes':True})
    export();print('Frozen6 settings,309 routes/query order,2942 page order and input/code hashes. No fused outcomes computed.')

def rrf(a,b,w):
    scores={}
    for branch,weight in [(a,w),(b,1-w)]:
        for rank,page in enumerate(branch,1):scores[int(page)]=scores.get(int(page),0.)+weight/(60+rank)
    return np.array(sorted(scores,key=lambda p:(-scores[p],p))[:100],dtype=np.int64)
def metric(rank,rels,pids,k):
    grades=[rels.get(pids[int(p)],0) for p in rank[:k]];found=[i+1 for i,g in enumerate(grades) if g>0]
    ideal=sum(g/math.log2(i+2) for i,g in enumerate(sorted(rels.values(),reverse=True)[:k]))
    return {'hit':float(bool(found)),'recall':len(found)/sum(g>0 for g in rels.values()),'mrr':1/found[0] if found else 0.,'ndcg':sum(g/math.log2(i+2) for i,g in enumerate(grades))/ideal if ideal else 0.}
def expand(frame):
    both=pd.concat([frame,frame.assign(cohort='full309')],ignore_index=True)
    return pd.concat([both.assign(scope='all'),both.assign(scope=np.where(both.routed,'routed','fallback'))],ignore_index=True)

def evaluate():
    check();start=time.perf_counter();z,qids,pids,routes=inputs();rels=read(S/'qrels.json')
    old=csv(C/'per_query_metrics.csv',dtype={'query_id':str});base=old[(old.condition=='filtered')&old.method.isin(BASES)]
    base=base[['query_id','cohort','routed','route_reason','method','k']+METRICS].copy();base['family']='constrained_standalone'
    ranks={m:z[m] for m in BASES};rows=[]
    for c in GRID:
        arr=np.stack([rrf(a,b,c['dense_weight']) for a,b in zip(z[c['dense']],z['bm25'])]);ranks[c['name']]=arr
        for i,r in enumerate(routes):
            for k in KS:rows.append({'query_id':qids[i],'cohort':r['cohort'],'routed':bool(r['route_doc_ids']),'route_reason':r['reason'],'method':c['name'],'family':'constrained_hybrid','k':k,**metric(arr[i],rels[qids[i]],pids,k)})
    frame=pd.concat([base,pd.DataFrame(rows)],ignore_index=True);frame.to_csv(OUT/'per_query_metrics.csv',index=False)
    np.savez_compressed(OUT/'rankings.npz',query_ids=np.array(qids),passage_ids=np.array(pids),**ranks)
    data=expand(frame);summary=data.groupby(['cohort','scope','method','k'],sort=False).agg(n=('query_id','size'),**{m:(m,'mean') for m in METRICS}).reset_index();summary.to_csv(OUT/'summary.csv',index=False)
    comparisons=[]
    for c in GRID:
        controls=[('same_mask_bm25','bm25'),('matching_dense',c['dense'])]
        if c['dense']!='question_dense':controls.append(('same_weight_question_hybrid',next(v['name'] for v in GRID if v['dense']=='question_dense' and v['dense_weight']==c['dense_weight'])))
        for key,g in data.groupby(['cohort','scope','k'],sort=False):
            a=g[g.method==c['name']].set_index('query_id')
            for relation,control in controls:
                delta=a[METRICS]-g[g.method==control].set_index('query_id')[METRICS]
                comparisons.append(dict(zip(['cohort','scope','k'],key))|{'method':c['name'],'control':control,'comparison':relation,'n':len(delta),**{m+'_delta':float(delta[m].mean()) for m in METRICS},'ndcg_wins':int((delta.ndcg>1e-12).sum()),'ndcg_losses':int((delta.ndcg< -1e-12).sum()),'ndcg_ties':int((abs(delta.ndcg)<=1e-12).sum())})
    pd.DataFrame(comparisons).to_csv(OUT/'matched_comparisons.csv',index=False)
    dump(OUT/'resources.json',{'api_calls':0,'new_embeddings':0,'new_generations':0,'new_models':0,'paid_cost_usd':0,'evaluation_seconds_before_validation_report':time.perf_counter()-start,'scope':'Cached-rank fusion and evaluation only'})
    validate();report()

def validate():
    check();z,qids,pids,routes=inputs();rels=read(S/'qrels.json');ranks=np.load(OUT/'rankings.npz')
    frame=csv(OUT/'per_query_metrics.csv',dtype={'query_id':str});assert len(frame)==309*10*len(KS) and not frame.duplicated(['query_id','method','k']).any()
    assert ranks['query_ids'].tolist()==qids and ranks['passage_ids'].tolist()==pids
    assert set(frame.method)==set(BASES+[c['name'] for c in GRID])
    old=csv(C/'per_query_metrics.csv',dtype={'query_id':str});old=old[(old.condition=='filtered')&old.method.isin(BASES)]
    joined=frame[frame.method.isin(BASES)].merge(old,on=['query_id','method','k'],suffixes=('','_old'),validate='one_to_one')
    assert len(joined)==309*4*len(KS)
    for m in METRICS:assert np.array_equal(joined[m].to_numpy(),joined[m+'_old'].to_numpy())
    for m in BASES:assert np.array_equal(ranks[m],z[m])
    pages=[json.loads(x) for x in CORPUS.read_text().splitlines() if x.strip()];assert pids==[str(p['corpus_id']) for p in pages]
    sources=np.array([p['doc_id'] for p in pages]);lookup=frame.set_index(['query_id','method','k']);checks=0
    for c in GRID:
        w=c['dense_weight'];arr=ranks[c['name']];assert arr.shape==(309,100)
        for i,r in enumerate(routes):
            scores=np.zeros(2942,dtype=np.float64);scores[z[c['dense']][i]]+=w/(60+np.arange(1,101));scores[z['bm25'][i]]+=(1-w)/(60+np.arange(1,101))
            assert np.array_equal(np.lexsort((np.arange(2942),-scores))[:100],arr[i]) and len(set(arr[i]))==100
            assert set(arr[i]).issubset(set(z[c['dense']][i])|set(z['bm25'][i]))
            if r['route_doc_ids']:assert set(sources[arr[i]]).issubset(set(r['route_doc_ids']))
            for k in KS:
                grades=np.array([rels[qids[i]].get(pids[int(p)],0) for p in arr[i,:k]],dtype=float);hits=grades>0;positions=np.flatnonzero(hits)
                ideal=np.sort(np.array(list(rels[qids[i]].values()),dtype=float))[::-1][:k];idcg=float(np.sum(ideal/np.log2(np.arange(len(ideal))+2)))
                computed={'hit':float(hits.any()),'recall':hits.sum()/sum(g>0 for g in rels[qids[i]].values()),'mrr':1/(positions[0]+1) if len(positions) else 0.,'ndcg':float(np.sum(grades/np.log2(np.arange(len(grades))+2)))/idcg if idcg else 0.}
                row=lookup.loc[(qids[i],c['name'],k)];assert all(abs(row[m]-v)<1e-12 for m,v in computed.items())
            checks+=1
    data=expand(frame);summary=csv(OUT/'summary.csv')
    for row in summary.to_dict('records'):
        sub=data[(data.cohort==row['cohort'])&(data.scope==row['scope'])&(data.method==row['method'])&(data.k==row['k'])]
        assert len(sub)==row['n'] and all(abs(sub[m].mean()-row[m])<1e-12 for m in METRICS)
    comparisons=csv(OUT/'matched_comparisons.csv');assert len(comparisons)==16*3*3*len(KS)
    for row in comparisons.to_dict('records'):
        sub=data[(data.cohort==row['cohort'])&(data.scope==row['scope'])&(data.k==row['k'])]
        delta=sub[sub.method==row['method']].set_index('query_id')[METRICS]-sub[sub.method==row['control']].set_index('query_id')[METRICS]
        assert len(delta)==row['n'] and all(abs(delta[m].mean()-row[m+'_delta'])<1e-12 for m in METRICS)
        assert row['ndcg_wins']==int((delta.ndcg>1e-12).sum()) and row['ndcg_losses']==int((delta.ndcg< -1e-12).sum()) and row['ndcg_ties']==int((abs(delta.ndcg)<=1e-12).sum())
    dump(OUT/'validation.json',{'passed':True,'api_calls':0,'new_embeddings':0,'new_pvalues':0,'all6_reported_without_selection':True,'independent_array_rrf_rank_checks':checks,'unchanged_baseline_metric_rows':len(joined),'same_frozen_bank_routes_verified':309,'original_full_qrels_IDCG_recall_denominators_verified':True,'all_summary_denominators_verified':True,'matched_comparisons_verified':len(comparisons),'input_and_code_hashes_unchanged':True,'metric_rows':len(frame),'summary_rows':len(summary),'status':'One fixed descriptive round; no tuning/new heldout claim'})

def report():
    os.environ.setdefault('MPLCONFIGDIR',str(OUT/'matplotlib_cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    s=csv(OUT/'summary.csv');primary=s[(s.cohort=='remaining249')&(s.scope=='all')&(s.k==10)].set_index('method')
    fig,ax=plt.subplots(figsize=(8.4,4.7));x=np.arange(3);width=.35
    for j,(w,color) in enumerate([(.25,'#69a4c5'),(.50,'#245a81')]):
        settings=[next(c for c in GRID if c['dense']==dense and c['dense_weight']==w) for _,dense in DENSES];values=[primary.loc[c['name'],'ndcg'] for c in settings]
        bars=ax.bar(x+(j-.5)*width,values,width,label=f'Dense weight {w:.2f}',color=color)
        for bar,v in zip(bars,values):ax.text(bar.get_x()+bar.get_width()/2,v+.005,f'{v:.3f}',ha='center',fontsize=9)
    bm=float(primary.loc['bm25','ndcg']);ax.axhline(bm,color='#bb7a2a',linestyle='--',label=f'Constrained BM25 ({bm:.3f})')
    ax.set_xticks(x,['Question hybrid','Focused mix hybrid','Short mix hybrid']);ax.set_ylabel('nDCG@10');ax.set_ylim(0,max(float(primary.ndcg.max()),bm)*1.23);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);ax.legend(frameon=False,fontsize=9)
    fig.suptitle('Bank-constrained dense + BM25 RRF: all remaining249');fig.tight_layout(rect=[0,.065,1,.94]);fig.text(.5,.02,'All six fixed; repeatedly inspected exploratory benchmark; no new p-values. Original qrels retained.',ha='center',fontsize=8);fig.savefig(OUT/'constrained_hybrid.png',dpi=180);plt.close(fig)
    lines=['# Bank-constrained dense + BM25 hybrids','','One fixed offline round on the repeatedly inspected ViDoRe Finance English309 benchmark. **All remaining249, including fallback, is primary descriptive.** All six settings are reported; no development selection or tuning. New API calls, generations, embeddings, models and p-values: **0**.','','RRF uses the preceding constrained BM25/dense Top100 ranks: dense weights 0.25/0.50, k=60, one-based ranks, absent contribution0, union Top100, stable corpus-index ties. This is truncated-rank fusion. Question–HyDE mixing stays0.50 and all309 bank routes stay unchanged. Every metric retains complete original qrels, IDCG and Recall denominators.','','| Constrained standalone, all249 | nDCG@10 | Hit@10 | Recall@10 |','|---|---:|---:|---:|']
    for m in BASES:
        r=primary.loc[m];lines.append(f'| {LABELS[m]} | {r.ndcg:.4f} | {r.hit:.1%} | {r.recall:.4f} |')
    lines+=['','| All249 hybrid | nDCG@10 | Δ vs BM25 | Δ vs matching dense | Δ vs same-weight question hybrid |','|---|---:|---:|---:|---:|']
    for c in GRID:
        r=primary.loc[c['name']];q=next(v for v in GRID if v['dense']=='question_dense' and v['dense_weight']==c['dense_weight'])
        lines.append(f"| {LABELS[c['dense']]} / w={c['dense_weight']:.2f} | {r.ndcg:.4f} | {r.ndcg-primary.loc['bm25','ndcg']:+.4f} | {r.ndcg-primary.loc[c['dense'],'ndcg']:+.4f} | {r.ndcg-primary.loc[q['name'],'ndcg']:+.4f} |")
    lines+=['','![All six fixed hybrids](constrained_hybrid.png)','','| Hybrid | Initial60 nDCG@10 | Full309 nDCG@10 |','|---|---:|---:|']
    for c in GRID:
        r=s[(s.scope=='all')&(s.method==c['name'])&(s.k==10)].set_index('cohort');lines.append(f"| {LABELS[c['dense']]} / w={c['dense_weight']:.2f} | {r.loc['initial60','ndcg']:.4f} | {r.loc['full309','ndcg']:.4f} |")
    lines+=['','Routed/fallback results, Hit/Recall/MRR at5/10/15/20/25 and all matched contrasts are in CSVs. The unchanged constraint routes202/249 queries and falls back on47;13/249 queries exclude some positive labels, which remain in metric denominators.','','Both cohorts have repeated outcome exposure and share six annual-report sources. These descriptive comparisons provide no new heldout confirmation or final-answer accuracy evidence. No aliases, weights or representations were changed after outcomes.','','Reproduce from retained caches:','','```bash','.venv/bin/python person3_meeting9_constrained_hybrid/run.py prepare','.venv/bin/python person3_meeting9_constrained_hybrid/run.py evaluate','```','','`validate` checks every fused rank independently, allowed-page membership, unchanged baselines, complete original qrel denominators, all comparisons, routes and input/code hashes.']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n');export();print((OUT/'README.md').read_text())

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','evaluate','validate','report']);{'prepare':prepare,'evaluate':evaluate,'validate':validate,'report':report}[p.parse_args().stage]()
