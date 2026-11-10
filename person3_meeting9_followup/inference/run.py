"""Frozen finite exploratory inference and one missing cached-rank RRF cell."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, csv, hashlib, json, math, re, shutil, time
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/person3_meeting9_followup/inference_v1'
EXPORT=ROOT.parent/'outputs/meeting9_followup_20261109/inference'
H=ROOT/'results/person3_meeting9_constrained_hybrid/v1'
R=ROOT/'results/person3_meeting9_refinement/v1'
D=ROOT/'results/person3_meeting9_fullpage/v1'
S=ROOT/'results/person3_meeting9_transfer/vidore_finance_full309_summary_v1'
HISTORY=ROOT.parent/'work/full_results_hashes.json'
KS=[5,10,15,20,25]; METRICS=['hit','recall','mrr','ndcg']
GRID=[{'name':f'{label}_bank_rrf_w{int(w*100):03d}','dense':dense,'dense_weight':w} for label,dense in [('question','question_dense'),('focusedmix','focused160_mix_dense'),('shortmix','short_30_50_mix_dense')] for w in [.25,.50]]
FAMILY=[{'id':c['name']+'_vs_bm25','method':c['name'],'control':'bm25'} for c in GRID]
FAMILY += [{'id':c['name']+'_vs_question_same_weight','method':c['name'],'control':f"question_bank_rrf_w{int(c['dense_weight']*100):03d}"} for c in GRID if c['dense']!='question_dense']
UFQ='question_unfiltered_rrf_w050'; UFF='focusedmix_unfiltered_rrf_w050'
FQ='question_bank_rrf_w050'; FF='focusedmix_bank_rrf_w050'
CELLS=[['no_bank_filter','question',UFQ],['no_bank_filter','focusedmix',UFF],['bank_filter','question',FQ],['bank_filter','focusedmix',FF]]
SEED=20261109; DRAWS=100000; BATCH=1000
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def frozen(p,v):
    if p.exists():assert read(p)==v,f'Frozen artifact changed: {p}'
    else:dump(p,v)
def metrics_file(p):
    with Path(p).open() as f:rows=list(csv.DictReader(f))
    return pd.DataFrame([r|{'k':int(r['k']),**{m:float(r[m]) for m in METRICS}} for r in rows])
def export():
    EXPORT.mkdir(parents=True,exist_ok=True)
    for p in OUT.iterdir():
        if p.is_file():shutil.copy2(p,EXPORT/p.name)
def history_check():
    original=read(HISTORY);assert len(original)==565
    changed=[rel for rel,h in original.items() if not (ROOT/rel).is_file() or sha(ROOT/rel)!=h]
    assert not changed,f'Historical artifacts changed: {changed}'
    dump(OUT/'historical_preservation.json',{'checked':len(original),'changed_or_missing':changed,'passed':True})
def inputs():
    z=np.load(H/'rankings.npz');qids=z['query_ids'].tolist();pids=z['passage_ids'].tolist()
    q=np.load(R/'evaluated_rankings.npz');f=np.load(D/'rankings.npz');s=np.load(S/'rankings.npz');dev=np.load(R/'development_rankings.npz')
    for other in [q,f,s]:assert other['query_ids'].tolist()==qids and other['passage_ids'].tolist()==pids
    assert dev['query_ids'].tolist()==qids[:60] and dev['passage_ids'].tolist()==pids
    assert len(qids)==len(set(qids))==309 and len(pids)==len(set(pids))==2942
    assert qids==[q['query_id'] for q in read(S/'selection.json')['queries']]
    return z,q,f,s,dev,qids,pids
def check():
    for name in ['source_hashes.json','implementation_hashes.json']:
        for rel,h in read(OUT/name).items():assert sha(ROOT/rel)==h,f'Changed: {rel}'
    receipt=read(OUT/'freeze_receipt.json')
    for name in ['protocol','selection','routes','data_quality','implementation_hashes']:
        assert receipt[name+'_sha256']==sha(OUT/(name+'.json'))

def prepare():
    OUT.mkdir(parents=True,exist_ok=True);EXPORT.mkdir(parents=True,exist_ok=True)
    *_,qids,pids=inputs()
    protocol={'status':'Exploratory inference after repeated benchmark inspection; no new holdout/confirmation',
      'only_new_retrieval_outcome':'Unfiltered fullpage Focusedmix RRF dense_weight0.50; compute309, verify initial60 against retained dev ranks',
      'rrf':{'one_based':True,'k':60,'dense_weight':.5,'branch_depth':100,'absent':0,'output':'union top100','ties':'ascending original corpus index','mix_question_weight':.5},
      'primary_exploratory_family':FAMILY,'family_size':10,'cohort':'remaining249 only, all queries including fallback',
      'endpoint':'Mean nDCG@10, original complete qrels and linear gains; original IDCG/Recall denominators',
      'test':{'kind':'Two-sided paired question-level sign-flip Monte Carlo','draws':DRAWS,'seed':SEED,'batch':BATCH,
        'statistic':'Absolute sum of paired per-question nDCG differences','extreme':'abs(flipped_sum)>=abs(observed_sum)-1e-12',
        'p':'(extreme_count+1)/(100000+1)','shared_signs':'Same deterministic sign vectors across10 comparisons in frozen query order'},
      'multiplicity':'Holm across exactly10 prelisted comparisons, stable family order for ties; no other p-values',
      'effect_reporting':'Mean difference; wins/losses/ties at tolerance1e-12; raw and Holm10 p for every member',
      'confidence_intervals':'None. Query-bootstrap CI omitted because six reports are shared and prior adaptive inspection is unaccounted for.',
      'two_by_two':{'cells':CELLS,'cohorts':['initial60','remaining249','full309'],
        'interaction':'(filtered Focused-filtered question)-(unfiltered Focused-unfiltered question)',
        'policy':'Descriptive only; no interaction p or transfer into Holm10'},
      'all_k_recheck':{'k':KS,'metrics':METRICS,'scope':'Recompute saved constrained10-method metrics and4 control cells from ranks; no retrieval rerun'},
      'source_sensitivity':{'grouping':'275 single-source,34 multisource kept separate; also six single-bank groups',
        'leave_one_report_out':'Exclude every query whose positive gold-source set contains that report, including multisource queries touching it; descriptive effects only',
        'analyses':'The10 comparisons plus2x2 interaction; initial60/remaining249/full309; no additional p or CI'},
      'data_quality_sensitivity':'Keepq39 in main analysis; one fixed exclude-only-q39 descriptive sensitivity for all11 effects, no additional p or route changes',
      'caveats':['Question-level sign-flip assumes sign exchangeability; shared-report dependence is not adjusted.',
        'Holm10 covers only this finite family, not earlier experiments, selections or repeated inspection.',
        'All cohorts/configurations were already inspected; small p-values do not create new confirmation.',
        'Source omission is a stress test, not clustered independent evidence; OCR retrieval does not establish answer accuracy.'],
      'stop_rule':'One missing configuration and this finite inference family only; no new hypotheses/configurations after outcomes',
      'resources':{'api_calls':0,'new_embeddings':0,'new_generations':0,'paid_cost_usd':0},'historical_manifest_sha256':sha(HISTORY)}
    frozen(OUT/'protocol.json',protocol)
    frozen(OUT/'selection.json',{'initial60_ids':qids[:60],'remaining249_ids':qids[60:],'full309_ids':qids,'passage_ids':pids})
    frozen(OUT/'routes.json',read(H/'routes.json'))
    questions=read(S/'selection.json')['queries'];marker=r'\banswer\s*:'
    flagged=[{'query_id':q['query_id'],'question':q['question']} for q in questions if re.search(marker,q['question'],re.I)]
    assert [r['query_id'] for r in flagged]==['39']
    frozen(OUT/'data_quality.json',{'detector':marker,'case_insensitive':True,'explicit_Answer_marker_queries':flagged,
      'main_analysis_preserved':True,'fixed_sensitivity':'Exclude onlyquery39; descriptive, no p or route changes',
      'scope_limit':'This marker check is not an exhaustive leakage audit and does not establish absence of other leakage.'})
    paths=[H/n for n in ['rankings.npz','per_query_metrics.csv','protocol.json','validation.json','routes.json','freeze_receipt.json','implementation_hashes.json']]
    paths += [R/n for n in ['evaluated_rankings.npz','development_rankings.npz','per_query_metrics.csv','query_source_groups.csv','protocol.json','freeze_receipt.json']]
    paths += [D/'rankings.npz',S/'rankings.npz',S/'qrels.json',S/'selection.json',HISTORY]
    frozen(OUT/'source_hashes.json',{str(p.relative_to(ROOT) if p.is_relative_to(ROOT) else Path('..')/p.relative_to(ROOT.parent)):sha(p) for p in paths})
    scripts=[Path(__file__).resolve(),Path(__file__).resolve().with_name('report.py')]
    frozen(OUT/'implementation_hashes.json',{str(p.relative_to(ROOT)):sha(p) for p in scripts})
    history_check()
    if not (OUT/'freeze_receipt.json').exists():
        assert not any((OUT/n).exists() for n in ['rankings.npz','primary_tests.csv','two_by_two_summary.csv'])
        dump(OUT/'freeze_receipt.json',{'utc':datetime.now(timezone.utc).isoformat(),
          **{n+'_sha256':sha(OUT/(n+'.json')) for n in ['protocol','selection','routes','data_quality','implementation_hashes']},
          'new_outcome_and_pvalues_not_computed':True,'finite_family_size':10})
    export();print('Frozen missing configuration,Holm10 family,descriptive2x2/source/q39 controls; no new outcomes/p computed.')

def rrf(a,b):
    scores={}
    for branch in [a,b]:
        for rank,page in enumerate(branch,1):scores[int(page)]=scores.get(int(page),0.)+.5/(60+rank)
    return np.array(sorted(scores,key=lambda p:(-scores[p],p))[:100],dtype=np.int64)
def score(rank,rel,pids,k):
    g=np.array([rel.get(pids[int(p)],0) for p in rank[:k]],dtype=float);positive=g>0;found=np.flatnonzero(positive)
    ideal=np.sort(np.array(list(rel.values()),dtype=float))[::-1][:k];idcg=float(np.sum(ideal/np.log2(np.arange(len(ideal))+2)))
    return {'hit':float(positive.any()),'recall':float(positive.sum()/sum(v>0 for v in rel.values())),
      'mrr':1/(found[0]+1) if len(found) else 0.,'ndcg':float(np.sum(g/np.log2(np.arange(len(g))+2)))/idcg if idcg else 0.}
def cohort_ids(qids,cohort):return qids[:60] if cohort=='initial60' else qids[60:] if cohort=='remaining249' else qids
def holm(p):
    p=np.asarray(p);order=np.argsort(p,kind='stable');corrected=np.empty(len(p));corrected[order]=np.minimum(1,np.maximum.accumulate((len(p)-np.arange(len(p)))*p[order]));return corrected
def arrays(frame,qids,k=10):
    return {m:group.set_index('query_id').loc[qids,METRICS].to_numpy() for m,group in frame[frame.k==k].groupby('method',sort=False)}

def evaluate():
    check();started=time.perf_counter();z,q,f,s,dev,qids,pids=inputs();rels=read(S/'qrels.json')
    new=np.stack([rrf(a,b) for a,b in zip(f['focused160_mix_dense'],s['bm25'])])
    assert np.array_equal(new[:60],dev['focusedmix_rrf_w050'])
    names=[n for n in z.files if n not in ['query_ids','passage_ids']]
    ranks={m:z[m] for m in names}|{UFQ:q['question_rrf_w050'],UFF:new}
    np.savez_compressed(OUT/'rankings.npz',query_ids=np.array(qids),passage_ids=np.array(pids),**ranks)
    rows=[]
    for method,arr in ranks.items():
        for i,qid in enumerate(qids):
            for k in KS:rows.append({'query_id':qid,'cohort':'initial60' if i<60 else 'remaining249','method':method,'k':k,**score(arr[i],rels[qid],pids,k)})
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'all_k_per_query_metrics.csv',index=False)
    summary=pd.concat([frame,frame.assign(cohort='full309')]).groupby(['cohort','method','k'],sort=False).agg(n=('query_id','size'),**{m:(m,'mean') for m in METRICS}).reset_index();summary.to_csv(OUT/'all_k_summary.csv',index=False)
    all_arrays={k:arrays(frame,qids,k) for k in KS};cell_rows=[];contrast_rows=[];contrast_values={}
    for k,a in all_arrays.items():
        values={'filter_effect_question':a[FQ]-a[UFQ],'filter_effect_focused':a[FF]-a[UFF],
          'hyde_effect_unfiltered':a[UFF]-a[UFQ],'hyde_effect_filtered':a[FF]-a[FQ],
          'interaction_filter_by_hyde':(a[FF]-a[FQ])-(a[UFF]-a[UFQ])}
        if k==10:contrast_values=values
        for cohort in ['initial60','remaining249','full309']:
            indices=[qids.index(v) for v in cohort_ids(qids,cohort)]
            for filtering,representation,method in CELLS:
                cell_rows.append({'cohort':cohort,'filter':filtering,'representation':representation,'method':method,'k':k,'n':len(indices),**{m:float(a[method][indices,j].mean()) for j,m in enumerate(METRICS)}})
            for name,values_ in values.items():
                d=values_[indices];contrast_rows.append({'cohort':cohort,'contrast':name,'k':k,'n':len(indices),**{m+'_delta':float(d[:,j].mean()) for j,m in enumerate(METRICS)}})
    pd.DataFrame(cell_rows).to_csv(OUT/'two_by_two_summary.csv',index=False);pd.DataFrame(contrast_rows).to_csv(OUT/'two_by_two_contrasts.csv',index=False)
    a=all_arrays[10];differences={c['id']:a[c['method']][:,3]-a[c['control']][:,3] for c in FAMILY}
    matrix=np.column_stack([differences[c['id']][60:] for c in FAMILY]);assert matrix.shape==(249,10)
    pd.DataFrame({'query_id':qids[60:],**{c['id']:matrix[:,j] for j,c in enumerate(FAMILY)}}).to_csv(OUT/'primary_per_query_differences.csv',index=False)
    observed=abs(matrix.sum(axis=0));counts=np.zeros(10,dtype=np.int64);rng=np.random.default_rng(SEED);signhash=hashlib.sha256()
    for _ in range(DRAWS//BATCH):
        signs=rng.integers(0,2,size=(BATCH,249),dtype=np.int8)*2-1;signhash.update(signs.tobytes());counts+=(abs(signs@matrix)>=observed-1e-12).sum(axis=0)
    raw=(counts+1)/(DRAWS+1);corrected=holm(raw);tests=[]
    for j,c in enumerate(FAMILY):
        d=matrix[:,j];tests.append(c|{'cohort':'remaining249','metric':'ndcg@10','n':249,'mean_delta':float(d.mean()),
          'wins':int((d>1e-12).sum()),'losses':int((d< -1e-12).sum()),'ties':int((abs(d)<=1e-12).sum()),
          'draws':DRAWS,'seed':SEED,'extreme_count':int(counts[j]),'p_raw':float(raw[j]),'p_holm10':float(corrected[j]),'interpretation':'exploratory,question-level,shared-source dependence unadjusted'})
    pd.DataFrame(tests).to_csv(OUT/'primary_tests.csv',index=False)
    dump(OUT/'signflip_audit.json',{'seed':SEED,'draws':DRAWS,'batch':BATCH,'query_count':249,'family_size':10,'sign_matrix_stream_sha256':signhash.hexdigest(),
      'difference_csv_sha256':sha(OUT/'primary_per_query_differences.csv'),'null':'Question-level sign exchangeability; source dependence unadjusted','bootstrap_or_other_pvalues':False})
    sensitivity(qids,differences|{'interaction_filter_by_hyde':contrast_values['interaction_filter_by_hyde'][:,3]})
    dump(OUT/'resources.json',{'api_calls':0,'new_embeddings':0,'new_generations':0,'paid_cost_usd':0,'elapsed_seconds_before_validation_report':time.perf_counter()-started})
    validate();history_check()
    from report import report
    report(OUT,EXPORT)

def sensitivity(qids,effects):
    groups=pd.read_csv(R/'query_source_groups.csv',dtype={'query_id':str}).set_index('query_id').loc[qids]
    assert (groups.source_count==1).sum()==275 and (groups.source_count>1).sum()==34
    report_sets=[set(v.split('|')) for v in groups.relevant_sources];reports=sorted(set.union(*report_sets));assert len(reports)==6
    rows=[]
    for cohort in ['initial60','remaining249','full309']:
        permitted=set(cohort_ids(qids,cohort));base=np.array([q in permitted for q in qids])
        masks=[('source_group','single_source',base&(groups.source_count.to_numpy()==1)),('source_group','multiple_sources',base&(groups.source_count.to_numpy()>1))]
        masks += [('single_bank',report,base&(groups.source_group.to_numpy()==report)) for report in reports]
        masks += [('leave_one_report_out',report,base&np.array([report not in sources for sources in report_sets])) for report in reports]
        masks += [('data_quality','exclude_only_q39',base&np.array([q!='39' for q in qids]))]
        for kind,label,mask in masks:
            for name,values in effects.items():
                d=values[mask];rows.append({'cohort':cohort,'sensitivity':kind,'group_or_omitted_report':label,'analysis':name,'n':len(d),
                  'mean_delta':float(d.mean()) if len(d) else None,'wins':int((d>1e-12).sum()),'losses':int((d< -1e-12).sum()),'ties':int((abs(d)<=1e-12).sum()),'p_or_CI':'none;descriptive only'})
    pd.DataFrame(rows).to_csv(OUT/'source_sensitivity.csv',index=False)

def validate():
    check();z,q,f,s,dev,qids,pids=inputs();ranks=np.load(OUT/'rankings.npz');rels=read(S/'qrels.json')
    assert ranks['query_ids'].tolist()==qids and ranks['passage_ids'].tolist()==pids
    for m in [n for n in z.files if n not in ['query_ids','passage_ids']]:assert np.array_equal(ranks[m],z[m])
    assert np.array_equal(ranks[UFQ],q['question_rrf_w050']) and np.array_equal(ranks[UFF][:60],dev['focusedmix_rrf_w050'])
    for i in range(309):
        scores=np.zeros(2942,dtype=np.float64);scores[f['focused160_mix_dense'][i]]+=.5/(60+np.arange(1,101));scores[s['bm25'][i]]+=.5/(60+np.arange(1,101))
        assert np.array_equal(np.lexsort((np.arange(2942),-scores))[:100],ranks[UFF][i])
    frame=metrics_file(OUT/'all_k_per_query_metrics.csv');assert len(frame)==309*12*len(KS) and not frame.duplicated(['query_id','method','k']).any()
    old=metrics_file(H/'per_query_metrics.csv');joined=frame.merge(old,on=['query_id','method','k'],suffixes=('','_old'),validate='one_to_one');assert len(joined)==15450
    for m in METRICS:assert np.allclose(joined[m],joined[m+'_old'],atol=1e-12,rtol=0)
    oldq=metrics_file(R/'per_query_metrics.csv');oldq=oldq[oldq.method=='question_rrf_w050'].assign(method=UFQ)
    joinedq=frame[frame.method==UFQ].merge(oldq,on=['query_id','method','k'],suffixes=('','_old'),validate='one_to_one');assert len(joinedq)==1545
    for m in METRICS:assert np.allclose(joinedq[m],joinedq[m+'_old'],atol=1e-12,rtol=0)
    lookup=frame.set_index(['query_id','method','k'])
    for i,qid in enumerate(qids):
        for k in KS:
            grades=[rels[qid].get(pids[int(p)],0) for p in ranks[UFF][i,:k]];found=[j+1 for j,g in enumerate(grades) if g>0]
            idcg=sum(g/math.log2(j+2) for j,g in enumerate(sorted(rels[qid].values(),reverse=True)[:k]))
            expected={'hit':float(bool(found)),'recall':len(found)/sum(g>0 for g in rels[qid].values()),'mrr':1/found[0] if found else 0.,'ndcg':sum(g/math.log2(j+2) for j,g in enumerate(grades))/idcg if idcg else 0.}
            row=lookup.loc[(qid,UFF,k)];assert all(abs(row[m]-v)<1e-12 for m,v in expected.items())
    tests=pd.read_csv(OUT/'primary_tests.csv');assert tests.id.tolist()==[c['id'] for c in FAMILY] and len(tests)==10
    assert set(tests.cohort)=={'remaining249'} and set(tests.n)=={249}
    matrix=pd.read_csv(OUT/'primary_per_query_differences.csv',dtype={'query_id':str});assert matrix.query_id.tolist()==qids[60:]
    a=arrays(frame,qids)
    for j,c in enumerate(FAMILY):
        d=matrix[c['id']].to_numpy();row=tests.iloc[j]
        assert np.allclose(d,(a[c['method']][:,3]-a[c['control']][:,3])[60:],atol=1e-12,rtol=0)
        assert abs(row.mean_delta-d.mean())<1e-12 and row.wins==int((d>1e-12).sum()) and row.losses==int((d< -1e-12).sum()) and row.ties==int((abs(d)<=1e-12).sum())
    assert np.allclose(tests.p_raw,(tests.extreme_count+1)/(DRAWS+1),atol=1e-15,rtol=0)
    assert np.allclose(tests.p_holm10,holm(tests.p_raw),atol=1e-15,rtol=0)
    sensitivity_rows=pd.read_csv(OUT/'source_sensitivity.csv');assert len(sensitivity_rows)==3*15*11
    q39=sensitivity_rows[(sensitivity_rows.sensitivity=='data_quality')&(sensitivity_rows.cohort=='remaining249')]
    assert set(q39.n)=={249}
    dump(OUT/'validation.json',{'passed':True,'api_calls':0,'new_embeddings':0,'finite_primary_tests':10,'new_configuration_rank_checks':309,
      'initial60_matches_existing_development_ranks':True,'existing_constrained_all_k_metric_rows_rechecked':len(joined),'unfiltered_question_all_k_rows_rechecked':len(joinedq),
      'new_focused_all_k_metric_rows_independently_checked':309*len(KS),'original_full_qrels_denominators':True,'Holm10_and_complete_family_verified':True,
      'no_interaction_p_or_CI':True,'source_sensitivity_rows':len(sensitivity_rows),'q39_main_preserved_sensitivity_descriptive':True,
      'source_and_code_hashes_unchanged':True,'metric_rows':len(frame),'interpretation':'Exploratory only; shared-source dependence and prior adaptive inspection unadjusted'})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','evaluate','validate','report']);stage=p.parse_args().stage
    if stage=='report':
        from report import report
        report(OUT,EXPORT)
    else:{'prepare':prepare,'evaluate':evaluate,'validate':validate}[stage]()
