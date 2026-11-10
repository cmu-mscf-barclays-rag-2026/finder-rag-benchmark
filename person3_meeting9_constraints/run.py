"""One fixed question-only metadata constraint; offline cached-score evaluation."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, importlib.util, json, math, os, shutil, time
import numpy as np
import pandas as pd
from router import ALIASES, AMBIGUITY, OPEN_COMPARISON, QUOTES, route

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/person3_meeting9_constraints/v1'
EXPORT = ROOT.parent/'outputs/meeting9_constraints_20261109'
D = ROOT/'results/person3_meeting9_fullpage/v1'
S = ROOT/'results/person3_meeting9_transfer/vidore_finance_full309_summary_v1'
A = ROOT/'results/person3_meeting9_transfer/vidore_finance_v1'
B = ROOT/'results/person3_meeting9_transfer/vidore_finance_remaining249_v1'
CORPUS = A/'data/corpus.jsonl'
METHODS = ['bm25','question_dense','focused160_mix_dense','short_30_50_mix_dense']
KS = [5,10,15,20,25]
METRICS = ['hit','recall','mrr','ndcg']
LABELS = {'bm25':'BM25','question_dense':'Question dense','focused160_mix_dense':'Focused mix','short_30_50_mix_dense':'Short mix'}

def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def frozen(p,v):
    if p.exists(): assert read(p)==v, f'Frozen artifact changed: {p}'
    else: dump(p,v)
def meta():
    qs=read(S/'selection.json')['queries']
    pages=[json.loads(x) for x in CORPUS.read_text().splitlines() if x.strip()]
    assert len(qs)==309 and len(pages)==2942
    return qs,pages
def export():
    EXPORT.mkdir(parents=True,exist_ok=True)
    for p in OUT.iterdir():
        if p.is_file(): shutil.copy2(p,EXPORT/p.name)
def check():
    for name in ['source_hashes.json','implementation_hashes.json']:
        for rel,h in read(OUT/name).items(): assert sha(ROOT/rel)==h, f'Changed: {rel}'
    receipt=read(OUT/'freeze_receipt.json')
    assert receipt['routes_sha256']==sha(OUT/'routes.json')
    assert receipt['protocol_sha256']==sha(OUT/'protocol.json')

def prepare():
    OUT.mkdir(parents=True,exist_ok=True); EXPORT.mkdir(parents=True,exist_ok=True)
    qs,pages=meta(); sources=np.array([p['doc_id'] for p in pages])
    assert set(sources)==set(ALIASES)
    routes=[]
    for i,q in enumerate(qs):
        # Only the question string is passed into the router. No qrels/answers are loaded.
        r=route(q['question']); docs=r['route_doc_ids']
        routes.append({'query_id':q['query_id'],'question':q['question'],
          'cohort':'initial60' if i<60 else 'remaining249',**r,
          'allowed_pages':int(np.isin(sources,docs).sum()) if docs else len(pages)})
    protocol={'status':'One fixed exploratory metadata-constraint experiment on previously inspected ViDoRe Finance English309; not new heldout confirmation',
      'development_diagnosis_only':'Initial60 source-mismatch diagnosis motivated the intervention; no filtered outcomes inspected before freeze',
      'primary_descriptive_cohort':'All remaining249, including routed and fallback queries',
      'secondary_cohorts':['initial60','full309','routed/fallback within each cohort'],
      'primary_descriptive_endpoint':'Mean nDCG@10, linear graded relevance gains',
      'methods':METHODS,'same_mask_for_all_methods':True,
      'aliases':ALIASES,'matching':'Question literal only; Unicode NFKC; case-insensitive Unicode word boundaries; spaces/hyphens/periods normalized only inside a name',
      'excluded_aliases':'No bare bank/Morgan/Chase/America or tickers C/MS/GS/JPM/BAC/WFC; no BofA or Citibank alias',
      'multiple_matches':'Union of all explicitly matched companies',
      'no_match':'Keep original full-corpus ranking',
      'open_comparison_regex':OPEN_COMPARISON.pattern,'open_comparison_scope':'Fallback only for exactly one distinct explicitly mentioned bank',
      'negation_or_exclusion_regex':AMBIGUITY.pattern,'negation_or_exclusion_scope':'Conservatively fallback if marker occurs anywhere in a question with an alias',
      'quoted_mention_regex':QUOTES.pattern,'quoted_mention_scope':'Fallback if a literal company alias overlaps a paired quoted span',
      'year_filter':False,'routing_forbidden_inputs':['query_id','query metadata','HyDE','answers','qrels','gold source','retrieval results'],
      'dense_scores':'Exact retained float32 question/original50-50 mix vectors @ all14017 cached chunk vectors, then page maximum over all2942 pages',
      'bm25_scores':'Existing helper BM25 same full-OCR index, k1=1.2,b=.75,binary query counts; retain full2942 scores, global corpus statistics unchanged',
      'ranking':'Rank all allowed pages by original method score; stable ascending original corpus index ties; save top100. Never filter an old top100 list.',
      'replay_gate':'All309 unfiltered top100 rankings for all4 methods must exactly reproduce retained results before filtered metrics',
      'metric_denominators':'Always original unfiltered complete qrels, original IDCG and original count of positive labels; never restrict qrels to allowed pages',
      'qrel_retention':'Separate post-freeze diagnostic of retained positive-page count and relevance-grade mass; never used for routing or tuning',
      'secondary_metrics':{'k':KS,'names':METRICS},
      'contrasts':'For every method: filtered minus matched unfiltered. Within each condition: Focusedmix and Shortmix minus question dense. No winner selection.',
      'inference':'All descriptive, no new p-values or significance search; cohorts previously inspected and annual-report sources shared',
      'stop_rule':'Exactly this one rule set. Stop after results, including no improvement. No alias/guard change, tuning or expanded search based on outcomes.',
      'resources':{'api_calls':0,'new_embeddings':0,'new_generations':0,'new_models':0,'paid_cost_usd':0}}
    frozen(OUT/'routes.json',routes); frozen(OUT/'protocol.json',protocol)
    paths=[CORPUS,S/'selection.json',S/'qrels.json',S/'rankings.npz',D/'rankings.npz',D/'per_query_metrics.csv',D/'chunks.json',D/'index_validation.json',ROOT/'person3_cross_dataset/run.py']
    paths += sorted((D/'chunk_vectors').glob('*.npy'))
    paths += [p/'vectors'/name for p in [A,B] for name in ['questions.npy','focused160.npy','short_30_50.npy']]
    frozen(OUT/'source_hashes.json',{str(p.relative_to(ROOT)):sha(p) for p in paths})
    frozen(OUT/'implementation_hashes.json',{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__).resolve(),Path(__file__).resolve().with_name('router.py')]})
    if not (OUT/'freeze_receipt.json').exists():
        assert not any((OUT/n).exists() for n in ['unfiltered_rankings.npz','filtered_rankings.npz','per_query_metrics.csv'])
        dump(OUT/'freeze_receipt.json',{'utc':datetime.now(timezone.utc).isoformat(),'routes_sha256':sha(OUT/'routes.json'),
          'protocol_sha256':sha(OUT/'protocol.json'),'implementation_hashes_sha256':sha(OUT/'implementation_hashes.json'),
          'no_filtered_or_replay_results_existed':True,'qrels_loaded_by_prepare':False})
    export(); print('Frozen309 question-only routes, one rule set and input/code hashes. No filtered outcomes computed.')

def replay():
    check(); qs,pages=meta(); start=time.perf_counter()
    chunkmeta=read(D/'chunks.json'); starts=np.array(chunkmeta['page_chunk_starts'])
    ce=np.concatenate([np.load(p) for p in sorted((D/'chunk_vectors').glob('*.npy'))])
    qe=np.concatenate([np.load(p/'vectors/questions.npy') for p in [A,B]])
    assert ce.shape==(14017,384) and qe.shape==(309,384)
    assert ce.dtype==qe.dtype==np.float32 and np.isfinite(ce).all() and np.isfinite(qe).all()
    vectors={'question_dense':qe}
    for style in ['focused160','short_30_50']:
        he=np.concatenate([np.load(p/'vectors'/f'{style}.npy') for p in [A,B]])
        mix=.5*qe+.5*he
        vectors[style+'_mix_dense']=mix/np.maximum(np.linalg.norm(mix,axis=-1,keepdims=True),1e-12)
    dense=np.load(D/'rankings.npz'); sparse=np.load(S/'rankings.npz')
    qids=[q['query_id'] for q in qs]; pids=[str(p['corpus_id']) for p in pages]
    assert qids==dense['query_ids'].tolist()==sparse['query_ids'].tolist()
    assert pids==dense['passage_ids'].tolist()==sparse['passage_ids'].tolist()==[p['page_id'] for p in chunkmeta['pages']]
    ranks={}; timings={}
    for method,vecs in vectors.items():
        t=time.perf_counter()
        assert vecs.shape==(309,384) and vecs.dtype==np.float32 and np.isfinite(vecs).all()
        scores=np.stack([np.maximum.reduceat(ce@vec,starts[:-1]) for vec in vecs])
        assert scores.shape==(309,2942) and np.isfinite(scores).all()
        rank=np.argsort(-scores,axis=1,kind='stable')[:,:100]
        assert np.array_equal(rank,dense[method]), f'Unfiltered replay mismatch: {method}'
        np.save(OUT/f'full_scores_{method}.npy',scores); ranks[method]=rank
        timings[method]=time.perf_counter()-t; print(f'Replayed {method}: all309 exact top100',flush=True)
    spec=importlib.util.spec_from_file_location('constraints_existing_bm25',ROOT/'person3_cross_dataset/run.py')
    helper=importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    t=time.perf_counter(); bm=helper.BM25([p['markdown'] for p in pages]); build=time.perf_counter()-t
    t=time.perf_counter(); scores=[]
    for q in qs:
        qc=bm.vec.transform([q['question']]).tocsr(); qc.data[:]=1
        scores.append((qc@bm.mat.T).toarray().ravel())
    scores=np.stack(scores); rank=np.argsort(-scores,axis=1,kind='stable')[:,:100]
    assert scores.shape==(309,2942) and np.isfinite(scores).all() and np.array_equal(rank,sparse['bm25'])
    ranks['bm25']=rank; np.save(OUT/'full_scores_bm25.npy',scores); timings['bm25_query_scores']=time.perf_counter()-t
    np.savez_compressed(OUT/'unfiltered_rankings.npz',query_ids=np.array(qids),passage_ids=np.array(pids),**ranks)
    dump(OUT/'replay_validation.json',{'passed':True,'queries':309,'methods':METHODS,'all_unfiltered_top100_exact':True,
      'rank_checks':309*4,'api_calls':0,'new_embeddings':0,'full_score_hashes':{m:sha(OUT/f'full_scores_{m}.npy') for m in METHODS},
      'routes_sha256':sha(OUT/'routes.json'),'replay_wall_seconds':time.perf_counter()-start,
      'method_score_seconds':timings,'bm25_build_seconds':build})
    export(); print('Replay gate passed for all1236 rankings; filtered evaluation not yet computed.')

def metric(rank,rels,pids,k):
    # Keep the COMPLETE original rels for both filtered and unfiltered denominators.
    grades=[rels.get(pids[int(p)],0) for p in rank[:k]]
    found=[i+1 for i,g in enumerate(grades) if g>0]
    ideal=sum(g/math.log2(i+2) for i,g in enumerate(sorted(rels.values(),reverse=True)[:k]))
    return {'hit':float(bool(found)),'recall':len(found)/sum(g>0 for g in rels.values()),
      'mrr':1/found[0] if found else 0.,'ndcg':sum(g/math.log2(i+2) for i,g in enumerate(grades))/ideal if ideal else 0.}

def allowed_indices(r,sources):
    return np.flatnonzero(np.isin(sources,r['route_doc_ids'])) if r['route_doc_ids'] else np.arange(len(sources))

def expand_scopes(frame):
    allcohorts=pd.concat([frame,frame.assign(cohort='full309')],ignore_index=True)
    return pd.concat([allcohorts.assign(scope='all'),allcohorts.assign(scope=np.where(allcohorts.routed,'routed','fallback'))],ignore_index=True)

def evaluate():
    check(); gate=read(OUT/'replay_validation.json'); assert gate['passed']
    assert gate['routes_sha256']==sha(OUT/'routes.json')
    start=time.perf_counter(); qs,pages=meta(); routes=read(OUT/'routes.json')
    qids=[q['query_id'] for q in qs]; pids=[str(p['corpus_id']) for p in pages]; sources=np.array([p['doc_id'] for p in pages])
    unfiltered=np.load(OUT/'unfiltered_rankings.npz'); rels=read(S/'qrels.json')
    assert qids==[r['query_id'] for r in routes]==unfiltered['query_ids'].tolist()
    filtered={}; rows=[]
    for method in METHODS:
        path=OUT/f'full_scores_{method}.npy'; assert sha(path)==gate['full_score_hashes'][method]
        scores=np.load(path); ranks=[]
        for i,r in enumerate(routes):
            allowed=allowed_indices(r,sources)
            ranks.append(allowed[np.argsort(-scores[i,allowed],kind='stable')[:100]])
        filtered[method]=np.stack(ranks)
        for condition,arr in [('unfiltered',unfiltered[method]),('filtered',filtered[method])]:
            for i,r in enumerate(routes):
                for k in KS:
                    rows.append({'query_id':qids[i],'cohort':r['cohort'],'routed':bool(r['route_doc_ids']),
                      'route_reason':r['reason'],'method':method,'condition':condition,'k':k,
                      **metric(arr[i],rels[qids[i]],pids,k)})
    np.savez_compressed(OUT/'filtered_rankings.npz',query_ids=np.array(qids),passage_ids=np.array(pids),**filtered)
    frame=pd.DataFrame(rows); frame.to_csv(OUT/'per_query_metrics.csv',index=False)
    expanded=expand_scopes(frame)
    keys=['cohort','scope','method','condition','k']
    summary=expanded.groupby(keys,sort=False).agg(n=('query_id','size'),**{m:(m,'mean') for m in METRICS}).reset_index()
    summary.to_csv(OUT/'summary.csv',index=False)
    differences=[]
    for keys_,group in expanded.groupby(['cohort','scope','method','k'],sort=False):
        a=group[group.condition=='filtered'].set_index('query_id'); b=group[group.condition=='unfiltered'].set_index('query_id')
        delta=a[METRICS]-b[METRICS]
        differences.append(dict(zip(['cohort','scope','method','k'],keys_))|{'n':len(delta),
          **{m+'_delta':float(delta[m].mean()) for m in METRICS},'ndcg_wins':int((delta.ndcg>1e-12).sum()),
          'ndcg_losses':int((delta.ndcg< -1e-12).sum()),'ndcg_ties':int((abs(delta.ndcg)<=1e-12).sum())})
    pd.DataFrame(differences).to_csv(OUT/'matched_deltas.csv',index=False)
    contrasts=[]
    for keys_,group in expanded[expanded.k==10].groupby(['cohort','scope','condition'],sort=False):
        q=group[group.method=='question_dense'].set_index('query_id')
        for method in ['focused160_mix_dense','short_30_50_mix_dense']:
            h=group[group.method==method].set_index('query_id');delta=h[METRICS]-q[METRICS]
            contrasts.append(dict(zip(['cohort','scope','condition'],keys_))|{'method':method,'control':'question_dense','k':10,'n':len(delta),**{m+'_delta':float(delta[m].mean()) for m in METRICS}})
    pd.DataFrame(contrasts).to_csv(OUT/'hyde_contrasts.csv',index=False)
    retained=[]
    for r in routes:
        q=r['query_id']; allowed=set(allowed_indices(r,sources)); positives={p:g for p,g in rels[q].items() if g>0}
        retained_pos={p:g for p,g in positives.items() if pids.index(p) in allowed}
        excluded={p:g for p,g in positives.items() if p not in retained_pos}
        retained.append({'query_id':q,'cohort':r['cohort'],'routed':bool(r['route_doc_ids']),'route_reason':r['reason'],
          'positive_pages':len(positives),'retained_positive_pages':len(retained_pos),'positive_page_retention':len(retained_pos)/len(positives),
          'positive_grade_mass':sum(positives.values()),'retained_positive_grade_mass':sum(retained_pos.values()),
          'grade_mass_retention':sum(retained_pos.values())/sum(positives.values()),'any_gold_excluded':bool(excluded),
          'excluded_positive_pages_json':json.dumps(excluded,sort_keys=True)})
    retention=pd.DataFrame(retained); retention.to_csv(OUT/'qrel_retention.csv',index=False)
    retention_summary=expand_scopes(retention).groupby(['cohort','scope'],sort=False).agg(n=('query_id','size'),
      queries_with_gold_excluded=('any_gold_excluded','sum'),mean_positive_page_retention=('positive_page_retention','mean'),
      mean_grade_mass_retention=('grade_mass_retention','mean'),positive_pages=('positive_pages','sum'),retained_positive_pages=('retained_positive_pages','sum'),
      positive_grade_mass=('positive_grade_mass','sum'),retained_positive_grade_mass=('retained_positive_grade_mass','sum')).reset_index()
    retention_summary.to_csv(OUT/'qrel_retention_summary.csv',index=False)
    route_stats=[]
    for cohort in ['initial60','remaining249','full309']:
        selected=[r for r in routes if cohort=='full309' or r['cohort']==cohort]
        route_stats.append({'cohort':cohort,'n':len(selected),'routed':sum(bool(r['route_doc_ids']) for r in selected),
          'multiple_banks':sum(len(r['route_doc_ids'])>1 for r in selected),'fallback':sum(not r['route_doc_ids'] for r in selected),
          **{reason:sum(r['reason']==reason for r in selected) for reason in sorted({r['reason'] for r in routes})}})
    pd.DataFrame(route_stats).to_csv(OUT/'routing_summary.csv',index=False)
    dump(OUT/'resources.json',{'api_calls':0,'new_embeddings':0,'new_generations':0,'new_models':0,'paid_cost_usd':0,
      'evaluation_wall_seconds_before_validation_and_report':time.perf_counter()-start,
      'full_score_float32_bytes':sum((OUT/f'full_scores_{m}.npy').stat().st_size for m in METHODS),
      'timing_note':'Local cached-vector/BM25 replay and evaluation; not controlled production latency or end-to-end RAG'})
    validate(); report()

def validate():
    check(); qs,pages=meta(); routes=read(OUT/'routes.json'); sources=np.array([p['doc_id'] for p in pages])
    pids=[str(p['corpus_id']) for p in pages]; rels=read(S/'qrels.json')
    old=np.load(D/'rankings.npz'); sparse=np.load(S/'rankings.npz')
    unfiltered=np.load(OUT/'unfiltered_rankings.npz'); filtered=np.load(OUT/'filtered_rankings.npz')
    gate=read(OUT/'replay_validation.json'); assert gate['passed'] and gate['all_unfiltered_top100_exact']
    assert len(routes)==309
    for q,r in zip(qs,routes):
        expected=route(q['question'])
        assert all(r[k]==v for k,v in expected.items())
        assert r['query_id']==q['query_id'] and r['allowed_pages']==len(allowed_indices(r,sources))
    frame=pd.read_csv(OUT/'per_query_metrics.csv',dtype={'query_id':str})
    assert len(frame)==309*4*2*len(KS) and not frame.duplicated(['query_id','method','condition','k']).any()
    lookup=frame.set_index(['query_id','method','condition','k'])
    rank_checks=0; fallback_checks=0
    for method in METHODS:
        assert np.array_equal(unfiltered[method],sparse['bm25'] if method=='bm25' else old[method])
        assert sha(OUT/f'full_scores_{method}.npy')==gate['full_score_hashes'][method]
        scores=np.load(OUT/f'full_scores_{method}.npy')
        for i,r in enumerate(routes):
            allowed=allowed_indices(r,sources)
            independent=sorted(allowed,key=lambda p:(-float(scores[i,p]),int(p)))[:100]
            got=filtered[method][i]
            assert np.array_equal(independent,got) and len(set(got))==100
            assert set(got).issubset(set(allowed))
            if not r['route_doc_ids']:
                assert np.array_equal(got,unfiltered[method][i]);fallback_checks+=1
            for condition,arr in [('unfiltered',unfiltered[method]),('filtered',filtered[method])]:
                for k in KS:
                    # Independent NumPy metric calculation retains ORIGINAL ideal/recall denominator.
                    grades=np.array([rels[r['query_id']].get(pids[int(p)],0) for p in arr[i,:k]],dtype=float)
                    positive=grades>0; positions=np.flatnonzero(positive)
                    ideal=np.sort(np.array(list(rels[r['query_id']].values()),dtype=float))[::-1][:k]
                    idcg=float(np.sum(ideal/np.log2(np.arange(len(ideal))+2)))
                    expected={'hit':float(positive.any()),'recall':positive.sum()/sum(g>0 for g in rels[r['query_id']].values()),
                      'mrr':1/(positions[0]+1) if len(positions) else 0.,'ndcg':float(np.sum(grades/np.log2(np.arange(len(grades))+2)))/idcg if idcg else 0.}
                    row=lookup.loc[(r['query_id'],method,condition,k)]
                    assert all(abs(row[m]-v)<1e-12 for m,v in expected.items())
            rank_checks+=1
    retained=pd.read_csv(D/'per_query_metrics.csv',dtype={'query_id':str})
    retained=retained[(retained.method.isin(METHODS)) & retained.representation.isin(['fullpage_chunkmax','full_ocr_bm25'])]
    joined=frame[frame.condition=='unfiltered'].merge(retained,on=['query_id','method','k'],suffixes=('','_old'),validate='one_to_one')
    assert len(joined)==309*4*len(KS)
    for m in METRICS: assert np.allclose(joined[m],joined[m+'_old'],rtol=0,atol=1e-12)
    expanded=expand_scopes(frame); summary=pd.read_csv(OUT/'summary.csv')
    for row in summary.to_dict('records'):
        sub=expanded[(expanded.cohort==row['cohort']) & (expanded.scope==row['scope']) & (expanded.method==row['method']) & (expanded.condition==row['condition']) & (expanded.k==row['k'])]
        assert len(sub)==row['n'] and all(abs(sub[m].mean()-row[m])<1e-12 for m in METRICS)
    retention=pd.read_csv(OUT/'qrel_retention.csv',dtype={'query_id':str}).set_index('query_id')
    for r in routes:
        all_rel={p:g for p,g in rels[r['query_id']].items() if g>0}; allowed=set(allowed_indices(r,sources))
        kept={p:g for p,g in all_rel.items() if pids.index(p) in allowed}; row=retention.loc[r['query_id']]
        assert row.positive_pages==len(all_rel) and row.retained_positive_pages==len(kept)
        assert abs(row.positive_page_retention-len(kept)/len(all_rel))<1e-12
        assert abs(row.grade_mass_retention-sum(kept.values())/sum(all_rel.values()))<1e-12
    dump(OUT/'validation.json',{'passed':True,'api_calls':0,'new_embeddings':0,'new_pvalues':0,
      'question_only_routes_reproduced':309,'full_corpus_unfiltered_replay_checks':309*4,
      'independent_allowed_page_rank_checks':rank_checks,'fallback_identical_rank_checks':fallback_checks,
      'original_complete_qrels_IDCG_and_recall_denominators_verified':True,'baseline_metric_rows_reproduced':len(joined),
      'same_frozen_mask_all_methods':True,'all_summary_denominators_verified':True,'qrel_retention_verified':309,
      'input_and_implementation_hashes_unchanged':True,'metric_rows':len(frame),'summary_rows':len(summary),
      'status':'Single fixed exploratory rule; no tuning or new heldout confirmation'})

def report():
    os.environ.setdefault('MPLCONFIGDIR',str(OUT/'matplotlib_cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary=pd.read_csv(OUT/'summary.csv'); deltas=pd.read_csv(OUT/'matched_deltas.csv')
    routing=pd.read_csv(OUT/'routing_summary.csv'); retention=pd.read_csv(OUT/'qrel_retention_summary.csv')
    data=summary[(summary.cohort=='remaining249') & (summary.scope=='all') & (summary.k==10)].set_index(['method','condition'])
    fig,ax=plt.subplots(figsize=(8.2,4.6)); x=np.arange(4);width=.36
    for j,(condition,label,color) in enumerate([('unfiltered','Original full-corpus','#9aa6b1'),('filtered','Question-only bank constraint','#347da5')]):
        values=[data.loc[(m,condition),'ndcg'] for m in METHODS]
        bars=ax.bar(x+(j-.5)*width,values,width,label=label,color=color)
        for bar,v in zip(bars,values): ax.text(bar.get_x()+bar.get_width()/2,v+.006,f'{v:.3f}',ha='center',fontsize=8)
    ax.set_xticks(x,[LABELS[m] for m in METHODS]);ax.set_ylabel('nDCG@10');ax.set_ylim(0,float(data.ndcg.max())*1.25)
    ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);ax.legend(frameon=False,fontsize=9)
    fig.suptitle('ViDoRe Finance: fixed company constraint (all remaining249)')
    fig.tight_layout(rect=[0,.07,1,.94]);fig.text(.5,.025,'Previously inspected benchmark; descriptive only. Original complete qrels/IDCG/Recall denominators.',ha='center',fontsize=8)
    fig.savefig(OUT/'company_constraint.png',dpi=180);plt.close(fig)
    lines=['# Fixed company constraint: offline evaluation','','One fixed, question-only metadata constraint on the previously inspected ViDoRe Finance English309 benchmark. **All remaining249** is primary descriptive; initial60/full309 are secondary. No new API calls, generations, embeddings, models or p-values. No rule tuning after outcomes.','',
      'Aliases: Bank of America; Citigroup/Citi; Goldman Sachs; JPMorgan/JPMorgan Chase/JPMorganChase/J.P. Morgan; Morgan Stanley; Wells Fargo. Matching ignores case and normalizes only name-internal spaces/hyphens/periods. Multiple names form a union. No match, quoted company mentions, negation/exclusion markers, or one named bank with an open comparison trigger fallback. Exact regexes and per-question reasons are retained. No year filter.','',
      'Routes and code/input hashes were frozen before score calculation. Four methods share the same mask and rank **all allowed pages**, using unchanged full-OCR BM25 scores or cached MiniLM chunk-max scores. All 1,236 original unfiltered Top100 rankings replayed exactly. Metrics always retain complete original qrels, IDCG and Recall denominators; no excluded labels are silently removed.','',
      '| All remaining249 | Original nDCG@10 | Constrained | Difference | Original→constrained Hit@10 | Original→constrained Recall@10 |','|---|---:|---:|---:|---:|---:|']
    for m in METHODS:
        a,b=data.loc[(m,'unfiltered')],data.loc[(m,'filtered')]
        lines.append(f'| {LABELS[m]} | {a.ndcg:.4f} | {b.ndcg:.4f} | {b.ndcg-a.ndcg:+.4f} | {a.hit:.1%}→{b.hit:.1%} | {a.recall:.4f}→{b.recall:.4f} |')
    lines += ['','![All249 matched comparison](company_constraint.png)','','| Secondary cohort | BM25 ΔnDCG@10 | Question Δ | Focused mix Δ | Short mix Δ |','|---|---:|---:|---:|---:|']
    for cohort in ['initial60','full309']:
        rows=deltas[(deltas.cohort==cohort)&(deltas.scope=='all')&(deltas.k==10)].set_index('method')
        lines.append('| '+cohort+' | '+' | '.join(f'{rows.loc[m,"ndcg_delta"]:+.4f}' for m in METHODS)+' |')
    lines += ['','| Cohort | Routed / all | Multi-bank | Fallback | Queries losing any positive label | Mean positive-page retention |','|---|---:|---:|---:|---:|---:|']
    for row in routing.to_dict('records'):
        gold=retention[(retention.cohort==row['cohort'])&(retention.scope=='all')].iloc[0]
        lines.append(f"| {row['cohort']} | {row['routed']}/{row['n']} | {row['multiple_banks']} | {row['fallback']} | {int(gold.queries_with_gold_excluded)} | {gold.mean_positive_page_retention:.1%} |")
    hyde=pd.read_csv(OUT/'hyde_contrasts.csv');h=hyde[(hyde.cohort=='remaining249')&(hyde.scope=='all')]
    for m in ['focused160_mix_dense','short_30_50_mix_dense']:
        rows=h[h.method==m].set_index('condition')
        lines += ['',f"All249 {LABELS[m]} minus question nDCG@10: original **{rows.loc['unfiltered','ndcg_delta']:+.4f}**, constrained **{rows.loc['filtered','ndcg_delta']:+.4f}**. This separates the metadata intervention from query representation."]
    lines += ['','Routed/fallback matched results and grade-mass retention are in the CSVs. Source mismatch against qrels is diagnostic, not proof that another page is factually incorrect. These reused cohorts share six annual-report sources; no generalization or final-answer accuracy claim follows. The single fixed experiment is complete; aliases/guards were not adjusted.','',
      'Reproduce from retained caches:','', '```bash','.venv/bin/python person3_meeting9_constraints/run.py prepare','.venv/bin/python person3_meeting9_constraints/run.py replay','.venv/bin/python person3_meeting9_constraints/run.py evaluate','```','',
      '`validate` independently checks full-allowed-page ranks, unchanged fallbacks, original metric denominators, exact baseline replay, gold retention, frozen routes and hashes.']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n');export();print((OUT/'README.md').read_text())

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','replay','evaluate','validate','report'])
    {'prepare':prepare,'replay':replay,'evaluate':evaluate,'validate':validate,'report':report}[p.parse_args().stage]()
