"""Meeting 11 controlled Legal MultiQuery study. Stages are resumable.

No upstream request is made outside the explicitly named generate stage.
Corpus and successful historical generations are immutable inputs.
"""
from pathlib import Path
from collections import Counter
import argparse
import concurrent.futures as cf
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
import platform
import sys
import time

for key, value in {"HF_HUB_DISABLE_TELEMETRY":"1", "DO_NOT_TRACK":"1",
    "HF_HUB_OFFLINE":"1", "TRANSFORMERS_OFFLINE":"1", "LANGSMITH_TRACING":"false",
    "LANGCHAIN_TRACING_V2":"false"}.items():
    os.environ.setdefault(key,value)

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT.parent
OUT=ROOT/'results/person3_meeting11_legal/v1'
EXPORT=BASE/'outputs/meeting11_20261123'
OLD=ROOT/'results/person3_meeting10_legal/v1'
MODEL='gpt-4o-mini-2024-07-18'
REV='1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
KS=[5,8,10]
BUDGETS=[5000,8000,10000]
DEPTH=100
VARIANTS=['question','default','legal']
BACKENDS=['bm25','dense','hybrid']
NEW_GUARD=.25
LEGAL_PROMPT='''You are an AI language model assistant helping retrieve passages from a legal corpus.
Generate 3 different search questions that preserve the original legal information need.
Keep the original facts, named entities, jurisdiction, procedural stage, legal relationships,
qualifications, and exceptions when they are stated. Do not invent facts, jurisdictions,
case names, statutes, or legal conclusions. Use complementary legal terminology or
alternative phrasing only where it expresses the same issue; avoid adjacent legal topics.
Write questions for searching, not hypothetical answers or evidence passages.
Provide exactly 3 alternative questions separated by newlines, with no heading or explanation.
Original question: {question}'''


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


old=module('meeting11_historical_helpers',ROOT/'person3_meeting10_legal/run.py')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w') as handle:
        json.dump(value,handle,indent=2,ensure_ascii=False,allow_nan=False)
        handle.write('\n');handle.flush();os.fsync(handle.fileno())
    tmp.replace(path)
def freeze(path,value):
    if path.exists():assert read(path)==value,f'Frozen artifact changed: {path}'
    else:dump(path,value)
def data():return old.data()
def check():
    for p,h in read(OUT/'source_hashes.json').items():assert sha(ROOT/p)==h,p
    assert sha(Path(__file__))==read(OUT/'implementation.json')['run_sha256'],'Frozen execution source changed'
def rank(scores):return np.argsort(-scores,kind='stable')[:DEPTH].astype(np.int32)
def fuse(routes,weights=None,depth=DEPTH):
    scores={}
    if weights is None:weights=[1.]*len(routes)
    assert len(routes)==len(weights) and routes
    for route,w in zip(routes,weights):
        assert len(route)==len(set(map(int,route)))
        for r,pid in enumerate(route,1):
            i=int(pid);scores[i]=scores.get(i,0.)+w/(60+r)
    return sorted(scores,key=lambda i:(-scores[i],i))[:depth]
def metrics(ids,gold):
    where=[i+1 for i,x in enumerate(ids) if x==gold]
    r=where[0] if where else 0
    return dict(hit=float(r>0),recall=float(r>0),mrr=1/r if r else 0.,ndcg=1/np.log2(r+1) if r else 0.)
def querysets():
    return {'default':read(OLD/'multiquery.json'),**({'legal':read(OUT/'legal_queries.json')} if (OUT/'legal_queries.json').exists() else {})}


def prepare():
    OUT.mkdir(parents=True,exist_ok=True);EXPORT.mkdir(parents=True,exist_ok=True)
    qa,pages=data();split=read(OLD/'selection.json');assert split==old.choose_groups(qa)
    cache=read(OLD/'multiquery.json');assert set(cache)=={q['id'] for q in qa}
    protocol_old=read(OLD/'protocol.json')
    prior=read(BASE/'outputs/CURRENT_API_BUDGET.json')
    assert prior['guard_usd']==7 and prior['unknown_reservations_usd']==0
    # Resume uses the initial budget snapshot, rather than counting this run twice.
    if (OUT/'budget_before.json').exists():prior=read(OUT/'budget_before.json')
    freeze(OUT/'budget_before.json',prior)
    assert all(len(cache[q['id']]['generated_queries'])==3 and cache[q['id']]['queries'][-1]==q['question'] for q in qa)
    protocol=dict(date='2026-11-23',dataset='Legal RAG Bench',n=100,primary_n=80,corpus_n=4876,
        split=split,previously_inspected=True,model=MODEL,historical_request_alias='gpt-4o-mini',
        historical_returned_snapshot=MODEL,temperature=0,max_completion_tokens=256,
        default_prompt=protocol_old['multiquery']['default_prompt'],legal_prompt=LEGAL_PROMPT,
        requested_alternatives=3,include_original=True,parser='Native LangChain LineListOutputParser; no outcome-dependent repair',
        encoder=dict(model='sentence-transformers/all-MiniLM-L6-v2',revision=REV,max_tokens=256,
                     pooling='attention-mask mean, then L2 normalization',backend='offline PyTorch CPU float32'),
        bm25=dict(k1=1.2,b=.75,binary_query_counts=True,document='title + full text'),
        ranking=dict(candidate_depth=DEPTH,ks=KS,constant=60,query_fusion='Equal RRF of per-query top k, output k; identical to Meeting10 BM25',
            hybrid='For each query fuse BM25 top100 and dense top100, equal weights, output100; then query fusion at k',
            fixed100_diagnostic='Separately fuse each query top100, then cut at k; not used to replace primary results',
            tie_break='Ascending corpus index',duplicate_queries='Retained exactly, as in native historical adapter'),
        design=dict(backends=BACKENDS,variants=VARIANTS,manual_prompt='Frozen before new outcomes; not DSPy/GEPA optimized',
            evaluation='Same80 questions; all100 only descriptive; no final reader calls'),
        delivery=dict(budgets=BUDGETS,tokenizer='tiktoken o200k_base',policy='Ranking order; truncate final passage prefix; never skip to later passages',
            context='Passage N:\\n<title + text>, separated by two newlines',
            budget_scope='Entire evidence context including labels and separators; excludes question and reader instructions',
            full_label='Primary delivered hit/recall requires entire labeled passage; partial touch reported separately; not span-level sufficiency'),
        statistics=dict(primary='Nine two-sided exact paired McNemar Hit@5 contrasts on test80; Holm9 across all nine',
            contrasts='For each backend: default vs question, legal vs question, legal vs default',
            secondary='Other k, delivered budgets and fixed100 diagnostic descriptive; no selected significant secondary contrasts',
            interpretation='Exploratory; no correction for historical adaptive searches; no independent generalization'),
        budget=dict(prior_usd=prior['cumulative_estimate_usd'],cumulative_guard_usd=7,new_guard_usd=NEW_GUARD,max_new_calls=100,
            prices_per_million={'input':.15,'output':.60},pricing_checked='2026-11-23',
            source='https://developers.openai.com/api/docs/models/gpt-4o-mini',automatic_retries=0,unknown_attempts_reserved=True),
        acceptance=['All100 default queries reused, zero default generation calls',
            '100 exact BM25 top100 replays; all300 default per-k Meeting10 rankings replay',
            'Encoder question vectors match stored values within1e-5',
            'All nine methods have80 scores at each k and all nine budget pairs',
            'Legal prompt100 successful outputs or explicitly incomplete, with costs saved per call',
            'No historical file overwritten; checksums verified before each stage'])
    freeze(OUT/'protocol.json',protocol);freeze(OUT/'selection.json',split)
    paths=[ROOT/'data/legal_rag_bench/qa.jsonl',ROOT/'data/legal_rag_bench/corpus.jsonl',OLD/'selection.json',OLD/'multiquery.json',OLD/'rankings.json',
        OLD/'api/api_checkpoint.sqlite',ROOT/'results/person3_openai_hyde/embeddings.npz',ROOT/'results/person3_openai_hyde/saved_rankings.npz',
        ROOT/'person3_meeting10_legal/run.py',ROOT/'person3_meeting10_legal/api.py']
    freeze(OUT/'source_hashes.json',{str(p.relative_to(ROOT)):sha(p) for p in paths})
    freeze(OUT/'implementation.json',dict(run_sha256=sha(Path(__file__))))
    if not (OUT/'freeze_receipt.json').exists():
        dump(OUT/'freeze_receipt.json',dict(utc=datetime.now(timezone.utc).isoformat(),outcomes_computed=False,
             hashes={p:sha(OUT/p) for p in ['protocol.json','selection.json','source_hashes.json','implementation.json','budget_before.json']}))
    print('Protocol frozen: same80 test questions, nine methods,3 depths x3 budgets; no API calls.',flush=True)


def encode_queries(texts):
    torch=__import__('torch')
    from transformers import AutoTokenizer,AutoModel
    torch.set_num_threads(4)
    tok=AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2',revision=REV,
        cache_dir=str(BASE/'model_cache'),local_files_only=True)
    model=AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2',revision=REV,
        cache_dir=str(BASE/'model_cache'),local_files_only=True).to(device='cpu',dtype=torch.float32).eval()
    result=np.empty((len(texts),384),dtype=np.float32)
    order=sorted(range(len(texts)),key=lambda i:len(texts[i]),reverse=True)
    for start in range(0,len(order),16):
        inds=order[start:start+16]
        tokens=tok([texts[i] for i in inds],padding=True,truncation=True,max_length=256,return_tensors='pt')
        with torch.inference_mode():
            h=model(**tokens).last_hidden_state
            mask=tokens['attention_mask'].unsqueeze(-1).float()
            pooled=(h*mask).sum(1)/mask.sum(1).clamp(min=1e-12)
            v=pooled.numpy();v=v/np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-12)
        result[inds]=v
    return result


def routes(variant):
    check();qa,pages=data();qsets=querysets();assert variant in qsets
    saved=qsets[variant];pids=[p['id'] for p in pages];byid={p:i for i,p in enumerate(pids)}
    vector_cache=OUT/f'{variant}_query_vectors.npz'
    alltexts=list(dict.fromkeys(t for q in qa for t in saved[q['id']]['queries']))
    # Question vectors are checked independently and then use exact historical values.
    z=np.load(ROOT/'results/person3_openai_hyde/embeddings.npz');qvectors=z['questions'];corpus=z['corpus']
    if vector_cache.exists():
        vz=np.load(vector_cache);assert vz['texts'].tolist()==alltexts;vectors=vz['vectors']
    else:
        vectors=encode_queries(alltexts);np.savez_compressed(vector_cache,texts=np.array(alltexts),vectors=vectors)
    vector_by_text=dict(zip(alltexts,vectors));delta=max(float(np.max(abs(vector_by_text[q['question']]-qvectors[i]))) for i,q in enumerate(qa))
    assert delta<=1e-5,delta
    bm=old.BM25(pages);result={};base=np.load(OLD/'baseline_rankings.npz');gold=base['gold']
    assert base['passage_ids'].tolist()==pids
    dense_gold=np.load(ROOT/'results/person3_openai_hyde/saved_rankings.npz')['question_dense']
    qreplay=[];old_route_checks=0
    for i,q in enumerate(qa):
        qid=q['id'];items=[]
        for query in saved[qid]['queries']:
            b=bm.indices(query,DEPTH)
            v=qvectors[i] if query==q['question'] else vector_by_text[query]
            d=rank(v@corpus.T)
            if query==q['question']:
                assert np.array_equal(b,base['bm25'][i]);assert np.array_equal(d,dense_gold[i])
                qreplay.append(qid)
            if variant=='default':
                oldroute=saved[qid]['per_query_rankings'][len(items)]['passage_ids']
                assert [pids[j] for j in b[:len(oldroute)]]==oldroute
                old_route_checks+=1
            items.append(dict(query=query,bm25=b.tolist(),dense=d.tolist(),hybrid=fuse([b,d],[.5,.5])))
        result[qid]=dict(gold=int(gold[i]),routes=items)
        if (i+1)%20==0:
            dump(OUT/f'{variant}_routes.partial.json',result)
            print(f'Ranked {variant} {i+1}/100',flush=True)
    dump(OUT/f'{variant}_routes.json',result)
    dump(OUT/f'{variant}_replay.json',dict(passed=True,question_encoder_max_abs_diff=delta,original_question_replays=len(qreplay),
        bm25_top100_exact=True,dense_top100_exact=True,historical_generated_bm25_routes_checked=old_route_checks,api_calls=0))
    evaluate([variant])


def api_proxy():
    api=module('meeting11_budget_gateway',ROOT/'person3_meeting10_legal/api.py');api.MODEL=MODEL
    p=read(OUT/'protocol.json')['budget']
    return api.BudgetProxy(OUT/'api',prior_usd=p['prior_usd'],new_guard_usd=NEW_GUARD,max_unique_attempts=100)


def generate():
    check();qa,_=data();path=OUT/'legal_queries.json';saved=read(path) if path.exists() else {}
    from langchain_classic.retrievers.multi_query import LineListOutputParser
    parser=LineListOutputParser();proxy=api_proxy()
    try:
        proxy.access_check();proxy.set_stage('legal_prompt_generation')
        def work(q):
            request=dict(model=MODEL,temperature=0.,max_completion_tokens=256,stream=False,
                messages=[dict(role='user',content=LEGAL_PROMPT.format(question=q['question']))])
            started=time.perf_counter();timestamp=time.time();cached=False
            response,cached=proxy._complete(request,'legal_prompt_generation')
            proxy._event(stage='legal_prompt_generation',body=request,cached=cached,status=200,
                timestamp=timestamp,elapsed=time.perf_counter()-started)
            assert response['model']==MODEL and response['choices'][0]['finish_reason']=='stop'
            text=response['choices'][0]['message']['content'];lines=parser.parse(text)
            assert lines and all(isinstance(t,str) and t.strip() for t in lines)
            return dict(question=q['question'],queries=lines+[q['question']],generated_queries=lines,
                generated_query_count=len(lines),raw_text=text,prompt=request['messages'][0]['content'],
                request=request,response_id=response['id'],model=response['model'],usage=response['usage'],
                elapsed_seconds=time.perf_counter()-started,cache_hit=cached)
        old.windowed([q for q in qa if q['id'] not in saved],work,
            lambda q,r:save_query(q,r,saved,path),concurrency=4)
    finally:
        summary=proxy.summary();proxy.close();update_budget(summary)
    assert len(saved)==100
    dump(OUT/'generation_validation.json',dict(passed=True,outputs=100,model=MODEL,
        query_counts=dict(Counter(str(x['generated_query_count']) for x in saved.values())),
        unique_response_ids=len({x['response_id'] for x in saved.values()}),reference_answers_in_prompts=False))


def save_query(q,r,saved,path):
    saved[q['id']]=r;dump(path,saved)
    print(f'Legal prompt checkpoint {len(saved)}/100',flush=True)


def update_budget(summary):
    before=read(OUT/'budget_before.json')
    dump(BASE/'outputs/CURRENT_API_BUDGET.json',dict(latest_round='Meeting11 Legal MultiQuery dense/hybrid and domain prompt',
        cumulative_estimate_usd=summary['cumulative_estimate_including_reserves_usd'],guard_usd=7,
        unknown_reservations_usd=summary['unresolved_reserved_usd'],prior_estimate_usd=before['cumulative_estimate_usd'],
        meeting11_new_usd=summary['known_actual_cost_usd'],source=str((OUT/'api/cost_summary.json').relative_to(BASE))))


def all_rankings(variants=None):
    qa,_=data();available=[v for v in ['default','legal'] if (OUT/f'{v}_routes.json').exists()]
    if variants is not None:available=[v for v in available if v in variants]
    result={q['id']:{} for q in qa};diagnostic={q['id']:{} for q in qa}
    for variant in available:
        source=read(OUT/f'{variant}_routes.json')
        for q in qa:
            qid=q['id'];record=source[qid];last=record['routes'][-1]
            for backend in BACKENDS:
                # The retained original question is the last native route.
                for k in KS:
                    result[qid].setdefault('question_'+backend,{})[str(k)]=last[backend][:k]
                    lists=[r[backend] for r in record['routes']]
                    result[qid].setdefault(variant+'_'+backend,{})[str(k)]=fuse([r[:k] for r in lists],depth=k)
                    diagnostic[qid].setdefault(variant+'_'+backend,{})[str(k)]=fuse(lists)[:k]
    return result,diagnostic


def evaluate(variants=None):
    check();qa,_=data();split=read(OUT/'selection.json');ranks,diag=all_rankings(variants)
    rows=[];diagnostic=[];old_ranks=read(OLD/'rankings.json');_,pages=data();pids=[p['id'] for p in pages]
    gold={q['id']:pids.index(q['relevant_passage_id']) for q in qa};replays=0
    for q in qa:
        qid=q['id'];cohort='test80' if qid in split['test_ids'] else 'dev20'
        for method,kmap in ranks[qid].items():
            for k,ids in kmap.items():
                if method=='default_bm25':
                    assert [pids[i] for i in ids]==old_ranks[qid]['multiquery'][k];replays+=1
                rows.append(dict(query_id=qid,cohort=cohort,method=method,k=int(k),**metrics(ids,gold[qid])))
        for method,kmap in diag[qid].items():
            for k,ids in kmap.items():diagnostic.append(dict(query_id=qid,cohort=cohort,method=method,k=int(k),**metrics(ids,gold[qid])))
    dump(OUT/'rankings.json',ranks);dump(OUT/'fixed100_rankings.json',diag)
    df=pd.DataFrame(rows);df.to_csv(OUT/'retrieval_per_query.csv',index=False)
    summary(df).to_csv(OUT/'retrieval_summary.csv',index=False)
    summary(pd.DataFrame(diagnostic)).to_csv(OUT/'fixed100_summary.csv',index=False)
    tests=[];test=df[(df.cohort=='test80') & (df.k==5)]
    for backend in BACKENDS:
        for candidate,control in [('default','question'),('legal','question'),('legal','default')]:
            c=test[test.method==candidate+'_'+backend].set_index('query_id').hit
            b=test[test.method==control+'_'+backend].set_index('query_id').hit
            if len(c)!=80 or len(b)!=80:continue
            assert set(c.index)==set(b.index);d=c-b
            w=int((d>0).sum());l=int((d<0).sum());p=binomtest(w,w+l).pvalue if w+l else 1.
            tests.append(dict(candidate=candidate+'_'+backend,control=control+'_'+backend,n=80,k=5,
                metric='hit',mean_delta=float(d.mean()),wins=w,losses=l,ties=int((d==0).sum()),p_raw=p))
    t=pd.DataFrame(tests)
    if len(t)==9:
        order=np.argsort(t.p_raw,kind='stable');adj=np.empty(9)
        adj[order]=np.minimum(1,np.maximum.accumulate(np.asarray(t.p_raw)[order]*(9-np.arange(9))))
        t['p_holm9']=adj;t['significant_holm9']=adj<.05
    if not t.empty:t.to_csv(OUT/'primary_tests.csv',index=False)
    dump(OUT/'ranking_status.json',dict(methods=len(ranks[qa[0]['id']]),queries=100,rows=len(df),
        historical_per_k_replays=replays,complete=len(ranks[qa[0]['id']])==9,api_calls_this_stage=0))
    print(summary(df[df.cohort=='test80']).query('k==5').to_string(index=False),flush=True)


def summary(df):
    return df.groupby(['cohort','method','k'],as_index=False).agg(n=('query_id','size'),**{m:(m,'mean') for m in ['hit','recall','mrr','ndcg']})


def budget_eval():
    check();qa,pages=data();ranks,_=all_rankings();assert all(len(ranks[q['id']])==9 for q in qa)
    os.environ['TIKTOKEN_CACHE_DIR']=str(BASE/'work/meeting11_20261123/tiktoken_cache')
    import tiktoken
    enc=tiktoken.get_encoding('o200k_base');texts=[(p['title']+' '+p['text']).strip() for p in pages]
    tokens=[enc.encode(t,disallowed_special=()) for t in texts]
    split=read(OUT/'selection.json');gold={q['id']:[p['id'] for p in pages].index(q['relevant_passage_id']) for q in qa}
    rows=[];delivery_records=[]
    for qi,q in enumerate(qa):
        qid=q['id'];cohort='test80' if qid in split['test_ids'] else 'dev20'
        for method,kmap in ranks[qid].items():
            for k,ids in kmap.items():
                for budget in BUDGETS:
                    parts=[];delivered=[];full=[];partial=None;context='';partial_tokens=0
                    for pos,pid in enumerate(ids,1):
                        prefix=('\n\n' if parts else '')+f'Passage {pos}:\n'
                        candidate=context+prefix+texts[pid]
                        if len(enc.encode(candidate,disallowed_special=()))<=budget:
                            context=candidate;parts.append(pid);delivered.append(pid);full.append(pid);continue
                        lo,hi=0,len(tokens[pid])
                        while lo<hi:
                            mid=(lo+hi+1)//2
                            fragment=enc.decode(tokens[pid][:mid],errors='ignore')
                            if len(enc.encode(context+prefix+fragment,disallowed_special=()))<=budget:lo=mid
                            else:hi=mid-1
                        if lo:
                            fragment=enc.decode(tokens[pid][:lo],errors='ignore')
                            assert texts[pid].startswith(fragment)
                            context+=prefix+fragment;delivered.append(pid);partial=pid;partial_tokens=len(enc.encode(fragment,disallowed_special=()))
                        break
                    used=len(enc.encode(context,disallowed_special=()));assert used<=budget
                    fullmetrics=metrics(full,gold[qid]);touchmetrics=metrics(delivered,gold[qid])
                    row=dict(query_id=qid,cohort=cohort,method=method,k=int(k),budget=budget,context_tokens=used,
                        full_passages=len(full),delivered_passages=len(delivered),partial_passages=int(partial is not None),
                        excluded_seed_passages=len(ids)-len(delivered),all_k_fit=int(len(full)==len(ids)),
                        full_hit=fullmetrics['hit'],full_recall=fullmetrics['recall'],full_mrr=fullmetrics['mrr'],full_ndcg=fullmetrics['ndcg'],
                        touched_hit=touchmetrics['hit'],touched_recall=touchmetrics['recall'],gold_partial=int(partial==gold[qid]))
                    rows.append(row)
                    delivery_records.append(dict(query_id=qid,method=method,k=int(k),budget=budget,full_ids=full,
                        partial_id=partial,partial_tokens=partial_tokens,context_tokens=used,
                        context_sha256=hashlib.sha256(context.encode()).hexdigest()))
        if (qi+1)%20==0:print(f'Packed {qi+1}/100 questions across all81 settings',flush=True)
    df=pd.DataFrame(rows);df.to_csv(OUT/'delivery_per_query.csv',index=False)
    stats={c:(c,'mean') for c in ['context_tokens','full_passages','delivered_passages','partial_passages',
        'excluded_seed_passages','all_k_fit','full_hit','full_recall','full_mrr','full_ndcg','touched_hit','touched_recall','gold_partial']}
    df.groupby(['cohort','method','k','budget'],as_index=False).agg(n=('query_id','size'),median_context_tokens=('context_tokens','median'),**stats).to_csv(OUT/'delivery_summary.csv',index=False)
    dump(OUT/'delivery_records.json',delivery_records)
    dump(OUT/'tokenizer.json',dict(encoding=enc.name,max_passage_tokens=max(map(len,tokens)),
        median_passage_tokens=float(np.median([len(t) for t in tokens])),vocab_size=enc.n_vocab,
        cache_hashes={p.name:sha(p) for p in (BASE/'work/meeting11_20261123/tiktoken_cache').iterdir() if p.is_file()}))
    print('Budget grid complete:',len(df),'rows; token budgets exact, including evidence labels/separators.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','default','generate','legal','evaluate','budget'])
    args=p.parse_args()
    if args.stage=='default':routes('default')
    elif args.stage=='legal':routes('legal')
    elif args.stage=='evaluate':evaluate()
    elif args.stage=='budget':budget_eval()
    else:globals()[args.stage]()
