"""One fresh Legal HyDE replication with the original model and prompt."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import concurrent.futures as cf
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import sqlite3
import shutil
import time

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/person3_legal_regeneration/fresh100_v1'
EXPORT=ROOT.parent/'outputs/legal_regeneration_20261117'
LEGAL=ROOT/'results/person3_openai_hyde'
SPARSE=ROOT/'results/person3_sparse_hyde/sparse_query_v1/full/legal'
MODEL='gpt-5.4-2026-03-05'
KS=[3,5]
METHODS=['q','old_h','old_qh','fresh_h','fresh_qh']
PAIRS=[dict(k=k,candidate=a,control=b) for k in KS for a,b in
       [('fresh_h','q'),('fresh_qh','q'),('fresh_h','old_h'),('fresh_qh','old_qh')]]
os.environ.setdefault('MPLCONFIGDIR',str(OUT/'matplotlib_cache'))


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    with tmp.open('w') as f:
        json.dump(value,f,indent=2,ensure_ascii=False);f.write('\n');f.flush();os.fsync(f.fileno())
    tmp.replace(p)
def frozen(p,value):
    if p.exists():assert read(p)==value,f'Frozen artifact changed: {p}'
    else:dump(p,value)
def data():
    qa=pd.read_csv(LEGAL/'evaluation_queries.csv',dtype=str).to_dict('records')
    pages=[json.loads(l) for l in (ROOT/'data/legal_rag_bench/corpus.jsonl').read_text().splitlines() if l.strip()]
    old={str(r['query_id']):r for r in map(json.loads,(LEGAL/'generated_passages.jsonl').read_text().splitlines()) if r['profile']=='focused160'}
    assert len(qa)==len(old)==100 and len(pages)==4876
    return qa,pages,old
def engine(pages):
    spec=importlib.util.spec_from_file_location('replication_bm25',ROOT/'person3_meeting10_legal/run.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.BM25(pages)
def check():
    for name in ['source_hashes.json','implementation_hashes.json']:
        for path,digest in read(OUT/name).items():assert sha(ROOT/path)==digest,path
    receipt=read(OUT/'freeze_receipt.json')
    for name,digest in receipt['hashes'].items():assert sha(OUT/name)==digest,name


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    qa,pages,old=data();p=read(LEGAL/'protocol.json')
    template=p['profiles']['focused160']['prompt']
    assert p['model']==MODEL and p['temperature']==0 and p['profiles']['focused160']['max_output_tokens']==160
    for q in qa:assert old[q['id']]['prompt']==template.format(q=q['question'])
    prior=read(ROOT/'results/person3_meeting10_legal/v1/api/cost_summary.json')
    assert prior['unresolved_reserved_usd']==0
    protocol=dict(date='2026-11-17',purpose='Fresh generation and full BM25 pipeline replication of previously inspected Legal100',
        queries=100,corpus=4876,model=MODEL,api='Responses',prompt=template,
        generation=dict(temperature=0,reasoning_effort='none',max_output_tokens=160,store=False,tools='none'),
        bm25='Unchanged full title+text corpus,k1=1.2,b=.75,binary query counts,stable corpus-index ties',
        replay_gate='Recompute Q,oldH,oldQH top100 for all100 and require exact agreement before fresh API calls',
        top_k=KS,methods=METHODS,pairs=PAIRS,
        inference='Eight fixed exploratory exact McNemar comparisons, Holm8 across all eight; no other significance tests',
        preservation='No original calls, generations or saved rankings overwritten',
        caveats=['One regenerated sample per question; temperature0 need not imply identical text.',
                 'Same previously inspected100 questions, not independent confirmation or a causal term-ablation test.',
                 'Question-level tests do not adjust shared source dependence or all historical comparisons.',
                 'A nonsignificant fresh-vs-old difference does not establish equivalence.'],
        budget=dict(prior_usd=prior['cumulative_estimate_including_reserves_usd'],guard_usd=7,new_guard_usd=.75,
                    price_input_per_million=2.5,price_output_per_million=15,
                    pricing_checked='2026-11-17',source='https://developers.openai.com/api/docs/models/gpt-5.4'),
        freshness='New Responses requests in an empty dedicated ledger; old generation cache is not used. Resume only reuses successful calls from this replication.')
    frozen(OUT/'protocol.json',protocol)
    paths=[LEGAL/n for n in ['evaluation_queries.csv','passage_index.csv','saved_rankings.npz','generated_passages.jsonl','protocol.json']]
    paths += [ROOT/'data/legal_rag_bench/corpus.jsonl',ROOT/'person3_meeting10_legal/run.py']+sorted(SPARSE.glob('*.json'))
    frozen(OUT/'source_hashes.json',{str(x.relative_to(ROOT)):sha(x) for x in paths})
    frozen(OUT/'implementation_hashes.json',{str(Path(__file__).resolve().relative_to(ROOT)):sha(Path(__file__))})
    bm=engine(pages);index=pd.read_csv(LEGAL/'passage_index.csv',dtype={'passage_id':str})
    assert bm.pids==index.passage_id.tolist()
    ranks={m:[] for m in METHODS[:3]}
    for q in qa:
        saved=read(SPARSE/f'{q["id"]}.json')['rankings']
        queries={'q':q['question'],'old_h':old[q['id']]['text'],'old_qh':q['question']+'\n'+old[q['id']]['text']}
        mapping={'q':'bm25_q','old_h':'bm25_focused160_h','old_qh':'bm25_focused160_qh'}
        for m,text in queries.items():
            ranking=bm.indices(text,100)
            assert np.array_equal(ranking,saved[mapping[m]]),(q['id'],m,'Replay failed')
            ranks[m].append(ranking)
    np.savez_compressed(OUT/'old_replayed_rankings.npz',query_ids=np.array([q['id'] for q in qa]),
                        passage_ids=np.array(bm.pids),**{m:np.stack(v) for m,v in ranks.items()})
    dump(OUT/'replay_validation.json',dict(passed=True,exact_top100_rankings=300,queries=100,api_calls=0))
    if not (OUT/'freeze_receipt.json').exists():
        assert not (OUT/'api_checkpoint.sqlite').exists()
        names=['protocol.json','source_hashes.json','implementation_hashes.json','replay_validation.json']
        dump(OUT/'freeze_receipt.json',dict(utc=datetime.now(timezone.utc).isoformat(),fresh_calls_not_started=True,
             hashes={n:sha(OUT/n) for n in names}))
    print('Replay passed: all300 old Q/H/Q+H top100 lists match exactly. Fresh round frozen.',flush=True)


def client():
    path=ROOT/'.env.local'
    assert path.is_file() and not path.is_symlink() and path.stat().st_mode&0o077==0
    values={line.split('=',1)[0].strip():line.split('=',1)[1].strip().strip('\"\'')
            for line in path.read_text().splitlines() if '=' in line and not line.lstrip().startswith('#')}
    from openai import OpenAI
    return OpenAI(api_key=values['OPENAI_API_KEY'],organization=values.get('OPENAI_ORG_ID') or os.getenv('OPENAI_ORG_ID','org-ya6BfJ99A55qzohO1pvOUaXO'),max_retries=0,timeout=120)


def generate():
    check();assert read(OUT/'replay_validation.json')['passed']
    settings=read(OUT/'protocol.json');budget=settings['budget'];qa,_,old=data()
    api=client();assert api.models.retrieve(MODEL).id==MODEL
    dump(OUT/'model_access.json',dict(model=MODEL,read_only_check='passed'))
    with sqlite3.connect(OUT/'api_checkpoint.sqlite') as db:
        os.chmod(OUT/'api_checkpoint.sqlite',0o600)
        db.execute('PRAGMA journal_mode=WAL');db.execute('PRAGMA synchronous=FULL')
        db.execute('CREATE TABLE IF NOT EXISTS calls(query_id TEXT PRIMARY KEY,state TEXT,reserve REAL,cost REAL,payload TEXT,created_at REAL)');db.commit()
        def export():
            charge=float(db.execute('SELECT coalesce(sum(coalesce(cost,reserve)),0) FROM calls').fetchone()[0])
            outputs={qid:json.loads(payload) for qid,payload in db.execute("SELECT query_id,payload FROM calls WHERE state='saved'")}
            dump(OUT/'fresh_generations.json',outputs)
            dump(OUT/'cost_summary.json',dict(new_estimate_and_reserves_usd=charge,
                prior_usd=budget['prior_usd'],cumulative_estimate_usd=budget['prior_usd']+charge,guard_usd=7,
                states=dict(db.execute('SELECT state,count(*) FROM calls')),billing_statement=False))
            return charge,outputs
        charge,saved=export()
        assert not db.execute("SELECT 1 FROM calls WHERE state!='saved'").fetchone(),'Unresolved prior attempt; not resending'
        todo=iter([q for q in qa if q['id'] not in saved]);futures={};errors=[]
        def request(q,prompt):
            start=time.perf_counter()
            r=api.responses.create(model=MODEL,input=prompt,max_output_tokens=160,
                reasoning={'effort':'none'},temperature=0,store=False)
            usage=r.usage.model_dump() if r.usage else {}
            text=r.output_text.strip()
            assert usage and text and r.model==MODEL and r.status in ['completed','incomplete']
            assert r.id!=old[q['id']]['response_id'],'Unexpected old response reused'
            cost=(usage['input_tokens']*2.5+usage['output_tokens']*15)/1e6
            return dict(query_id=q['id'],prompt=prompt,text=text,model_returned=r.model,response_id=r.id,
                status=r.status,usage=usage,cost_usd_estimate=cost,seconds=time.perf_counter()-start,
                incomplete_details=r.incomplete_details.model_dump() if r.incomplete_details else None),cost
        def submit(pool):
            nonlocal charge
            try:q=next(todo)
            except StopIteration:return
            prompt=settings['prompt'].format(q=q['question'])
            reserve=((len(prompt.encode())+256)*2.5+160*15)/1e6
            charge,_=export()
            assert charge+reserve<=budget['new_guard_usd'] and budget['prior_usd']+charge+reserve<=7,'Budget reached'
            db.execute('INSERT INTO calls VALUES(?,?,?,?,?,?)',(q['id'],'pending',reserve,None,None,time.time()));db.commit()
            futures[pool.submit(request,q,prompt)]=q['id']
        with cf.ThreadPoolExecutor(max_workers=4) as pool:
            for _ in range(4):submit(pool)
            while futures:
                done,_=cf.wait(futures,return_when=cf.FIRST_COMPLETED)
                for future in done:
                    qid=futures.pop(future)
                    try:
                        payload,cost=future.result()
                        db.execute("UPDATE calls SET state='saved',cost=?,payload=? WHERE query_id=?",(cost,json.dumps(payload),qid))
                    except Exception as error:
                        db.execute("UPDATE calls SET state='failed_or_unknown' WHERE query_id=?",(qid,));errors.append(type(error).__name__)
                    db.commit()
                charge,saved=export();print(f'Fresh H checkpoint {len(saved)}/100; round estimate ${charge:.6f}',flush=True)
                if not errors:
                    for _ in done:submit(pool)
        if errors:raise RuntimeError('Unknown attempt retained; no automatic retry. Types: '+','.join(set(errors)))
        assert len(saved)==100


def evaluate():
    check();qa,pages,old=data();fresh=read(OUT/'fresh_generations.json')
    assert set(fresh)=={q['id'] for q in qa}
    bm=engine(pages);base=np.load(OUT/'old_replayed_rankings.npz')
    ranks={m:base[m] for m in METHODS[:3]}
    ranks['fresh_h']=np.stack([bm.indices(fresh[q['id']]['text'],100) for q in qa])
    ranks['fresh_qh']=np.stack([bm.indices(q['question']+'\n'+fresh[q['id']]['text'],100) for q in qa])
    np.savez_compressed(OUT/'rankings.npz',query_ids=base['query_ids'],passage_ids=base['passage_ids'],**ranks)
    rows=[];stability=[]
    for i,q in enumerate(qa):
        gold=bm.pids.index(q['relevant_passage_id'])
        for m in METHODS:
            for k in KS:
                positions=np.flatnonzero(ranks[m][i,:k]==gold)
                position=int(positions[0]+1) if len(positions) else 0
                rows.append(dict(query_id=q['id'],method=m,k=k,hit=int(position>0),gold_rank=position,
                    mrr=1/position if position else 0.,ndcg=1/math.log2(position+1) if position else 0.))
        stability.append(dict(query_id=q['id'],identical_text=fresh[q['id']]['text']==old[q['id']]['text'],
            distinct_response_id=fresh[q['id']]['response_id']!=old[q['id']]['response_id'],
            h_top5_overlap=len(set(ranks['fresh_h'][i,:5])&set(ranks['old_h'][i,:5]))/5,
            status=fresh[q['id']]['status']))
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'per_query_metrics.csv',index=False)
    summary=frame.groupby(['method','k'],sort=False).agg(n=('query_id','size'),hit=('hit','mean'),mrr=('mrr','mean'),ndcg=('ndcg','mean')).reset_index()
    summary.to_csv(OUT/'summary.csv',index=False);pd.DataFrame(stability).to_csv(OUT/'generation_stability.csv',index=False)
    tests=[]
    for pair in PAIRS:
        wide=frame[frame.k==pair['k']].pivot(index='query_id',columns='method',values='hit')
        delta=wide[pair['candidate']]-wide[pair['control']]
        wins,losses=int((delta>0).sum()),int((delta<0).sum());n=wins+losses
        p=float(binomtest(wins,n,.5).pvalue) if n else 1.
        exact=min(1.,2*sum(math.comb(n,j) for j in range(min(wins,losses)+1))/2**n) if n else 1.
        assert abs(p-exact)<1e-14
        tests.append(pair|dict(n=100,delta=float(delta.mean()),wins=wins,losses=losses,ties=int((delta==0).sum()),p_raw=p))
    raw=np.array([r['p_raw'] for r in tests]);order=np.argsort(raw,kind='stable');adjusted=np.empty(8)
    adjusted[order]=np.minimum(1,np.maximum.accumulate((8-np.arange(8))*raw[order]))
    for row,p in zip(tests,adjusted):row['p_holm8']=float(p)
    pd.DataFrame(tests).to_csv(OUT/'paired_tests.csv',index=False)
    assert (summary[summary.k==5].set_index('method').loc[['q','old_h','old_qh'],'hit'].to_numpy()==[.31,.50,.49]).all()
    for q in qa:
        value=fresh[q['id']]
        assert value['prompt']==old[q['id']]['prompt'] and value['model_returned']==MODEL
        assert value['response_id']!=old[q['id']]['response_id']
    check()
    dump(OUT/'validation.json',dict(passed=True,fresh_calls=100,recomputed_old_rankings=300,
        metric_rows=len(frame),tests=8,all_response_ids_new=True,source_hashes_unchanged=True,
        identical_texts=sum(s['identical_text'] for s in stability),completed=sum(s['status']=='completed' for s in stability)))
    EXPORT.mkdir(parents=True,exist_ok=True)
    for p in OUT.iterdir():
        if p.is_file() and p.suffix in ['.csv','.json'] and p.name!='fresh_generations.json':shutil.copy2(p,EXPORT/p.name)
    print(summary.to_string(index=False));print(pd.DataFrame(tests).to_string(index=False));print(read(OUT/'validation.json'));print(EXPORT)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['prepare','generate','evaluate']);stage=parser.parse_args().stage
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'.runner.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);globals()[stage]()
