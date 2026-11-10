"""One durable API budget ledger. No automatic retries; successful calls are reused."""
from pathlib import Path
import argparse, concurrent.futures as cf, fcntl, hashlib, json, os, sqlite3, time
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/person3_meeting9_workflow/v1'
MODEL='gpt-5.4-2026-03-05'
ORG=os.environ.get('OPENAI_ORG_ID')
PRIOR=2.4225025
GUARD=7.

def dump(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w') as f:
        json.dump(obj,f,indent=2,ensure_ascii=False);f.flush();os.fsync(f.fileno())
    tmp.replace(path)

def identity(j):
    return hashlib.sha256(json.dumps(dict(model=MODEL,prompt=j['prompt'],cap=160,
        temperature=0,reasoning='none',tools=[],store=False),sort_keys=True).encode()).hexdigest()

def load_jobs(paths):
    jobs=[]
    for p in paths:
        for line in p.read_text().splitlines():
            if not line.strip():continue
            j=json.loads(line)
            assert all(k in j for k in ['dataset','query_id','style','prompt'])
            assert j.get('model',MODEL)==MODEL and j.get('max_output_tokens',160)==160
            jobs.append((p,j,identity(j)))
    return jobs

def total(db):
    return PRIOR+db.execute('SELECT coalesce(sum(coalesce(cost,reserve)),0) FROM calls').fetchone()[0]

def export(db,jobs):
    saved={k:json.loads(p) for k,p in db.execute("SELECT key,payload FROM calls WHERE state='saved'")}
    for path in dict.fromkeys(p for p,_,_ in jobs):
        dump(path.parent/'successful_api_outputs.json',[j|saved[k] for p,j,k in jobs if p==path and k in saved])
    dump(OUT/'cost_summary.json',dict(prior_cumulative_estimate_usd=PRIOR,
        new_cost_and_reserves_usd=total(db)-PRIOR,cumulative_estimate_usd=total(db),guard_usd=GUARD,
        states=dict(db.execute('SELECT state,count(*) FROM calls GROUP BY state')),
        prices_per_million=dict(input=2.5,output=15),cached_discount_ignored=True,
        not_billing_statement=True,latency_scope='wall seconds around Responses create; excludes queue, embeddings, retrieval'))

def request(client,j,concurrency):
    start=time.perf_counter()
    r=client.responses.create(model=MODEL,input=j['prompt'],max_output_tokens=160,
        temperature=0,reasoning={'effort':'none'},store=False)
    elapsed=time.perf_counter()-start;u=r.usage.model_dump() if r.usage else {};s=r.output_text.strip()
    p=dict(text=s,model_returned=r.model,response_id=r.id,status=r.status,usage=u,
        word_count=len(s.split()),cap=160,latency_seconds=elapsed,request_concurrency=concurrency,
        incomplete_details=r.incomplete_details.model_dump() if r.incomplete_details else None)
    cost=(u.get('input_tokens',0)*2.5+u.get('output_tokens',0)*15)/1e6 if u else None
    return p,cost,bool(s and u and r.model==MODEL and r.status in ['completed','incomplete'])

def run(paths,paid=False,concurrency=4):
    jobs=load_jobs(paths);OUT.mkdir(parents=True,exist_ok=True)
    db=sqlite3.connect(OUT/'api_checkpoint.sqlite')
    db.execute('PRAGMA journal_mode=WAL');db.execute('PRAGMA synchronous=FULL')
    db.execute('CREATE TABLE IF NOT EXISTS calls (key TEXT PRIMARY KEY,state TEXT,reserve REAL,cost REAL,payload TEXT,created_at REAL)');db.commit()
    unique=dict((k,j) for _,j,k in jobs)
    pending=[(k,j) for k,j in unique.items() if not db.execute('SELECT 1 FROM calls WHERE key=?',(k,)).fetchone()]
    estimate=sum(((len(j['prompt'].encode())+256)*2.5+160*15)/1e6 for _,j in pending)
    dump(OUT/'preflight.json',dict(model=MODEL,logical_jobs=len(jobs),unique_jobs=len(unique),pending=len(pending),
        maximum_new_reservation_usd=estimate,cumulative_with_reservation_usd=total(db)+estimate,guard_usd=GUARD,
        queues=[str(p.relative_to(ROOT)) for p in paths],pricing_checked='2026-11-09',
        pricing_source='https://developers.openai.com/api/docs/models/gpt-5.4'))
    export(db,jobs);print('Preflight:',len(pending),'new calls; reservation USD',round(estimate,6),flush=True)
    assert total(db)+estimate<=GUARD,'Cumulative budget exceeded'
    if not paid or not pending:return
    assert not db.execute("SELECT 1 FROM calls WHERE state NOT IN ('saved','unknown_reserved_acknowledged')").fetchone(),'Unresolved attempt; no retry'
    secret=ROOT/'.env.local'
    assert secret.exists() and not secret.is_symlink() and secret.stat().st_mode&0o077==0
    values=dict(line.split('=',1) for line in secret.read_text().splitlines() if '=' in line)
    from openai import OpenAI
    client=OpenAI(api_key=values['OPENAI_API_KEY'].strip().strip('\"\''),organization=ORG,max_retries=0,timeout=120)
    client.models.retrieve(MODEL)
    dump(OUT/'model_access.json',dict(model=MODEL,read_only_access='passed',checked_at_unix=time.time()))
    iterator=iter(pending);futures={};stop=False
    def submit(ex):
        try:k,j=next(iterator)
        except StopIteration:return
        reserve=((len(j['prompt'].encode())+256)*2.5+160*15)/1e6
        db.execute('BEGIN IMMEDIATE')
        if total(db)+reserve>GUARD:db.rollback();raise RuntimeError('Budget guard reached')
        db.execute('INSERT INTO calls VALUES (?,?,?,?,?,?)',(k,'pending',reserve,None,None,time.time()));db.commit()
        futures[ex.submit(request,client,j,concurrency)]=k
    with cf.ThreadPoolExecutor(max_workers=concurrency) as ex:
        for _ in range(concurrency):submit(ex)
        while futures:
            done,_=cf.wait(futures,return_when=cf.FIRST_COMPLETED)
            for f in done:
                k=futures.pop(f)
                try:
                    p,c,ok=f.result();p['key']=k
                    db.execute('UPDATE calls SET state=?,cost=?,payload=? WHERE key=?',('saved' if ok else 'invalid_response',c,json.dumps(p),k))
                    if not ok:stop=True
                except Exception as exc:
                    db.execute("UPDATE calls SET state='failed_or_unknown' WHERE key=?",(k,));stop=True
                    print('HALTED:',type(exc).__name__,'no automatic retry',flush=True)
                db.commit()
            export(db,jobs)
            print('Saved',db.execute("SELECT count(*) FROM calls WHERE state='saved'").fetchone()[0],
                'calls; cumulative USD',round(total(db),6),flush=True)
            if not stop:
                for _ in done:submit(ex)
    if stop:raise RuntimeError('Unresolved attempt retained; no automatic retry')

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('queues',type=Path,nargs='+');a.add_argument('--paid',action='store_true');a.add_argument('--concurrency',type=int,choices=[1,2,4],default=4);x=a.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'.runner.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);run([p.resolve() for p in x.queues],x.paid,x.concurrency)
