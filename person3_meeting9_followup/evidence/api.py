"""Independent resumable reader/judge ledger. No retries or secret logging.

Reads original shared ledger read-only; never mutates its code or database.
"""
from pathlib import Path
import argparse, concurrent.futures as cf, fcntl, hashlib, json, os, sqlite3, time
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/person3_meeting9_followup/evidence_v1'
OLD=ROOT/'results/person3_meeting9_workflow/v1/api_checkpoint.sqlite'
MODEL='gpt-5.4-2026-03-05'
ORG=os.environ.get('OPENAI_ORG_ID')
PRIOR=2.4225025
GUARD=7.
def read(p):return json.loads(Path(p).read_text())
def dump(p,o):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
 with tmp.open('w') as f:json.dump(o,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 tmp.replace(p)
 fd=os.open(str(p.parent),os.O_RDONLY)
 try:os.fsync(fd)
 finally:os.close(fd)
def original_base():
 db=sqlite3.connect(f'file:{OLD}?mode=ro',uri=True)
 try:
  states=dict(db.execute('SELECT state,count(*) FROM calls GROUP BY state'))
  assert all(s in ['saved','unknown_reserved_acknowledged'] for s in states),'Original ledger unresolved'
  return PRIOR+db.execute('SELECT coalesce(sum(coalesce(cost,reserve)),0) FROM calls').fetchone()[0]
 finally:db.close()
def identity(j):
 return hashlib.sha256(json.dumps({'model':MODEL,'prompt':j['prompt'],'max_output_tokens':j['max_output_tokens'],'temperature':0,'reasoning':'none','tools':[],'store':False},sort_keys=True).encode()).hexdigest()
def reserve(j):return ((len(j['prompt'].encode())+256)*2.5+j['max_output_tokens']*15)/1e6
def ledger_total(db):return db.execute('SELECT coalesce(sum(coalesce(cost,reserve)),0) FROM calls').fetchone()[0]
def load(stage):
 path=OUT/'reader_calls.jsonl' if stage=='reader' else OUT/'judge/judge_calls.jsonl'
 jobs=[json.loads(s) for s in path.read_text().splitlines() if s.strip()]
 assert len(jobs)==(80 if stage=='reader' else 20)
 assert all(j['model']==MODEL and j['max_output_tokens']==(256 if stage=='reader' else 512) for j in jobs)
 return path,jobs
def export(db):
 for stage,path in [('reader',OUT/'successful_api_outputs.json'),('judge',OUT/'judge/successful_api_outputs.json')]:
  saved={k:json.loads(p) for k,p in db.execute("SELECT key,payload FROM calls WHERE state='saved' AND stage=?",(stage,))}
  queue=OUT/'reader_calls.jsonl' if stage=='reader' else OUT/'judge/judge_calls.jsonl'
  if saved and queue.exists():
   _,jobs=load(stage);rows=[saved[identity(j)]|j for j in jobs if identity(j) in saved];dump(path,rows)
 base=original_base()
 dump(OUT/'cost_summary.json',{'original_dynamic_base_usd':base,'local_cost_and_reserves_usd':ledger_total(db),'cumulative_estimate_usd':base+ledger_total(db),'guard_usd':GUARD,'states':dict(db.execute('SELECT state,count(*) FROM calls GROUP BY state')),'prices_per_million':{'input':2.5,'output':15},'cached_discount_ignored':True,'not_billing_statement':True,'original_ledger_read_only':True})
def request(client,j,concurrency):
 start=time.perf_counter();r=client.responses.create(model=MODEL,input=j['prompt'],max_output_tokens=j['max_output_tokens'],temperature=0,reasoning={'effort':'none'},store=False)
 elapsed=time.perf_counter()-start;usage=r.usage.model_dump() if r.usage else {};text=r.output_text.strip()
 payload=j|{'text':text,'model_returned':r.model,'response_id':r.id,'status':r.status,'usage':usage,'cap':j['max_output_tokens'],'latency_seconds':elapsed,'request_concurrency':concurrency,'incomplete_details':r.incomplete_details.model_dump() if r.incomplete_details else None}
 cost=(usage.get('input_tokens',0)*2.5+usage.get('output_tokens',0)*15)/1e6 if usage else None
 return payload,cost,bool(text and usage and r.model==MODEL and r.status=='completed')
def run(stage,paid):
 path,jobs=load(stage);db=sqlite3.connect(OUT/'api_checkpoint.sqlite');db.execute('PRAGMA journal_mode=WAL');db.execute('PRAGMA synchronous=FULL')
 db.execute('CREATE TABLE IF NOT EXISTS calls(key TEXT PRIMARY KEY,stage TEXT,state TEXT,reserve REAL,cost REAL,payload TEXT,created_at REAL)');db.commit()
 unique={identity(j):j for j in jobs};pending=[(k,j) for k,j in unique.items() if not db.execute('SELECT 1 FROM calls WHERE key=?',(k,)).fetchone()]
 base=original_base();stage_reserve=sum(reserve(j) for _,j in pending)
 future_judge_reserve=read(OUT/'preflight.json')['judge_reserve_upper_usd'] if stage=='reader' else 0.
 preflight={'stage':stage,'logical_jobs':len(jobs),'unique_jobs':len(unique),'new_calls':len(pending),'concurrency':2 if stage=='reader' else 1,'original_dynamic_base_usd':base,'local_already_spent_or_reserved_usd':ledger_total(db),'new_stage_reserve_usd':stage_reserve,'future_judge_reserve_upper_usd':future_judge_reserve,'cumulative_with_all_remaining_reserves_usd':base+ledger_total(db)+stage_reserve+future_judge_reserve,'guard_usd':GUARD,'queue_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'paid_requested':paid}
 dump(OUT/(stage+'_execution_preflight.json'),preflight);export(db);print(json.dumps(preflight,indent=2),flush=True)
 assert preflight['cumulative_with_all_remaining_reserves_usd']<=GUARD,'All-stage budget guard exceeded'
 if not paid or not pending:return
 receipt=read(OUT/'freeze_receipt.json')
 for f,h in receipt['artifact_hashes'].items():assert hashlib.sha256((OUT/f).read_bytes()).hexdigest()==h,f'Frozen artifact changed: {f}'
 for f,h in read(OUT/'implementation_hashes.json').items():assert hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h,f'Implementation changed: {f}'
 for f,h in read(OUT/'source_hashes.json').items():assert hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h,f'Source changed: {f}'
 approval=read(OUT/'paid_approval.json');assert approval['approved'] and approval['reader_calls_upper']==80 and approval['judge_calls_upper']==20 and approval['reader_cap']==256 and approval['judge_cap']==512
 assert approval['protocol_sha256']==hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest(),'Approval bound to another protocol'
 assert approval['implementation_hashes_sha256']==hashlib.sha256((OUT/'implementation_hashes.json').read_bytes()).hexdigest(),'Approval bound to another implementation'
 assert approval['reader_queue_sha256']==hashlib.sha256((OUT/'reader_calls.jsonl').read_bytes()).hexdigest(),'Approval bound to another reader queue'
 assert not db.execute("SELECT 1 FROM calls WHERE state!='saved'").fetchone(),'Unresolved/invalid attempt retained; no retry'
 secret=ROOT/'.env.local';assert secret.exists() and not secret.is_symlink() and secret.stat().st_mode&0o077==0
 values=dict(line.split('=',1) for line in secret.read_text().splitlines() if '=' in line)
 from openai import OpenAI
 client=OpenAI(api_key=values['OPENAI_API_KEY'].strip().strip('\"\''),organization=ORG,max_retries=0,timeout=120)
 access=client.models.retrieve(MODEL);assert access.id==MODEL
 dump(OUT/'model_access.json',{'model_id':access.id,'read_only_access':'passed'})
 concurrency=2 if stage=='reader' else 1;iterator=iter(pending);futures={};stop=False
 def submit(ex):
  try:k,j=next(iterator)
  except StopIteration:return
  db.execute('BEGIN IMMEDIATE')
  if original_base()+ledger_total(db)+reserve(j)+future_judge_reserve>GUARD:db.rollback();raise RuntimeError('Budget guard reached')
  db.execute('INSERT INTO calls VALUES(?,?,?,?,?,?,?)',(k,stage,'pending',reserve(j),None,None,time.time()));db.commit()
  futures[ex.submit(request,client,j,concurrency)]=(k,j)
 with cf.ThreadPoolExecutor(max_workers=concurrency) as ex:
  for _ in range(concurrency):submit(ex)
  while futures:
   done,_=cf.wait(futures,return_when=cf.FIRST_COMPLETED)
   for future in done:
    k,j=futures.pop(future)
    try:
     payload,cost,ok=future.result();payload['key']=k
     db.execute('UPDATE calls SET state=?,cost=?,payload=? WHERE key=?',('saved' if ok else 'invalid_response',cost,json.dumps(payload,ensure_ascii=False),k))
     if not ok:stop=True
    except Exception as exc:
     db.execute("UPDATE calls SET state='failed_or_unknown' WHERE key=?",(k,));stop=True;print('HALTED:',type(exc).__name__,'no automatic retry',flush=True)
    db.commit();export(db)
   print('Saved',db.execute("SELECT count(*) FROM calls WHERE state='saved'").fetchone()[0],'cumulativeUSD',round(original_base()+ledger_total(db),6),flush=True)
   if not stop:
    for _ in done:submit(ex)
 if stop:raise RuntimeError('Unresolved/invalid attempt retained; no automatic retry')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['reader','judge']);p.add_argument('--paid',action='store_true');a=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
 with (OUT/'.runner.lock').open('w') as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);run(a.stage,a.paid)
