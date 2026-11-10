"""Independent v2 paid-call ledger. Old ledgers are read-only; no retries."""
from pathlib import Path
import argparse, concurrent.futures as cf, fcntl, hashlib, json, os, sqlite3, time
from run import OUT, ROOT, MODEL, base, check, identity, parse, reserve, read
ORG=os.environ.get('OPENAI_ORG_ID')
GUARD=7.
def dump(p,o):
 p=Path(p);t=p.with_suffix(p.suffix+'.tmp')
 with t.open('w') as f:json.dump(o,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 t.replace(p);fd=os.open(str(p.parent),os.O_RDONLY)
 try:os.fsync(fd)
 finally:os.close(fd)
def total(db):return db.execute('SELECT coalesce(sum(coalesce(cost,reserve)),0) FROM calls').fetchone()[0]
def export(db,jobs):
 saved={k:json.loads(p) for k,p in db.execute("SELECT key,payload FROM calls WHERE state='saved'")}
 if saved:dump(OUT/'successful_api_outputs.json',[saved[identity(j)]|j for j in jobs if identity(j) in saved])
 dump(OUT/'cost_summary.json',{'dynamic_prior_base_usd':base(),'new_cost_and_reserves_usd':total(db),'cumulative_estimate_usd':base()+total(db),'guard_usd':GUARD,'states':dict(db.execute('SELECT state,count(*) FROM calls GROUP BY state')),'cached_discount_ignored':True,'not_billing_statement':True,'old_ledgers_read_only':True})
def request(client,j):
 start=time.perf_counter();r=client.responses.create(model=MODEL,input=j['prompt'],max_output_tokens=512,temperature=0,reasoning={'effort':'none'},store=False)
 usage=r.usage.model_dump() if r.usage else {};text=r.output_text.strip();p=j|{'text':text,'model_returned':r.model,'response_id':r.id,'status':r.status,'usage':usage,'latency_seconds':time.perf_counter()-start,'request_concurrency':1,'incomplete_details':r.incomplete_details.model_dump() if r.incomplete_details else None}
 valid=bool(text and usage and r.model==MODEL and r.status=='completed')
 if valid:
  try:parse(text)
  except Exception:valid=False
 cost=(usage.get('input_tokens',0)*2.5+usage.get('output_tokens',0)*15)/1e6 if usage else None
 return p,cost,valid
def run(paid):
 check();jobs=[json.loads(s) for s in (OUT/'judge_calls.jsonl').read_text().splitlines() if s.strip()];unique={identity(j):j for j in jobs};assert len(jobs)==80 and len(unique)==66
 db=sqlite3.connect(OUT/'api_checkpoint.sqlite');db.execute('PRAGMA journal_mode=WAL');db.execute('PRAGMA synchronous=FULL');db.execute('CREATE TABLE IF NOT EXISTS calls(key TEXT PRIMARY KEY,state TEXT,reserve REAL,cost REAL,payload TEXT,created_at REAL)');db.commit()
 pending=[(k,j) for k,j in unique.items() if not db.execute('SELECT 1 FROM calls WHERE key=?',(k,)).fetchone()]
 pre={'logical_jobs':80,'unique_jobs':66,'new_calls':len(pending),'concurrency':1,'dynamic_prior_base_usd':base(),'already_spent_or_reserved_usd':total(db),'new_reserve_usd':sum(reserve(j) for _,j in pending),'cumulative_all_reserves_usd':base()+total(db)+sum(reserve(j) for _,j in pending),'guard_usd':GUARD,'paid_requested':paid};dump(OUT/'execution_preflight.json',pre);export(db,jobs);print(json.dumps(pre,indent=2),flush=True)
 assert pre['cumulative_all_reserves_usd']<=GUARD
 if not paid or not pending:return
 approval=read(OUT/'paid_approval.json');assert approval['approved'] and approval['unique_calls_upper']==66 and approval['max_output_tokens']==512
 for f in ['protocol.json','implementation_hashes.json','judge_calls.jsonl']:assert approval['bound_hashes'][f]==hashlib.sha256((OUT/f).read_bytes()).hexdigest(),f
 assert not db.execute("SELECT 1 FROM calls WHERE state!='saved'").fetchone(),'Unresolved/invalid attempt retained; no automatic retry'
 secret=ROOT/'.env.local';assert secret.exists() and not secret.is_symlink() and secret.stat().st_mode&0o077==0
 values=dict(s.split('=',1) for s in secret.read_text().splitlines() if '=' in s)
 from openai import OpenAI
 client=OpenAI(api_key=values['OPENAI_API_KEY'].strip().strip('\"\''),organization=ORG,max_retries=0,timeout=120)
 access=client.models.retrieve(MODEL);assert access.id==MODEL;dump(OUT/'model_access.json',{'model_id':access.id,'read_only_access':'passed'})
 for k,j in pending:
  db.execute('BEGIN IMMEDIATE')
  if base()+total(db)+reserve(j)>GUARD:db.rollback();raise RuntimeError('Budget guard reached')
  db.execute('INSERT INTO calls VALUES(?,?,?,?,?,?)',(k,'pending',reserve(j),None,None,time.time()));db.commit()
  try:
   payload,cost,valid=request(client,j);payload['key']=k;db.execute('UPDATE calls SET state=?,cost=?,payload=? WHERE key=?',('saved' if valid else 'invalid_response',cost,json.dumps(payload,ensure_ascii=False),k));db.commit();export(db,jobs)
   if not valid:raise RuntimeError('Invalid response retained, no retry')
  except Exception as exc:
   if db.execute('SELECT state FROM calls WHERE key=?',(k,)).fetchone()[0]=='pending':db.execute("UPDATE calls SET state='failed_or_unknown' WHERE key=?",(k,));db.commit();export(db,jobs)
   print('HALTED:',type(exc).__name__,'no retry, reservation retained',flush=True);raise RuntimeError('Unresolved/invalid attempt retained') from None
  print('Saved',db.execute("SELECT count(*) FROM calls WHERE state='saved'").fetchone()[0],'cumulativeUSD',round(base()+total(db),6),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--paid',action='store_true');a=p.parse_args()
 with (OUT/'.runner.lock').open('w') as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);run(a.paid)
