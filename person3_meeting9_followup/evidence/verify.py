"""Independent local audit of fixed sample, source prefixes, blinding and budget."""
from pathlib import Path
import hashlib, json, re, sqlite3
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/person3_meeting9_followup/evidence_v1'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def verify():
 for f in ['source_hashes.json','implementation_hashes.json']:
  if (OUT/f).exists():
   for path,h in read(OUT/f).items():assert sha(ROOT/path)==h,(f,path)
 receipt=read(OUT/'freeze_receipt.json') if (OUT/'freeze_receipt.json').exists() else None
 if receipt:
  for f,h in receipt['artifact_hashes'].items():assert sha(OUT/f)==h,f
 sel=read(OUT/'selection.json')['queries'];allq=read(ROOT/'results/person3_meeting9_transfer/vidore_finance_full309_summary_v1/selection.json')['queries']
 assert sel==sorted(allq,key=lambda q:hashlib.sha256(('meeting9-reader-v1|'+q['query_id']).encode()).hexdigest())[:20]
 z=np.load(ROOT/'results/person3_meeting9_constrained_hybrid/v1/rankings.npz');qids=z['query_ids'].tolist();pids=z['passage_ids'].tolist()
 corpus={str(c['corpus_id']):c for c in map(json.loads,(ROOT/'results/person3_meeting9_transfer/vidore_finance_v1/data/corpus.jsonl').read_text().splitlines())}
 records=read(OUT/'context_records.json');assert len(records)==80
 jobs=[json.loads(s) for s in (OUT/'reader_calls.jsonl').read_text().splitlines()]
 assert len(jobs)==80 and len(set(j['prompt'] for j in jobs))==66
 index={(j['query_id'],j['style']):j for j in jobs};assert len(index)==80
 for r in records:
  assert hashlib.sha256(r['context'].encode()).hexdigest()==r['context_sha256']
  pieces=[];tokens=0
  for i,s in enumerate(r['delivered_spans']):
   assert s['passage_id']==pids[int(z[r['method']][qids.index(r['query_id']),i])]
   p=corpus[s['passage_id']];assert s['char_start']==0 and s['source_sha256']==hashlib.sha256(p['markdown'].encode()).hexdigest()
   pieces.append(f"[P{s['passage_id']}] {p['doc_id']} page_index={p['page_number_in_doc']}\n"+p['markdown'][:s['char_end']]);tokens+=s['tokens']
  assert r['context']==('\n\n'.join(pieces) if pieces else '(No report excerpts supplied.)')
  assert tokens==r['used_wordpiece_tokens']==(0 if r['method']=='no_context' else 1536)
  j=index[(r['query_id'],r['method'])];assert j['prompt'].endswith(r['context']+'\n\nANSWER:') and '\nQUESTION:\n'+r['question']+'\n' in j['prompt']
 for q in sel:
  sub=[r for r in records if r['query_id']==q['query_id']];order=sorted([r['method'] for r in sub],key=lambda m:hashlib.sha256(('reader-blind-v1|'+q['query_id']+'|'+m).encode()).hexdigest())
  assert all(r['blind_label']=='ABCD'[order.index(r['method'])] for r in sub)
 summary=read(OUT/'reader_execution_preflight.json');assert summary['cumulative_with_all_remaining_reserves_usd']<=7 and summary['new_calls']<=66
 human=list(__import__('csv').DictReader((OUT/'human_review_template.csv').open()));assert len(human)==80
 # Human scores may legitimately change in a later real review; this audit never writes them.
 reviewed=sum(bool(r['reviewer_id'] and r['reviewed_at']) for r in human)
 if (OUT/'successful_api_outputs.json').exists():
  outputs=read(OUT/'successful_api_outputs.json');assert len(outputs)==80
  keys={}
  for o in outputs:
   key=(o['query_id'],o['style']);assert key in index and o['prompt']==index[key]['prompt'] and o['model_returned']=='gpt-5.4-2026-03-05'
   if o['key'] in keys:assert o['text']==keys[o['key']]
   keys[o['key']]=o['text']
  assert len(keys)==66
 if (OUT/'judge/alias_mapping.json').exists():
  aliases=read(OUT/'judge/alias_mapping.json');assert len(aliases)==80
  lookup={(r['query_id'],r['blind_label']):(next(o['text'] for o in outputs if o['query_id']==r['query_id'] and o['style']==r['method']),r['context']) for r in records}
  for a in aliases:assert lookup[(a['query_id'],a['blind_label'])]==lookup[(a['query_id'],a['judged_label'])]
 result={'passed':True,'fixed_hash_selected_questions':20,'reconstructed_actual_contexts':80,'retrieval_context_budget_wordpiece':1536,'no_context_controls':20,'unique_reader_prompts':66,'human_reviewed':reviewed,'reference_marker_screen_only':True,'paid_calls_triggered_by_verify':0}
 (OUT/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':verify()
