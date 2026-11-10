"""Post-review candidate-isolated repair. Offline preparation/reporting only."""
from pathlib import Path
import argparse, csv, hashlib, json, os, re, shutil, sqlite3
from datetime import datetime, timezone
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/person3_meeting9_followup/evidence_isolated_v2'
OLD=ROOT/'results/person3_meeting9_followup/evidence_v1'
EXPORT=ROOT.parent/'outputs/meeting9_followup_20261109/evidence_isolated'
MODEL='gpt-5.4-2026-03-05'
METHODS=['bm25','question_bank_rrf_w050','focusedmix_bank_rrf_w050','no_context']
LABELS=['BM25','Question hybrid .50','Focused hybrid .50','No-context control']
METRICS=['ai_reference_consistency','ai_coverage','ai_grounding','ai_citation','ai_complete_reference_consistent','ai_supported_reference_consistent']
RULES='''Evaluate ONE financial QA candidate. The question and released reference are comparison data, not grounding evidence. Only OWN REPORT EXCERPTS can support claims. You see no other candidate. Do not assume unseen pages or use outside knowledge. C reference consistency:0 wrong/no substantive requested answer,1 partly consistent with material error,2 fully consistent. V requested-fact coverage:0 no requested answer,1 partial,2 all essential facts. G own-context grounding:0 unsupported/contradicted material claims,1 partially supported,2 all claims supported. T citations:0 absent/invalid IDs/materially unsupported,1 partially supported with gaps,2 citations support all material claims. An abstention giving no substantive requested answer MUST be answer_status=abstain and C0,V0,T0. If it makes NO factual assertions, G2 is vacuous; if it asserts explanatory facts, score their G using only own excerpts. Reference may affect C/V, NEVER G/T. Partial substantive answers use partial, not abstain. Every missing citation or any cited ID absent from OWN EXCERPTS MUST have T0. Never treat reference agreement as source support. Correctly reporting insufficient evidence does not require evidence from some other candidate. Flag clear reference issues, do not force agreement with a faulty reference. Return ONLY JSON {"C":0,"V":0,"G":2,"T":0,"answer_status":"abstain","reference_issue":false,"sourcequotes":[{"source_id":"P123","quote":"exact contiguous quote"}],"reason":"short explanation"}. C,V,G,T integer0/1/2. answer_status answer/partial/abstain. Sourcequotes at most2, each at most12 words copied EXACTLY contiguously from own excerpts, no ellipses or added quotation marks; empty list for abstention/no support. Reason at most30 words. Max512 tokens.'''
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,o):Path(p).write_text(json.dumps(o,ensure_ascii=False,indent=2)+'\n')
def frozen(p,o):
 if Path(p).exists():assert read(p)==o,f'Frozen changed: {p}'
 else:dump(p,o)
def prompt(r,reference,answer):return RULES+'\nQUESTION:\n'+r['question']+'\nRELEASED REFERENCE (comparison only):\n'+reference+'\nCANDIDATE ANSWER:\n'+answer+'\nOWN REPORT EXCERPTS:\n'+r['context']
def identity(j):return hashlib.sha256(json.dumps({'model':MODEL,'prompt':j['prompt'],'max_output_tokens':512,'temperature':0,'reasoning':'none','store':False},sort_keys=True).encode()).hexdigest()
def reserve(j):return ((len(j['prompt'].encode())+256)*2.5+512*15)/1e6
def base():
 total=2.4225025
 for p in [ROOT/'results/person3_meeting9_workflow/v1/api_checkpoint.sqlite',OLD/'api_checkpoint.sqlite']:
  db=sqlite3.connect(f'file:{p}?mode=ro',uri=True)
  try:
   assert all(s in ['saved','unknown_reserved_acknowledged'] for s, in db.execute('SELECT DISTINCT state FROM calls'))
   total+=db.execute('SELECT coalesce(sum(coalesce(cost,reserve)),0) FROM calls').fetchone()[0]
  finally:db.close()
 return total
def export():
 EXPORT.mkdir(parents=True,exist_ok=True)
 for p in OUT.iterdir():
  if p.is_file() and not p.name.startswith('.') and '.sqlite' not in p.name:shutil.copy2(p,EXPORT/p.name)
def check():
 receipt=read(OUT/'freeze_receipt.json')
 for n,h in receipt['artifact_hashes'].items():assert sha(OUT/n)==h,n
 for n in ['source_hashes.json','implementation_hashes.json']:
  for p,h in read(OUT/n).items():assert sha(ROOT/p)==h,p
def prepare():
 OUT.mkdir(parents=True,exist_ok=True)
 records=read(OLD/'context_records.json');outputs={(r['query_id'],r['style']):r for r in read(OLD/'successful_api_outputs.json')};refs={r['query_id']:r for r in read(OLD/'reference_answers_judge_only.json')}
 assert len(records)==len(outputs)==80
 jobs=[];inputs=[]
 for r in records:
  answer=outputs[(r['query_id'],r['method'])]['text'];j={'dataset':'vidore_reader20_isolated_judge_v2','query_id':r['query_id'],'style':r['method'],'prompt':prompt(r,refs[r['query_id']]['reference_answer'],answer),'model':MODEL,'max_output_tokens':512};jobs.append(j)
  inputs.append({k:r[k] for k in ['query_id','method','question','context','delivered_spans','used_wordpiece_tokens']}|{'answer':answer,'reference_answer':refs[r['query_id']]['reference_answer'],'original_qrels':refs[r['query_id']]['original_qrels'],'judge_key':identity(j),'original_reader_key':outputs[(r['query_id'],r['method'])]['key']})
 protocol={'status':'Post-review repair of rejected joint-rater v1; AI-only exploratory, not independent confirmatory validation','selection':'Same original SHA256-selected20; all80 logical answers across all four conditions rejudged; no reader calls, no performance-dependent selection','model':MODEL,'temperature':0,'reasoning':'none','max_output_tokens':512,'candidate_isolation':'Each request contains exactly one original question, released reference, saved answer and its own actual delivered context. No method label, other candidate, other context or v1 score in API input. Reference is comparison data only, never grounding evidence.','reader_reuse':'All original reader outputs unchanged; three retrieval contexts1536 WordPiece, no-context control no excerpts','deduplication':'Exact full judge-prompt identity; every duplicate mapped to identical saved result','rubric':'C reference-consistency, V requested-fact coverage, G own-context grounding, T citations; each integer0/1/2; semantic model judgment remains AI-only','abstention_rule':'No substantive requested answer means C0/V0/T0. Pure abstention with no positive factual assertions has vacuous G2. If explanatory facts are asserted, G is judged against own context; no mechanical G2 upgrade.','guard':'Preserve raw JSON. Only frozen20 exact whole-answer SHA entries (all pure no-context abstentions) force0/0/2/0. Other model-declared abstention forces C0/V0/T0 while preserving own-fact G; this classification is AI-dependent, not mechanical semantic identification. Missing/any invalid cited ID forces T0. Raw-to-guard changes/reasons are separate.','quote_verification':'Exact contiguous sourcequote and source ID mechanically verified against own original delivered page prefix. Raw/guard scores retained; extra strict-supported requires all provided quotes valid and at least one valid quote.','primary_fields':METRICS,'failure_policy':'Unknown/failed attempt retains reservation and halts; no auto retry. Invalid judge JSON retained and reported; no hidden deletion or substitution.','inference':'All20 per method, descriptive paired contrasts only; no new p-values, no human accuracy claim','human_reviewed':0,'human_review_status':'pending','stop_rule':'Exactly these80 logical isolated judges,66 unique full prompts, no new reader/generation/embedding method or question tuning','rejected_v1':'Retained read-only as rater-calibration failure evidence; never mixed into v2 primary scores'}
 pure=[{'query_id':i['query_id'],'method':i['method'],'answer_sha256':hashlib.sha256(i['answer'].encode()).hexdigest(),'judge_key':i['judge_key']} for i in inputs if i['method']=='no_context']
 assert len(pure)==20 and all(not i['delivered_spans'] and i['answer'].startswith('Insufficient evidence') for i in inputs if i['method']=='no_context')
 frozen(OUT/'pure_abstention_cases.json',{'cases':pure,'determination':'AI inspection of entire saved answers before v2 calls: each says no excerpts supplied, restates question targets/entities/periods, and gives no positive substantive answer or external factual assertion. Exact answer SHA matching, not keyword semantics. This is a frozen AI classification, not human validation; q178 explanatory-fact retrieval answers are excluded.'})
 frozen(OUT/'protocol.json',protocol);frozen(OUT/'inputs.json',inputs)
 queue=''.join(json.dumps(j,ensure_ascii=False)+'\n' for j in jobs);p=OUT/'judge_calls.jsonl'
 if p.exists():assert p.read_text()==queue
 else:p.write_text(queue)
 unique={identity(j):j for j in jobs};assert len(unique)==66
 frozen(OUT/'preflight.json',{'logical_jobs':80,'unique_calls':66,'unique_utf8_input_bytes':sum(len(j['prompt'].encode()) for j in unique.values()),'reserve_usd':sum(reserve(j) for j in unique.values()),'dynamic_prior_base_usd':base(),'cumulative_worst_usd':base()+sum(reserve(j) for j in unique.values()),'guard_usd':7,'prices_per_million':{'input':2.5,'output':15},'cached_discount_ignored':True,'not_billing_statement':True,'paid_calls':0})
 sources=[OLD/n for n in ['context_records.json','successful_api_outputs.json','reference_answers_judge_only.json','selection.json','protocol.json','freeze_receipt.json']]+[ROOT/'results/person3_meeting9_transfer/vidore_finance_v1/data/corpus.jsonl']
 frozen(OUT/'source_hashes.json',{str(p.relative_to(ROOT)):sha(p) for p in sources})
 frozen(OUT/'implementation_hashes.json',{str(p.relative_to(ROOT)):sha(p) for p in Path(__file__).parent.glob('*.py')})
 if not (OUT/'freeze_receipt.json').exists():dump(OUT/'freeze_receipt.json',{'utc':datetime.now(timezone.utc).isoformat(),'no_v2_outputs_existed':not (OUT/'successful_api_outputs.json').exists(),'artifact_hashes':{n:sha(OUT/n) for n in ['protocol.json','inputs.json','pure_abstention_cases.json','judge_calls.jsonl','preflight.json','source_hashes.json','implementation_hashes.json']}})
 export();print(json.dumps(read(OUT/'preflight.json'),indent=2))
def parse(text):
 r=json.loads(text);assert set(r)=={'C','V','G','T','answer_status','reference_issue','sourcequotes','reason'}
 assert all(type(r[k])==int and r[k] in [0,1,2] for k in 'CVGT')
 assert r['answer_status'] in ['answer','partial','abstain'] and type(r['reference_issue'])==bool and isinstance(r['reason'],str)
 assert isinstance(r['sourcequotes'],list) and len(r['sourcequotes'])<=2 and all(set(q)=={'source_id','quote'} and all(isinstance(v,str) for v in q.values()) for q in r['sourcequotes'])
 return r
def report():
 check();inputs=read(OUT/'inputs.json');outs=read(OUT/'successful_api_outputs.json');lookup={r['key']:r for r in outs};assert len(lookup)==66
 pure={(r['query_id'],r['method'],r['answer_sha256']) for r in read(OUT/'pure_abstention_cases.json')['cases']}
 corpus={str(r['corpus_id']):r for r in map(json.loads,(ROOT/'results/person3_meeting9_transfer/vidore_finance_v1/data/corpus.jsonl').read_text().splitlines())};rows=[];invalid=[];packets=[]
 for item in inputs:
  o=lookup[item['judge_key']]
  try:r=parse(o['text']);assert o['status']=='completed'
  except Exception:invalid.append({'query_id':item['query_id'],'method':item['method'],'key':item['judge_key']});continue
  ids={s['passage_id'] for s in item['delivered_spans']};cites=re.findall(r'\[P(\d+)\]',item['answer']);idvalid=bool(cites) and all(pid in ids for pid in cites)
  forced_empty_abstain=(item['query_id'],item['method'],hashlib.sha256(item['answer'].encode()).hexdigest()) in pure;raw=[r[k] for k in 'CVGT'];guard=raw.copy();reasons=[]
  if forced_empty_abstain:
   guard=[0,0,2,0]
   if guard!=raw:reasons.append('frozen_exact_pure_abstention_0_0_2_0')
  elif r['answer_status']=='abstain':
   guard=[0,0,raw[2],0]
   if guard!=raw:reasons.append('AI_classified_abstention_C0_V0_T0_preserve_G')
  if not idvalid:
   if guard[3]!=0:reasons.append('missing_or_invalid_citation_T0')
   guard[3]=0
  verified=[]
  for q in r['sourcequotes']:
   pid=q['source_id'].removeprefix('P');verified.append(bool(q['quote']) and any(s['passage_id']==pid and q['quote'] in corpus[pid]['markdown'][s['char_start']:s['char_end']] for s in item['delivered_spans']))
  quotevalid=bool(verified) and all(verified)
  row={'query_id':item['query_id'],'method':item['method'],'judge_key':item['judge_key'],'answer_status':r['answer_status'],'frozen_exact_pure_abstention':forced_empty_abstain,'guard_abstention_classification_source':'frozen_AI_whole_answer_SHA_list' if forced_empty_abstain else ('AI_judge_answer_status' if r['answer_status']=='abstain' else 'none'),'reference_issue_flag':r['reference_issue'],'reader_has_citation':bool(cites),'reader_citation_ids_valid':idvalid,'ai_source_quote_mechanically_verified':quotevalid,'sourcequotes_count':len(verified),'sourcequotes_verified_count':sum(verified),'rule_guard_applied':guard!=raw,'rule_guard_reasons':'|'.join(reasons),'ai_reason':r['reason'],'ai_sourcequotes':json.dumps(r['sourcequotes'],ensure_ascii=False),'human_status':'pending'}
  for prefix,values in [('raw_',raw),('rule_guard_',guard),('',guard)]:
   for k,v in zip(METRICS[:4],values):row[prefix+k]=v/2
   row[prefix+METRICS[4]]=int(values[0]==values[1]==2);row[prefix+METRICS[5]]=int(values==[2,2,2,2])
  row['ai_strict_supported_reference_consistent']=int(guard==[2,2,2,2] and quotevalid and idvalid)
  rows.append(row);packets.append({k:item[k] for k in ['query_id','question','reference_answer','answer','context','delivered_spans']}|{'review_id':hashlib.sha256((item['query_id']+'|'+item['method']+'|'+item['judge_key']).encode()).hexdigest()[:12],'human_status':'pending'})
 frame=pd.DataFrame(rows);frame.to_csv(OUT/'per_query_metrics.csv',index=False)
 summary=frame.groupby('method').agg(n=('query_id','size'),**{m:(m,'mean') for m in METRICS}).reindex(METHODS).reset_index();summary.to_csv(OUT/'summary.csv',index=False)
 rawcols=['raw_'+m for m in METRICS];frame.groupby('method').agg(n=('query_id','size'),**{m:(m,'mean') for m in rawcols}).reindex(METHODS).reset_index().to_csv(OUT/'raw_summary.csv',index=False)
 diagnostic=['reader_has_citation','reader_citation_ids_valid','ai_source_quote_mechanically_verified','ai_strict_supported_reference_consistent','rule_guard_applied'];frame.groupby('method').agg(n=('query_id','size'),**{m:(m,'mean') for m in diagnostic}).reindex(METHODS).reset_index().to_csv(OUT/'mechanical_diagnostics.csv',index=False)
 frame[frame.rule_guard_applied].to_csv(OUT/'raw_rubric_violations.csv',index=False);frame[(frame.sourcequotes_count>0)&~frame.ai_source_quote_mechanically_verified].to_csv(OUT/'source_quote_verification_failures.csv',index=False)
 pairs=[]
 for a,b in [(METHODS[1],METHODS[0]),(METHODS[2],METHODS[0]),(METHODS[2],METHODS[1])]+[(m,'no_context') for m in METHODS[:3]]:
  d=frame[frame.method==a].set_index('query_id')[METRICS]-frame[frame.method==b].set_index('query_id')[METRICS]
  for m in METRICS:pairs.append({'method':a,'control':b,'metric':m,'n':len(d),'mean_delta':d[m].mean(),'wins':int((d[m]>0).sum()),'losses':int((d[m]<0).sum()),'ties':int((d[m]==0).sum())})
 pd.DataFrame(pairs).to_csv(OUT/'paired_comparisons.csv',index=False)
 dump(OUT/'blinded_human_review_packet.json',packets)
 if not (OUT/'human_review_template.csv').exists():
  with (OUT/'human_review_template.csv').open('w',newline='') as f:
   fields=['query_id','review_id','human_reference_consistency','human_coverage','human_grounding','human_citation','reference_issue','notes','reviewer_id','reviewed_at','status'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
   for p in packets:w.writerow({'query_id':p['query_id'],'review_id':p['review_id'],'status':'pending'})
 unique=list(lookup.values());usage={'new_reader_calls':0,'judge_unique_calls':len(unique),'judge_logical_jobs':80,'judge_unique_calls_verified':66,'input_tokens':sum(r['usage']['input_tokens'] for r in unique),'output_tokens':sum(r['usage']['output_tokens'] for r in unique),'new_cost_estimate_usd':sum((r['usage']['input_tokens']*2.5+r['usage']['output_tokens']*15)/1e6 for r in unique),'dynamic_prior_base_usd':base(),'cached_discount_ignored':True,'billing_statement':False};usage['cumulative_estimate_usd']=base()+usage['new_cost_estimate_usd'];dump(OUT/'usage.json',usage)
 q178=frame[frame.query_id=='178'];q178.to_csv(OUT/'q178_repair_diagnostic.csv',index=False)
 no=frame[frame.method=='no_context'];passed=not invalid and len(frame)==80 and len(no)==20 and all((no[METRICS[:4]].to_numpy()==[0,0,1,0]).all(axis=1)) and all(q178[q178.method.isin(METHODS[1:3])].answer_status=='abstain')
 dump(OUT/'completion_status.json',{'engineering_complete':len(frame)==80 and not invalid,'critical_high_repair_checks_passed':bool(passed),'invalid_rows':invalid,'human_reviewed':0,'human_review_status':'pending','human_validated_accuracy_complete':False,'scoring_status':'AI-only post-review repair; requires independent audit; never human validation'})
 table='| Method | n | AI reference consistency | AI coverage | AI grounding | AI citation | AI complete reference consistent |\n|---|---:|---:|---:|---:|---:|---:|\n'+'\n'.join(f'| {LABELS[METHODS.index(r.method)]} | {r.n} | {r.ai_reference_consistency:.3f} | {r.ai_coverage:.3f} | {r.ai_grounding:.3f} | {r.ai_citation:.3f} | {r.ai_complete_reference_consistent:.1%} |' for r in summary.itertuples())
 text='# Isolated-candidate AI evaluation: post-review repair\n\n**AI-only; human-reviewed0/pending. Post-review repair, not independent confirmatory validation.** v1 joint-rater scores are retained as rejected calibration evidence and are not mixed into this primary table.\n\n'+table+'\n\nPrimary columns contain rule_guard scores. raw_summary.csv and raw_* per-query fields retain every original model score. Missing/any invalid page citation forces T0; the frozen20 exact pure-abstention answers force C0/V0/G2/T0; other AI-classified abstention forces C0/V0/T0 while preserving own-fact G. The latter classification depends on AI, not mechanical semantic identification. Grounding2 for abstention is vacuous claim support, not answer quality. Strict supported consistency additionally needs valid own-page quotes and valid citation IDs.\n\nEvery request contains ONE saved answer with only its question, released reference and own1536-WordPiece OCR context (or no excerpts for control). No other candidate, method label, context or v1 score is supplied. All80 original logical answers are rejudged; full-prompt duplicates share one response. No reader is rerun. References measure agreement and cannot supply grounding evidence. Same-model and faulty-reference risks remain; no human accuracy claim, p-value or heldout confirmation is supported.\n\nThe q178 regression diagnostic must show both original unsupported-context answers treated as abstentions rather than accused using BM25-only18%/9% evidence. No-context answers must all score0/0/2/0 after deterministic guards. Raw rubric violations and quote failures remain first-class diagnostics. Mechanical ID/quote validity does not establish complete semantic evidence sufficiency.\n\nThe blind human packet and CSV remain pending. Original Legal20 fact/quote labels also remain AI drafts, human0/pending.\n\nReproduce locally after approved calls:\n```bash\n.venv/bin/python person3_meeting9_followup/evidence_isolated/run.py prepare\n.venv/bin/python person3_meeting9_followup/evidence_isolated/run.py report\n.venv/bin/python person3_meeting9_followup/evidence_isolated/run.py verify\n```\n\n'+json.dumps(usage,indent=2)+'\n'
 (OUT/'README.md').write_text(text);export();print(table);print(json.dumps(read(OUT/'completion_status.json'),indent=2))
def verify():
 check();jobs=[json.loads(s) for s in (OUT/'judge_calls.jsonl').read_text().splitlines()];inputs=read(OUT/'inputs.json');assert len(jobs)==len(inputs)==80 and len(set(identity(j) for j in jobs))==66
 for j,r in zip(jobs,inputs):assert j['prompt']==prompt(r,r['reference_answer'],r['answer']) and identity(j)==r['judge_key']
 result={'passed':True,'single_candidate_prompt_reconstructions':80,'unique_full_prompts':66,'all_original_reader_outputs_reused_unchanged':True,'new_reader_calls':0,'human_reviewed':0}
 if (OUT/'per_query_metrics.csv').exists():
  frame=pd.read_csv(OUT/'per_query_metrics.csv',dtype={'query_id':str});assert len(frame)==80
  for row in read(OUT/'successful_api_outputs.json'):parse(row['text'])
  for qid,g in frame.groupby('judge_key'):
   assert all(g[m].nunique()==1 for m in METRICS)
  result['logical_scores_and_duplicate_identity_verified']=80
 dump(OUT/'validation.json',result);export();print(json.dumps(result,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','report','verify']);a=p.parse_args();{'prepare':prepare,'report':report,'verify':verify}[a.stage]()
