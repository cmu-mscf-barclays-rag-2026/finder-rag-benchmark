"""Offline, source-grounded evidence-quote availability prototype.
No API/reader calls. AI-assisted fact drafts require independent human review.
"""
from pathlib import Path
import os,json,hashlib,shutil,bisect
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',DO_NOT_TRACK='1')
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from annotation_draft import DRAFT
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/person3_meeting9_evidence/legal20_v1'
EXPORT=ROOT.parent/'outputs/meeting9_20261109/evidence'
RANKS=ROOT/'results/person3_prompt_topk/prompt_styles_v1/pilot/legal_rankings.npz'
SEL=ROOT/'results/person3_prompt_topk/prompt_styles_v1/selection.json'
CORPUS=ROOT/'data/legal_rag_bench/corpus.jsonl'
METHODS=['question_dense','focused160_mix_dense','short_30_50_mix_dense']
LABELS=dict(zip(METHODS,['Question','Focused mix','Short mix']))
KS=[5,10,15,20,25];BUDGETS=[256,512,1024,2048]
REV='1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,o):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(o,ensure_ascii=False,indent=2)+'\n');t.replace(p)
def frozen(p,o):
 if p.exists():assert json.loads(p.read_text())==o, 'Frozen artifact changed: '+str(p)
 else:dump(p,o)
def build_annotations(selection,corpus):
 annotations=[]
 for q in selection:
  d=DRAFT[q['query_id']];facts=[]
  for fid,requirement,quotes in d['facts']:
   evidence=[]
   for pid,quote in quotes:
    pid=pid or q['relevant_passage_id'];assert quote in corpus[pid]['text'],(q['query_id'],pid,quote)
    # Record identical source quotations in other corpus passages as alternatives.
    # No semantic automatic matching and no query-rank/score dependent selection.
    for source in corpus.values():
     start=0
     while (offset:=source['text'].find(quote,start))!=-1:
      evidence.append({'passage_id':source['id'],'char_start':offset,'char_end':offset+len(quote),'quote':quote,'discovery':'specified primary source' if source['id']==pid else 'identical quotation elsewhere in corpus'})
      start=offset+1
   facts.append({'fact_id':fid,'requirement_draft':requirement,'alternative_support_spans':evidence})
  branches=d.get('branches',[[f['fact_id'] for f in facts]])
  annotations.append({'query_id':q['query_id'],'question':q['question'],'reference_answer':q['answer'],'published_qrel':q['relevant_passage_id'],'draft_type':'alternative sufficient routes' if len(branches)>1 else ('complementary facts' if len(facts)>1 else 'single fact'),'sufficient_branches_draft':branches,'facts':facts,'eligible_for_aggregate':d.get('eligible',True),'annotation_status':'AI-assisted draft; independent human review pending','note':d.get('note','Required rule(s) drafted from question, released reference answer and actual corpus source. Scenario facts come from the question, not from retrieved evidence.')})
 return annotations
def logic(covered,branches):
 values=[sum(fid in covered for fid in b)/len(b) for b in branches]
 return max(values),int(any(all(fid in covered for fid in b) for b in branches))
def main():
 OUT.mkdir(parents=True,exist_ok=True);EXPORT.mkdir(parents=True,exist_ok=True)
 sources=[RANKS,SEL,CORPUS,Path(__file__).with_name('annotation_draft.py')]
 hashes={str(p.relative_to(ROOT)):sha(p) for p in sources}
 frozen(OUT/'source_hashes.json',hashes)
 protocol={'scope':'Meeting 9 evidence-sufficiency engineering prototype, not a validated benchmark','dataset':'Legal RAG Bench','selection':'All 20 previously fixed hash-selected development questions, unchanged; already inspected benchmark; no new random selection or test holdout','methods':METHODS,'k':KS,'budgets':BUDGETS,'tokenizer':'sentence-transformers/all-MiniLM-L6-v2','tokenizer_revision':REV,'token_unit':'MiniLM WordPiece tokens, no special tokens; not GPT tokens','document_text':'original corpus text field, including its embedded headings, excluding separate title/footnotes fields','assembly':'ranked top-k document text prefixes; fill each document before moving on; final document prefix truncated at tokenizer offset, then stop; no skipping','overhead':'document text only; no prompt, question, separators or reader overhead','annotation_method':'AI-assisted source-grounded facts and exact quotations, prepared before these metric results; prior project findings known, not independently blinded; human review pending','logic':'AND within a sufficient branch; OR across alternative branches; OR across source quotations for each fact. Maximum branch-completion fraction is reported as draft quote coverage. Do not require all alternative explanations or duplicated sources.','matching':'exact complete source quotation contained in delivered original text; candidate spans have verified original char offsets; no semantic entailment, no unsupported paraphrase credit','metric_limits':'Quote availability is a reproducible proxy for availability of draft supporting rules, not validated fact recall, true answerability, legal correctness, or reader answer accuracy. Exact quote matching misses paraphrases and incomplete drafts can overstate sufficiency. Full text reference judgments are not fact labels.','eligibility':'Query 92 retained but excluded from aggregate because released answer conflicts with stated permanent-retention intent; decision recorded in annotation draft before scoring. Query 32 supported by actual non-qrel DNA passage.','primary_inference':'none; descriptive exploratory prototype, no p-values or cross-dataset generalization','api_calls':0,'reader_calls':0,'new_embeddings':0}
 frozen(OUT/'protocol.json',protocol)
 selection=json.loads(SEL.read_text())['datasets']['legal'];assert len(selection)==20 and set(q['query_id'] for q in selection)==set(DRAFT)
 frozen(OUT/'selection.json',selection)
 corpus={x['id']:x for x in map(json.loads,CORPUS.read_text().splitlines())}
 annotations=build_annotations(selection,corpus);frozen(OUT/'annotations.json',annotations)
 # A first-class exclusion does not delete a query or silently replace a failed example.
 eligible={x['query_id'] for x in annotations if x['eligible_for_aggregate']};assert len(eligible)==19
 z=np.load(RANKS);ids=z['passage_ids'].tolist();assert z['query_ids'].tolist()==[q['query_id'] for q in selection]
 needed=sorted(set(ids[int(j)] for m in METHODS for ranks in z[m] for j in ranks[:max(KS)]))
 tokenpath=OUT/'token_offsets.json'
 if tokenpath.exists():tokenized=json.loads(tokenpath.read_text())
 else:
  tokenizer=AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2',revision=REV,cache_dir=str(ROOT.parent/'model_cache'),local_files_only=True);tokenizer.model_max_length=10**9;tokenized={}
  for st in range(0,len(needed),128):
   batch=needed[st:st+128];t=tokenizer([corpus[pid]['text'] for pid in batch],add_special_tokens=False,truncation=False,return_offsets_mapping=True)
   for pid,ts,offs in zip(batch,t['input_ids'],t['offset_mapping']):tokenized[pid]={'n_tokens':len(ts),'offsets':offs}
  dump(tokenpath,tokenized)
 assert set(tokenized)==set(needed)
 rows=[];factrows=[];deliveries=[];minimums=[]
 for qi,a in enumerate(annotations):
  qid=a['query_id'];branches=a['sufficient_branches_draft']
  for m in METHODS:
   for k in KS:
    ranked=[ids[int(j)] for j in z[m][qi,:k]];fullrankhit=int(a['published_qrel'] in ranked)
    lengths=[tokenized[pid]['n_tokens'] for pid in ranked];cumul=np.cumsum(lengths)
    earliest={}
    for f in a['facts']:
     points=[]
     for span in f['alternative_support_spans']:
      pid=span['passage_id']
      if pid not in ranked:continue
      ri=ranked.index(pid);before=int(cumul[ri-1]) if ri else 0
      ends=[pair[1] for pair in tokenized[pid]['offsets']];n=bisect.bisect_left(ends,span['char_end'])+1
      assert 1<=n<=len(ends) and ends[n-1]>=span['char_end']
      points.append(before+n)
     earliest[f['fact_id']]=min(points) if points else None
    branch_min=[max(earliest[fid] for fid in branch) if all(earliest[fid] is not None for fid in branch) else None for branch in branches]
    valid=[v for v in branch_min if v is not None];minimum=min(valid) if valid else None
    minimums.append({'query_id':qid,'method':m,'k':k,'minimum_prefix_tokens_for_draft_quote_sufficiency':minimum,'eligible':a['eligible_for_aggregate']})
    for budget in BUDGETS:
     remain=budget;docends={};docrecord=[];used=0
     for ri,pid in enumerate(ranked):
      n=tokenized[pid]['n_tokens'];take=min(remain,n)
      if take<=0:break
      charend=len(corpus[pid]['text']) if take==n else tokenized[pid]['offsets'][take-1][1]
      docends[pid]=charend;docrecord.append({'passage_id':pid,'rank':ri+1,'tokens':take,'full_document':take==n,'char_start':0,'char_end':charend});used+=take;remain-=take
      if take<n:break
     assert used==min(budget,sum(lengths))
     covered=set();supports={}
     for f in a['facts']:
      found=[sp for sp in f['alternative_support_spans'] if docends.get(sp['passage_id'],-1)>=sp['char_end']]
      for sp in found:assert corpus[sp['passage_id']]['text'][sp['char_start']:sp['char_end']]==sp['quote']
      if found:covered.add(f['fact_id'])
      supports[f['fact_id']]=found
      factrows.append({'query_id':qid,'method':m,'k':k,'budget':budget,'fact_id':f['fact_id'],'requirement_draft':f['requirement_draft'],'quote_available':int(bool(found)),'supporting_spans':json.dumps(found,ensure_ascii=False),'eligible':a['eligible_for_aggregate']})
     frac,sufficient=logic(covered,branches)
     # Independent earliest-token boundary check for every fact and logical outcome.
     alt={fid for fid,t in earliest.items() if t is not None and t<=budget};assert covered==alt
     assert bool(sufficient)==(minimum is not None and minimum<=budget)
     gold=a['published_qrel'];touched=int(gold in docends);goldfull=int(any(d['passage_id']==gold and d['full_document'] for d in docrecord))
     rows.append({'query_id':qid,'method':m,'k':k,'budget':budget,'eligible':a['eligible_for_aggregate'],'draft_type':a['draft_type'],'used_tokens':used,'retrieval_qrel_hit':fullrankhit,'delivered_qrel_any_prefix':touched,'delivered_qrel_full':goldfull,'draft_quote_coverage':frac,'draft_quote_sufficient':sufficient,'qrel_hit_but_draft_incomplete':int(fullrankhit and not sufficient),'draft_complete_without_qrel_hit':int(sufficient and not fullrankhit),'matched_fact_quotes':len(covered),'annotated_facts':len(a['facts']),'covered_draft_facts_per_1000_tokens':1000*len(covered)/used if used else 0})
     deliveries.append({'query_id':qid,'method':m,'k':k,'budget':budget,'documents':docrecord,'covered_fact_ids':sorted(covered),'branch_coverage':[[fid in covered for fid in b] for b in branches]})
 f=pd.DataFrame(rows);ff=pd.DataFrame(factrows);assert len(f)==1200
 f.to_csv(OUT/'per_query_metrics.csv',index=False);ff.to_csv(OUT/'per_fact_support.csv',index=False);pd.DataFrame(minimums).to_csv(OUT/'minimum_prefix_budget.csv',index=False);dump(OUT/'delivery_records.json',deliveries)
 metrics=['used_tokens','retrieval_qrel_hit','delivered_qrel_any_prefix','delivered_qrel_full','draft_quote_coverage','draft_quote_sufficient','qrel_hit_but_draft_incomplete','draft_complete_without_qrel_hit']
 summary=f[f.eligible].groupby(['method','k','budget']).agg(n=('query_id','size'),**{v:(v,'mean') for v in metrics}).reset_index();summary.to_csv(OUT/'summary.csv',index=False)
 # Explicit independent sanity checks for complementary vs alternative evidence.
 assert logic({'a'},[['a','b']])==(.5,0)
 assert logic({'a'},[['a'],['b','c']])==(1.,1)
 for _,g in f.groupby(['query_id','method','k']):
  for v in ['draft_quote_coverage','draft_quote_sufficient','used_tokens']:assert (g.sort_values('budget')[v].diff().dropna()>=-1e-12).all()
 for _,g in f.groupby(['query_id','method','budget']):
  for v in ['draft_quote_coverage','draft_quote_sufficient','used_tokens']:assert (g.sort_values('k')[v].diff().dropna()>=-1e-12).all()
 review=[]
 for a in annotations:
  for fact in a['facts']:
   review.append({'query_id':a['query_id'],'question':a['question'],'reference_answer':a['reference_answer'],'fact_id':fact['fact_id'],'requirement_draft':fact['requirement_draft'],'sufficient_branches_draft':json.dumps(a['sufficient_branches_draft']),'source_quotes':json.dumps(fact['alternative_support_spans'],ensure_ascii=False),'ai_eligible':a['eligible_for_aggregate'],'issue_note':a['note'],'human_fact_required':'','human_source_support_valid':'','human_corrected_fact_or_logic':'','reviewer':'','review_status':'pending'})
 pd.DataFrame(review).to_csv(OUT/'human_review_template.csv',index=False)
 assert all(sha(p)==hashes[str(p.relative_to(ROOT))] for p in sources)
 validation={'engineering_checks_passed':True,'selected_questions':20,'aggregate_eligible':19,'fact_drafts':len(review),'source_spans':sum(len(f['alternative_support_spans']) for a in annotations for f in a['facts']),'metric_rows':len(f),'fact_support_rows':len(ff),'offsets_and_quotes_verified':True,'independent_token_boundary_verification':True,'AND_OR_logic_checks':True,'budget_and_k_monotonicity':True,'sources_unchanged':True,'api_calls':0,'human_reviewed_questions':0,'human_validation':'pending; never interpret engineering pass as scientific validation','completion':'offline prototype complete; independent human validation remains outstanding'}
 dump(OUT/'validation.json',validation)
 report(annotations,f,summary,validation)
 cases(annotations,corpus,deliveries,f,ff)
 for p in OUT.iterdir():
  if p.suffix in ['.json','.csv'] and p.name not in ['token_offsets.json','delivery_records.json']:shutil.copy2(p,EXPORT/p.name)
 print(summary[(summary.k==10)&summary.budget.isin([512,1024,2048])].to_string(index=False));print(json.dumps(validation,indent=2));print(EXPORT)
def table(df):
 return '| '+' | '.join(df.columns)+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+'\n'.join('| '+' | '.join(f'{x:.3f}' if isinstance(x,float) else str(x) for x in row)+' |' for row in df.itertuples(index=False,name=None))
def report(annotations,f,s,v):
 os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,2,figsize=(10,4.4),layout='constrained')
 for m in METHODS:
  d=s[(s.method==m)&(s.k==10)].sort_values('budget');axes[0].plot(d.budget,100*d.draft_quote_coverage,'o-',label=LABELS[m]);axes[1].plot(d.budget,100*d.draft_quote_sufficient,'o-',label=LABELS[m])
 for ax in axes:
  ax.set(xlabel='Delivered document budget (MiniLM WordPiece tokens)',xticks=BUDGETS,ylim=(0,100));ax.grid(alpha=.2)
 axes[0].set(ylabel='Draft quote coverage (%)',title='Availability of required source quotations');axes[1].set(ylabel='Draft quote completeness (%)',title='At least one complete sufficient branch');axes[0].legend(fontsize=8)
 fig.suptitle('Legal prototype: 19 eligible questions, Top-10\nAI-assisted annotations; human validation pending',fontsize=11);fig.savefig(EXPORT/'evidence_quote_coverage.png',dpi=180);plt.close(fig)
 showcase=f[(f.k==10)&(f.budget==1024)&f.eligible].copy()
 mismatch=showcase[showcase.qrel_hit_but_draft_incomplete.eq(1)|showcase.draft_complete_without_qrel_hit.eq(1)]
 cols=['query_id','method','retrieval_qrel_hit','draft_quote_coverage','draft_quote_sufficient','used_tokens']
 mismatch[cols].to_csv(EXPORT/'metric_disagreements.csv',index=False)
 display=s[(s.k==10)&s.budget.isin([512,1024,2048])][['method','budget','n','retrieval_qrel_hit','draft_quote_coverage','draft_quote_sufficient','used_tokens']].copy();display['method']=display.method.map(LABELS)
 lines=['# Evidence sufficiency: offline prototype','', '**Status:** engineering prototype complete; independent human validation pending. No API, reader, or new embedding calls.','', 'Twenty unchanged Legal development questions were annotated using the question, released reference answer, and actual corpus quotations. Nineteen enter descriptive aggregates. This is an AI-assisted draft, not new human-validated ground truth or a held-out evaluation. FiQA was not assigned fabricated fact labels: its available qrels identify documents, not required facts.','', '## What is measured','', 'A fact is counted only if one of its verified source quotations is completely present in the delivered real document text. Different quotations for the same fact are alternatives (OR); facts inside a sufficient route are complementary (AND); different sufficient routes are alternatives (OR). Repeated support for one fact is counted once. Draft quote coverage is the maximum fraction covered within any sufficient route. Draft quote completeness requires one route to be fully covered.','', 'This measures **availability of draft source quotations**, not semantic entailment, true answerability, legal correctness, or reader answer accuracy. It misses valid paraphrases and omitted alternative evidence. Token budgets use pinned MiniLM WordPiece tokens and document text only, excluding prompt/question overhead.','', '## Top-10 results','',table(display),'', '![Draft quote coverage](evidence_quote_coverage.png)','', 'The same top-k rankings are used at every budget. A qrel Hit can remain constant while a needed quotation is truncated or absent. See `metric_disagreements.csv` for actual examples; this demonstrates a mismatch in these draft measurements, not proof of human-judged insufficiency.','', '## Annotation issues caught before scoring','', '- **Query 32:** the released relevant passage describes jury directions; the rule about nuclear DNA uniqueness occurs in another real corpus passage (`4.13.2-c1-s1`). The draft accepts that source, rather than treating the original qrel as the only possible evidence.','- **Query 53:** two possible explanatory routes are represented: unborn victim (two complementary rules) OR qualifying duress (two complementary rules). The annotation does not assume either factual circumstance is established in the scenario.','- **Query 92:** excluded from aggregate completeness. The question says the money is intended to be kept permanently, while the released answer says there is no intention to permanently deprive. Supporting blackmail rules are retained, but do not cure the contradictory explanation.','', '## Human review required','', 'Use `human_review_template.csv` to check whether each fact is required, whether the exact source really supports the interpretation, whether alternative sources/routes are missing, and whether reference answers contain errors. Human fields remain blank and status pending. A second reviewer should inspect a sample and disagreements should be adjudicated before using these labels for scientific claims. Candidate-span matching and token assembly have passed engineering checks only.','', '## Reproduce','', 'From the repository root:','```bash','.venv/bin/python person3_meeting9_evidence/run.py','```','', 'Requires existing Legal corpus, fixed 20-query saved rankings, and cached pinned MiniLM tokenizer. Exact source hashes, annotation draft, source character offsets, per-fact support, context-delivery records, and minimum prefix budgets are retained under `results/person3_meeting9_evidence/legal20_v1/`. Existing successful generations are reused and never rerun.','', f"Engineering verification: {v['metric_rows']} query/method/depth/budget rows; {v['fact_support_rows']} fact checks; every source quotation and offset checked, token thresholds independently reconstructed, and AND/OR plus monotonicity checks passed. Human-reviewed questions: 0."]
 (EXPORT/'README.md').write_text('\n'.join(lines)+'\n')
 (OUT/'completion_status.json').write_text(json.dumps({'offline_prototype_complete':True,'human_validated_benchmark_complete':False,'pending':'Independent human fact/source/logic review','api_calls':0},indent=2)+'\n')
def cases(annotations,corpus,deliveries,metrics,facts):
 amap={a['query_id']:a for a in annotations};lookup={(d['query_id'],d['method'],d['k'],d['budget']):d for d in deliveries}
 choices=[('51','short_30_50_mix_dense',10,512),('83','focused160_mix_dense',10,256),('32','focused160_mix_dense',10,1024)]
 lines=['# Concrete examples: draft quotation availability','','These cases illustrate the implemented AND/OR and source-offset measurements. They are not independent human judgments of answerability.']
 for qid,m,k,b in choices:
  a=amap[qid];row=metrics[(metrics.query_id==qid)&(metrics.method==m)&(metrics.k==k)&(metrics.budget==b)].iloc[0]
  lines += ['',f'## Query {qid}: {LABELS[m]}, Top-{k}, budget {b} WordPiece tokens','',a['question'],'',f"Published-qrel Hit: {int(row.retrieval_qrel_hit)}. Draft quotation coverage: {row.draft_quote_coverage:.1%}. Complete draft route: {bool(row.draft_quote_sufficient)}.",'','| Required fact draft | Source quotation available |','| --- | --- |']
  for fact in a['facts']:
   matched=facts[(facts.query_id==qid)&(facts.method==m)&(facts.k==k)&(facts.budget==b)&(facts.fact_id==fact['fact_id'])].iloc[0].quote_available
   lines.append('| '+fact['requirement_draft']+' | '+('Yes' if matched else 'No exact complete quote')+' |')
  if a['note']:lines += ['',a['note']]
 (EXPORT/'case_examples.md').write_text('\n'.join(lines)+'\n')
 # Ready-to-review real contexts with deterministic method-blind IDs.
 # Only one fixed setting is included to keep the first human-review task small.
 blinded=[];mapping=[]
 for row in metrics[(metrics.k==10)&(metrics.budget==1024)&metrics.eligible].to_dict('records'):
  a=amap[row['query_id']];d=lookup[(row['query_id'],row['method'],10,1024)]
  case_id=hashlib.sha256(('evidence-review-v1|'+row['query_id']+'|'+row['method']).encode()).hexdigest()[:16]
  blinded.append({'case_id':case_id,'query_id':row['query_id'],'question':a['question'],'documents':[{'source_id':x['passage_id'],'text':corpus[x['passage_id']]['text'][:x['char_end']]} for x in d['documents']],'reviewer':'','semantic_context_sufficient':'','supported_facts_and_source_spans':[],'review_status':'pending'})
  mapping.append({'case_id':case_id,'query_id':row['query_id'],'method':row['method'],'k':10,'budget':1024})
 blinded.sort(key=lambda x:x['case_id'])
 (EXPORT/'context_review_blinded.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in blinded))
 dump(OUT/'context_review_method_mapping.json',mapping)
if __name__=='__main__':main()
