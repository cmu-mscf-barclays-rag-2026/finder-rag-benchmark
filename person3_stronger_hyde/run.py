"""Controlled local HyDE generator comparison with resumable per-query checkpoints."""
from __future__ import annotations
import argparse, collections, gc, hashlib, json, platform, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from person3_hyde.run_hyde_experiment import load_benchmark,sha256,metrics_at_k,paired_test
from person3_hyde.run_robustness import top,fuse,EMBED,EMBED_REV,SOURCE_HASHES
from legal_rag_hybrid_experiment import build_bm25
OUT=ROOT/'results/person3_stronger_hyde';DATA=ROOT/'data/legal_rag_bench'
MODELS={'qwen06':('Qwen/Qwen3-0.6B','c1899de289a04d12100db370d81485cdf75e47ca'),'qwen17':('Qwen/Qwen3-1.7B','70d244cc86ccca08cf5af4e1e306ecf908b1ad5e')}
PROMPT='Generate the type of legal passage that would contain the evidence needed to answer this question about Victorian criminal law. Question: {q} Legal passage:'
CONTRACT={'models':MODELS,'embedding':[EMBED,EMBED_REV],'embedding_max_length':256,'prompt':PROMPT,'max_new_tokens':96,'decoding':'greedy, repetition_penalty1.2, no_repeat_ngram_size3','thinking':False,'threads':4,'dtype':'bfloat16 (generator); float32 (embeddings)','candidate_depth':100,'rrf_constant':60,'bm25_weight':.75,'data_hashes':SOURCE_HASHES,'scope':'exploratory previously inspected 100 questions','answer_pilot':'8 lowest sha256(str(query_id)) values; fixed before new generation; all five methods on same IDs'}

def write_json(path,value):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)
def data():
 for n,h in SOURCE_HASHES.items():assert sha256(DATA/n)==h,n
 return load_benchmark(DATA)
def embedder():
 m=SentenceTransformer(EMBED,revision=EMBED_REV,device='cpu',local_files_only=True);m.max_seq_length=256;return m
def enc(m,texts):return m.encode(texts,batch_size=32,normalize_embeddings=True,show_progress_bar=False,convert_to_numpy=True)
def prepare():
 cr,qa,ids,corpus,questions,answers,gold=data();m=embedder()
 cache=DATA/'stronger_hyde_corpus_embeddings.npy'
 if cache.exists():ce=np.load(cache)
 else:ce=enc(m,corpus);np.save(cache,ce)
 qe=enc(m,questions);qrank=top(qe@ce.T);oracle=top(enc(m,answers)@ce.T)
 vec,mat=build_bm25(corpus);qc=vec.transform(questions).tocsr();qc.data[:]=1;brank=top((qc@mat.T).toarray())
 ranks={'question_dense':qrank,'bm25':brank,'standard_hybrid':fuse([brank,qrank],[.75,.25]),'reference_oracle':oracle}
 old=ROOT/'results/person3_hyde_robustness';gf=pd.read_csv(old/'generated_evidence_passage_len96_t0_seed20261006.csv',keep_default_na=False);gf=gf[gf.hypothesis==0].set_index('query_id').loc[[q['id'] for q in qa]]
 hr=top(enc(m,gf.hyde_text.tolist())@ce.T);ranks['flan_dense']=hr;ranks['flan_hybrid']=fuse([brank,hr],[.75,.25])
 np.savez_compressed(OUT/'baseline_rankings.npz',gold=gold,**ranks)
 prev=np.load(old/'saved_rankings.npz');mapping={'question_dense':'question_dense','bm25':'question_bm25','standard_hybrid':'standard_hybrid','reference_oracle':'reference_answer_oracle','flan_dense':'evidence_passage_len96_t0_seed20261006__hyde_single'}
 write_json(OUT/'baseline_replay_check.json',{a:{'current_hit5':metrics_at_k(ranks[a],gold,5)['hit_rate_at_k'],'old_hit5':metrics_at_k(prev[b],gold,5)['hit_rate_at_k'],'identical_top100':bool(np.array_equal(ranks[a],prev[b]))} for a,b in mapping.items()})
 pilot=sorted(qa,key=lambda q:hashlib.sha256(str(q['id']).encode()).hexdigest())[:8]
 write_json(OUT/'pilot_ids.json',[q['id'] for q in pilot]);pd.DataFrame({'query_id':[q['id'] for q in qa],'question':questions,'gold_passage_id':[q['relevant_passage_id'] for q in qa]}).to_csv(OUT/'queries.csv',index=False)
 print('PREPARED controls and fixed pilot',flush=True)

def load_generator(key):
 name,rev=MODELS[key];tok=AutoTokenizer.from_pretrained(name,revision=rev,local_files_only=True);model=AutoModelForCausalLM.from_pretrained(name,revision=rev,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').eval();return tok,model

def generate_one(tok,model,prompt,limit):
 text=tok.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
 inp=tok(text,return_tensors='pt');start=time.perf_counter()
 with torch.inference_mode():out=model.generate(**inp,max_new_tokens=limit,do_sample=False,repetition_penalty=1.2,no_repeat_ngram_size=3,pad_token_id=tok.eos_token_id)
 elapsed=time.perf_counter()-start;new=out[0,inp.input_ids.shape[1]:];return {'text':tok.decode(new,skip_special_tokens=True),'seconds':elapsed,'input_tokens':int(inp.input_ids.shape[1]),'output_tokens':len(new),'hit_length_cap':len(new)==limit,'precision':'bfloat16'}

def generate(key):
 cr,qa,*_=data();cache=OUT/f'generated_{key}.json';done=json.loads(cache.read_text()) if cache.exists() else {}
 # An initial float32 timing pilot is preserved separately, never mixed into the bfloat16 comparison.
 pilot=OUT/'generated_qwen06_float32_timing_pilot.json'
 if key=='qwen06' and pilot.exists():
  for qid,row in json.loads(pilot.read_text()).items():
   if done.get(qid)==row:done.pop(qid)
 for row in done.values():row.setdefault('precision','bfloat16')
 if len(done)==len(qa):
  write_json(cache,done);pd.DataFrame(done.values()).to_csv(OUT/f'generated_{key}.csv',index=False);print(key,'all cached; no generation');return
 tok,model=load_generator(key)
 # One warm-up excluded from timing; real queries are checkpointed individually.
 generate_one(tok,model,'Write one word meaning law.',4)
 for q in qa:
  qid=str(q['id'])
  if qid in done:continue
  done[qid]=generate_one(tok,model,PROMPT.format(q=q['question']),96)|{'query_id':q['id'],'question':q['question']};write_json(cache,done)
  print(key,len(done),'/100',round(done[qid]['seconds'],2),'s',flush=True)
 pd.DataFrame(done.values()).to_csv(OUT/f'generated_{key}.csv',index=False)

def retrieve():
 cr,qa,ids,corpus,questions,answers,gold=data();m=embedder();ce=np.load(DATA/'stronger_hyde_corpus_embeddings.npy');base=np.load(OUT/'baseline_rankings.npz');ranks={k:base[k] for k in base.files if k!='gold'};times=[]
 for key in MODELS:
  cache=OUT/f'generated_{key}.json'
  if not cache.exists():continue
  done=json.loads(cache.read_text())
  if len(done)!=100:continue
  texts=[done[str(q['id'])]['text'] for q in qa]
  # Measure query encoding + search serially, without generation/model loading.
  rr=[];serial=[]
  enc(m,['warmup'])
  for text in texts:
   t=time.perf_counter();r=top(enc(m,[text])@ce.T)[0];serial.append((time.perf_counter()-t)*1000);rr.append(r)
  dense=np.asarray(rr);ranks[key+'_dense']=dense;ranks[key+'_hybrid']=fuse([base['bm25'],dense],[.75,.25])
  for q,ms in zip(qa,serial):times.append({'model':key,'query_id':q['id'],'retrieval_ms':ms,'generation_seconds':done[str(q['id'])]['seconds']})
 np.savez_compressed(OUT/'saved_rankings.npz',gold=gold,**ranks)
 rows=[metrics_at_k(r,gold,k)|{'method_id':mid,'evaluation_scope':CONTRACT['scope']} for mid,r in ranks.items() for k in [1,3,5,10]];df=pd.DataFrame(rows);df.to_csv(OUT/'retrieval_metrics.csv',index=False);df[df.k==5].to_csv(OUT/'comparison_k5.csv',index=False)
 comparisons=[]
 for key in MODELS:
  for suffix,bases in [('dense',['question_dense','flan_dense']),('hybrid',['standard_hybrid','flan_hybrid'])]:
   mid=key+'_'+suffix
   if mid not in ranks:continue
   for b in bases:comparisons.append(paired_test(ranks[mid],ranks[b],gold,5)|{'candidate':mid,'baseline':b})
 pd.DataFrame(comparisons).to_csv(OUT/'paired_tests.csv',index=False);pd.DataFrame(times).to_csv(OUT/'latency_per_query.csv',index=False)
 # Mechanical output-quality checks, never mislabelled semantic correctness.
 audit=[]
 for key in MODELS:
  if key+'_dense' not in ranks:continue
  for r in json.loads((OUT/f'generated_{key}.json').read_text()).values():
   tokens=m.tokenizer(r['text'])['input_ids'];audit.append({'model':key,'query_id':r['query_id'],'words':len(r['text'].split()),'empty':not bool(r['text'].strip()),'very_short':len(r['text'].split())<=3,'minilm_truncated':len(tokens)>256,'generation_capped':r['hit_length_cap']})
 pd.DataFrame(audit).to_csv(OUT/'generation_audit.csv',index=False)
 print(df[df.k==5].to_string(index=False),flush=True)

def answer():
 cr,qa,ids,corpus,questions,answers,gold=data();ranks=np.load(OUT/'saved_rankings.npz');pilot=json.loads((OUT/'pilot_ids.json').read_text());cache=OUT/'final_answers.json';done=json.loads(cache.read_text()) if cache.exists() else {};tok,model=load_generator('qwen06')
 methods=['standard_hybrid','flan_hybrid','qwen06_hybrid','qwen17_hybrid','gold_context']
 write_json(OUT/'answer_protocol.json',{'model':MODELS['qwen06'],'prompt':'Use only the supplied evidence to answer the question. Give a concise answer and cite supporting passage numbers like [1]. If the evidence is insufficient, say so.','context_budget_tokens':2048,'context_selection':'top5 title+text packed sequentially with final passage truncated to budget; model tokenizer; gold_context diagnostic','max_new_tokens':128,'thinking':False,'pilot_ids':pilot,'methods':methods,'evaluation':'manual rubric in answer_reviews.csv; lexical scores only proxies; no independent held-out claim'})
 for qid in pilot:
  i=next(i for i,q in enumerate(qa) if q['id']==qid)
  for method in methods:
   if method!='gold_context' and method not in ranks.files:continue
   cachekey=f'{qid}:{method}'
   if cachekey in done:continue
   indices=[int(gold[i])] if method=='gold_context' else ranks[method][i,:5].tolist();pieces=[];remaining=2048;used=[]
   for j,ix in enumerate(indices,1):
    item=f'[{j}] {corpus[ix]}';tokens=tok.encode(item,add_special_tokens=False);take=tokens[:remaining]
    if not take:break
    pieces.append(tok.decode(take));used.append({'label':j,'passage_id':ids[ix],'corpus_index':ix,'tokens':len(take),'truncated':len(take)<len(tokens)});remaining-=len(take)
   context='\n\n'.join(pieces);prompt='Use only the supplied evidence to answer the question. Give a concise answer and cite supporting passage numbers like [1]. If the evidence is insufficient, say so.\nQuestion: '+questions[i]+'\nEvidence:\n'+context+'\nAnswer:'
   generated=generate_one(tok,model,prompt,128)
   done[cachekey]=generated|{'query_id':qid,'method':method,'question':questions[i],'reference_answer':answers[i],'context':context,'passages':used,'gold_in_selected_top5':int(gold[i]) in indices,'gold_tokens_in_context':sum(x['tokens'] for x in used if x['corpus_index']==int(gold[i]))};write_json(cache,done)
   print('ANSWER',cachekey,round(generated['seconds'],2),'s',flush=True)
 pd.DataFrame([{k:v for k,v in r.items() if k not in ['passages']} for r in done.values()]).to_csv(OUT/'final_answers.csv',index=False)

def main():
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','generate','retrieve','answer']);p.add_argument('--model',choices=list(MODELS),default='qwen06');args=p.parse_args();OUT.mkdir(exist_ok=True);torch.set_num_threads(4);torch.manual_seed(20261020)
 contract=OUT/'experiment_contract.json'
 if contract.exists():assert json.loads(contract.read_text())==json.loads(json.dumps(CONTRACT)),'Protocol changed: use separate output directory'
 else:write_json(contract,CONTRACT)
 if args.stage=='generate':generate(args.model)
 else:globals()[args.stage]()
if __name__=='__main__':main()
