"""Independent read-only audit of full-page OCR control. No model/API execution.
Tokenizer-only source offset replay; saved float32 score replay on fixed queries.
"""
from pathlib import Path
import os,json,math,hashlib
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',DO_NOT_TRACK='1',TOKENIZERS_PARALLELISM='false')
import numpy as np,pandas as pd
from transformers import AutoTokenizer
R=Path(__file__).resolve().parents[1];P=R/'results/person3_meeting9_fullpage/v1';A=R/'results/person3_meeting9_transfer/vidore_finance_v1';B=A.with_name('vidore_finance_remaining249_v1');OUT=R.parent/'outputs/meeting9_20261109/independent_fullpage_audit.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def main():
 transfer=read(OUT.with_name('independent_transfer_audit.json'));assert all(x['status']=='passed' for x in transfer['cohorts'].values()) and transfer['full309_aggregate']['status']=='passed'
 names=['protocol.json','chunks.json','index_validation.json','rankings.npz','per_query_metrics.csv','summary.csv','source_hashes.json','query_vector_source_hashes.json','validation.json','completion_status.json']
 hashes={name:sha(P/name) for name in names};protocol=read(P/'protocol.json');assert (protocol['chunk_tokens'],protocol['overlap_tokens'],protocol['stride_tokens'])==(224,32,192)
 assert protocol['dataset_revision']=='7f432c176d82e27546501ad8064a713ac3071809' and protocol['mix_question_weight']==.5
 assert 'maximum cosine' in protocol['page_score'] and protocol['encoder_revision']=='1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
 for name in ['source_hashes.json','query_vector_source_hashes.json']:
  for relative,digest in read(P/name).items():assert sha(R/relative)==digest
 corpus=[json.loads(x) for x in (A/'data/corpus.jsonl').read_text().splitlines()];m=read(P/'chunks.json');index=read(P/'index_validation.json');assert sha(P/'chunks.json')==index['chunk_manifest_sha256']
 assert len(m['pages'])==len(corpus)==2942 and len(m['chunk_texts'])==14017 and len(m['page_chunk_starts'])==2943
 assert [x['page_id'] for x in m['pages']]==[str(x['corpus_id']) for x in corpus]
 tokenizer=AutoTokenizer.from_pretrained(protocol['encoder'],revision=protocol['encoder_revision'],cache_dir=str(R.parent/'model_cache'),local_files_only=True);tokenizer.model_max_length=10**9
 offsets=[]
 for start in range(0,len(corpus),64):
  batch=tokenizer([x['markdown'] for x in corpus[start:start+64]],add_special_tokens=False,truncation=False,return_offsets_mapping=True,verbose=False)
  offsets.extend(batch['offset_mapping'])
 total_tokens=0;empty_pages=0;seenchunks=[]
 for page,source,off in zip(m['pages'],corpus,offsets):
  text=source['markdown'];assert page['ocr_sha256']==hashlib.sha256(text.encode()).hexdigest();n=len(off);assert page['source_tokens']==n
  total_tokens+=n;empty_pages+=int(n==0);assert page['chunk_start']==m['page_chunk_starts'][page['page_index']] and page['chunk_end']==m['page_chunk_starts'][page['page_index']+1]
  chunks=page['chunks'];assert len(chunks)==page['chunk_end']-page['chunk_start'] and len(chunks)>0
  covered=np.zeros(n+1,dtype=np.int16)
  for ci,c in enumerate(chunks):
   expected_start=ci*192 if n else 0;expected_end=min(expected_start+224,n)
   assert (c['source_token_start'],c['source_token_end'])==(expected_start,expected_end)
   assert c['chunk_index']==page['chunk_start']+ci
   if n:
    assert (c['char_start'],c['char_end'])==(off[expected_start][0],off[expected_end-1][1]);covered[expected_start]+=1;covered[expected_end]-=1
    assert all(c['char_start']<=aa<=bb<=c['char_end'] for aa,bb in off[expected_start:expected_end])
   else:assert (c['char_start'],c['char_end'])==(0,0) and len(chunks)==1
   t=text[c['char_start']:c['char_end']];assert t==m['chunk_texts'][c['chunk_index']] and hashlib.sha256(t.encode()).hexdigest()==c['text_sha256'];seenchunks.append(c['chunk_index'])
  assert n==0 or np.all(np.cumsum(covered[:-1])>0)
  assert chunks[0]['source_token_start']==0 and chunks[-1]['source_token_end']==n
 assert seenchunks==list(range(14017)) and total_tokens==2489816 and empty_pages==19
 lengths=[]
 for start in range(0,14017,256):lengths.extend(tokenizer(m['chunk_texts'][start:start+256],add_special_tokens=True,truncation=False,return_length=True,verbose=False)['length'])
 saved_lengths=[c['encoder_tokens_including_special'] for page in m['pages'] for c in page['chunks']];assert lengths==saved_lengths and min(lengths)==2 and max(lengths)==228 and max(lengths)<=256
 # Saved vector integrity and independent page maximum replay; no encoder execution.
 paths=sorted((P/'chunk_vectors').glob('*.npy'))
 for q in paths:assert sha(q)==index['vector_hashes'][q.name]
 ce=np.concatenate([np.load(q) for q in paths]);assert ce.shape==(14017,384) and ce.dtype==np.float32 and np.isfinite(ce).all() and np.allclose(np.linalg.norm(ce,axis=1),1,atol=1e-5)
 z=np.load(P/'rankings.npz');base=np.load(P/'baseline_rankings.npz');za=np.load(A/'rankings.npz');zb=np.load(B/'rankings.npz');qids=za['query_ids'].tolist()+zb['query_ids'].tolist();ids=z['passage_ids'].tolist();assert z['query_ids'].tolist()==qids and len(set(qids))==309 and ids==[str(i) for i in range(2942)]
 assert np.array_equal(base['question_dense'],z['question_dense'])
 qe=np.concatenate([np.load(x/'vectors/questions.npy') for x in [A,B]]);vectors={'question_dense':qe}
 for style in ['focused160','short_30_50']:
  he=np.concatenate([np.load(x/'vectors'/(style+'.npy')) for x in [A,B]]);mixed=.5*qe+.5*he;mixed=mixed/np.maximum(np.linalg.norm(mixed,axis=1,keepdims=True),1e-12);vectors[style+'_pure_dense']=he;vectors[style+'_mix_dense']=mixed
 sample=[0,59,60,308];replays=[]
 for method,vectorset in vectors.items():
  for qi in sample:
   scores=ce@vectorset[qi];page_scores=np.array([np.max(scores[page['chunk_start']:page['chunk_end']]) for page in m['pages']]);rank=np.lexsort((np.arange(2942),-page_scores))[:100]
   assert np.array_equal(rank,z[method][qi]),(method,qi)
   replays.append({'method':method,'query_id':qids[qi],'pages_replayed':2942,'top100_equal':True})
 # Every saved metric from all 5 query vectors / 2 document representations + BM25.
 official=pd.read_parquet(A/'data/qrels.parquet');rels={q:{} for q in qids}
 for row in official.to_dict('records'):
  if str(row['query_id']) in rels:rels[str(row['query_id'])][str(row['corpus_id'])]=int(row['score'])
 f=pd.read_csv(P/'per_query_metrics.csv',dtype={'query_id':str});assert len(f)==16995 and not f.duplicated(['query_id','method','representation','k']).any();lookup=f.set_index(['query_id','method','representation','k']);metricchecks=0
 methods=list(vectors);representations={'page_first256':{name:np.concatenate([za[name],zb[name]]) for name in methods},'fullpage_chunkmax':{name:z[name] for name in methods},'full_ocr_bm25':{'bm25':np.concatenate([za['bm25'],zb['bm25']])}}
 for rep,items in representations.items():
  for method,arr in items.items():
   assert arr.shape==(309,100)
   for qi,(qid,rank) in enumerate(zip(qids,arr)):
    assert len(set(rank))==100 and min(rank)>=0 and max(rank)<2942;rel=rels[qid];ideal=sorted(rel.values(),reverse=True)
    for k in protocol['top_k']:
     gains=[rel.get(ids[int(j)],0) for j in rank[:k]];found=[j+1 for j,g in enumerate(gains) if g>0];den=sum(g/math.log2(j+2) for j,g in enumerate(ideal[:k]));values={'hit':int(bool(found)),'recall':len(found)/len(rel),'mrr':1/min(found) if found else 0.,'ndcg':sum(g/math.log2(j+2) for j,g in enumerate(gains))/den}
     row=lookup.loc[(qid,method,rep,k)];assert row['cohort']==('initial60' if qi<60 else 'remaining249')
     for metric,value in values.items():assert math.isclose(row[metric],value,abs_tol=1e-13)
     metricchecks+=1
 summary=pd.read_csv(P/'summary.csv');assert len(summary)==165
 for row in summary.to_dict('records'):
  g=f[f.method.eq(row['method'])&f.representation.eq(row['representation'])&f.k.eq(row['k'])]
  if row['cohort']!='full309':g=g[g.cohort.eq(row['cohort'])]
  assert len(g)==row['n']=={'initial60':60,'remaining249':249,'full309':309}[row['cohort']]
  for metric in ['hit','recall','mrr','ndcg']:assert math.isclose(g[metric].mean(),row[metric],abs_tol=1e-13)
 baseline=summary[summary.cohort.eq('full309')&summary.method.eq('question_dense')&summary.representation.eq('fullpage_chunkmax')&summary.k.eq(10)].iloc[0];assert round(float(baseline.ndcg),6)==.438501
 complete=read(P/'completion_status.json');assert complete['complete'] and complete['api_calls']==0 and complete['new_pvalues']==0 and complete['secondary_descriptive_only']
 assert all(sha(P/name)==digest for name,digest in hashes.items())
 result={'passed':True,'scope':'Independent local full-page OCR control audit; secondary descriptive only','input_sha256':hashes,'pages':2942,'chunks':14017,'empty_pages_retained':19,'source_tokens_independently_retokenized':total_tokens,'224token_chunks_32token_overlap_192stride_verified':True,'all_source_token_offsets_covered':True,'all_chunk_text_and_character_offsets_match_source':True,'chunk_lengths_independently_retokenized':True,'max_encoder_tokens_including_specials':max(lengths),'no_document_chunk_encoder_truncation':True,'normalized_float32_vectors_and_hashes_verified':True,'page_max_aggregation_replays':replays,'page_scores_independently_replayed':len(replays)*2942,'baseline_question_rankings_unchanged':True,'metric_rows_independently_recomputed_from_rankings_and_official_qrels':metricchecks,'summary_rows_independently_recomputed':len(summary),'question_full309_ndcg10':float(baseline.ndcg),'generation_calls':0,'encoder_calls':0,'reader_calls':0,'new_pvalues':0,'tokenizer_only_replay':True,'limitations':['Scores replayed for 4 fixed queries in each of 5 query representations (20 top100 comparisons), not every dense score; metrics replay covers all saved rankings.','Page maximum offers more matching opportunities on longer pages and changes document representation.','Text-only supplied OCR, no visual model or final-answer reader evaluation.','Human evidence-sufficiency validation remains pending; this engineering audit does not validate fact annotations.']}
 OUT.parent.mkdir(parents=True,exist_ok=True);OUT.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['input_sha256','page_max_aggregation_replays','limitations']},indent=2));print(OUT)
if __name__=='__main__':main()
