"""Question-only HyDE robustness grid, fixed MiniLM, official Legal RAG Bench.
All 100 previously inspected questions are exploratory, not a new held-out test.
"""
from __future__ import annotations
import argparse, hashlib, json, platform, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from person3_hyde.run_hyde_experiment import load_benchmark, sha256, metrics_at_k, first_gold_rank, paired_test
from legal_rag_hybrid_experiment import build_bm25, retrieve_bm25, download_if_missing
import legal_rag_hybrid_experiment as legal_benchmark
DATA_REV = 'db0b31dc6d195ce9916897e1ac5e4e6209736c8a'
EMBED = 'sentence-transformers/all-MiniLM-L6-v2'
EMBED_REV = '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
GEN = 'google/flan-t5-base'
GEN_REV = '7bcac572ce56db69c1ea7c8af255c5d7c9672fc2'
SOURCE_HASHES = {
 'corpus.jsonl': '3a3565bc5429f6cead90548e81f87352b449927e4be1cdd804c28d766bb9c246',
 'qa.jsonl': 'e3b869a4e293d081ec5f5b39c2058c8d27b36f611aa9f8275eb1877a7c8b38b0'}
GENERATION_RULES = {'num_beams':1,'repetition_penalty':1.2,'no_repeat_ngram_size':3,'top_p_when_sampling':.9,'input_max_length':512}
BENCHMARK = {'owner':'Cheryl','dataset_id':'isaacus/legal-rag-bench','corpus_sha256':SOURCE_HASHES['corpus.jsonl'],'qa_sha256':SOURCE_HASHES['qa.jsonl'],'evaluation_scope':'exploratory previously inspected 100 questions','answer_status':'not_evaluated'}
PROMPTS = {
 'short_answer': [
  'Generate a concise answer to this Australian criminal-law question. Question: {q} Answer:',
  'State the legal rule needed to answer this Australian criminal-law question. Question: {q} Answer:',
  'Explain the legal procedure relevant to this Australian criminal-law question. Question: {q} Answer:'],
 'evidence_passage': [
  'Generate the type of legal passage that would contain the evidence needed to answer this question about Victorian criminal law. Question: {q} Legal passage:',
  'Write a passage from a Victorian criminal-law guide explaining the relevant rule and its conditions. Question: {q} Legal passage:',
  'Write a legal evidence passage describing the procedure and exceptions relevant to this Victorian criminal-law question. Question: {q} Legal passage:']}

def top(scores, depth=100):
    # Stable ties follow corpus order, so exact rank output is reproducible.
    return np.argsort(-scores, axis=1, kind='stable')[:, :depth].astype(np.int32)

def fuse(rankings, weights, depth=100, rrf_k=60):
    result=[]
    for rows in zip(*rankings):
        scores={}
        for row,w in zip(rows,weights):
            if w == 0: continue
            for r,d in enumerate(row,1): scores[int(d)]=scores.get(int(d),0)+w/(rrf_k+r)
        result.append(sorted(scores,key=lambda d:(-scores[d],d))[:depth])
    return np.asarray(result,dtype=np.int32)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-dir',type=Path,default=ROOT/'data/legal_rag_bench')
    p.add_argument('--output-dir',type=Path,default=ROOT/'results/person3_hyde_robustness')
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--replay',action='store_true',help='Recompute metrics from saved rankings without models/data.')
    args=p.parse_args(); out=args.output_dir; out.mkdir(parents=True,exist_ok=True)
    if args.replay:
        replay(out)
        from person3_hyde import summarize_robustness
        summarize_robustness.OUT=out; summarize_robustness.main()
        return
    torch.set_num_threads(args.threads); torch.manual_seed(20261006)
    legal_benchmark.DATA_URL=f'https://huggingface.co/datasets/isaacus/legal-rag-bench/resolve/{DATA_REV}'
    download_if_missing(args.data_dir)
    for name, expected_hash in SOURCE_HASHES.items():
        if sha256(args.data_dir/name) != expected_hash:
            raise ValueError(f'{name} differs from the recorded benchmark; do not silently compare a different dataset.')
    cr,qa,ids,corpus,questions,answers,gold=load_benchmark(args.data_dir)
    embed=SentenceTransformer(EMBED,revision=EMBED_REV,device='cpu')
    enc=lambda texts: embed.encode(texts,batch_size=32,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False)
    cache=args.data_dir/'all_MiniLM_L6_v2_corpus_embeddings.npy'
    # Verify content/model provenance of the inherited corpus cache.
    previous=json.loads((ROOT/'results/person3_hyde/run_summary.json').read_text())
    if cache.exists() and previous['corpus_sha256']==sha256(args.data_dir/'corpus.jsonl') and previous['embedding_revision']==EMBED_REV:

        try:
            ce=np.load(cache)
            if ce.shape != (len(corpus),embed.get_sentence_embedding_dimension()): raise ValueError('Invalid corpus cache shape')
        except (ValueError, OSError):
            ce=enc(corpus); np.save(cache,ce)
    else: ce=enc(corpus); np.save(cache,ce)
    start=time.perf_counter(); qe=enc(questions); qrank=top(qe@ce.T); qms=(time.perf_counter()-start)*1000/len(qa)
    orank=top(enc(answers)@ce.T)
    vec,mat=build_bm25(corpus); brank,bms=retrieve_bm25(vec,mat,questions,100)
    # Stable BM25 rank ties too.
    qc=vec.transform(questions).tocsr(); qc.data[:]=1
    brank=top((qc@mat.T).toarray())
    standard=fuse([brank,qrank],[.75,.25])
    rankings={}; metadata={}; rows=[]; generation_rows=[]; audits=[]
    def add(mid,ranking,config,representation,latency,alpha=None,depth=100):
        rankings[mid]=ranking; metadata[mid]={'config':config,'representation':representation,'dense_weight':alpha,'candidate_depth':depth,'latency_ms_per_query':latency}
        for k in (1,3,5,10): rows.append(metrics_at_k(ranking,gold,k)|{'method_id':mid}|metadata[mid]|BENCHMARK)
    add('question_bm25',brank,'baseline','bm25',bms)
    add('question_dense',qrank,'baseline','question_dense',qms)
    add('standard_hybrid',standard,'baseline','hybrid',bms+qms,.25)
    add('reference_answer_oracle',orank,'oracle_only','oracle',None)
    tokenizer=AutoTokenizer.from_pretrained(GEN,revision=GEN_REV)
    generator=AutoModelForSeq2SeqLM.from_pretrained(GEN,revision=GEN_REV).eval()
    configs=[(style,length,temp,20261006) for style in PROMPTS for length in (32,96) for temp in (0.,.7)]
    # Repeat stochastic generation for the same default setup with two further seeds.
    configs += [('evidence_passage',96,.7,seed) for seed in (20261007,20261008)]
    for style,length,temp,seed in configs:
        config=f'{style}_len{length}_t{temp:g}_seed{seed}'
        cachefile=out/f'generated_{config}.csv'
        expected={'generator':GEN,'revision':GEN_REV,'style':style,'length':length,'temperature':temp,'seed':seed,'prompts':PROMPTS[style],'question_hash':hashlib.sha256(json.dumps(questions).encode()).hexdigest(),'batch_size':args.batch_size,'generation_rules':GENERATION_RULES}
        signature=hashlib.sha256(json.dumps(expected,sort_keys=True).encode()).hexdigest()
        if cachefile.exists():
            gf=pd.read_csv(cachefile,keep_default_na=False)
            if not gf.signature.eq(signature).all() or len(gf)!=len(qa)*3: raise ValueError(f'Stale generation cache: {cachefile}')
            genms=float(gf.generation_ms_per_query.iloc[0])
        else:
            torch.manual_seed(seed); outputs=[]; start=time.perf_counter()
            # Three different question-only prompt formulations supply 3 hypotheses.
            for h,template in enumerate(PROMPTS[style]):
                prompts=[template.format(q=q) for q in questions]
                for offset in range(0,len(prompts),args.batch_size):
                    tokens=tokenizer(prompts[offset:offset+args.batch_size],return_tensors='pt',padding=True,truncation=True,max_length=GENERATION_RULES['input_max_length'])
                    kwargs={'max_new_tokens':length,'do_sample':temp>0,**{k:GENERATION_RULES[k] for k in ('num_beams','repetition_penalty','no_repeat_ngram_size')}}
                    if temp>0: kwargs.update(temperature=temp,top_p=GENERATION_RULES['top_p_when_sampling'])
                    with torch.inference_mode(): generated=generator.generate(**tokens,**kwargs)
                    texts=tokenizer.batch_decode(generated,skip_special_tokens=True)
                    outputs.extend({'query_id':qa[i]['id'],'question':questions[i],'hypothesis':h,'hyde_text':text} for i,text in enumerate(texts,offset))
                print(f'{config}: hypothesis {h+1}/3 complete',flush=True)
            genms=1000*(time.perf_counter()-start)/len(qa)
            gf=pd.DataFrame(outputs); gf['signature']=signature; gf['generation_ms_per_query']=genms
            gf.to_csv(cachefile,index=False)
        # Never depend on CSV row ordering when associating hypotheses with queries.
        if gf.duplicated(['hypothesis','query_id']).any():
            raise ValueError(f'Duplicate hypothesis/query pair in {cachefile}')
        index=pd.MultiIndex.from_product([range(3),[q['id'] for q in qa]],names=['hypothesis','query_id'])
        gf=gf.set_index(['hypothesis','query_id']).loc[index].reset_index()
        texts=[gf[gf.hypothesis==h].hyde_text.astype(str).tolist() for h in range(3)]
        t=time.perf_counter(); he=np.stack([enc(x) for x in texts]); embedms=1000*(time.perf_counter()-t)/len(qa)
        hranks=[top(e@ce.T) for e in he]
        avg=he.mean(axis=0); avg/=np.maximum(np.linalg.norm(avg,axis=1,keepdims=True),1e-12)
        representations={'hyde_single':hranks[0],'question_plus_hyde':top(enc([q+'\n'+h for q,h in zip(questions,texts[0])])@ce.T),'multi_average':top(avg@ce.T),'multi_rrf':fuse(hranks,[1/3]*3)}
        for rep,ranking in representations.items():
            # Amortized batched experiment timing: single uses 1/3 of measured 3-hypothesis cost.
            multi=rep.startswith('multi'); latency=(genms+embedms)*(1 if multi else 1/3)
            add(config+'__'+rep,ranking,config,rep,latency)
            for depth in (20,100):
                for alpha in (.25,.5,.75):
                    t=time.perf_counter(); fused=fuse([brank[:,:depth],ranking[:,:depth]],[1-alpha,alpha],depth)
                    fusems=1000*(time.perf_counter()-t)/len(qa)
                    add(config+'__'+rep+f'__hybrid_a{alpha:g}_d{depth}',fused,config,rep+'_hybrid',latency+bms+fusems,alpha,depth)
        for i,row in enumerate(qa):
            for h in range(3):
                text=texts[h][i]; tok=embed.tokenizer(text,add_special_tokens=True)['input_ids']
                audits.append({'config':config,'query_id':row['id'],'hypothesis':h,'word_count':len(text.split()),'embedding_tokens':len(tok),'truncated_by_minilm':len(tok)>embed.max_seq_length,'empty':not text.strip(),'question_word_overlap':len(set(text.lower().split())&set(questions[i].lower().split()))/max(1,len(set(questions[i].lower().split())))})
        generation_rows.append(expected|{'config':config,'generation_3hyp_ms_per_query':genms,'mean_hypothesis_words':gf.hyde_text.str.split().str.len().mean()})
        pd.DataFrame(rows).to_csv(out/'retrieval_metrics.csv',index=False)
    np.savez_compressed(out/'saved_rankings.npz',gold=gold,**rankings)
    (out/'method_metadata.json').write_text(json.dumps(metadata,indent=2))
    pd.DataFrame(generation_rows).to_csv(out/'generation_configs.csv',index=False)
    (out/'generation_configs.json').write_text(json.dumps(generation_rows,indent=2))
    pd.DataFrame(audits).to_csv(out/'generation_audit.csv',index=False)
    pd.DataFrame({'query_id':[q['id'] for q in qa],'question':questions,'gold_passage_id':[q['relevant_passage_id'] for q in qa]}).to_csv(out/'queries.csv',index=False)
    pd.DataFrame({'index':range(len(ids)),'passage_id':ids}).to_csv(out/'passage_ids.csv',index=False)
    import transformers, sentence_transformers, scipy, sklearn
    manifest={'dataset':'isaacus/legal-rag-bench','n_questions':len(qa),'n_passages':len(ids),'corpus_sha256':sha256(args.data_dir/'corpus.jsonl'),'qa_sha256':sha256(args.data_dir/'qa.jsonl'),'embedding_model':EMBED,'embedding_revision':EMBED_REV,'embedding_max_seq_length':embed.max_seq_length,'generator':GEN,'generator_revision':GEN_REV,'batch_size':args.batch_size,'threads':args.threads,'rrf_k':60,'evaluation':'exploratory official 100-question test; already inspected in previous work; no held-out claim','latency':'batched CPU throughput including generation; single hypothesis estimated as one third of 3-hypothesis generation/embedding; not interactive latency; hardware dependent','python':platform.python_version(),'versions':{m.__name__:m.__version__ for m in [np,pd,torch,transformers,sentence_transformers,scipy,sklearn]}}
    manifest.update({'corpus_text':'title + space + text','gold_protocol':'single labelled relevant_passage_id per question','bm25':{'k1':1.2,'b':.75},'generation_rules':GENERATION_RULES,'cache_signature_version':2})
    manifest['dataset_revision']=DATA_REV
    manifest['generator_tie_word_embeddings']=generator.config.tie_word_embeddings
    manifest['generator_shared_and_lm_head_are_same_tensor']=generator.shared.weight.data_ptr()==generator.lm_head.weight.data_ptr()
    (out/'run_manifest.json').write_text(json.dumps(manifest,indent=2))
    replay(out)
    from person3_hyde import summarize_robustness
    summarize_robustness.OUT=out; summarize_robustness.main()

def replay(out):
    z=np.load(out/'saved_rankings.npz'); gold=z['gold']; meta=json.loads((out/'method_metadata.json').read_text()); rows=[]; pairs=[]; details=[]
    for mid,m in meta.items():
        for k in (1,3,5,10): rows.append(metrics_at_k(z[mid],gold,k)|{'method_id':mid}|m|BENCHMARK)
        if m['config'] not in ('baseline','oracle_only'):
            baseline='standard_hybrid' if m['dense_weight'] is not None else 'question_dense'
            pairs.append(paired_test(z[mid],z[baseline],gold,5)|{'method_id':mid,'baseline':baseline})
        for i,(r,g) in enumerate(zip(z[mid],gold)):
            details.append({'method_id':mid,'query_index':i,'gold_rank':first_gold_rank(r,int(g)),'top5_indices':json.dumps(r[:5].tolist()),'top10_indices':json.dumps(r[:10].tolist())})
    df=pd.DataFrame(rows); df.to_csv(out/'retrieval_metrics.csv',index=False)
    k5=df[df.k==5].copy(); q=float(k5[k5.method_id=='question_dense'].recall_at_k.iloc[0]); o=float(k5[k5.method_id=='reference_answer_oracle'].recall_at_k.iloc[0])
    k5['oracle_gap_closed']=(k5.recall_at_k-q)/(o-q); k5.to_csv(out/'summary_k5.csv',index=False)
    pd.DataFrame(pairs).to_csv(out/'paired_comparisons_k5.csv',index=False)
    pd.DataFrame(details).to_csv(out/'per_query_results.csv',index=False)
    # Audit, not selection on a held-out set: report range over all prespecified settings.
    dense=k5[k5.config.str.startswith(('short_answer','evidence_passage')) & k5.dense_weight.isna()]
    dense.groupby('representation').agg(hit5_min=('recall_at_k','min'),hit5_mean=('recall_at_k','mean'),hit5_max=('recall_at_k','max'),n_settings=('method_id','count')).to_csv(out/'robustness_by_representation.csv')
    best=k5[k5.config!='oracle_only'].sort_values(['recall_at_k','mrr_at_k'],ascending=False).head(12)
    best.to_csv(out/'exploratory_top_configurations.csv',index=False)
    print(best[['method_id','recall_at_k','mrr_at_k','ndcg_at_k']].to_string(index=False),flush=True)
if __name__=='__main__': main()
