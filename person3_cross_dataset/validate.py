"""Independently recompute artifact metrics, identities, evidence spans and API totals."""
from pathlib import Path
import json, re, sqlite3, hashlib
import numpy as np
import pandas as pd
from scipy.stats import binomtest

BASE=Path(__file__).resolve().parents[1]
OUT=BASE/'results/person3_cross_dataset/gpt54_focused160_v1'
DATA=BASE/'data/external_benchmarks'

def validate(fiqa_only=False, available_dense=False):
    protocol=json.loads((OUT/'protocol.json').read_text());methods=protocol['methods']
    fq=[json.loads(l) for l in (OUT/'fiqa_queries.jsonl').read_text().splitlines()]
    rels={q['query_id']:q['qrels'] for q in fq}
    summary=pd.read_csv(OUT/'fiqa_metrics.csv')
    details=pd.read_csv(OUT/'fiqa_per_query_metrics.csv',dtype={'query_id':str})
    assert len(fq)==648 and len(details)==648*9*5
    assert sum(len(v) for v in rels.values())==1706
    rows=[]
    with np.load(OUT/'fiqa_rankings.npz') as saved:
        ids=saved['passage_ids'];qids=saved['query_ids']
        assert len(ids)==57638 and len(set(ids))==57638 and len(qids)==648
        for method in methods:
            ranks=saved[method];assert ranks.shape==(648,100)
            assert all(len(set(r))==100 and min(r)>=0 and max(r)<len(ids) for r in ranks)
            for k in [1,3,5,10,100]:
                computed=[]
                for qid,ranking in zip(qids,ranks):
                    rel=rels[str(qid)]
                    labels=[rel.get(str(ids[i]),0) for i in ranking[:k]]
                    positions=[i+1 for i,g in enumerate(labels) if g>0]
                    dcg=sum(g/np.log2(i+2) for i,g in enumerate(labels))
                    idcg=sum(g/np.log2(i+2) for i,g in enumerate(sorted(rel.values(),reverse=True)[:k]))
                    computed.append([int(bool(positions)),len(positions)/len(rel),len(positions)/k,
                      1/positions[0] if positions else 0,dcg/idcg])
                avg=np.mean(computed,axis=0);row=summary[(summary.method==method)&(summary.k==k)].iloc[0]
                assert np.allclose(avg,row[['hit','recall','precision','mrr','ndcg']].astype(float),rtol=1e-10,atol=1e-12)
                rows.append((method,k))
    if fiqa_only:
        result=dict(status='passed',queries=648,corpus_documents=57638,positive_judgments=1706,recomputed_metric_rows=len(rows))
        (OUT/'fiqa_validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));return
    lb=json.loads((DATA/'longbench2.json').read_text());byid={q['_id']:q for q in lb}
    outputs=[json.loads(l) for l in (OUT/'api_outputs.jsonl').read_text().splitlines()]
    calls={o['key']:o for o in outputs};assert len(calls)==len(outputs)
    generated=[o for o in outputs if o['kind'].startswith('hyde_')]
    assert len(generated)==648+503
    for o in outputs:
        assert o['text'] and o['model_returned']==protocol['model']
        assert o['usage']['output_tokens']<=(160 if o['kind'].startswith('hyde_') else 32)
        assert o['status'] in ['completed','incomplete']
        cost=(o['usage']['input_tokens']*2.5+o['usage']['output_tokens']*15)/1e6
        assert abs(cost-o['cost_usd_estimate'])<1e-12
    for ds,qa in [('fiqa',fq),('longbench2',lb)]:
        for q in qa:
            qid=q.get('query_id',q.get('_id'));question=q.get('question')
            o=calls[f'hyde::{ds}::{qid}']
            assert o['prompt']==protocol['profiles']['focused160']['prompts'][ds].format(q=question)
    mappings=json.loads((OUT/'longbench2_answer_mapping.json').read_text())
    predictions=pd.read_csv(OUT/'longbench2_predictions.csv',keep_default_na=False)
    assert len(predictions)==len(mappings)==503*10
    assert len(predictions.groupby(['method','query_id']))==5030
    for qid in byid:
        r=json.loads((OUT/'longbench2_retrieval'/(qid+'.json')).read_text());q=byid[qid]
        assert hashlib.sha256(q['context'].encode()).hexdigest()==r['context_sha256']
        for m in methods:
            rr=r['methods'][m];assert len(set(rr['top100']))==len(rr['top100'])
            assert all(0<=j<r['n_chunks'] for j in rr['top100'])
            assert len(rr['top5'])==min(5,r['n_chunks'])
            assert len(rr['top100'])==min(100,r['n_chunks'])
            for i,c in enumerate(rr['top5']):
                assert c['chunk_id']==rr['top100'][i]
                assert 0<=c['chunk_id']<r['n_chunks']
                a,b=c['span'];assert q['context'][a:b]==c['text']
    cache=DATA/'cache_v1';seen_contexts=set();dense_checks=0;fusion_checks=0;near_ties=[];missing_dense_queries=[]
    with np.load(OUT/'longbench2_query_embeddings.npz') as saved:
        qe=saved['questions'];he=saved['hyde']
    assert qe.shape==he.shape==(503,384)
    assert np.allclose(np.linalg.norm(qe,axis=1),1,atol=1e-6) and np.allclose(np.linalg.norm(he,axis=1),1,atol=1e-6)
    for i,q in enumerate(lb):
        r=json.loads((OUT/'longbench2_retrieval'/(q['_id']+'.json')).read_text())
        context_hash=r['context_sha256'];local=cache/'longbench2'/context_hash
        if not (local/'embeddings.npy').exists() and available_dense:
            missing_dense_queries.append(q['_id'])
        else:
            ce=np.load(local/'embeddings.npy',mmap_mode='r')
            assert ce.dtype==np.float32 and ce.shape==(r['n_chunks'],384)
            if context_hash not in seen_contexts:
                seen_contexts.add(context_hash)
                chunks=json.loads((local/'chunks.json').read_text())
                assert len(chunks['chunks'])==len(chunks['spans'])==r['n_chunks']
                assert all(a[0]<=b[0]<=a[1]<=b[1] for a,b in zip(chunks['spans'],chunks['spans'][1:]))
                assert np.allclose(np.linalg.norm(ce,axis=1),1,atol=1e-6)
            mix=(qe[i]+he[i])/2;mix=mix/np.linalg.norm(mix)
            for m,v in [('question_dense',qe[i]),('focused160_pure_dense',he[i]),('focused160_mix_dense',mix)]:
                scores=np.dot(v,ce.T)
                expected=np.lexsort((np.arange(len(scores)),-scores))[:min(100,len(scores))]
                stored=r['methods'][m]['top100']
                if expected.tolist()!=stored:
                    # Initial query vectors were encoded with the equivalent raw graph,
                    # before the cache was refreshed with O2. Allow only FP32-scale
                    # near-tie reorderings, without changing any saved experimental ranks.
                    values=scores[stored];inversion=float(np.max(values-np.minimum.accumulate(values))) if len(values)>1 else 0.
                    cutoff_shortfall=float(max(0,scores[expected[-1]]-values.min()))
                    assert inversion<=1e-6 and cutoff_shortfall<=1e-6,(q['_id'],m,inversion,cutoff_shortfall)
                    near_ties.append(dict(query_id=q['_id'],method=m,maximum_order_inversion=inversion,cutoff_shortfall=cutoff_shortfall))
                dense_checks+=1
        for m,dense,w in [('standard_hybrid_d25','question_dense',.25),('standard_hybrid_d50','question_dense',.5),
                         ('focused160_pure_hybrid_d25','focused160_pure_dense',.25),
                         ('focused160_mix_hybrid_d25','focused160_mix_dense',.25),
                         ('focused160_mix_hybrid_d50','focused160_mix_dense',.5)]:
            score=np.zeros(r['n_chunks'],dtype=np.float64)
            for source,weight in [('bm25',1-w),(dense,w)]:
                indices=r['methods'][source]['top100']
                score[indices]+=weight/(60+np.arange(1,len(indices)+1))
            expected=np.lexsort((np.arange(len(score)),-score))[:min(100,len(score))]
            assert expected.tolist()==r['methods'][m]['top100'],(q['_id'],m)
            fusion_checks+=1
    retrieval_validation=dict(status='passed_for_declared_scope',dense_scope='available caches' if available_dense else 'all queries',missing_dense_queries=missing_dense_queries,contexts=len(seen_contexts),dense_rankings_recomputed=dense_checks,
      fusion_rankings_recomputed=fusion_checks,query_count=503,dense_score_tolerance=1e-6,
      exact_dense_matches=dense_checks-len(near_ties),fp32_near_tie_reorderings=near_ties)
    (OUT/('longbench2_available_retrieval_validation.json' if available_dense else 'longbench2_retrieval_validation.json')).write_text(json.dumps(retrieval_validation,indent=2))
    for mapping in mappings:
        q=byid[mapping['query_id']];o=calls[mapping['call_key']]
        r=json.loads((OUT/'longbench2_retrieval'/(q['_id']+'.json')).read_text())
        chunks=[] if mapping['method']=='no_context' else r['methods'][mapping['method']]['top5']
        context='\n\n'.join(f'Retrieved chunk {i+1}: {c["text"]}' for i,c in enumerate(chunks))
        prompt=protocol['longbench2']['reader_prompt'].format(context=context,**{k:q[k] for k in ['question','choice_A','choice_B','choice_C','choice_D']})
        assert o['prompt']==prompt
        match=re.search(r'The correct answer is\s*\(?([A-D])\)?(?=[\s.,:;!?)]|$)',o['text'],re.I)
        if not match:match=re.search(r'^\s*\(?([A-D])\)?[.\s]*$',o['text'],re.I)
        answer=match.group(1).upper() if match else ''
        pred=predictions[(predictions.query_id==q['_id'])&(predictions.method==mapping['method'])].iloc[0]
        assert pred['pred']==answer and int(pred.correct)==int(answer==q['answer'])
    acc=pd.read_csv(OUT/'longbench2_accuracy.csv')
    for r in acc.to_dict('records'):
        df=predictions[predictions.method==r['method']]
        if r['group_by']!='overall':df=df[df[r['group_by']]==r['group']]
        assert len(df)==r['n'] and abs(df.correct.mean()-r['accuracy'])<1e-12
    tests=pd.read_csv(OUT/'paired_tests.csv');assert len(tests)==12
    for r in tests[tests.dataset=='longbench2'].to_dict('records'):
        pp=predictions.pivot(index='query_id',columns='method',values='correct')
        a,b=pp[r['candidate']],pp[r['control']]
        won=int(((a==1)&(b==0)).sum());lost=int(((a==0)&(b==1)).sum())
        p=binomtest(won,won+lost,.5).pvalue if won+lost else 1.
        assert won==r['gained'] and lost==r['lost'] and abs(p-r['p_raw'])<1e-12
    order=np.argsort(tests.p_raw);running=0
    for j,i in enumerate(order):
        running=max(running,min(1,(12-j)*tests.iloc[i].p_raw))
        assert abs(running-tests.iloc[i].p_holm)<1e-12
    db=sqlite3.connect(OUT/'api_checkpoint.sqlite')
    states=dict(db.execute('SELECT status,COUNT(*) FROM calls GROUP BY status'))
    assert states.get('saved')==len(outputs) and set(states)<= {'saved','unresolved_transport'}
    unresolved=[(key,reserve,json.loads(payload)) for key,reserve,payload in db.execute("SELECT key,reserve,payload FROM calls WHERE status='unresolved_transport'")]
    for key,reserve,payload in unresolved:
        assert payload['original_call_key'] in calls and payload['error_type']=='APIConnectionError' and payload['http_status'] is None
        assert reserve>0 and calls[payload['original_call_key']]['text']
    cost=sum(o['cost_usd_estimate'] for o in outputs)
    assert abs(cost-db.execute('SELECT SUM(cost) FROM calls').fetchone()[0])<1e-9
    result=dict(status='passed',scope='results_and_available_dense' if available_dense else 'full',dense_rankings_recomputed=dense_checks,dense_rankings_total=1509,fusion_rankings_recomputed=fusion_checks,missing_dense_queries=missing_dense_queries,fiqa_queries=648,fiqa_corpus_documents=57638,
      fiqa_recomputed_metric_rows=len(rows),longbench2_queries=503,
      longbench2_recomputed_predictions=5030,hyde_outputs=1151,
      unique_api_calls=len(outputs),estimated_api_cost_usd=cost,
      unresolved_transport_attempts=len(unresolved),unresolved_upper_allowance_usd=sum(x[1] for x in unresolved),
      checked=['dataset IDs and counts','multi-relevance metrics from all saved rankings',
        'HyDE prompts equal question-only templates','reader prompts contain only retrieved spans and question/choices',
        'retrieved spans match the correct question context','all RRF rankings recomputed; dense coverage explicitly recorded',
        'chunk overlap checked for contexts with retained indexes','answer extraction and exact accuracy',
        'all LongBench paired tests and Holm correction','API usage and budget totals'])
    (OUT/('results_validation.json' if available_dense else 'validation.json')).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--fiqa-only',action='store_true');p.add_argument('--available-dense',action='store_true');a=p.parse_args();validate(a.fiqa_only,a.available_dense)
