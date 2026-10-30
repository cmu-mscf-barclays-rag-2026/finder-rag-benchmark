"""Independent replay of saved retrieval metrics; no API calls or model downloads."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binomtest
p=argparse.ArgumentParser();p.add_argument('result_dir',type=Path);a=p.parse_args();out=a.result_dir
protocol=json.loads((out/'protocol.json').read_text())
state=json.loads((out/'generation_checkpoint.json').read_text())
queries=pd.read_csv(out/'evaluation_queries.csv',keep_default_na=False)
assert len(queries)==100 and queries.id.is_unique
assert len(state['outputs'])==200 and len(state['attempts'])==200
for key,o in state['outputs'].items():
 q=queries.iloc[o['query_index']]
 assert str(q.id)==o['query_id']
 assert o['prompt']==protocol['profiles'][o['profile']]['prompt'].format(q=q.question)
 assert o['model_returned']==protocol['model']
 assert o['text'].strip() and o['status'] in ['completed','incomplete']
 assert o['usage']['input_tokens']>0 and o['usage']['output_tokens']>0
 assert o['usage']['output_tokens']<=protocol['profiles'][o['profile']]['max_output_tokens']
ranks=np.load(out/'saved_rankings.npz');gold=ranks['gold']
table=pd.read_csv(out/'retrieval_metrics.csv');per=pd.read_csv(out/'per_query_metrics.csv')
assert len(gold)==100 and len(table)==56 and len(per)==1400
hits={}
for name in ranks.files:
 if name=='gold':continue
 r=ranks[name];assert r.shape==(100,100)
 assert all(len(set(row))==100 for row in r)
 assert r.min()>=0 and r.max()<4876
 for k in [1,3,5,10]:
  scores=[];hit=[];rr=[];dcg=[]
  for row,g in zip(r,gold):
   pos=list(row[:k]).index(g)+1 if g in row[:k] else None
   hit.append(pos is not None);rr.append(1/pos if pos else 0);dcg.append(1/np.log2(pos+1) if pos else 0)
  expected={'hit':np.mean(hit),'recall':np.mean(hit),'precision':np.mean(hit)/k,'mrr':np.mean(rr),'ndcg':np.mean(dcg)}
  actual=table[(table.method==name)&(table.k==k)].iloc[0]
  for metric,v in expected.items():assert abs(actual[metric]-v)<1e-12,(name,metric)
  if k==5:hits[name]=np.array(hit)
tests=pd.read_csv(out/'paired_tests.csv');assert len(tests)==12
for _,row in tests.iterrows():
 c,b=hits[row.candidate],hits[row.control]
 won=int((c&~b).sum());lost=int((~c&b).sum())
 expected=binomtest(won,won+lost,.5).pvalue if won+lost else 1
 assert won==row.gained and lost==row.lost and abs(row.mcnemar_p-expected)<1e-12
assert abs(table[(table.method=='question_dense')&(table.k==5)].iloc[0]['hit']-.29)<1e-12
cost=sum(x['cost_usd_estimate'] for x in state['attempts'].values())
report={'validation':'passed','generated_passages':200,'ranked_methods':14,'metric_rows':56,'per_query_rows':1400,'planned_paired_tests':12,'cost_usd_estimate':cost,'note':'API and local decoding differ; prior question set reused; no final-answer evaluation.'}
(out/'validation.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
