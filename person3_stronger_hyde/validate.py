"""Independent arithmetic checks of saved results; no models or generation."""
import json,math,csv
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parents[1]/'results/person3_stronger_hyde'
def main():
 z=np.load(P/'saved_rankings.npz');gold=z['gold'];assert len(gold)==100
 count=0
 for row in csv.DictReader((P/'retrieval_metrics.csv').open()):
  k=int(row['k']);r=z[row['method_id']];assert r.shape==(100,100)
  for a in r:assert len(set(a))==100 and min(a)>=0 and max(a)<4876
  ranks=[next((j+1 for j,x in enumerate(a) if x==g),0) for a,g in zip(r,gold)]
  hit=sum(0<x<=k for x in ranks)/100
  expected={'precision_at_k':hit/k,'hit_rate_at_k':hit,'recall_at_k':hit,'mrr_at_k':sum(1/x for x in ranks if 0<x<=k)/100,'ndcg_at_k':sum(1/math.log2(x+1) for x in ranks if 0<x<=k)/100}
  for key,v in expected.items():assert abs(float(row[key])-v)<1e-12,(row['method_id'],key)
  count+=1
 for row in csv.DictReader((P/'paired_tests.csv').open()):
  a,b=int(row['candidate_only_hits']),int(row['baseline_only_hits']);n=a+b
  p=min(1,2*sum(math.comb(n,j) for j in range(min(a,b)+1))/2**n) if n else 1
  assert abs(p-float(row['mcnemar_exact_p']))<1e-12
 for key in ['qwen06','qwen17']:
  d=json.loads((P/f'generated_{key}.json').read_text());assert set(d)=={str(x) for x in range(1,101)}
  assert all(v.get('precision')=='bfloat16' for v in d.values())
 pilot=json.loads((P/'pilot_ids.json').read_text());ans=json.loads((P/'final_answers.json').read_text());assert len(ans)==40
 for v in ans.values():assert v['query_id'] in pilot and sum(x['tokens'] for x in v['passages'])<=2048
 reviews=list(csv.DictReader((P/'answer_reviews.csv').open()));assert len(reviews)==40 and len({(r['query_id'],r['method']) for r in reviews})==40
 for r in reviews:assert int(r['correctness']) in (0,1,2) and int(r['grounding']) in (0,1,2) and r['rationale']
 print(f'PASS: {count} metric rows, paired tests,200 complete generations and40 reviewed answers.')
if __name__=='__main__':main()
