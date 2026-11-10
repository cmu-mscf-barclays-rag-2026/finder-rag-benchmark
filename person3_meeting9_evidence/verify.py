"""Independent cached-artifact verification; no tokenizer/model/API execution."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/person3_meeting9_evidence/legal20_v1'
def main():
 for rel,digest in json.loads((OUT/'source_hashes.json').read_text()).items():
  assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==digest,rel
 corpus={r['id']:r for r in map(json.loads,(ROOT/'data/legal_rag_bench/corpus.jsonl').read_text().splitlines())}
 ann=json.loads((OUT/'annotations.json').read_text());a_byid={a['query_id']:a for a in ann}
 for a in ann:
  for f in a['facts']:
   assert f['alternative_support_spans']
   for sp in f['alternative_support_spans']:
    assert corpus[sp['passage_id']]['text'][sp['char_start']:sp['char_end']]==sp['quote']
 metrics=pd.read_csv(OUT/'per_query_metrics.csv',dtype={'query_id':str});facts=pd.read_csv(OUT/'per_fact_support.csv',dtype={'query_id':str});deliveries=json.loads((OUT/'delivery_records.json').read_text())
 lookup={(r['query_id'],r['method'],r['k'],r['budget']):r for r in deliveries};counts=0
 for row in metrics.to_dict('records'):
  key=(row['query_id'],row['method'],row['k'],row['budget']);a=a_byid[row['query_id']];delivered=lookup[key];docends={d['passage_id']:d['char_end'] for d in delivered['documents']}
  present=[]
  for f in a['facts']:
   supported=any(sp['char_end']<=docends.get(sp['passage_id'],-1) for sp in f['alternative_support_spans'])
   if supported:present.append(f['fact_id'])
   v=facts[(facts.query_id==row['query_id'])&(facts.method==row['method'])&(facts.k==row['k'])&(facts.budget==row['budget'])&(facts.fact_id==f['fact_id'])]
   assert len(v)==1 and int(v.iloc[0].quote_available)==int(supported);counts+=1
  branches=a['sufficient_branches_draft'];numerators=[sum(fid in present for fid in b) for b in branches]
  fraction=max(n/len(b) for n,b in zip(numerators,branches));complete=any(n==len(b) for n,b in zip(numerators,branches))
  assert np.isclose(row['draft_quote_coverage'],fraction) and row['draft_quote_sufficient']==int(complete)
  assert sum(d['tokens'] for d in delivered['documents'])==row['used_tokens']<=row['budget']
  assert len(delivered['documents'])<=row['k']
  assert all(d['full_document'] for d in delivered['documents'][:-1])
 eligible=metrics[metrics.eligible];saved=pd.read_csv(OUT/'summary.csv');columns=['used_tokens','retrieval_qrel_hit','delivered_qrel_any_prefix','delivered_qrel_full','draft_quote_coverage','draft_quote_sufficient','qrel_hit_but_draft_incomplete','draft_complete_without_qrel_hit']
 grouped=eligible.groupby(['method','k','budget']);assert len(saved)==len(grouped)
 for r in saved.to_dict('records'):
  g=grouped.get_group((r['method'],r['k'],r['budget']));assert len(g)==r['n']==19
  for c in columns:assert np.isclose(g[c].mean(),r[c]),c
 template=pd.read_csv(OUT/'human_review_template.csv');assert template.review_status.eq('pending').all() and template.reviewer.isna().all()
 assert len(ann)==20 and sum(x['eligible_for_aggregate'] for x in ann)==19 and len(template)==31
 print(f'Verified 20 annotations, 1200 context rows, {counts} fact rows, 60 aggregate rows; human reviews remain 0.')
if __name__=='__main__':main()
