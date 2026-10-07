"""Independent CSV-only check: standard library, no data/model downloads."""
import csv, json, math
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/person3_hyde_robustness'

def read(name):
    with (OUT/name).open(newline='',encoding='utf-8') as f: return list(csv.DictReader(f))

def main():
    manifest=json.loads((OUT/'run_manifest.json').read_text())
    n=manifest['n_questions']; assert n==100
    passage_index={r['passage_id']:int(r['index']) for r in read('passage_ids.csv')}
    gold=[passage_index[r['gold_passage_id']] for r in read('queries.csv')]
    details=defaultdict(list)
    seen=defaultdict(set)
    for row in read('per_query_results.csv'):
        i=int(row['query_index']); assert 0<=i<n and i not in seen[row['method_id']]
        seen[row['method_id']].add(i)
        top5=json.loads(row['top5_indices']); assert len(top5)==5 and len(set(top5))==5
        assert all(0<=d<manifest['n_passages'] for d in top5)
        top10=json.loads(row['top10_indices']); assert len(top10)==10 and len(set(top10))==10
        assert all(0<=d<manifest['n_passages'] for d in top10) and top10[:5]==top5
        rank=int(row['gold_rank'])
        actual=top10.index(gold[i])+1 if gold[i] in top10 else 0
        assert (rank if 0<rank<=10 else 0)==actual
        details[row['method_id']].append((i,rank))
    details={method:[rank for _,rank in sorted(pairs)] for method,pairs in details.items()}
    rows=read('retrieval_metrics.csv')
    assert len(rows)==4*len(details)
    for row in rows:
        assert row['dataset_id']=='isaacus/legal-rag-bench'
        assert row['corpus_sha256']==manifest['corpus_sha256'] and row['qa_sha256']==manifest['qa_sha256']
        assert row['answer_status']=='not_evaluated'
        ranks=details[row['method_id']]; assert len(ranks)==n
        k=int(row['k']); hits=[r for r in ranks if 0<r<=k]
        expected={'precision_at_k':len(hits)/(n*k),'recall_at_k':len(hits)/n,'hit_rate_at_k':len(hits)/n,'mrr_at_k':sum(1/r for r in hits)/n,'ndcg_at_k':sum(1/math.log2(r+1) for r in hits)/n}
        for key,val in expected.items(): assert math.isclose(float(row[key]),val,abs_tol=1e-12),(row['method_id'],key)
    for pair in read('paired_comparisons_k5.csv'):
        a=[0<r<=5 for r in details[pair['method_id']]]; b=[0<r<=5 for r in details[pair['baseline']]]
        aw=sum(x and not y for x,y in zip(a,b)); bw=sum(y and not x for x,y in zip(a,b)); d=aw+bw
        p=min(1.,2*sum(math.comb(d,i) for i in range(min(aw,bw)+1))/(2**d)) if d else 1.
        assert aw==int(pair['candidate_only_hits']) and bw==int(pair['baseline_only_hits'])
        assert math.isclose(p,float(pair['mcnemar_exact_p']),abs_tol=1e-12)
    for path in OUT.glob('generated_*.csv'):
        with path.open(newline='',encoding='utf-8') as f: gen=list(csv.DictReader(f))
        assert len(gen)==3*n
        assert len({(r['query_id'],r['hypothesis']) for r in gen})==3*n
        assert len({r['signature'] for r in gen})==1
    print(f'PASS: {len(details)} methods, {len(rows)} metric rows, paired tests and generation records independently verified.')
if __name__=='__main__': main()
