"""Create per-question comparison and paired uncertainty estimates for held-out Hit@5."""
import argparse
import csv
import json
import random
from pathlib import Path


def analyze(result):
    rows=list(csv.DictReader((result/'per_query.csv').open()))
    k5={(r['method'],r['query_id']):r for r in rows if r['k']=='5'}
    records=[json.loads(line) for line in (result/'retrievals.jsonl').read_text().splitlines()]
    questions={str(r['query_id']):r['question'] for r in records}
    gold={str(r['query_id']):r['gold_parent_id'] for r in records}
    comparison=[]
    for qid in sorted(questions,key=int):
        hits={m:int(float(k5[m,qid]['hit_rate'])) for m in ('bm25','dense','hybrid')}
        comparison.append({'query_id':qid, 'question':questions[qid], **{m+'_hit5':v for m,v in hits.items()},
                           'hybrid_rescues_bm25':int(hits['hybrid'] and not hits['bm25']),
                           'hybrid_loses_bm25_hit':int(hits['bm25'] and not hits['hybrid']),
                           'all_miss':int(not any(hits.values()))})
    with (result/'question_comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(comparison[0])); w.writeheader(); w.writerows(comparison)
    rng=random.Random(42)
    bootstraps=[]
    for left,right in [('hybrid','bm25'),('hybrid','dense'),('bm25','dense')]:
        deltas=[r[left+'_hit5']-r[right+'_hit5'] for r in comparison]
        n=len(deltas)
        groups={}
        for row,delta in zip(comparison,deltas):
            groups.setdefault(gold[row['query_id']],[]).append(delta)
        clusters=list(groups.values())
        means=[]
        for _ in range(10000):
            sample=[d for cluster in rng.choices(clusters,k=len(clusters)) for d in cluster]
            means.append(sum(sample)/len(sample))
        means.sort()
        bootstraps.append({'comparison':left+' minus '+right, 'n_queries':n,
                           'hit5_difference':sum(deltas)/n,'paired_bootstrap_95pct_low':means[249],
                           'paired_bootstrap_95pct_high':means[9749]})
    with (result/'paired_hit5_intervals.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(bootstraps[0])); w.writeheader(); w.writerows(bootstraps)
    selected=json.loads((result/'selected.json').read_text())
    lines=['', '## Interpretation of this run', '']
    for method,cfg in selected.items():
        found=sum(r[method+'_hit5'] for r in comparison)
        lines.append(f'- {method}: {found}/{len(comparison)} labeled-parent hits at K=5; selected configuration `{cfg["config_id"]}`.')
    rescues=sum(r['hybrid_rescues_bm25'] for r in comparison)
    losses=sum(r['hybrid_loses_bm25_hit'] for r in comparison)
    misses=sum(r['all_miss'] for r in comparison)
    interval=bootstraps[0]
    lines += ['', f'Hybrid rescues {rescues} BM25 misses and loses {losses} BM25 hits; all three miss {misses} questions.',
              f'The hybrid-minus-BM25 Hit@5 difference is {interval["hit5_difference"]:.1%}; its exploratory paired gold-passage-group bootstrap interval is [{interval["paired_bootstrap_95pct_low"]:.1%}, {interval["paired_bootstrap_95pct_high"]:.1%}]. This does not establish a clear hybrid advantage.',
              'BM25 returns whole passages while the selected dense and hybrid configurations return smaller chunks. Gold-text coverage and context size therefore matter alongside the hit count.',
              'See question_comparison.csv for concrete rescues/regressions and paired_hit5_intervals.csv for uncertainty estimates.', '']
    report=result/'report.md'
    base=report.read_text().split('\n## Interpretation of this run')[0].rstrip()
    report.write_text(base+'\n'+'\n'.join(lines))
    print(json.dumps({'all_three_miss':sum(r['all_miss'] for r in comparison),
                      'hybrid_rescues_bm25':sum(r['hybrid_rescues_bm25'] for r in comparison),
                      'hybrid_loses_bm25_hit':sum(r['hybrid_loses_bm25_hit'] for r in comparison),
                      'paired_hit5_intervals':bootstraps},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',required=True,type=Path)
    analyze(p.parse_args().results)
