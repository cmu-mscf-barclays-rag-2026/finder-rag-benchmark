"""Independently verify the fresh-generation replication, then write its report."""
from pathlib import Path
import hashlib
import json
import math
import os
import shutil
import sqlite3

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/person3_legal_regeneration/fresh100_v1'
EXPORT=ROOT.parent/'outputs/legal_regeneration_20261117'
os.environ.setdefault('MPLCONFIGDIR',str(OUT/'matplotlib_cache'))


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    import run as original
    original.check()
    qa,pages,old=original.data()
    fresh=read(OUT/'fresh_generations.json')
    expected_ids={q['id'] for q in qa}
    assert set(fresh)==expected_ids
    protocol=read(OUT/'protocol.json')
    old_ids={x['response_id'] for x in old.values()}
    fresh_ids={x['response_id'] for x in fresh.values()}
    assert len(fresh_ids)==100 and not fresh_ids&old_ids
    db=sqlite3.connect(f'file:{OUT/"api_checkpoint.sqlite"}?mode=ro',uri=True)
    ledger=list(db.execute('SELECT query_id,state,reserve,cost,payload FROM calls'));db.close()
    assert len(ledger)==100 and {r[0] for r in ledger}==expected_ids
    total=0.
    for qid,state,reserve,cost,payload in ledger:
        assert state=='saved' and json.loads(payload)==fresh[qid]
        value=fresh[qid]
        q=next(q for q in qa if q['id']==qid)
        assert value['prompt']==old[qid]['prompt']==protocol['prompt'].format(q=q['question'])
        assert value['model_returned']==protocol['model'] and value['text'].strip()
        u=value['usage'];computed=(u['input_tokens']*2.5+u['output_tokens']*15)/1e6
        assert abs(computed-cost)<1e-14 and cost<=reserve
        total+=cost
    budget=read(OUT/'cost_summary.json')
    assert budget['states']=={'saved':100}
    assert abs(total-budget['new_estimate_and_reserves_usd'])<1e-12
    assert protocol['budget']['prior_usd']+total<=7 and total<=.75
    ranks=np.load(OUT/'rankings.npz')
    assert ranks['query_ids'].tolist()==[q['id'] for q in qa]
    bm=original.engine(pages)
    replayed=0
    for i,q in enumerate(qa):
        for name,text in [('fresh_h',fresh[q['id']]['text']),('fresh_qh',q['question']+'\n'+fresh[q['id']]['text'])]:
            counts=bm.vec.transform([text]).tocsr();counts.data[:]=1
            scores=(counts@bm.mat.T).toarray().reshape(-1)
            expected=sorted(range(len(pages)),key=lambda j:(-float(scores[j]),j))[:100]
            assert expected==ranks[name][i].tolist(),(q['id'],name)
            replayed+=1
    frame=pd.read_csv(OUT/'per_query_metrics.csv',dtype={'query_id':str})
    assert len(frame)==1000 and not frame.duplicated(['query_id','method','k']).any()
    pids=ranks['passage_ids'].tolist();gold={q['id']:pids.index(q['relevant_passage_id']) for q in qa}
    lookup={qid:i for i,qid in enumerate(ranks['query_ids'].tolist())}
    for row in frame.itertuples(index=False):
        prefix=ranks[row.method][lookup[row.query_id],:row.k].tolist()
        position=prefix.index(gold[row.query_id])+1 if gold[row.query_id] in prefix else 0
        assert row.gold_rank==position and row.hit==int(position>0)
        assert abs(row.mrr-(1/position if position else 0))<1e-12
        assert abs(row.ndcg-(1/math.log2(position+1) if position else 0))<1e-12
    tests=pd.read_csv(OUT/'paired_tests.csv')
    assert len(tests)==8
    raw=[]
    for row in tests.itertuples(index=False):
        wide=frame[frame.k==row.k].pivot(index='query_id',columns='method',values='hit')
        a,b=wide[row.candidate],wide[row.control]
        wins=int(((a==1)&(b==0)).sum());losses=int(((a==0)&(b==1)).sum())
        assert (wins,losses)==(row.wins,row.losses)
        n=wins+losses
        p=min(1.,2*sum(math.comb(n,j) for j in range(min(wins,losses)+1))/2**n) if n else 1.
        assert abs(p-row.p_raw)<1e-12
        raw.append(p)
    adjusted=np.empty(8);previous=0.
    for rank,index in enumerate(sorted(range(8),key=lambda j:raw[j])):
        previous=min(1.,max(previous,(8-rank)*raw[index]));adjusted[index]=previous
    assert np.allclose(adjusted,tests.p_holm8,atol=1e-12,rtol=0)
    summary=pd.read_csv(OUT/'summary.csv')
    for row in summary.itertuples(index=False):
        sub=frame[(frame.method==row.method)&(frame.k==row.k)]
        assert len(sub)==row.n==100
        for metric in ['hit','mrr','ndcg']:assert abs(sub[metric].mean()-getattr(row,metric))<1e-12
    original.check()
    identical=sum(v['text']==old[qid]['text'] for qid,v in fresh.items())
    statuses={s:sum(v['status']==s for v in fresh.values()) for s in sorted({v['status'] for v in fresh.values()})}
    audit=dict(passed=True,fresh_responses=100,new_response_ids=True,usage_and_costs_checked=True,
        old_rank_replays=300,independent_new_rank_replays=replayed,metric_rows=1000,exact_tests=8,
        identical_texts=identical,status_counts=statuses,new_cost_usd=total,cumulative_estimate_usd=protocol['budget']['prior_usd']+total,
        frozen_inputs_unchanged=True,model_api_calls_during_audit=0)
    original.dump(OUT/'independent_validation.json',audit)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    methods=['q','old_h','fresh_h','old_qh','fresh_qh']
    names={'q':'Q','old_h':'Old H','fresh_h':'Fresh H','old_qh':'Old Q+H','fresh_qh':'Fresh Q+H'}
    table=summary.set_index(['method','k'])
    fig,ax=plt.subplots(figsize=(9,4.4));x=np.arange(5)
    for offset,k,color in [(-.18,3,'#147d8d'),(.18,5,'#9eafb9')]:
        values=[table.loc[(m,k),'hit'] for m in methods]
        bars=ax.bar(x+offset,values,width=.34,label=f'Hit@{k}',color=color)
        ax.bar_label(bars,labels=[f'{v:.0%}' for v in values],padding=3)
    ax.set_xticks(x,[names[m] for m in methods]);ax.set_ylim(0,.7);ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_ylabel('Labeled-passage Hit rate');ax.legend(frameon=False,loc='upper left');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    ax.set_title('Legal Q-to-H: old generation vs one fresh repeat')
    fig.tight_layout(rect=[0,.07,1,1]);fig.text(.5,.02,'Same100 questions, same prompt/model/BM25. Exploratory replication, not a causal mechanism test.',ha='center',fontsize=9)
    fig.savefig(OUT/'replication.png',dpi=180);plt.close(fig)
    lines=['# Legal Q-to-H fresh-generation replication','','One fresh H per question, with the original prompt and model.','',
        '| Query | Hit@3 | Hit@5 |','|---|---:|---:|']
    for m in methods:lines.append(f'| {names[m]} | {table.loc[(m,3),"hit"]:.0%} | {table.loc[(m,5),"hit"]:.0%} |')
    lines+=['',f'All300 original Q/H/Q+H Top100 lists replay exactly. All100 new response IDs differ from the original run; **{identical}/100 generated texts are identical**. New response status counts: `{statuses}`.','',
        '| k | Paired comparison | Wins | Losses | Raw p | Holm8 p |','|---:|---|---:|---:|---:|---:|']
    for row in tests.itertuples(index=False):
        lines.append(f'| {row.k} | {names[row.candidate]} vs {names[row.control]} | {row.wins} | {row.losses} | {row.p_raw:.5f} | {row.p_holm8:.5f} |')
    lines+=['','![Replication results](replication.png)','',
        'Settings: gpt-5.4-2026-03-05, Responses API, temperature0, reasoning none, output cap160, original focused160 prompt requesting70–100 words. Full4876-passage corpus, binary-query BM25 k1=1.2/b=.75, stable corpus-index ties. No prompt, model, ranking or question selection changes.','',
        'Holm8 covers these eight fixed post-hoc comparisons only. The same100 questions were inspected before; this is a computational repeat, not independent dataset confirmation. Similar fresh/old scores or nonsignificant differences do not establish equivalence. Regeneration does not prove which terms caused the gain, or that generated/final answers are correct.','',
        f'New cost estimate: **${total:.6f}**; cumulative **${audit["cumulative_estimate_usd"]:.6f} / $7**. Saved usage at $2.50/$15 per million input/output tokens, ignoring cached-input discounts; not a billing statement. [Official model pricing](https://developers.openai.com/api/docs/models/gpt-5.4).','',
        '[All metrics](summary.csv) · [All tests](paired_tests.csv) · [Text/ranking stability](generation_stability.csv) · [Independent validation](independent_validation.json)','',
        'Recheck retained results without new API calls, from the repository root:','','```bash',
        'OPENBLAS_NUM_THREADS=1 ../work/meeting10_venv/bin/python person3_legal_regeneration/run.py evaluate',
        'OPENBLAS_NUM_THREADS=1 ../work/meeting10_venv/bin/python person3_legal_regeneration/verify_and_report.py','```']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')
    EXPORT.mkdir(parents=True,exist_ok=True)
    for p in OUT.iterdir():
        if p.is_file() and p.suffix in ['.json','.csv','.md','.png'] and p.name!='fresh_generations.json':shutil.copy2(p,EXPORT/p.name)
    original.dump(ROOT.parent/'outputs/CURRENT_API_BUDGET.json',dict(latest_round='Legal fresh-H replication',
        cumulative_estimate_usd=audit['cumulative_estimate_usd'],guard_usd=7,unknown_reservations_usd=0,
        prior_including_meeting10_usd=protocol['budget']['prior_usd'],fresh_replication_usd=total,
        source='finder-rag-benchmark/results/person3_legal_regeneration/fresh100_v1/cost_summary.json'))
    print(json.dumps(audit,indent=2));print(EXPORT)


if __name__=='__main__':main()
