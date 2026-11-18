"""Check original/fresh Legal scores and all three existing test families offline."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.stats import binomtest

D=Path(__file__).resolve().parent/'data'


def holm(raw):
    raw=np.asarray(raw);order=np.argsort(raw,kind='stable');result=np.empty(len(raw))
    result[order]=np.minimum(1,np.maximum.accumulate((len(raw)-np.arange(len(raw)))*raw[order]))
    return result


def main():
    scores=pd.read_csv(D/'replication_per_query.csv',dtype={'query_id':str})
    summary=pd.read_csv(D/'replication_summary.csv')
    assert len(scores)==1000 and not scores.duplicated(['query_id','method','k']).any()
    assert scores.query_id.nunique()==100
    for row in summary.itertuples(index=False):
        subset=scores[(scores.method==row.method)&(scores.k==row.k)]
        assert len(subset)==row.n==100
        for metric in ['hit','mrr','ndcg']:
            assert abs(subset[metric].mean()-getattr(row,metric))<1e-12
    families=[('original_top5_tests_holm10.csv','p_holm',10),
              ('original_top3_tests_holm3.csv','p_holm3',3),
              ('regenerated_tests_holm8.csv','p_holm8',8)]
    frames={}
    for filename,adjusted,n in families:
        f=pd.read_csv(D/filename);assert len(f)==n
        raw=[]
        for row in f.itertuples(index=False):
            discordant=int(row.wins+row.losses)
            p=float(binomtest(int(row.wins),discordant,.5).pvalue) if discordant else 1.
            assert abs(p-row.p_raw)<1e-12
            raw.append(p)
        assert np.allclose(holm(raw),f[adjusted],atol=1e-12,rtol=0)
        frames[n]=f
    aliases={'bm25_q':'q','bm25_focused160_h':'old_h','bm25_focused160_qh':'old_qh'}
    for size in [3,8]:
        for row in frames[size].itertuples(index=False):
            wide=scores[scores.k==row.k].pivot(index='query_id',columns='method',values='hit')
            a,b=aliases.get(row.candidate,row.candidate),aliases.get(row.control,row.control)
            delta=wide[a]-wide[b]
            assert (int((delta>0).sum()),int((delta<0).sum()),int((delta==0).sum()))==(row.wins,row.losses,row.ties)
    for row in frames[10].itertuples(index=False):
        if row.candidate not in aliases or row.control!='bm25_q':continue
        wide=scores[scores.k==5].pivot(index='query_id',columns='method',values='hit')
        delta=wide[aliases[row.candidate]]-wide.q
        assert (int((delta>0).sum()),int((delta<0).sum()))==(row.wins,row.losses)
    displayed=pd.read_csv(D/'comparisons_vs_q.csv')
    assert len(displayed)==8 and not displayed.duplicated(['k','method']).any()
    for row in displayed.itertuples(index=False):
        f=frames[row.family_size]
        src=f[(f.candidate==row.source_candidate)&(f.control==row.source_control)&(f.k==row.k)].iloc[0]
        column={3:'p_holm3',8:'p_holm8',10:'p_holm'}[row.family_size]
        assert abs(src.p_raw-row.p_raw)<1e-12 and abs(src[column]-row.p_adjusted)<1e-12
        values=summary[(summary.method==row.method)&(summary.k==row.k)].iloc[0]
        baseline=summary[(summary.method=='q')&(summary.k==row.k)].iloc[0]
        assert abs(values.hit-row.hit)<1e-12 and abs(baseline.hit-row.q_baseline)<1e-12
        assert bool(row.significant)==(row.p_adjusted<.05)
    stability=pd.read_csv(D/'generation_stability.csv')
    assert len(stability)==100 and stability.query_id.nunique()==100
    assert stability.distinct_response_id.all() and not stability.identical_text.any()
    print(json.dumps(dict(passed=True,queries=100,metric_rows=1000,displayed_comparisons=8,
        original_families_verified=[10,3,8],exact_tests_checked=21,
        correction_scope='Original separate families retained; no pooled correction or new tests',
        api_calls=0),indent=2))


if __name__=='__main__':main()
