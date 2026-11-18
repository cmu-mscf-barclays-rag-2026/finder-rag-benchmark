"""Top-3 follow-up from retained Legal BM25 Q/H/Q+H rankings; no API calls."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import shutil

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/person3_legal_small_k/k3_v1'
EXPORT = ROOT.parent / 'outputs/legal_top3_20261117'
SAVED = ROOT / 'results/person3_sparse_hyde/sparse_query_v1'
LEGAL = ROOT / 'results/person3_openai_hyde'
METHODS = ['bm25_q', 'bm25_focused160_h', 'bm25_focused160_qh']
LABELS = dict(zip(METHODS, ['BM25(Q)', 'BM25(H)', 'BM25(Q+H)']))
PAIRS = [(METHODS[1], METHODS[0]), (METHODS[2], METHODS[0]), (METHODS[1], METHODS[2])]


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p, value):
    Path(p).write_text(json.dumps(value, indent=2) + '\n')


def frozen(p, value):
    if p.exists():
        assert read(p) == value, f'Frozen artifact changed: {p}'
    else:
        dump(p, value)


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = dict(date='2026-11-17', dataset='Same previously inspected Legal100,4876 passages',
        endpoint='Hit@3', methods=METHODS, pairs=PAIRS, family_size=3,
        test='Two-sided exact paired McNemar; Holm across all three Top3 comparisons',
        secondary='Top5 retained as descriptive historical reference; reproduce existing31/50/49 and paired counts',
        settings='Reuse complete saved top100 lists and focused160 generations; no retrieval/model rerun',
        caveat='Post-hoc exploratory smaller-k follow-up. Holm3 does not adjust earlier Holm10 or historical inspection.',
        resources=dict(api_calls=0, new_generations=0, new_embeddings=0, paid_usd=0))
    frozen(OUT/'protocol.json', protocol)
    paths = [LEGAL/'evaluation_queries.csv', LEGAL/'passage_index.csv', LEGAL/'saved_rankings.npz',
             ROOT/'results/person3_offline_followup/step1_v1/legal_sparse_tests.csv']
    paths += sorted((SAVED/'full/legal').glob('*.json'))
    assert len(paths) == 104
    frozen(OUT/'source_hashes.json', {str(p.relative_to(ROOT)): sha(p) for p in paths})
    frozen(OUT/'implementation_hashes.json', {str(Path(__file__).resolve().relative_to(ROOT)): sha(Path(__file__))})
    frozen(OUT/'freeze_receipt.json', dict(new_top3_outcomes_not_computed=True,
        protocol_sha256=sha(OUT/'protocol.json'), source_manifest_sha256=sha(OUT/'source_hashes.json')))
    print('Frozen three Top3 comparisons before calculating outcomes.')


def evaluate():
    for manifest in ['source_hashes.json', 'implementation_hashes.json']:
        for path, digest in read(OUT/manifest).items():
            assert sha(ROOT/path) == digest, path
    qa = pd.read_csv(LEGAL/'evaluation_queries.csv', dtype=str)
    passages = pd.read_csv(LEGAL/'passage_index.csv', dtype={'passage_id':str})
    pids = passages.passage_id.tolist()
    assert passages.corpus_index.tolist() == list(range(4876))
    assert len(qa) == qa.id.nunique() == 100
    previous = np.load(LEGAL/'saved_rankings.npz')
    rows, hits = [], {k:{} for k in [3,5]}
    for i,q in enumerate(qa.itertuples(index=False)):
        saved = read(SAVED/'full/legal'/f'{q.id}.json')
        assert saved['query_id']==q.id and saved['baseline_replay_exact']
        assert np.array_equal(saved['rankings']['bm25_q'],previous['bm25'][i])
        gold = pids.index(q.relevant_passage_id)
        assert gold == int(previous['gold'][i])
        for method in METHODS:
            rank = saved['rankings'][method]
            assert len(rank)==len(set(rank))==100 and all(0<=v<4876 for v in rank)
            for k in [3,5]:
                position = rank.index(gold)+1 if gold in rank[:k] else 0
                hit = int(position>0)
                hits[k].setdefault(method,[]).append(hit)
                rows.append(dict(query_id=q.id,method=method,k=k,gold_passage_id=q.relevant_passage_id,
                    gold_rank_within_k=position,hit=hit,mrr=1/position if position else 0.,
                    ndcg=1/math.log2(position+1) if position else 0.))
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT/'per_query_metrics.csv',index=False)
    summary = frame.groupby(['method','k'],sort=False).agg(n=('query_id','size'),hits=('hit','sum'),
        hit=('hit','mean'),mrr=('mrr','mean'),ndcg=('ndcg','mean')).reset_index()
    summary.to_csv(OUT/'summary.csv',index=False)
    assert [sum(hits[5][m]) for m in METHODS]==[31,50,49]
    tests=[]
    for candidate,control in PAIRS:
        delta=np.array(hits[3][candidate])-np.array(hits[3][control])
        wins,losses=int((delta>0).sum()),int((delta<0).sum())
        n=wins+losses
        p=float(binomtest(wins,n,.5).pvalue) if n else 1.
        # Independent exact binomial tail check using integer arithmetic.
        exact=min(1.,2*sum(math.comb(n,j) for j in range(min(wins,losses)+1))/2**n) if n else 1.
        assert abs(p-exact)<1e-14
        tests.append(dict(candidate=candidate,control=control,k=3,n=100,mean_delta=float(delta.mean()),
            wins=wins,losses=losses,ties=int((delta==0).sum()),p_raw=p))
    raw=np.array([r['p_raw'] for r in tests]);order=np.argsort(raw,kind='stable');adjusted=np.empty(3)
    adjusted[order]=np.minimum(1,np.maximum.accumulate((3-np.arange(3))*raw[order]))
    for row,p in zip(tests,adjusted):row['p_holm3']=float(p)
    pd.DataFrame(tests).to_csv(OUT/'paired_tests.csv',index=False)
    old=pd.read_csv(ROOT/'results/person3_offline_followup/step1_v1/legal_sparse_tests.csv')
    for method in METHODS[1:]:
        d=np.array(hits[5][method])-np.array(hits[5]['bm25_q'])
        row=old[(old.candidate==method)&(old.control=='bm25_q')].iloc[0]
        assert (int((d>0).sum()),int((d<0).sum()))==(row.wins,row.losses)
    for path,digest in read(OUT/'source_hashes.json').items():assert sha(ROOT/path)==digest
    dump(OUT/'validation.json',dict(passed=True,queries=100,metric_rows=len(frame),
        complete_original_ranks_preserved=True,top5_baselines_and_paired_counts_reproduced=True,
        exact_tests_independently_checked=True,source_hashes_unchanged=True,api_calls=0,paid_usd=0))
    os.environ.setdefault('MPLCONFIGDIR',str(OUT/'matplotlib_cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7.5,4.3))
    x=np.arange(3)
    for offset,k,color in [(-.18,3,'#147d8d'),(.18,5,'#9eafb9')]:
        bars=ax.bar(x+offset,[np.mean(hits[k][m]) for m in METHODS],width=.34,color=color,label=f'Hit@{k}')
        ax.bar_label(bars,labels=[f'{np.mean(hits[k][m]):.0%}' for m in METHODS],padding=3)
    ax.set_xticks(x,[LABELS[m] for m in METHODS]);ax.set_ylim(0,.65);ax.set_ylabel('Labeled-passage Hit rate')
    from matplotlib.ticker import PercentFormatter
    ax.yaxis.set_major_formatter(PercentFormatter(1));ax.legend(frameon=False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    ax.set_title('Legal BM25: smaller retrieval depth, same100 questions')
    fig.tight_layout(rect=[0,.07,1,1]);fig.text(.5,.02,'Saved Focused HyDE rankings; exploratory. No API calls or new retrieval.',ha='center',fontsize=9)
    fig.savefig(OUT/'top3_vs_top5.png',dpi=180);plt.close(fig)
    lines=['# Legal BM25 at Top3','','Same100 questions and saved focused160 rankings; no API calls.','',
        '| Query | Hit@3 | Hit@5, historical reference |','|---|---:|---:|']
    for m in METHODS:lines.append(f'| {LABELS[m]} | {np.mean(hits[3][m]):.0%} | {np.mean(hits[5][m]):.0%} |')
    lines+=['','| Top3 paired comparison | Wins | Losses | Raw p | Holm3 p |','|---|---:|---:|---:|---:|']
    for r in tests:lines.append(f'| {LABELS[r["candidate"]]} vs {LABELS[r["control"]]} | {r["wins"]} | {r["losses"]} | {r["p_raw"]:.5f} | {r["p_holm3"]:.5f} |')
    lines+=['','![Smaller retrieval depth](top3_vs_top5.png)','',
        'Holm3 covers these three post-hoc Top3 comparisons only; it does not adjust earlier Top5 tests or historical benchmark inspection. Hit measures retrieval of a labeled passage, not answer accuracy.','',
        'Reproduce from retained rankings, from the repository root:','','```bash',
        '.venv/bin/python person3_legal_small_k/run.py prepare',
        'OPENBLAS_NUM_THREADS=1 .venv/bin/python person3_legal_small_k/run.py evaluate','```']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')
    EXPORT.mkdir(parents=True,exist_ok=True)
    for p in OUT.iterdir():
        if p.is_file():shutil.copy2(p,EXPORT/p.name)
    print(summary.to_string(index=False));print(pd.DataFrame(tests).to_string(index=False));print(EXPORT)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','evaluate'])
    globals()[parser.parse_args().stage]()
