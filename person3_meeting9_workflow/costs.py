"""Rebuild generation token/cost/request-latency tables from retained outputs."""
from pathlib import Path
import hashlib,json,os
import numpy as np,pandas as pd
from api import OUT,dump,ROOT
EXPORT=ROOT.parent/'outputs/meeting9_20261109/costs'

def main():
    paths=[ROOT/'results/person3_meeting9_ablation/factorial20_v1/successful_api_outputs.json',
        ROOT/'results/person3_meeting9_transfer/vidore_finance_v1/successful_api_outputs.json',
        ROOT/'results/person3_meeting9_transfer/vidore_finance_remaining249_v1/successful_api_outputs.json']
    rows=[]
    for p in paths:
        for x in json.loads(p.read_text()):
            u=x['usage'];cost=(u['input_tokens']*2.5+u['output_tokens']*15)/1e6
            rows.append(dict(dataset=x['dataset'],style=x['style'],query_id=x['query_id'],
                key=x['key'],input_tokens=u['input_tokens'],output_tokens=u['output_tokens'],
                total_tokens=u['total_tokens'],words=x['word_count'],latency_seconds=x['latency_seconds'],
                request_concurrency=x.get('request_concurrency',4),estimated_usd=cost,status=x['status']))
    f=pd.DataFrame(rows);assert len(f)==778
    logical_count=len(f);f=f.drop_duplicates('key').copy()
    assert f.status.eq('completed').all() and np.isfinite(f.latency_seconds).all()
    EXPORT.mkdir(parents=True,exist_ok=True);f.to_csv(EXPORT/'per_call.csv',index=False)
    g=f.groupby(['dataset','style']).agg(n=('query_id','size'),mean_input_tokens=('input_tokens','mean'),
        mean_output_tokens=('output_tokens','mean'),mean_total_tokens=('total_tokens','mean'),
        mean_words=('words','mean'),request_p50_seconds=('latency_seconds','median'),
        request_p95_seconds=('latency_seconds',lambda x:float(np.quantile(x,.95))),
        total_estimated_usd=('estimated_usd','sum'),mean_estimated_usd=('estimated_usd','mean')).reset_index()
    g.to_csv(EXPORT/'generation_cost_latency.csv',index=False)
    ledger=json.loads((OUT/'cost_summary.json').read_text())
    assert ledger['states'].get('saved')==len(f)
    assert not (set(ledger['states'])-{'saved','unknown_reserved_acknowledged'})
    unknown=ledger['new_cost_and_reserves_usd']-float(f.estimated_usd.sum())
    recovery=json.loads((OUT/'timeout_recovery_1.json').read_text())
    assert abs(unknown-recovery['conservative_retained_reservations_usd'])<1e-9
    concurrency_counts=f.request_concurrency.value_counts().sort_index().to_dict()
    concurrency_text=', '.join(f'{v} calls at concurrency{k}' for k,v in concurrency_counts.items())
    dump(EXPORT/'validation.json',dict(passed=True,logical_outputs=logical_count,unique_calls=len(f),new_cost_usd=float(f.estimated_usd.sum()),
        unknown_timeout_reserves_usd=unknown,cumulative_estimate_usd=ledger['cumulative_estimate_usd'],historical_latency_available=False,
        latency_scope='Observed Responses API duration: '+concurrency_text+'; not end-to-end RAG latency. Concurrency retained per call.',
        source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}))
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t=g[g.dataset=='vidore_finance_en'];labels=['Focused','Short']
    t=t.set_index('style').loc[['focused160','short_30_50']]
    fig,axes=plt.subplots(1,2,figsize=(7.4,4.1),layout='constrained')
    fig.get_layout_engine().set(rect=(0,.08,1,.92))
    axes[0].bar(labels,t.mean_output_tokens,color=['#6f8aa2','#21918c'])
    axes[0].set(ylabel='Mean generated tokens',title='Generation size')
    axes[1].bar(labels,t.request_p50_seconds,color=['#6f8aa2','#21918c'])
    axes[1].set(ylabel='Observed request median (seconds)',title='API request duration')
    for a in axes:a.spines[['top','right']].set_visible(False)
    fig.suptitle('ViDoRe Finance: 309 queries per frozen prompt')
    fig.text(.5,.012,'Descriptive request timing; mixed concurrency. Excludes embedding and retrieval.',ha='center',fontsize=8)
    fig.savefig(EXPORT/'generation_cost_latency.png',dpi=180);plt.close(fig)
    lines=['# Generation cost and latency','',
        f"{len(f)} successful new calls; estimated successful-call spend **USD{f.estimated_usd.sum():.6f}**, plus **USD{unknown:.6f}** conservatively reserved for four timeout attempts. Cumulative estimate **USD{ledger['cumulative_estimate_usd']:.6f} / USD7**.",'',
        'Rates: USD2.50/M input and USD15/M output; cached-input discounts ignored. Software estimates, not a billing statement. Pricing checked 9 November 2026 against [official model documentation](https://developers.openai.com/api/docs/models/gpt-5.4).','',
        'Request latency measures wall time around Responses create: '+concurrency_text+'. Concurrency is retained per call and mixed in these descriptive summaries, so these are not a controlled prompt-latency comparison. It excludes queueing, embedding and retrieval. No historical end-to-end latency was recovered or invented. Local stage timings are reported separately.','',
        '| Dataset | Prompt | n | Mean output tokens | Mean cost/query (USD) | Request p50 (s) | p95 (s) |',
        '|---|---|---:|---:|---:|---:|---:|']
    for _,r in g.iterrows():lines.append(f'| {r.dataset} | {r["style"]} | {r.n} | {r.mean_output_tokens:.1f} | {r.mean_estimated_usd:.6f} | {r.request_p50_seconds:.2f} | {r.request_p95_seconds:.2f} |')
    lines+=['','![Generation size and request duration](generation_cost_latency.png)','','Reproduce from repository root: `.venv/bin/python person3_meeting9_workflow/costs.py`. Successful API calls are never repeated.']
    (EXPORT/'README.md').write_text('\n'.join(lines)+'\n')
    print('Verified',len(f),'calls; new estimate USD',round(float(f.estimated_usd.sum()),6),flush=True)

if __name__=='__main__':main()
