"""Presentation for the frozen finite inference family; reads retained CSVs only."""
import json, os, shutil
import numpy as np
import pandas as pd

def label(method):
    if method=='bm25':return 'BM25'
    rep='Question' if method.startswith('question') else 'Focused' if method.startswith('focused') else 'Short'
    return f'{rep} w={int(method[-3:])/100:.2f}'

def report(out,export):
    os.environ.setdefault('MPLCONFIGDIR',str(out/'matplotlib_cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cells=pd.read_csv(out/'two_by_two_summary.csv');contrasts=pd.read_csv(out/'two_by_two_contrasts.csv');tests=pd.read_csv(out/'primary_tests.csv')
    d=cells[(cells.cohort=='remaining249')&(cells.k==10)].set_index(['filter','representation'])
    fig,ax=plt.subplots(figsize=(7.6,4.5));x=np.arange(2);width=.34
    for j,(rep,name,color) in enumerate([('question','Question hybrid','#9aa6b1'),('focusedmix','Focused mix hybrid','#347da5')]):
        values=[d.loc[(filter_,rep),'ndcg'] for filter_ in ['no_bank_filter','bank_filter']]
        bars=ax.bar(x+(j-.5)*width,values,width,label=name,color=color)
        for bar,v in zip(bars,values):ax.text(bar.get_x()+bar.get_width()/2,v+.006,f'{v:.3f}',ha='center',fontsize=10)
    ax.set_xticks(x,['No bank constraint','Fixed bank constraint']);ax.set_ylabel('nDCG@10');ax.set_ylim(0,float(d.ndcg.max())*1.2);ax.legend(frameon=False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('2×2 descriptive control: RRF dense weight0.50, all249');fig.tight_layout(rect=[0,.06,1,.94]);fig.text(.5,.018,'Repeatedly inspected benchmark. Interaction descriptive only; no new heldout confirmation.',ha='center',fontsize=8);fig.savefig(out/'two_by_two.png',dpi=180);plt.close(fig)
    lines=['# Exploratory inference and 2×2 hybrid control','','This follow-up adds only the missing unfiltered Focusedmix RRF w=0.50 cell. Existing runs and all565 historical artifacts remain unchanged. API calls/new embeddings/generations: **0**. All249 is primary; initial60/full309 and all K=5/10/15/20/25 are retained.','',
      '**Every p-value is exploratory.** Both cohorts and settings were repeatedly inspected. Question-level sign-flip does not adjust dependence from six shared annual reports. Holm10 covers only this fixed family, not earlier adaptive experiments. No bootstrap CI, independent clustered inference or new holdout claim is made.','',
      '## Table 1: finite inference family','','Two-sided paired sign-flip,100,000 draws, seed20261109; shared sign vectors in frozen query order. Original complete qrels/IDCG/Recall denominators. W/L/T are per-question nDCG@10 outcomes.','',
      '| Comparison (all249) | Mean Δ | W/L/T | Raw p | Holm10 p |','|---|---:|---:|---:|---:|']
    for r in tests.to_dict('records'):
        lines.append(f"| {label(r['method'])} vs {label(r['control'])} | {r['mean_delta']:+.4f} | {r['wins']}/{r['losses']}/{r['ties']} | {r['p_raw']:.5f} | {r['p_holm10']:.5f} |")
    lines+=['','## Table 2: bank filter × HyDE, fixed RRF w=0.50','','| Bank constraint | Query representation | nDCG@10 | Hit@10 | Recall@10 |','|---|---|---:|---:|---:|']
    for filtering,rep in [('no_bank_filter','question'),('no_bank_filter','focusedmix'),('bank_filter','question'),('bank_filter','focusedmix')]:
        r=d.loc[(filtering,rep)];lines.append(f'| {filtering} | {rep} | {r.ndcg:.4f} | {r.hit:.1%} | {r.recall:.4f} |')
    c=contrasts[(contrasts.cohort=='remaining249')&(contrasts.k==10)].set_index('contrast')
    lines+=['',f"Descriptive filter×HyDE interaction: **{c.loc['interaction_filter_by_hyde','ndcg_delta']:+.4f} nDCG@10**. No interaction test was added to the10-test family.",'','![Descriptive 2×2](two_by_two.png)','',
      'Source sensitivity keeps275 single-source and34 multi-source queries separate and also lists each bank. Each leave-one-report-out stress test excludes every query whose positive gold-source set contains that report, including multi-source queries; effects are descriptive, without added p-values or CIs. Full effects and W/L/T are in source_sensitivity.csv.','',
      'The original query39 explicitly contains an appended “Answer:” string. Main results retain it; a fixed exclude-only-q39 descriptive sensitivity is included. Query39 is in initial60, so excluding it leaves all249 unchanged. data_quality.json documents the marker; this is not an exhaustive leakage audit or a claim that all leakage was removed.','',
      'Reproduce from retained ranks:','','```bash','.venv/bin/python person3_meeting9_followup/inference/run.py prepare','.venv/bin/python person3_meeting9_followup/inference/run.py evaluate','```','',
      'The frozen protocol lists all10 tests and the separate descriptive interaction. Validation checks the missing ranks, full-K metrics, original denominators, complete test family, Holm arithmetic, source hashes and the565-file historical manifest.']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    export.mkdir(parents=True,exist_ok=True)
    for p in out.iterdir():
        if p.is_file():shutil.copy2(p,export/p.name)
    print((out/'README.md').read_text())
