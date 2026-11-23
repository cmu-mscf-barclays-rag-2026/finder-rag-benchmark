"""Publish concise English tables and charts; replayable from exported CSVs."""
from pathlib import Path
import argparse
import json
import os
import shutil
import sqlite3

os.environ.setdefault('MPLCONFIGDIR','/tmp/meeting11-report-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT.parent
OUT=ROOT/'results/person3_meeting11_legal/v1'
EXPORT=BASE/'outputs/meeting11_20261123'
LABELS={'question':'Original question','default':'Default MultiQuery','legal':'Legal MultiQuery'}
COLORS={'question':'#8798A5','default':'#126B84','legal':'#CF8C38'}
BACKENDS=['bm25','dense','hybrid']


def read(p):return json.loads(Path(p).read_text())
def csv(p):return pd.read_csv(p,dtype={'query_id':str})
def md(headers,rows):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in rows)
def save(fig,output,name):
    fig.savefig(output/(name+'.png'),dpi=200,bbox_inches='tight',pad_inches=.12)
    fig.savefig(output/(name+'.pdf'),bbox_inches='tight',pad_inches=.12)
    plt.close(fig)


def diagnostic(source):
    split=read(source/'selection.json');records=[]
    for variant in ['default','legal']:
        routes=read(source/f'{variant}_routes.json')
        for qid in split['test_ids']:
            item=routes[qid];gold=item['gold']
            for backend in BACKENDS:
                lists=[r[backend][:5] for r in item['routes']]
                ranks=read(source/'rankings.json')[qid][variant+'_'+backend]['5']
                records.append(dict(query_id=qid,prompt=variant,backend=backend,
                    original_hit5=int(gold in lists[-1]),any_route_hit5=int(any(gold in ids for ids in lists)),
                    fused_hit5=int(gold in ranks),query_count=len(lists),candidate_union=len(set(i for ids in lists for i in ids)),
                    found_but_lost=int(any(gold in ids for ids in lists) and gold not in ranks)))
    pd.DataFrame(records).to_csv(source/'query_fusion_per_query.csv',index=False)
    pd.DataFrame(records).groupby(['prompt','backend'],as_index=False).agg(
        n=('query_id','size'),question_hit5=('original_hit5','mean'),any_route_hit5=('any_route_hit5','mean'),
        fused_hit5=('fused_hit5','mean'),found_but_lost_queries=('found_but_lost','sum'),
        mean_candidate_union=('candidate_union','mean')).to_csv(source/'query_fusion_diagnostics.csv',index=False)
    costs=[]
    for variant,path in [('default',ROOT/'results/person3_meeting10_legal/v1/api/api_checkpoint.sqlite'),('legal',source/'api/api_checkpoint.sqlite')]:
        c=sqlite3.connect(f'file:{path}?mode=ro',uri=True)
        stage='multiquery_generation' if variant=='default' else 'legal_prompt_generation'
        rows=c.execute('SELECT cost,latency_seconds,usage_json FROM calls WHERE state=? AND created_stage=?',('saved',stage)).fetchall();c.close()
        assert len(rows)==100
        usage=[json.loads(r[2]) for r in rows]
        costs.append(dict(prompt=variant,n=100,fresh_generation_usd_per_query=sum(r[0] for r in rows)/100,
            generation_latency_p50_seconds=float(np.median([r[1] for r in rows])),
            mean_input_tokens=float(np.mean([u['prompt_tokens'] for u in usage])),
            mean_output_tokens=float(np.mean([u['completion_tokens'] for u in usage])),
            newly_incurred_usd=sum(r[0] for r in rows) if variant=='legal' else 0.,
            note='API generation latency only; includes historical measurement for default; not end-to-end RAG latency'))
    pd.DataFrame(costs).to_csv(source/'generation_costs.csv',index=False)


def figures(source,output):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'axes.axisbelow':True,'grid.color':'#E6E9ED','pdf.fonttype':42})
    data=csv(source/'retrieval_summary.csv').query('cohort=="test80"')
    test=data.query('k==5').set_index('method')
    fig,ax=plt.subplots(figsize=(7.4,4.3));fig.subplots_adjust(top=.76,bottom=.20)
    fig.suptitle('Legal: MultiQuery helps BM25 most at Top 5',x=.125,y=.99,ha='left',fontsize=13,weight='bold')
    fig.text(.125,.915,'Same 80 questions | MiniLM fixed | No new reader evaluation',fontsize=10)
    x=np.arange(3);width=.23
    for j,variant in enumerate(LABELS):
        vals=[test.loc[variant+'_'+b,'hit'] for b in BACKENDS]
        bars=ax.bar(x+(j-1)*width,vals,width,color=COLORS[variant],label=LABELS[variant])
        ax.bar_label(bars,labels=[f'{100*v:g}%' for v in vals],padding=4,fontsize=9)
    ax.set(xticks=x,xticklabels=['BM25','Dense','Hybrid'],ylim=(0,.58),ylabel='Hit@5')
    ax.yaxis.set_major_formatter(PercentFormatter(1));ax.grid(axis='y')
    ax.legend(loc='lower center',bbox_to_anchor=(.5,1.02),frameon=False,ncol=3,fontsize=9)
    fig.text(.125,.045,'No primary contrast passes the prespecified Holm9 correction.',fontsize=9)
    save(fig,output,'backend_comparison')
    fig,axes=plt.subplots(1,3,figsize=(10.2,3.9),sharey=True);fig.subplots_adjust(top=.73,bottom=.23,wspace=.16)
    fig.suptitle('Legal: changing the prompt does not help consistently across k',x=.125,y=.99,ha='left',fontsize=13,weight='bold')
    for ax,backend in zip(axes,BACKENDS):
        for variant in LABELS:
            part=data[data.method==variant+'_'+backend].sort_values('k')
            ax.plot(part.k,part.hit,marker={'question':'o','default':'s','legal':'^'}[variant],
                color=COLORS[variant],label=LABELS[variant],lw=1.8)
        ax.set(xticks=[5,8,10],ylim=(0,.60),xlabel='Top k');ax.set_title(backend.upper() if backend=='bm25' else backend.title(),fontsize=11)
        ax.grid(axis='y');ax.yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].set_ylabel('Hit@k')
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,ncol=3,loc='upper center',bbox_to_anchor=(.55,.90),frameon=False)
    fig.text(.125,.045,'80 evaluation questions. Query rankings are fused separately at each k; depths 8/10 are descriptive.',fontsize=9)
    save(fig,output,'retrieval_depth')
    d=csv(source/'delivery_summary.csv').query('cohort=="test80"')
    methods=[v+'_'+b for b in BACKENDS for v in LABELS]
    labels=[LABELS[v]+' | '+('BM25' if b=='bm25' else b) for b in BACKENDS for v in LABELS]
    cols=[(k,b) for k in [5,8,10] for b in [5000,8000,10000]]
    pivot=d.set_index(['method','k','budget'])
    matrix=np.array([[pivot.loc[(m,k,b),'full_hit'] for k,b in cols] for m in methods])
    fig,ax=plt.subplots(figsize=(9,5.5));fig.subplots_adjust(left=.31,right=.91,top=.81,bottom=.16)
    fig.suptitle('Legal: more context budget adds no labeled-evidence hits',x=.02,y=.99,ha='left',fontsize=13,weight='bold')
    fig.text(.02,.925,'80 questions per method | Cells: fully delivered labeled-passage Hit@k',fontsize=10)
    im=ax.imshow(matrix,cmap='Blues',vmin=0,vmax=.60,aspect='auto')
    ax.set(yticks=range(9),yticklabels=labels,xticks=range(9),xticklabels=[f'k={k}\n{b//1000}k' for k,b in cols])
    ax.tick_params(axis='both',length=0,labelsize=9)
    for i in range(9):
        for j in range(9):ax.text(j,i,f'{100*matrix[i,j]:g}%',ha='center',va='center',fontsize=8,color='white' if matrix[i,j]>.36 else '#20313F')
    for pos in [2.5,5.5]:ax.axvline(pos,color='white',lw=3)
    cb=fig.colorbar(im,ax=ax,fraction=.035,pad=.025);cb.ax.yaxis.set_major_formatter(PercentFormatter(1))
    fig.text(.02,.065,'At Top 10, 719/720 method-question lists fit fully in 5k tokens; the one partial list changes no hit.',fontsize=9)
    fig.text(.02,.025,'Exact o200k_base evidence tokens include labels/separators. Minimum tested budget is not a proven minimum.',fontsize=9)
    save(fig,output,'context_budget')


def report(source,output):
    source=source.resolve();output=output.resolve();output.mkdir(parents=True,exist_ok=True)
    if (source/'default_routes.json').exists():diagnostic(source)
    keep=['retrieval_summary.csv','retrieval_per_query.csv','primary_tests.csv','fixed100_summary.csv',
        'delivery_summary.csv','delivery_per_query.csv','query_fusion_diagnostics.csv','query_fusion_per_query.csv',
        'generation_costs.csv','protocol.json','selection.json','validation.json','default_replay.json','legal_replay.json',
        'generation_validation.json','tokenizer.json','ranking_status.json','source_hashes.json','implementation.json','freeze_receipt.json',
        'ranking_latency_samples.csv','ranking_latency_summary.csv','ranking_latency_protocol.json']
    if source!=output:
        for name in keep:shutil.copy2(source/name,output/name)
        shutil.copy2(source/'api/cost_summary.json',output/'cost_summary.json')
        shutil.copy2(Path(__file__).with_name('README.md'),output/'CODE_README.md')
    validation=read(output/'validation.json');assert validation['passed']
    df=csv(output/'retrieval_summary.csv').query('cohort=="test80"').set_index(['method','k'])
    rows=[]
    for backend in BACKENDS:
        for variant in LABELS:
            method=variant+'_'+backend
            rows.append([backend.upper() if backend=='bm25' else backend.title(),LABELS[variant]]+
                [f"{df.loc[(method,k),'hit']:.2%}" for k in [5,8,10]])
    table=md(['Backend','Query','Hit@5','Hit@8','Hit@10'],rows)
    text=f'''# Legal MultiQuery with dense and hybrid retrieval

23 November 2026 | Cheryl | Meeting 11

**Main finding:** MultiQuery helps BM25 most. The legal prompt gives no consistent advantage; larger context budgets add no labeled-passage hits.

Same 80 evaluation questions, 20 development questions and 4,876 passages; unchanged passage-grouped split. Model: `gpt-4o-mini-2024-07-18`; dense encoder: pinned MiniLM-256.

## 1. Retrieval results

{table}
Recall equals Hit here because each question has one labeled passage. Other ranking metrics are in [the full table](retrieval_summary.csv).

![Matched backends](backend_comparison.png)

## 2. Default versus legal prompt

Both prompts generate three alternatives and retain the original question. The legal version preserves stated facts, jurisdiction, procedural stage and exceptions, while prohibiting invented legal details. It was manually written and frozen before new outcomes, not DSPy-optimized. [Exact prompts](protocol.json)

The legal prompt ties default BM25 at Top 5 and modestly improves dense retrieval, but hybrid results vary with k. It does not establish a better general prompt.

![Retrieval depth](retrieval_depth.png)

## 3. Context budget

All nine methods were tested at Top-5/8/10 × 5k/8k/10k tokens: **6,480 evaluation records**, plus 1,620 development records. Packing follows rank order and truncates only the final passage; exact evidence tokens include labels and separators.

At Top 5/8, all lists fit in 5k. At Top 10, 719/720 method-question lists fit fully; one needs 5,010 tokens. Completing it changes no hit. Increasing the budget leaves every method's delivered Hit/Recall, MRR and nDCG unchanged. A smaller minimum budget was not tested.

![Evidence budget grid](context_budget.png)

## 4. Statistics and interpretation

**None of nine primary Hit@5 contrasts passes Holm9.** The earlier BM25 raw p remains 0.02148; its adjusted p is 0.19336 in this nine-test family. Top-8/10 and budget comparisons are descriptive. [All paired tests](primary_tests.csv)

Query fusion preserves the earlier per-k RRF60 rule; each hybrid query first fuses BM25/dense top100 at equal weights. Alternative candidate-depth fusion changes results and remains a [secondary diagnostic](fixed100_summary.csv). The [candidate-union analysis](query_fusion_diagnostics.csv) shows evidence can be found by individual queries but lost during fusion; that larger union is not an equal-budget competitor.

Results remain exploratory on previously inspected questions, with shared-source dependence unadjusted. No new reader answers were evaluated; retrieval gains do not establish answer correctness.

[Validation](validation.json) · [Code and resume commands](CODE_README.md)
'''
    (output/'README.md').write_text(text)
    figures(output,output)
    if source!=output:shutil.copy2(Path(__file__),output/'reproduce.py')
    print('Report written:',output,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=OUT);p.add_argument('--output',type=Path,default=EXPORT)
    a=p.parse_args();report(a.input,a.output)
