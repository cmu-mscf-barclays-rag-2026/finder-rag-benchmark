"""Regenerate publication charts from committed metric CSVs. No API calls."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
ROOT=Path(__file__).resolve().parents[1]
L=ROOT/'results/person3_openai_hyde'
C=ROOT/'results/person3_cross_dataset/gpt54_focused160_v1'
OUT=ROOT/'presentation/openai_hyde'; OUT.mkdir(parents=True,exist_ok=True)
def rows(p):
 with p.open() as f:return list(csv.DictReader(f))
legal={r['method']:float(r['hit']) for r in rows(L/'comparison_k5.csv')}
fiqa={r['method']:float(r['ndcg']) for r in rows(C/'fiqa_metrics.csv') if r['k']=='10'}
lb={r['method']:float(r['accuracy']) for r in rows(C/'longbench2_accuracy.csv') if r['group_by']=='overall'}
methods=['question_dense','bm25','standard_hybrid_d25','standard_hybrid_d50','focused160_pure_dense','focused160_mix_dense','focused160_pure_hybrid_d25','focused160_mix_hybrid_d25','focused160_mix_hybrid_d50']
labels=['Question dense','BM25','Standard hybrid · d25','Standard hybrid · d50','Pure HyDE dense','Question + HyDE dense','Pure HyDE hybrid · d25','Mixed HyDE hybrid · d25','Mixed HyDE hybrid · d50']
colors=['#52657b']*4+['#157f82']*5
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False})
fig,axes=plt.subplots(1,3,figsize=(16,6.4),sharey=True)
for ax,data,title,metric in zip(axes,[legal,fiqa,lb],['Legal · 100 questions','FiQA · 648 questions','LongBench v2 · 503 questions'],['Hit@5','nDCG@10','Answer accuracy']):
 vals=[data[m] for m in methods]
 ax.barh(range(9),vals,color=colors,height=.64)
 ax.set_title(title,fontweight='bold',pad=17)
 ax.set_xlabel(metric+'  ↑',labelpad=12)
 ax.set_xlim(0,.58 if data is not fiqa else .46)
 ax.set_yticks(range(9),labels);ax.tick_params(axis='y',length=0)
 ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
 if data is not fiqa:ax.xaxis.set_major_formatter(PercentFormatter(1,decimals=0))
 for i,v in enumerate(vals):ax.text(v+.008,i,f'{v:.4f}' if data is fiqa else f'{v:.1%}',va='center',fontsize=10,fontweight='bold' if v==max(vals) else 'normal')
 if data is lb:
  ax.axvline(lb['no_context'],color='#a24f30',ls='--',lw=1.4)
  ax.text(.02,-.95,f"No context: {lb['no_context']:.2%} (dashed)",fontsize=9,color='#a24f30')
axes[0].invert_yaxis()
fig.suptitle('GPT-5.4 HyDE: gains on legal questions did not transfer consistently',x=.02,ha='left',fontweight='bold',fontsize=18,y=.99)
fig.text(.02,.03,'Gray = original-question controls   •   Teal = focused HyDE   •   d25 / d50 = dense RRF weight 25% / 50%\nMetrics differ by dataset; compare methods within each panel. Bars are descriptive, not significance tests.',fontsize=10,color='#46566a')
fig.tight_layout(rect=(0,.10,1,.94))
for ext in ['svg','png']:fig.savefig(OUT/f'cross_dataset.{ext}',dpi=160,bbox_inches='tight',facecolor='white')
plt.close(fig)
fig,ax=plt.subplots(figsize=(10,4.5))
ms=['question_dense','original96_pure_dense','original96_mix_dense','focused160_pure_dense','focused160_mix_dense']
# Method names are validated against the recorded CSV.
assert all(m in legal for m in ms),list(legal)
vs=[legal[m] for m in ms]
ax.barh(range(5),vs,color=['#52657b','#82aeaf','#82aeaf','#157f82','#157f82'])
ax.set_yticks(range(5),['Question dense','Original / 96 · pure','Original / 96 · mixed','Focused / 160 · pure','Focused / 160 · mixed']); ax.invert_yaxis()
for i,v in enumerate(vs):ax.text(v+.008,i,f'{v:.0%}',va='center',fontweight='bold')
ax.set_xlim(0,.6);ax.xaxis.set_major_formatter(PercentFormatter(1,decimals=0));ax.set_xlabel('Legal Hit@5');ax.set_title('Prompt + output budget: original96 → focused160',loc='left',fontweight='bold',pad=20)
ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
fig.text(.02,.015,'Length-capped outputs: original96 = 94/100; focused160 = 0/100. Prompt and length changed together.',fontsize=10)
fig.tight_layout(rect=(0,.07,1,1))
for ext in ['svg','png']:fig.savefig(OUT/f'legal_prompt.{ext}',dpi=160,bbox_inches='tight',facecolor='white')
plt.close(fig)
