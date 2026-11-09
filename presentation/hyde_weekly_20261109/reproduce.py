"""Reproduce fixed primary tests and three charts from the accompanying CSVs."""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR',str(Path('/tmp')/'hyde-weekly-matplotlib'))
import numpy as np
import pandas as pd
from scipy.stats import binomtest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
D=Path(__file__).resolve().parent
wide=pd.read_csv(D/'fiqa_ndcg10_paired.csv',dtype={'query_id':str}).set_index('query_id')
tests=pd.read_csv(D/'fiqa_primary_tests.csv')
assert len(wide)==628 and len(tests)==12
computed=[]
for row in tests.to_dict('records'):
 delta=(wide[row['candidate']]-wide[row['control']]).to_numpy()
 assert abs(delta.mean()-row['mean_delta'])<1e-12
 rng=np.random.default_rng(20261109);count=0;observed=abs(delta.sum())
 for _ in range(100):
  signs=rng.integers(0,2,size=(1000,len(delta)),dtype=np.int8)*2-1
  count+=int((np.abs(signs@delta)>=observed-1e-12).sum())
 computed.append((count+1)/100001)
assert np.allclose(computed,tests.p_raw,atol=1e-12,rtol=0)
order=np.argsort(computed,kind='stable');adjusted=np.minimum(1,np.maximum.accumulate((12-np.arange(12))*np.asarray(computed)[order]))
assert np.allclose(adjusted,tests.p_holm.to_numpy()[order],atol=1e-12,rtol=0)
for name in ['fiqa_sparse_tests.csv','legal_sparse_tests.csv','fiqa_secondary_depth_tests.csv']:
 frame=pd.read_csv(D/name);n=len(frame);order=np.argsort(frame.p_raw.to_numpy(),kind='stable')
 expected=np.minimum(1,np.maximum.accumulate((n-np.arange(n))*frame.p_raw.to_numpy()[order]))
 assert np.allclose(expected,frame.p_holm.to_numpy()[order],atol=1e-12,rtol=0)
 if name=='legal_sparse_tests.csv':
  for row in frame.to_dict('records'):
   n=int(row['wins']+row['losses']);v=binomtest(int(row['wins']),n,.5).pvalue if n else 1.
   assert abs(v-row['p_raw'])<1e-12
s=pd.read_csv(D/'fiqa_628_metrics.csv')
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
fig,ax=plt.subplots(figsize=(8,4.2))
fig.subplots_adjust(left=.29,right=.98,bottom=.24,top=.85)
methods=['question_dense','focused160_mix_dense','short_30_50_mix_dense','answer_first_70_100_mix_dense'];labels=['Question only','Original focused mix','Short mix','Answer-first mix'];vals=[float(s[(s.method==m)&(s.k==10)].ndcg.iloc[0]) for m in methods]
bars=ax.barh(labels,vals,color=['#9AA5B1','#7A96AD','#167D8D','#B4C6CE']);ax.invert_yaxis();ax.set(xlim=(0,.46),xlabel='nDCG@10',title='FiQA: Short + question mixing improves retrieval')
for b,v in zip(bars,vals):ax.text(v+.005,b.get_y()+b.get_height()/2,f'{v:.4f}',va='center')
fig.text(.02,.025,'628 queries; 50/50 normalized embedding mix. Short vs question: Holm p=0.0048; vs focused mix: p=0.0166.',fontsize=9)
fig.savefig(D/'fiqa_main.png',dpi=180);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
for ax,names,values,title in [(axes[0],['Question','HyDE','Question + HyDE'],[.31,.50,.49],'Legal n=100: focused HyDE / Hit@5'),(axes[1],['Question','Short HyDE','Question + Short'],[.226836,.191790,.196507],'FiQA n=628: Short HyDE / nDCG@10')]:
 bars=ax.bar(names,values,color=['#9AA5B1','#167D8D','#7A96AD']);ax.set(title=title,ylim=(0,.6 if ax==axes[0] else .3));ax.tick_params(axis='x',labelsize=9)
 for b,v in zip(bars,values):ax.text(b.get_x()+b.get_width()/2,v+.008,f'{v:.3f}',ha='center')
fig.suptitle('Generated text is not universally a better BM25 query');fig.savefig(D/'sparse_transfer.png',dpi=180);plt.close(fig)
b=pd.read_csv(D/'context_budget_metrics.csv');fig,axes=plt.subplots(1,3,figsize=(11,3.7),layout='constrained')
for ax,budget in zip(axes,[2000,4000,8000]):
 for m,label,c in [('question_dense','Question','#9AA5B1'),('focused160_mix_dense','Focused mix','#CF8F35'),('short_30_50_mix_dense','Short mix','#167D8D')]:
  x=b[(b.method==m)&(b.budget==budget)].sort_values('k');ax.plot(x.k,x.full_recall,marker='o',label=label,color=c)
 ax.set(title=f'{budget:,} token budget',xlabel='Top-k',ylim=(0,.65),xticks=[5,10,15,20,25]);ax.grid(alpha=.15)
axes[0].set_ylabel('Full relevant-document Recall');axes[0].legend(fontsize=8);fig.suptitle('FiQA: depth and context budget impose different limits')
fig.get_layout_engine().set(rect=(0,.10,1,.90))
fig.text(.01,.015,'MiniLM WordPiece tokens; document text only. Document-level qrels, not answer-span coverage. Descriptive analysis.',fontsize=9)
fig.savefig(D/'depth_budget.png',dpi=180);plt.close(fig)
print('Verified 12 primary tests and Holm corrections; regenerated three charts.')
