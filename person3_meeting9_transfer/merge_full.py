"""Combine completed disjoint frozen stages; full309 results are descriptive only."""
from pathlib import Path
import json,os,hashlib,shutil
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'results/person3_meeting9_transfer';A=BASE/'vidore_finance_v1';B=BASE/'vidore_finance_remaining249_v1';OUT=BASE/'vidore_finance_full309_summary_v1';EXPORT=ROOT.parent/'outputs/meeting9_20261109/transfer/full309'
def read(p):return json.loads(p.read_text())
def main():
 OUT.mkdir(parents=True,exist_ok=True);EXPORT.mkdir(parents=True,exist_ok=True)
 for p in [A,B]:assert read(p/'completion_status.json')['complete'] and read(p/'validation.json')['passed']
 sa=read(A/'selection.json')['queries'];sb=read(B/'selection.json')['queries'];idsa={q['query_id'] for q in sa};idsb={q['query_id'] for q in sb};assert len(idsa)==60 and len(idsb)==249 and not idsa&idsb
 original=pd.read_parquet(A/'data/queries.parquet');full={str(x) for x in original[original.language=='english'].query_id};assert idsa|idsb==full and len(full)==309
 (OUT/'selection.json').write_text(json.dumps({'n':309,'queries':sa+sb,'stages':{'initial60':60,'remaining249':249},'order':'concatenated original frozen60 and remaining249 query order; equivalent to complete hash ordering'},ensure_ascii=False,indent=2))
 (OUT/'qrels.json').write_text(json.dumps({**read(A/'qrels.json'),**read(B/'qrels.json')},indent=2))
 provenance=read(B/'stage_provenance.json');assert provenance['prior60_hyde_evaluated_at_freeze'] is False and provenance['full309_merge']=='descriptive only, no new significance family'
 frames=[pd.read_csv(p/'per_query_metrics.csv',dtype={'query_id':str}) for p in [A,B]];f=pd.concat(frames,ignore_index=True);assert len(f)==309*6*5 and not f.duplicated(['query_id','method','k']).any();f.to_csv(OUT/'per_query_metrics.csv',index=False)
 summary=f.groupby(['method','k']).agg(n=('query_id','size'),**{m:(m,'mean') for m in ['hit','recall','precision','mrr','ndcg']}).reset_index();summary.to_csv(OUT/'summary.csv',index=False)
 tests=pd.concat([pd.read_csv(p/'primary_tests.csv').assign(stage=stage) for p,stage in [(A,'initial60'),(B,'remaining249')]],ignore_index=True);tests.to_csv(OUT/'separate_primary_families.csv',index=False)
 za=np.load(A/'rankings.npz');zb=np.load(B/'rankings.npz');assert np.array_equal(za['passage_ids'],zb['passage_ids']);methods=[k for k in za.files if k not in ['query_ids','passage_ids']];np.savez_compressed(OUT/'rankings.npz',query_ids=np.concatenate([za['query_ids'],zb['query_ids']]),passage_ids=za['passage_ids'],**{m:np.concatenate([za[m],zb[m]]) for m in methods})
 audit=pd.concat([pd.read_csv(p/'generation_audit.csv',dtype={'query_id':str}) for p in [A,B]],ignore_index=True);assert len(audit)==618;audit.to_csv(OUT/'generation_audit.csv',index=False)
 validation={'passed':True,'queries':309,'all_english_queries_covered':True,'query_sets_disjoint':True,'methods':6,'metric_rows':len(f),'successful_generations':618,'original60_unchanged':True,'remaining249_frozen_before60_hyde_results':True,'full309_significance_tests':0,'primary_test_families':['initial60 Holm4','remaining249 Holm4'],'visual_model_evaluated':False,'answer_reader_evaluated':False,'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for d in [A,B] for p in [d/'protocol.json',d/'selection.json',d/'validation.json',d/'summary.csv',d/'primary_tests.csv']}}
 (OUT/'validation.json').write_text(json.dumps(validation,indent=2));(OUT/'completion_status.json').write_text(json.dumps({'complete':True,**validation},indent=2))
 labels={'question_dense':'Question dense','bm25':'BM25','focused160_pure_dense':'Focused: pure HyDE','focused160_mix_dense':'Focused: 50/50 mix','short_30_50_pure_dense':'Short: pure HyDE','short_30_50_mix_dense':'Short: 50/50 mix'};order=list(labels);cols=['#8b929a','#b4bbc3','#a6c8de','#5395c4','#94d2b3','#27956c'];v=summary[summary.k==10].set_index('method').loc[order]
 os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'));import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,2,figsize=(11,4.8))
 for ax,col,title in [(axes[0],'ndcg','nDCG@10'),(axes[1],'hit','Hit@10')]:
  ax.barh([labels[m] for m in order],v[col],color=cols);ax.invert_yaxis();ax.set_xlim(0,v[col].max()*1.23);ax.set_title(title);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
  for i,value in enumerate(v[col]):ax.text(value+.003,i,f'{value:.3f}',va='center',fontsize=9)
 fig.suptitle('ViDoRe V3 Finance EN: all 309 English queries / 2,942 pages');fig.tight_layout(rect=[0,.07,1,.94]);fig.text(.5,.02,'Descriptive full-split aggregate. Text-only page OCR; dense input capped at 256 MiniLM tokens.',ha='center',fontsize=9);fig.savefig(EXPORT/'transfer_full309.png',dpi=180);fig.savefig(EXPORT/'transfer_full309.svg');plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(9,4.5))
 for ax,col,title in [(axes[0],'hit','Hit@k'),(axes[1],'recall','Recall@k')]:
  for m,c in [('question_dense','#8b929a'),('bm25','#454c55'),('focused160_mix_dense','#5395c4'),('short_30_50_mix_dense','#27956c')]:
   g=summary[summary.method==m].sort_values('k');ax.plot(g.k,g[col],marker='o',label=labels[m],color=c)
  ax.set_title(title);ax.set_xlabel('Retrieval depth k');ax.set_xticks([5,10,15,20,25]);ax.set_ylim(0,1);ax.grid(alpha=.15)
 handles,ll=axes[0].get_legend_handles_labels();fig.legend(handles,ll,loc='lower center',ncol=2,fontsize=9);fig.suptitle('ViDoRe Finance: retrieval depth (309 queries, descriptive)');fig.tight_layout(rect=[0,.13,1,.94]);fig.savefig(EXPORT/'transfer_depth309.png',dpi=180);fig.savefig(EXPORT/'transfer_depth309.svg');plt.close(fig)
 lines=['# HyDE transfer: complete English split','','All 309 English queries search all 2,942 supplied page OCR documents. Two frozen stages: initial 60 queries and remaining 249; the latter was frozen before the 60-query HyDE retrieval results were read. No prompt changes or significance stopping. Full309 aggregates below are descriptive.','','| Method | nDCG@10 | Hit@5 | Hit@10 | Recall@10 | MRR@10 |','|---|---:|---:|---:|---:|---:|']
 for m in order:
  g=summary[summary.method==m].set_index('k');lines.append(f'| {labels[m]} | {g.loc[10,"ndcg"]:.4f} | {g.loc[5,"hit"]:.1%} | {g.loc[10,"hit"]:.1%} | {g.loc[10,"recall"]:.4f} | {g.loc[10,"mrr"]:.4f} |')
 lines+=['','![Full English split](transfer_full309.png)','','| Method | Hit@5 | Hit@10 | Hit@15 | Hit@20 | Hit@25 |','|---|---:|---:|---:|---:|---:|']
 for m in order:
  g=summary[summary.method==m].set_index('k');lines.append('| '+labels[m]+' | '+' | '.join(f'{g.loc[k,"hit"]:.1%}' for k in [5,10,15,20,25])+' |')
 lines+=['','![Depth sweep](transfer_depth309.png)','','## Separate predeclared primary families','','| Stage | Comparison | Delta nDCG@10 | Holm p (4 tests within stage) |','|---|---|---:|---:|']
 for a in tests.to_dict('records'):lines.append(f'| {a["stage"]} | {labels[a["candidate"]]} vs {labels[a["control"]]} | {a["mean_delta"]:+.4f} | {a["p_holm"]:.5f} |')
 lines+=['','Each stage uses question-level two-sided paired sign-flip tests (100,000 draws, seed 20261109), Holm-adjusted over its own four planned tests. No new p-values are computed on the pooled 309 queries. Shared annual-report sources are not cluster-adjusted.','','The fixed MiniLM encoder keeps only the first 256 tokens of page OCR; 2,764/2,942 pages (93.95%) exceed that limit. BM25 uses full OCR. Comparisons within the dense route are matched; cross-route comparisons also reflect unequal input coverage. No image encoder, final-answer reader or fact-level evidence-sufficiency benchmark is evaluated here.','','## Reproduce from cached outputs','','```bash','.venv/bin/python person3_meeting9_transfer/run.py evaluate','.venv/bin/python person3_meeting9_transfer/validate.py','.venv/bin/python person3_meeting9_transfer/run.py evaluate --cohort remaining249','.venv/bin/python person3_meeting9_transfer/validate.py --cohort remaining249','.venv/bin/python person3_meeting9_transfer/merge_full.py','```','','No API calls occur during reproduction. Pinned corpus vectors and successful generation outputs must be retained. Source dataset: [ViDoRe V3 Finance EN](https://huggingface.co/datasets/vidore/vidore_v3_finance_en), revision 7f432c176d82e27546501ad8064a713ac3071809, CC-BY-4.0.']
 text='\n'.join(lines)+'\n';(OUT/'findings.md').write_text(text);(EXPORT/'README.md').write_text(text)
 for name in ['summary.csv','per_query_metrics.csv','separate_primary_families.csv','generation_audit.csv','validation.json','completion_status.json','selection.json']:shutil.copy2(OUT/name,EXPORT/name)
 print('Merged all309queries/618outputs; descriptive aggregate only',flush=True)
if __name__=='__main__':main()
