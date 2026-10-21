"""Build compact tables from saved outputs only; never runs a model."""
import json,re
from pathlib import Path
import numpy as np,pandas as pd
P=Path(__file__).resolve().parents[1]/'results/person3_stronger_hyde'
LABELS={'question_dense':'Original-question dense','bm25':'BM25','standard_hybrid':'Standard hybrid','reference_oracle':'Reference-answer oracle (diagnostic)','flan_dense':'FLAN-T5-base HyDE dense (matched)','flan_hybrid':'FLAN-T5-base HyDE hybrid (matched)','qwen06_dense':'Qwen3-0.6B HyDE dense','qwen06_hybrid':'Qwen3-0.6B HyDE hybrid','qwen17_dense':'Qwen3-1.7B HyDE dense','qwen17_hybrid':'Qwen3-1.7B HyDE hybrid'}
def main():
 df=pd.read_csv(P/'comparison_k5.csv');pairs=pd.read_csv(P/'paired_tests.csv')
 # Exploratory Holm correction across the eight planned paired comparisons.
 order=np.argsort(pairs.mcnemar_exact_p.to_numpy());vals=np.maximum.accumulate((len(pairs)-np.arange(len(pairs)))*pairs.mcnemar_exact_p.to_numpy()[order]);adj=np.zeros(len(pairs));adj[order]=np.minimum(1,vals);pairs['holm_adjusted_p']=adj;pairs.to_csv(P/'paired_tests.csv',index=False)
 lat=pd.read_csv(P/'latency_per_query.csv');audit=pd.read_csv(P/'generation_audit.csv');cost=[]
 for key in ['qwen06','qwen17']:
  t=lat[lat.model==key];a=audit[audit.model==key]
  cost.append({'model':key,'n':len(t),'generation_median_s':t.generation_seconds.median(),'generation_p95_s':t.generation_seconds.quantile(.95),'generation_mean_s':t.generation_seconds.mean(),'retrieval_median_ms':t.retrieval_ms.median(),'mean_generated_words':a.words.mean(),'very_short_rate':a.very_short.mean(),'generation_capped_rate':a.generation_capped.mean(),'minilm_truncation_rate':a.minilm_truncated.mean()})
 pd.DataFrame(cost).to_csv(P/'quality_latency_summary.csv',index=False)
 answers=json.loads((P/'final_answers.json').read_text());review=pd.read_csv(P/'answer_reviews.csv');summary=[]
 for method,g in review.groupby('method',sort=False):
  subset=[r for r in answers.values() if r['method']==method];valid=[];has_cite=[]
  for a in subset:
   citations=[int(x) for x in re.findall(r'\[(\d{1,2})\]',a['text'])];labels={r['label'] for r in a['passages']};has_cite.append(bool(citations));valid.append(bool(citations) and all(x in labels for x in citations))
  summary.append({'method':method,'n':len(g),'fully_correct_rate':(g.correctness==2).mean(),'partly_or_fully_correct_rate':(g.correctness>=1).mean(),'mean_correctness_0_to_2':g.correctness.mean(),'fully_grounded_rate':(g.grounding==2).mean(),'citation_present_rate':np.mean(has_cite),'all_citations_valid_rate':np.mean(valid),'mean_answer_seconds':np.mean([r['seconds'] for r in subset]),'numerical_accuracy':'not_applicable','review_type':'model-assisted qualitative review; not independent expert annotation'})
 ans=pd.DataFrame(summary);ans.to_csv(P/'answer_summary.csv',index=False)
 rows='\n'.join(f'| {LABELS[r.method_id]} | {r.precision_at_k:.1%} | {r.hit_rate_at_k:.0%} | {r.mrr_at_k:.4f} | {r.ndcg_at_k:.4f} |' for r in df.itertuples())
 tests='\n'.join(f'| {LABELS[r.candidate]} vs {LABELS[r.baseline]} | {r.candidate_only_hits} | {r.baseline_only_hits} | {r.mcnemar_exact_p:.4f} | {r.holm_adjusted_p:.4f} |' for r in pairs.itertuples())
 timing='\n'.join(f'| {r["model"]} | {r["generation_median_s"]:.2f} | {r["generation_p95_s"]:.2f} | {r["mean_generated_words"]:.1f} | {r["generation_capped_rate"]:.0%} | {r["minilm_truncation_rate"]:.0%} |' for r in cost)
 ar='\n'.join(f'| {r.method} | {r.n} | {r.fully_correct_rate:.0%} | {r.partly_or_fully_correct_rate:.0%} | {r.fully_grounded_rate:.0%} | {r.all_citations_valid_rate:.0%} |' for r in ans.itertuples())
 ix=df.set_index('method_id')
 pair17=pairs[(pairs.candidate=='qwen17_dense')&(pairs.baseline=='flan_dense')].iloc[0]
 report=f'''# Cheryl — stronger HyDE generators

**Question:** Does replacing FLAN-T5-base with a newer local instruction model improve evidence retrieval and final answers?

100 previously inspected Legal RAG Bench questions;4,876 passages. MiniLM fixed; one hypothesis; identical evidence-passage prompt;96-token generation cap;greedy;BM25/dense weights75%/25%;RRF60;depth100. Qwen uses official non-thinking chat templates and CPU bfloat16. FLAN outputs are reused, not regenerated.

## 1. Retrieval comparison

| Method | Precision@5 | Hit/Recall@5 | MRR@5 | nDCG@5 |
|---|---:|---:|---:|---:|
{rows}

Matched FLAN means the same single-passage/96-token/greedy setup. The earlier39% multi-HyDE result used a different selected setup and is not a matched control. The72% oracle uses real reference answers and is diagnostic only.

**Finding:** Qwen3-1.7B raises matched HyDE dense Hit@5 from{ix.loc['flan_dense','hit_rate_at_k']:.0%} to{ix.loc['qwen17_dense','hit_rate_at_k']:.0%}, while question dense is{ix.loc['question_dense','hit_rate_at_k']:.0%}. Its hybrid Hit@5 is{ix.loc['qwen17_hybrid','hit_rate_at_k']:.0%} versus{ix.loc['standard_hybrid','hit_rate_at_k']:.0%} for standard hybrid; MRR@5 ({ix.loc['qwen17_hybrid','mrr_at_k']:.4f} vs{ix.loc['standard_hybrid','mrr_at_k']:.4f}) and nDCG@5 ({ix.loc['qwen17_hybrid','ndcg_at_k']:.4f} vs{ix.loc['standard_hybrid','ndcg_at_k']:.4f}) are numerically higher. This is a ranking/coverage tradeoff, not a clear overall win.

## 2. Paired evidence gains/losses

| Comparison | Gained | Lost | McNemar p | Holm-adjusted p |
|---|---:|---:|---:|---:|
{tests}

Eight planned comparisons; all exploratory. Higher Hit@5 alone does not establish higher ranking quality or general superiority.

## 3. Generation quality and cost

| Model | Median generation seconds/query | P95 seconds/query | Mean words | Reached96-token cap | Truncated by MiniLM |
|---|---:|---:|---:|---:|---:|
{timing}

Serial CPU query measurements exclude model loading, warm-up, downloads and corpus encoding. The two generation jobs shared the same CPU host, so these are indicative timings with contention, not isolated hardware benchmarks. FLAN's old batched timing is not directly comparable. More words and fewer short outputs are mechanical quality checks, not semantic correctness.

**Limit:** {cost[0]['generation_capped_rate']:.0%} of0.6B and{cost[1]['generation_capped_rate']:.0%} of1.7B outputs reached the96-token generation cap; none was truncated by MiniLM. The fixed generation budget may prevent complete hypotheses. Do not interpret these as unconstrained model-quality results.

## 4. Final-answer pilot

Fixed eight IDs selected by hash before the new outcomes; same Qwen3-0.6B answer model, prompt,128-token cap,top5 packing and2048 passage-token budget for all methods. One question is12.5 percentage points. These scores come from **model-assisted qualitative review**, not independent legal-expert annotation or an automatic truth metric.

| Retrieved context | Questions | Fully correct | At least partly correct | Fully grounded | Citations present and all labels valid |
|---|---:|---:|---:|---:|---:|
{ar}

Grounding is judged against the actual supplied context; a valid citation number does not prove support. A supported abstention can be grounded but not correct. No pilot question asks for a numerical calculation, so numerical accuracy is N/A. Gold-context answers are diagnostic.

## Conclusion and next step

Keep standard hybrid as the main control. Qwen3-1.7B is worth a targeted follow-up for ranking quality, but does not yet establish a retrieval improvement: its dense difference versus FLAN has unadjusted McNemar p={pair17.mcnemar_exact_p:.4f}; see the paired table for the other comparisons. No planned Hit@5 comparison is significant at5%, even before adjustment.

Next test a larger generation budget and a prompt that discourages invented legal citations, one change at a time. The answer pilot also requires a stronger or better-constrained answer generator and independent review before treating its scores as final answer accuracy. Freeze promising settings and evaluate on new questions. All current results are exploratory, not an independent held-out evaluation.

**Pilot finding:** The local answer generator remains unreliable even with gold-only context. For example, Q39 includes the correct alibi passage but hybrid answers miss the required application for leave; Q91 includes the case name but answers do not extract R v Donnelly; Q50's gold-context answer contradicts the explicit statement that Victoria has no standalone accident defence. These examples are diagnostic, not independent legal-expert judgments.

[Reviewed answers and rationale](answer_reviews.csv) · [Exact contexts and outputs](final_answers.csv) · [Full retrieval metrics](retrieval_metrics.csv) · [Per-query timing](latency_per_query.csv) · [Reproduction and rubric](../../person3_stronger_hyde/README.md)
'''
 (P/'findings.md').write_text(report)
 print(report)
if __name__=='__main__':main()
