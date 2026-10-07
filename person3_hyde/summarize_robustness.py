"""Create a short report from the saved sweep; never calls an LLM."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
OUT=Path(__file__).resolve().parents[1]/'results/person3_hyde_robustness'
def main():
    df=pd.read_csv(OUT/'summary_k5.csv'); pairs=pd.read_csv(OUT/'paired_comparisons_k5.csv')
    # Holm adjustment across the entire exploratory comparison family.
    order=np.argsort(pairs.mcnemar_exact_p.to_numpy()); adjusted=np.maximum.accumulate((len(pairs)-np.arange(len(pairs)))*pairs.mcnemar_exact_p.to_numpy()[order])
    vals=np.zeros(len(pairs)); vals[order]=np.minimum(1,adjusted); pairs['holm_adjusted_p']=vals; pairs.to_csv(OUT/'paired_comparisons_k5.csv',index=False)
    base=df[df.config.isin(['baseline','oracle_only'])]
    fixed=df[(df.config!='baseline')&(df.config!='oracle_only')&((df.dense_weight.isna())|((df.dense_weight==.25)&(df.candidate_depth==100)))]
    fixed.to_csv(OUT/'fixed_weight_comparison_k5.csv',index=False)
    fixed[fixed.dense_weight.isna()].pivot(index='config',columns='representation',values='recall_at_k').to_csv(OUT/'generation_comparison_k5.csv')
    seeds=fixed[fixed.config.str.contains('evidence_passage_len96_t0.7')].groupby('representation').agg(hit5_mean=('recall_at_k','mean'),hit5_min=('recall_at_k','min'),hit5_max=('recall_at_k','max'),n_seeds=('method_id','count')); seeds.to_csv(OUT/'seed_robustness_k5.csv')
    audit=pd.read_csv(OUT/'generation_audit.csv'); audit.assign(very_short=audit.word_count.le(3)).groupby('config').agg(mean_words=('word_count','mean'),mean_tokens=('embedding_tokens','mean'),truncation_rate=('truncated_by_minilm','mean'),empty_rate=('empty','mean'),three_words_or_less_rate=('very_short','mean')).to_csv(OUT/'generation_audit_summary.csv')
    # Descriptive paired examples for the strongest observed fixed-weight configuration.
    cand=fixed[fixed.dense_weight==.25].sort_values(['recall_at_k','mrr_at_k'],ascending=False).iloc[0]
    details=pd.read_csv(OUT/'per_query_results.csv'); queries=pd.read_csv(OUT/'queries.csv')
    a=details[details.method_id==cand.method_id].set_index('query_index'); b=details[details.method_id=='standard_hybrid'].set_index('query_index')
    example=[]
    for i in range(len(queries)):
        ar=int(a.loc[i,'gold_rank']); br=int(b.loc[i,'gold_rank']); ah=0<ar<=5; bh=0<br<=5
        if ah != bh: example.append(queries.iloc[i].to_dict()|{'candidate':cand.method_id,'candidate_gold_rank':ar,'standard_hybrid_gold_rank':br,'category':'helps' if ah else 'hurts'})
    pd.DataFrame(example).to_csv(OUT/'help_hurt_examples.csv',index=False)
    def table(frame):
        return '\n'.join(f'| {r.method_id} | {r.recall_at_k:.2f} | {r.mrr_at_k:.4f} | {r.ndcg_at_k:.4f} |' for r in frame.itertuples())
    bestdense=fixed[fixed.dense_weight.isna()].sort_values(['recall_at_k','mrr_at_k'],ascending=False).iloc[0]
    paired=pairs[pairs.method_id==cand.method_id].iloc[0]
    q=float(base[base.method_id=='question_dense'].recall_at_k.iloc[0]); oracle=float(base[base.method_id=='reference_answer_oracle'].recall_at_k.iloc[0]); standard=float(base[base.method_id=='standard_hybrid'].recall_at_k.iloc[0])
    control=base[base.method_id=='standard_hybrid'].iloc[0]
    presentation=[]
    for r in base.to_dict('records'):
        presentation.append(r|{'display_name':r['method_id'],'selection_status':'control' if r['config']=='baseline' else 'oracle diagnostic'})
    for rep in ('hyde_single','question_plus_hyde','multi_average','multi_rrf'):
        r=fixed[(fixed.representation==rep)&fixed.dense_weight.isna()].sort_values(['recall_at_k','mrr_at_k'],ascending=False).iloc[0].to_dict()
        presentation.append(r|{'display_name':rep,'selection_status':'observed sweep maximum; exploratory'})
    presentation.append(cand.to_dict()|{'display_name':'HyDE + BM25 hybrid','selection_status':'observed fixed-weight sweep maximum; exploratory'})
    pd.DataFrame(presentation).to_csv(OUT/'presentation_k5.csv',index=False)
    ranges=pd.read_csv(OUT/'robustness_by_representation.csv')
    report=f'''# Cheryl — question-only HyDE robustness (week of 6 October 2026)

## Protocol and limits

100 Legal RAG Bench questions; 4,876 passages; fixed MiniLM and FLAN-T5-base,
pinned model revisions. Generation sees only the original question. Eight prompt ×
length × decoding configurations plus two repeated stochastic seeds. Four dense
representations per configuration, plus BM25 fusion weights/depths. This yields
{len(df)} method/configuration rows at each K. The same 100 questions were inspected
previously: **all sweep results are exploratory, not new held-out test results**.
No Qwen or teammate dependency. Final answer correctness was not evaluated here.

## Recomputed controls

| Method | Hit/Recall@5 | MRR@5 | nDCG@5 |
|---|---:|---:|---:|
{table(base)}

## Formulation comparison

Read `generation_comparison_k5.csv` for one row per generation setup and columns for
single-HyDE, question+HyDE, average-embedding multi-HyDE, and multi-ranking RRF.
Read `fixed_weight_comparison_k5.csv` for their BM25-hybrid variants at dense weight
0.25 and candidate depth 100, the same control setting as last week.

The strongest **observed** dense representation was `{bestdense.method_id}`:
Hit@5 {bestdense.recall_at_k:.2f}, versus question dense {q:.2f}. Its descriptive
oracle-gap fraction is {bestdense.oracle_gap_closed:.1%} relative to the oracle
{oracle:.2f}. This is a sweep maximum, not an independently validated winner.

The strongest **observed fixed-weight** HyDE hybrid was `{cand.method_id}`:
Hit@5 {cand.recall_at_k:.2f}, versus standard hybrid {standard:.2f}. Paired gains/losses:
{int(paired.candidate_only_hits)} gained and {int(paired.baseline_only_hits)} lost;
unadjusted exact McNemar p={paired.mcnemar_exact_p:.4g}, Holm-adjusted p={paired.holm_adjusted_p:.4g}
across all {len(pairs)} exploratory comparisons. Do not claim significance from a
selected sweep maximum. Full parameter sweeps are in `summary_k5.csv`.

The higher Hit@5 is also a ranking tradeoff: the selected hybrid's MRR@5 is
{cand.mrr_at_k:.4f}, versus {control.mrr_at_k:.4f} for standard hybrid; nDCG@5 is
{cand.ndcg_at_k:.4f}, versus {control.ndcg_at_k:.4f}. More Top-5 hits therefore
does not establish better ranking quality. Read `presentation_k5.csv` for a
compact descriptive table; maxima are explicitly labelled exploratory.

## Stability

{ranges.to_csv(index=False)}

Seed checks for sampled evidence-passage generation (length cap 96):

{seeds.to_csv()}

These ranges cover this small local generator and this benchmark only. They do
not establish that HyDE broadly succeeds or fails with stronger generators.

## Timing and audits

`generation_audit_summary.csv` records empty outputs and generated-text truncation
under MiniLM's original token limit. `generation_configs.csv` records three-hypothesis
generation throughput. Timing columns are batched CPU estimates; single-generation
cost is approximated by dividing total three-hypothesis time by three and extra
question+HyDE encoding is not timed. They are not interactive latency measurements.

## Easy meeting notes

- I kept MiniLM fixed and tested question-only HyDE generation, so the embedding
  model stayed the same across all comparisons.
- I compared short answers, evidence-style passages, the question plus generated
  text, and three hypotheses combined in two ways.
- I also changed output length, decoding, fusion weights and candidate depth,
  and repeated one sampled setting with three seeds.
- Question dense retrieved the correct passage for {q*100:.0f} out of 100 questions;
  the reference-answer oracle retrieved it for {oracle*100:.0f}. The oracle uses the
  dataset answer and is only a diagnostic.
- The best observed dense setup reached {bestdense.recall_at_k*100:.0f} hits, and the
  best observed hybrid at the original weight reached {cand.recall_at_k*100:.0f},
  compared with {standard*100:.0f} for standard hybrid.
- These are exploratory results. I will show the full ranges and seed checks,
  rather than treating the highest score as a reliable final improvement.
- The next step is to freeze a configuration and validate it on new questions,
  then test a stronger generator if the local model remains the bottleneck.

## Files to open during the meeting

1. `generation_comparison_k5.csv` — prompt/length/decoding and formulation comparison.
2. `fixed_weight_comparison_k5.csv` — fair comparison at unchanged hybrid weight.
3. `seed_robustness_k5.csv` — consistency across random seeds.
4. `help_hurt_examples.csv` and `generated_*.csv` — concrete question examples.
5. `paired_comparisons_k5.csv` — paired tests and multiplicity adjustment.

Run `python person3_hyde/validate_robustness.py` to independently verify CSV metrics
with no model download. Model-free replay is documented in `ROBUSTNESS_README.md`.
'''
    (OUT/'findings.md').write_text(report)
    print(report[:3500])
if __name__=='__main__': main()
