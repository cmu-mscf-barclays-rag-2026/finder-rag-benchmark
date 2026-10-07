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
    labels = {
        'hyde_single': 'Single HyDE answer',
        'multi_average': 'Three HyDE answers: average embeddings',
        'multi_rrf': 'Three HyDE answers: fuse rankings (RRF)',
        'question_plus_hyde': 'Original question + one HyDE answer',
    }
    dense_rows = '\n'.join(
        f"| {labels[r.representation]} | {r.hit5_min:.0%}–{r.hit5_max:.0%} | {r.hit5_mean:.1%} | {r.hit5_max:.0%} |"
        for r in ranges.itertuples()
    )
    previous = pd.read_csv(OUT.parent/'person3_hyde'/'summary_k5.csv')
    previous_labels = {
        'Question BM25': 'BM25 baseline',
        'Question Dense MiniLM': 'Original-question dense baseline',
        'Question BM25 + Question Dense RRF': 'Standard hybrid: BM25 + question dense',
        'Reference Answer Dense (oracle)': 'Reference-answer dense — diagnostic only',
        'Question-only HyDE Dense': 'Single HyDE answer: dense',
        'Question BM25 + HyDE Dense RRF': 'HyDE hybrid: BM25 + HyDE dense',
    }
    previous_rows = '\n'.join(
        f"| {previous_labels[r.method]} | {r.precision_at_k:.1%} | {r.hit_rate_at_k:.0%} | {r.mrr_at_k:.4f} | {r.ndcg_at_k:.4f} |"
        for r in previous.itertuples()
    )
    report=f'''# Cheryl — HyDE generation and parameter robustness

**Goal:** Following last week's feedback, test whether better HyDE generation improves dense retrieval and, in turn, hybrid retrieval.

Legal RAG Bench: **100 questions / 4,876 passages**. Generator: **FLAN-T5-base**. Embeddings: **MiniLM-L6-v2**, fixed throughout. Generation uses only the question.

## Last week: starting point

Same benchmark: **100 questions / 4,876 passages**, FLAN-T5-base + MiniLM. These are last week's saved primary results.

| Method | Precision@5 | Hit/Recall@5 | MRR@5 | nDCG@5 |
|---|---:|---:|---:|---:|
{previous_rows}

**Why this week's follow-up:** Test whether improving the 15% HyDE dense result could also improve hybrid retrieval. The 72% result uses the dataset's gold answer; it is a diagnostic, not a usable HyDE result. Generation software/settings differ between weeks, so cross-week HyDE gains are descriptive; this week's recomputed controls provide the main comparison.

[Last week's source metrics](../person3_hyde/summary_k5.csv)

## 1. What I tested this week

| Parameter / method | Settings tested |
|---|---|
| Prompt style | Short answer; evidence passage |
| Maximum generated length | 32; 96 tokens |
| Decoding | Greedy; sampling with temperature 0.7 and top-p 0.9 |
| Random-seed check | Three seeds for sampled evidence passages at 96 tokens |
| Retrieval input | One HyDE answer; question + HyDE; three-answer embedding average; three-answer ranking fusion |
| Hybrid weights | BM25 / dense: 75% / 25%, 50% / 50%, 25% / 75% |
| Candidates before fusion | Top 20; Top 100 per retrieval list; RRF constant = 60 |
| Evaluation | Precision, Recall/Hit Rate, MRR, nDCG at K = 1, 3, 5, 10; paired McNemar tests |
| Experiment size | 10 generation settings; 3,000 generated hypotheses; {len(df)} method/configuration rows per K |

Three-answer methods use three prompt variants, so they test both prompt diversity and the number of hypotheses.

## 2. Dense retrieval results

**Hit@5:** percentage of questions with the labelled evidence in the first five results. With one labelled passage per question, Recall@5 equals Hit@5.

| Retrieval input | Hit@5 range across 10 settings | Mean Hit@5 | Highest observed Hit@5 |
|---|---:|---:|---:|
| Original question — baseline | — | — | {q:.0%} |
{dense_rows}
| Dataset reference answer — diagnostic only | — | — | {oracle:.0%} |

**Finding:** No tested HyDE dense method exceeded the original-question baseline. The 72% reference-answer result uses the gold answer and cannot be deployed.

## 3. Hybrid retrieval results

Both rows use **75% BM25 / 25% dense**, candidate depth **100**, and RRF constant **60**.

| Method | Precision@5 | Hit/Recall@5 | MRR@5 | nDCG@5 |
|---|---:|---:|---:|---:|
| Standard hybrid: BM25 + original-question dense | {control.precision_at_k:.1%} | {standard:.0%} | {control.mrr_at_k:.4f} | {control.ndcg_at_k:.4f} |
| Highest observed HyDE hybrid: BM25 + three-answer RRF | {cand.precision_at_k:.1%} | **{cand.recall_at_k:.0%}** | {cand.mrr_at_k:.4f} | {cand.ndcg_at_k:.4f} |
| Change | +{(cand.precision_at_k-control.precision_at_k)*100:.1f} pp | +{(cand.recall_at_k-standard)*100:.0f} pp | {cand.mrr_at_k-control.mrr_at_k:+.4f} | {cand.ndcg_at_k-control.ndcg_at_k:+.4f} |

Selected HyDE setup: **short-answer prompt / 32 tokens / greedy / three hypotheses**. Paired comparison: **{int(paired.candidate_only_hits)} gained, {int(paired.baseline_only_hits)} lost**; McNemar **p = {paired.mcnemar_exact_p:.4f}**, Holm-adjusted **p = {paired.holm_adjusted_p:.0f}** across {len(pairs)} comparisons.

**Finding:** More Top-5 hits, but worse ranking quality and no statistically significant improvement. Improving standalone HyDE dense did not consistently improve hybrid retrieval.

## Conclusion and next step

**Conclusion:** These generation and parameter changes did not establish a reliable HyDE improvement with the current models.

**Next:** Freeze a configuration, validate on new questions, then test a stronger generator.

**Limits:** The same 100 questions were previously inspected; ranges and selected maxima are exploratory. Final answer correctness was not evaluated. Saved timings are batched CPU estimates, not interactive latency.

[Full generation comparison](generation_comparison_k5.csv) · [All weights and configurations](summary_k5.csv) · [Seed checks](seed_robustness_k5.csv) · [Paired tests](paired_comparisons_k5.csv) · [Help/hurt examples](help_hurt_examples.csv) · [Reproduction instructions](../../person3_hyde/ROBUSTNESS_README.md)
'''
    (OUT/'findings.md').write_text(report)
    print(report[:3500])
if __name__=='__main__': main()
