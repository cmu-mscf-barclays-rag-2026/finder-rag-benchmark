"""Write the final findings from validated results; no new API work."""
from pathlib import Path
import json,hashlib
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/person3_cross_dataset/gpt54_focused160_v1'
LABELS={'question_dense':'Original question dense','bm25':'BM25',
 'standard_hybrid_d25':'Standard hybrid (dense .25)',
 'standard_hybrid_d50':'Standard hybrid (dense .50)',
 'focused160_pure_dense':'Pure HyDE dense',
 'focused160_mix_dense':'Question + HyDE dense',
 'focused160_pure_hybrid_d25':'Pure HyDE hybrid (dense .25)',
 'focused160_mix_hybrid_d25':'Question + HyDE hybrid (dense .25)',
 'focused160_mix_hybrid_d50':'Question + HyDE hybrid (dense .50)',
 'no_context':'Reader without context'}
def pct(v):return f'{v*100:.2f}%'
def table(headers,rows):
 return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def main(available_dense=False):
 v=json.loads((OUT/('results_validation.json' if available_dense else 'validation.json')).read_text());assert v['status']=='passed'
 audit_note=(f"All 5,030 LongBench predictions, evidence spans, reader prompts, accuracy tables and 2,515 RRF rankings passed independent checks. Dense ranks were recomputed for {v['dense_rankings_recomputed']} of 1,509 query/method pairs using available caches. Earlier cache files were lost during environment resets; exhaustive dense-rank rechecking is not complete. This limitation concerns the additional audit, not missing benchmark questions or missing predictions." if available_dense else 'All predictions, source evidence, dense rankings and RRF rankings passed the complete independent artifact checks.')
 f=pd.read_csv(OUT/'fiqa_metrics.csv');l=pd.read_csv(OUT/'longbench2_accuracy.csv')
 tests=pd.read_csv(OUT/'paired_tests.csv');audit=pd.read_csv(OUT/'api_audit.csv')
 fq5=f[f.k==5].set_index('method');fq10=f[f.k==10].set_index('method')
 lb=l[l.group_by=='overall'].set_index('method')
 fbest=fq10.ndcg.idxmax();lbest=lb.drop('no_context').accuracy.idxmax()
 methods=list(fq5.index)
 text=f'''# GPT-5.4 focused HyDE transfer to FiQA and LongBench v2

## Main findings

Both complete collections finished: **648 FiQA test queries against 57,638 documents**, and **503 LongBench v2 questions across all six domains and all three length groups**. No test questions were sampled away, and each LongBench question searches its own complete context before taking the top five chunks. {audit_note}

On FiQA, the highest observed nDCG@10 was **{fq10.loc[fbest,'ndcg']:.4f}** using **{LABELS[fbest]}**; original question dense scored **{fq10.loc['question_dense','ndcg']:.4f}**, and standard hybrid at dense weight .25 scored **{fq10.loc['standard_hybrid_d25','ndcg']:.4f}**. The legal dataset's large HyDE improvement did not repeat: pure HyDE was lower, and mixed HyDE was close to question-only dense rather than better on the primary metric.

On LongBench v2, the highest observed retrieval-backed answer accuracy was **{pct(lb.loc[lbest,'accuracy'])}** using **{LABELS[lbest]}**; original question dense scored **{pct(lb.loc['question_dense','accuracy'])}**, standard hybrid at dense weight .25 scored **{pct(lb.loc['standard_hybrid_d25','accuracy'])}**, and the same reader without retrieved context scored **{pct(lb.loc['no_context','accuracy'])}**. These are descriptive best results, not independently selected methods or official leaderboard numbers.

## FiQA: retrieval evaluation

{table(['Method','Hit@5','Recall@5','MRR@5','nDCG@10'],[[LABELS[m],pct(fq5.loc[m,'hit']),pct(fq5.loc[m,'recall']),f'{fq5.loc[m,"mrr"]:.4f}',f'{fq10.loc[m,"ndcg"]:.4f}'] for m in methods])}

FiQA has multiple relevant answers per query. Hit@5 means at least one positive judgment was retrieved; Recall@5 counts the fraction of all positive judgments retrieved. nDCG@10 is the primary metric and is reported on a 0–1 scale. Unjudged documents are treated as nonrelevant. Dense vectors represent the first 256 MiniLM tokens of each document; BM25 uses the full document. Coincident query/document ID strings are not filtered, so this is not an exact reproduction of BEIR evaluator defaults.

## LongBench v2: retrieval plus fixed reader

{table(['Method','Correct / 503','Answer accuracy'],[[LABELS[m],f'{round(lb.loc[m,"accuracy"]*503)}/503',pct(lb.loc[m,'accuracy'])] for m in methods+['no_context']])}

These are final multiple-choice answer accuracies, not evidence retrieval Hit@5. LongBench v2 does not supply chunk relevance labels. The same GPT-5.4 reader sees only the selected five chunks, question and choices. HyDE generation and retrieval see only the question. Gold labels are used only after predictions are recorded. The no-context control uses an empty text section in the same reader template.

This is a custom RAG adaptation with 224 MiniLM token chunks and 32-token overlap; it does not replicate the official full-context or 512-token RAG leaderboard setting. The public Hugging Face distribution calls its single split `train`; all 503 benchmark items are used only for evaluation here, with no fitting. Full contexts are searched locally; they are not passed whole to the API. Tasks requiring information from many locations or many demonstrations can be poorly served by five short chunks. A reader can also answer from prior knowledge, so final accuracy alone cannot isolate evidence retrieval quality.

## Planned paired comparisons

{table(['Dataset / primary metric','Candidate','Control','Delta','Raw p','Holm p'],[[('FiQA / nDCG@10' if r.dataset=='fiqa' else 'LongBench / accuracy'),LABELS[r.candidate],LABELS[r.control],f'{r.delta:+.4f}',f'{r.p_raw:.6g}',f'{r.p_holm:.6g}'] for r in tests.itertuples()])}

The six candidate/control comparisons for each dataset were fixed in the protocol before outcomes. FiQA tests use two-sided paired sign-flip permutation tests of nDCG@10 (100,000 draws, seed 20261028); LongBench tests use exact two-sided paired McNemar tests of answer accuracy. Holm adjustment covers all 12 tests jointly. Other metrics and subgroup results are descriptive. This family does not cover the earlier legal experiments, nor a post-hoc comparison of the observed best methods. LongBench has 462 distinct contexts for 503 questions; question-level tests do not model correlation among questions sharing a context, so their p-values should be treated as exploratory benchmark evidence.

'''
 significant=tests[tests.p_holm<.05]
 text+=('No planned comparison remains below p=.05 after Holm correction.\n\n' if significant.empty else 'Planned comparisons below Holm-adjusted p=.05:\n\n'+'\n'.join(f'- {r.dataset}: {LABELS[r.candidate]} versus {LABELS[r.control]}, delta {r.delta:+.4f}, adjusted p={r.p_holm:.6g}.' for r in significant.itertuples())+'\n\n')
 text+='No planned LongBench comparison was significant after Holm correction. The significant FiQA comparison of mixed HyDE hybrid at dense weight .50 against standard hybrid at .25 changes both the query and the fusion weight; it cannot isolate a HyDE benefit. The corresponding matched-.50 comparison was not significant.\n\n'
 text+='## LongBench subgroups\n\n'
 for field in ['domain','difficulty','length']:
  subset=l[l.group_by==field];groups=list(subset['group'].unique())
  pivot=subset.pivot(index='method',columns='group',values='accuracy');counts=subset.groupby('group')['n'].first()
  text+=f'### {field}\n\n'+table(['Method']+[f'{g} (n={counts[g]})' for g in groups],[[LABELS[m]]+[pct(pivot.loc[m,g]) for g in groups] for m in methods+['no_context']])+'\n\n'
 cost=audit.cost_usd_estimate.sum();h=audit[audit.kind.str.startswith('hyde_')];reader=audit[audit.kind=='reader_longbench2']
 accounting=json.loads((OUT/'cost_accounting.json').read_text())
 recovery=accounting.get('checkpoint_recovery') or {}
 recovery_note=(f"\n- Environment recovery restored a persistent snapshot at 353 retrieval questions, after the old environment had reached approximately 422. A later reset restored 454 questions after 460 had been observed. Outputs after the snapshots were lost and some requests had to be repeated. The budget includes an additional ${recovery.get('lost_checkpoint_cost_allowance_usd',0):.2f} allowance for spending absent from the restored database. This is an estimated allowance, not verified usage or a billing statement.\n" if recovery else '')
 text+=f'''## Execution and costs

{recovery_note}

- HyDE outputs: {len(h)} (648 financial questions + 503 long-context questions); {int((h.status=='incomplete').sum())} length-capped/incomplete outputs retained under the fixed 160-token cap.
- Logical LongBench predictions: 5,030 (10 methods/controls × 503 questions); {len(reader)} unique reader requests after sharing identical complete prompts.
- Total successful API calls: {len(audit)}. Estimated usage cost **${cost:.6f}** (HyDE ${h.cost_usd_estimate.sum():.6f}, readers ${reader.cost_usd_estimate.sum():.6f}). This is a token-based estimate at $2.50/$15 per million input/output tokens, without cache discounts, not a billing statement.
- Transport recovery: {accounting['unresolved_attempts']} prior connection-failed attempts had no OpenAI response or token usage. They were retained separately and the missing logical calls were manually reissued after a successful read-only connectivity check. Their possible additional cost is conservatively reserved at **${accounting['unresolved_upper_allowance_usd']:.6f}**, giving a budget-accounting estimate, including the environment-recovery allowance, of **${accounting['conservative_total_upper_estimate_usd']:.6f}**. All intended logical outputs are available. Successful calls retained in the checkpoints were reused; workers were refreshed between complete batches to avoid stale long-running connections.
- API model: gpt-5.4-2026-03-05, reasoning none, temperature 0, no tools, store false; HyDE cap 160, answer cap 32; no intended prediction remains missing.
- Encoder: pinned all-MiniLM-L6-v2, normalized float32 embeddings; question/HyDE mixed 50/50 before normalization. Equivalent PyTorch/ONNX float32 caches were checked against each other. After a runtime safety review rejected an unknown Microsoft telemetry connection, encoding resumed with entirely offline PyTorch, no ONNX Runtime import and explicit Hugging Face telemetry opt-outs; existing vectors were reused and compatibility rechecked. No quantized outputs were used. Indexing processes were also resumed after a memory limit; bounded text loading/tokenization preserved chunk spans and all contexts.
- BM25 k1=1.2, b=.75, binary query counts. RRF k=60, top-100 input lists, fixed .25/.50 dense weights. BM25 always receives the original question.

## Interpretation limits

The focused prompt was transferred with domain-specific wording before new test outcomes. The original96 profile was intentionally omitted because 94/100 earlier legal generations hit its cap. Thus this is a focused-profile transfer study, not a full repeat of every earlier prompt ablation. No prompt, weight, model, or chunk budget was tuned on these new test labels. However, both datasets are public and can overlap model pretraining; these are not contamination-free private holdouts. The legal, financial and long-context numbers measure different tasks and must not be placed in one undifferentiated accuracy column. The retriever is still a small fixed MiniLM model, and the LongBench result additionally depends on the reader and five-chunk budget.

## Artifacts

`protocol.json` and source hashes fix the experiment; `fiqa_metrics.csv`, `fiqa_per_query_metrics.csv` and `fiqa_rankings.npz` support financial retrieval checks; `longbench2_accuracy.csv`, `longbench2_predictions.csv`, per-query retrieval JSON and answer mappings support long-context checks. `api_outputs.jsonl` records generated text, exact prompts and usage; `api_checkpoint.sqlite` supports resume without duplicate requests. `paired_tests.csv`, `cost_summary.csv`, backend checks and `validation.json` record inference and verification. Source datasets and large corpus/index caches are excluded from the result ZIP and can be restored from pinned downloads using the supplied code. No API key is included.
'''
 (OUT/'findings.md').write_text(text)
 code=ROOT/'person3_cross_dataset'
 (OUT/'code_hashes.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(code.glob('*')) if p.is_file()},indent=2))
 partial=OUT/'fiqa_findings.md'
 if partial.exists():partial.write_text(partial.read_text().replace('the joint paired-test report is produced after both benchmark evaluations finish.','the joint paired-test results are in `paired_tests.csv` and the complete `findings.md`.'))
 print('Report saved',OUT/'findings.md')

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--available-dense',action='store_true');a=p.parse_args();main(a.available_dense)
