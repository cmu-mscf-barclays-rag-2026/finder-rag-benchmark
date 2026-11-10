"""Assemble the final Meeting9 review from completed, saved stage results."""
from pathlib import Path
import hashlib,json
import pandas as pd
from api import ROOT,dump
EXPORT=ROOT.parent/'outputs/meeting9_20261109'
def read(p):return json.loads(p.read_text())
LABEL={'question_dense':'Question dense','bm25':'BM25 (full OCR)',
    'focused160_pure_dense':'Focused: pure HyDE','focused160_mix_dense':'Focused: 50/50 mix',
    'short_30_50_pure_dense':'Short: pure HyDE','short_30_50_mix_dense':'Short: 50/50 mix'}

def main():
    a=ROOT/'results/person3_meeting9_ablation/factorial20_v1'
    b=ROOT/'results/person3_meeting9_transfer/vidore_finance_full309_summary_v1'
    c=ROOT/'results/person3_meeting9_fullpage/v1'
    assert read(a/'validation.json')['status']=='passed'
    assert read(b/'completion_status.json')['complete'] and read(c/'validation.json')['passed']
    cost=read(EXPORT/'costs/validation.json')
    full=pd.read_csv(b/'summary.csv');chunk=pd.read_csv(c/'summary.csv')
    tests=pd.read_csv(b/'separate_primary_families.csv')
    primary=pd.read_csv(a/'primary_tests.csv')
    assert len(primary)==12 and not (primary.p_holm12<.05).any()
    history=read(ROOT.parent/'work/full_results_hashes.json')
    changed=[p for p,h in history.items() if not (ROOT/p).exists() or hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h]
    assert not changed,'Historical full results changed'
    dump(EXPORT/'historical_preservation.json',dict(checked=len(history),changed_or_missing=changed,passed=True))
    lines=['# Meeting9: HyDE findings and evaluation workflow','',
        '9 November 2026. Three parallel agents; shared API ledger; independent cross-checks. Historical full experiments were preserved.','',
        'On the frozen remaining 249 queries, Focused and Short question-plus-HyDE mixes improved the first-256-token dense baseline (Holm p=0.00936 and 0.04158). Short did not establish an advantage over Focused. A separate full-OCR representation control produced a larger descriptive change than prompt choice; full-OCR BM25 remained the strongest tested configuration by aggregate nDCG.','',
        '| Stage | Completed work | Interpretation |','|---|---|---|',
        '| Prompt ablation | Legal 20 + FiQA 20; 160 generations; 2×2 length/instruction grid | All 12 planned Holm-adjusted comparisons were nonsignificant |',
        '| New benchmark | All 309 English queries / all 2,942 ViDoRe Finance pages; 618 generations | Frozen prompts; initial 60 and remaining 249 have separate Holm4 families |',
        '| Full-page text control | Complete OCR; 14,017 chunks; same cached queries/generations | Secondary descriptive representation comparison; no new p-values |',
        '| Evidence prototype | Legal 20 annotated, 19 eligible; 31 fact drafts, 34 source spans | Quote availability only; human validation pending |',
        '| HNSW | FiQA 628; seven indexes/settings; two query representations | Index-only speed/quality/storage and isolated RSS measurements |','',
        '```mermaid','flowchart LR','  A[Recover caches and freeze protocols] --> B[Prompt ablation]',
        '  A --> C[New benchmark: 60 then 249]','  A --> D[Evidence prototype]',
        '  A --> E[Offline HNSW comparison]','  C --> F[Full OCR representation control]',
        '  B --> G[Independent audits]','  D --> G','  E --> G','  F --> G','  G --> H[Tables, charts and reproduction]','```','',
        '## New dataset: complete English split','',
        'All 309 queries search the same 2,942 financial-report pages. The original FiQA Focused and Short prompts were transferred without tuning on these outcomes. The aggregate below is descriptive.','',
        '| Method | nDCG@10 | Hit@5 | Hit@10 | Recall@10 |','|---|---:|---:|---:|---:|']
    for m in LABEL:
        g=full[full.method.eq(m)].set_index('k')
        lines.append(f'| {LABEL[m]} | {g.loc[10,"ndcg"]:.4f} | {g.loc[5,"hit"]:.1%} | {g.loc[10,"hit"]:.1%} | {g.loc[10,"recall"]:.4f} |')
    lines+=['','![All 309 English queries](transfer/full309/transfer_full309.png)','',
        '| Frozen stage | Comparison | Delta nDCG@10 | Holm p within 4 |','|---|---|---:|---:|']
    for _,r in tests.iterrows():lines.append(f'| {r.stage} | {LABEL[r.candidate]} vs {LABEL[r.control]} | {r.mean_delta:+.4f} | {r.p_holm:.5f} |')
    lines+=['','No pooled 309 significance tests were added. Tests are question-level and do not adjust for the six shared annual-report sources. Detailed [results and Top-k curves](transfer/full309/README.md).','',
        '## Document representation matters','',
        '2,764/2,942 pages exceed 256 MiniLM tokens. Frozen primary dense runs encode only the page beginning; BM25 indexes full OCR. A separate control encodes all OCR in 224-token chunks with 32-token overlap, then ranks each page by its highest chunk cosine score. This changes both coverage and aggregation; it does not isolate coverage alone.','',
        '| Query representation | Page first256 nDCG@10 | Full-page chunk-max nDCG@10 |','|---|---:|---:|']
    for m in ['question_dense','focused160_pure_dense','focused160_mix_dense','short_30_50_pure_dense','short_30_50_mix_dense']:
        old=full[full.method.eq(m)&full.k.eq(10)].iloc[0].ndcg
        new=chunk[chunk.cohort.eq('full309')&chunk.method.eq(m)&chunk.representation.eq('fullpage_chunkmax')&chunk.k.eq(10)].iloc[0].ndcg
        lines.append(f'| {LABEL[m]} | {old:.4f} | {new:.4f} |')
    lines+=['','This remains text-only retrieval. No image encoder, text/image fusion or final-answer reader was evaluated. See the [full-page control](fullpage/README.md).','',
        '## Prompt ablation','',
        'The shared template changes only requested length (30–50 versus 70–100 words) and one constraint-preservation sentence. Mixed50 is primary; pure HyDE is descriptive. Legal 20 and FiQA 20 are reused exploratory questions, not new holdouts. No planned contrast passed Holm12. Out-of-range lengths were retained without regeneration. [Matched tables, exact tests and realized lengths](ablation/README.md).','',
        '![Factorial prompt comparison](ablation/factorial_primary.png)','',
        '## Evidence sufficiency prototype','',
        'At Top-10 / 1,024 WordPiece tokens, draft quote completeness was 21.1% for Question, 42.1% for Focused mix and 36.8% for Short mix (n=19). A qrel Hit can coexist with missing quotations. One contradictory Legal reference answer was excluded explicitly.','',
        'Annotations are AI-assisted drafts with exact source offsets, not human-validated fact labels. Exact quote availability misses paraphrases and unannotated alternatives; it does not prove answer correctness. Human-reviewed questions: 0. [Protocol, cases and blinded human-review materials](evidence/README.md).','',
        '![Draft quote availability](evidence/evidence_quote_coverage.png)','',
        '## Search efficiency and generation cost','',
        'For cached Short50/50 queries, HNSW M32/efSearch64 recovered 99.3% of exact Top-10 neighbors, with local warm median search 0.095 ms versus 1.154 ms for exact search. nDCG@10 changed from 0.385824 to 0.383559. These are index-only measurements; generation is measured separately in seconds. [All settings, raw timing audit, storage and isolated memory](hnsw/README.md).','',
        '![HNSW tradeoff](hnsw/hnsw_tradeoff.png)','',
        f"Successful new calls: **{cost['unique_calls']}**. Estimated successful-call spend: **USD{cost['new_cost_usd']:.6f}**; four timeout attempts retain **USD{cost['unknown_timeout_reserves_usd']:.6f}** as possible spend. Current extension ledger: **USD{cost['cumulative_estimate_usd']:.6f}/USD7**. Estimates are not a billing statement.",'',
        'Successful outputs were never regenerated. Timeout recovery was investigated and explicitly recorded; SDK retries stayed disabled. Latency summaries mix recorded concurrency settings and are descriptive, not a controlled causal timing study or end-to-end RAG latency. [Per-call tokens, cost and request duration](costs/README.md).','',
        '## Reproduce from retained results','',
        'From the existing repository with pinned datasets/model and retained caches:','',
        '```bash','.venv/bin/python person3_meeting9_ablation/run.py evaluate',
        '.venv/bin/python person3_meeting9_transfer/run.py evaluate',
        '.venv/bin/python person3_meeting9_transfer/run.py evaluate --cohort remaining249',
        '.venv/bin/python person3_meeting9_transfer/merge_full.py',
        '.venv/bin/python person3_meeting9_fullpage/run.py evaluate',
        '.venv/bin/python person3_meeting9_evidence/verify.py',
        '.venv/bin/python person3_meeting9_workflow/independent_transfer_audit.py',
        '.venv/bin/python person3_meeting9_workflow/costs.py','```','',
        'These commands make no API calls. Each stage README documents preparation and index recreation. Do not delete retained successful generations. All 565 historical full-run hashes remained unchanged.','',
        'Remaining team work: independently validate fact requirements and sufficiency annotations; evaluate image and combined text/image retrieval if assigned; compare stronger encoders under a matched protocol. Graph-RAG refinements belong to the separate team workstream.']
    text='\n'.join(lines)+'\n'
    (EXPORT/'README.md').write_text(text)
    dump(EXPORT/'completion_status.json',dict(computational_stages_complete=True,
        ablation_generations=160,transfer_generations=618,full_english_queries=309,
        corpus_pages=2942,human_validated_questions=0,visual_model_evaluated=False,
        old_full_result_hashes_preserved=len(history),costs=cost))
    print('Meeting9reportassembled',flush=True)

if __name__=='__main__':main()
