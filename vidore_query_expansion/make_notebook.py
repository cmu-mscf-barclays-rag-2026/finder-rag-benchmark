"""Build a presentation notebook that reads the completed experiment artifacts."""

from pathlib import Path
import json

import nbformat as nbf


ROOT = Path(__file__).resolve().parent


def build():
    notebook = nbf.v4.new_notebook()
    cells = []

    def md(text):
        cells.append(nbf.v4.new_markdown_cell(text))

    def code(text):
        cells.append(nbf.v4.new_code_cell(text))

    md("""# Query expansion for financial page retrieval

**Research question:** Can alternative formulations of a question improve retrieval from financial reports, and does optimizing those formulations help beyond a manually written prompt?

The experiment extends the existing ViDoRe Finance benchmark. It reuses the same page corpus, page labels and Qwen3-VL page embeddings. The new component is the search question: the original wording, LangChain MultiQueryRetriever alternatives, or DSPy alternatives before and after prompt optimization. No answer reader or answer judge is evaluated.

All results below are loaded from saved, executed runs. Reopening this notebook does not make API calls or regenerate embeddings.""")
    code("""from pathlib import Path
import json
import pandas as pd
from IPython.display import display

ROOT = Path.cwd()
if not (ROOT / 'results/manifest.json').exists():
    ROOT = ROOT / 'outputs/vidore_query_expansion'
RESULTS = ROOT / 'results'
load = lambda name: json.loads((RESULTS / name).read_text(encoding='utf-8'))
manifest = load('manifest.json')
audit = load('data_audit.json')
protocol = load('protocol.json')
selection = load('selection.json')
summary = pd.read_csv(RESULTS / 'summary.csv')
assert manifest['queries'] == 1854 and manifest['pages'] == 2942
display(pd.DataFrame(audit['split_counts']).T.rename(columns={'english':'English questions', 'all_variants':'All language variants'}))
print('Search pool:', manifest['pages'], 'pages from six financial reports')
print('Generator:', protocol['generator_model'])""")
    md("""## 1. Data and evaluation split

There are **309 English questions**, each with five translated versions, for **1,854 query variants**. Every query searches the complete **2,942-page** English corpus. Translations are repeated information needs, not 1,854 independent English questions.

Questions with identical full relevance-label signatures stay together. This conservative grouping keeps translations together and also groups some distinct questions sharing the same labels. There are 31 English training questions, 31 validation questions and 247 optimizer-held-out test questions. Training examples support DSPy's instruction proposal; validation selects prompts and retrieval settings. The headline table uses the 247 English test questions.

These queries were evaluated in the previous benchmark, so the test set is held out from this optimizer rather than a newly unseen benchmark. Six shared reports also limit the breadth of generalization.""")
    code("""splits = pd.read_csv(RESULTS / 'splits.csv')
assert splits.groupby('group')['split'].nunique().max() == 1
display(splits.groupby(['split', 'language']).size().unstack().reindex(['train','validation','test']))
display(pd.Series({'Relevance-signature groups': audit['relevance_groups'],
                   'Groups crossing splits': 0,
                   'Pages available for every search': manifest['pages']}))""")
    md(r"""## 2. Retrieval methods and query generators

| Search backend | What is indexed | Representation |
|---|---|---|
| BM25 | Full OCR/Markdown page text | Lexical term matching |
| Dense OCR | OCR/Markdown page text | Qwen3-VL-Embedding-2B vectors |
| Dense image | Page images | Qwen3-VL-Embedding-2B visual vectors |
| OCR/image fusion | Both dense rankings | Weighted reciprocal rank fusion |

The page vectors are reused from the pinned prior run: a 4,096-token text cap and 786,432-pixel image cap. Generated questions use the same query encoder and financial retrieval instruction as the original questions. No embedding model is fine-tuned. OCR/image fusion is dense multimodal fusion; it does not combine BM25 and dense scores.

**Original:** search once with the user's question. **MultiQuery:** use the actual LangChain `MultiQueryRetriever.generate_queries()` interface with general and financial prompt candidates. **DSPy initial:** a typed `question -> list[str]` program. **DSPy optimized:** MIPROv2 selects its instruction using retrieval nDCG, rather than generated-answer accuracy. This is a retrieval-specific DSPy experiment.

Each generator produces five alternatives. Development tuning chooses whether to use the first 1, 3 or 5 and whether to retain the original question. Candidate retrieval and fusion use one shared ranked-search implementation for all generators. LangChain's unordered document union is therefore replaced with an explicit, reproducible ranking:

\[
S(d)=\sum_{j=0}^{m}\frac{w_j\,\mathbf{1}[d\in L_j[:D]]}{c+\operatorname{rank}_j(d)}.
\]

Here, $m$ is the number of alternatives, $D$ is per-query candidate depth, $c$ is the RRF constant, and $w_0$ is the original-question weight. Alternative-query weights are 1. An original weight of 0 excludes the original. Each list uses stable page-ID tie-breaking. Dense OCR/image fusion happens within each query first, with its own fixed constant of 60 and depth of 100.""")
    md("""## 3. Development-only parameter tuning

The primary selection metric is **English validation nDCG@10**, with Recall@10 as the first tie-breaker. Further ties favor fewer alternatives, shallower candidate sets, retaining the original, and a constant near 60.

The grid covers BM25 term-frequency saturation and length normalization; OCR/image fusion weight; general versus financial prompts; number of query alternatives; candidate depth; RRF constant; and original-query weight. Five alternatives are generated once per prompt/question, and prefix subsets are reused across configurations, so the grid does not require an API call for every setting.

DSPy uses two optimizer seeds, six instruction candidates, 12 search trials per seed plus the default-program evaluation, and zero demonstrations in the final programs. Its optimization uses BM25 with three alternatives, original weight 1, depth 100 and RRF constant 60. The selected instruction then transfers to the other backends; dense/image retrieval is not used inside MIPROv2's scoring loop.""")
    code("""grid = {
    'BM25 k1': protocol['bm25_k1'], 'BM25 b': protocol['bm25_b'],
    'OCR weight in OCR/image fusion': protocol['ocr_fusion_weights'],
    'Alternatives used': protocol['expansion_counts'],
    'Candidates per query': protocol['candidate_depths'],
    'Query-fusion RRF constant': protocol['rrf_constants'],
    'Original-question weight': protocol['original_query_weights'],
}
display(pd.DataFrame([{'Parameter': name, 'Values tested': str(values)} for name, values in grid.items()]))
print('Selected common baseline settings:', selection['baseline'])
tuning = pd.read_csv(RESULTS / 'expansion_tuning.csv')
print('Expansion configurations evaluated on validation:', len(tuning))
selected_rows = [dict(Backend=backend, Family=family, **params)
                 for backend, methods in selection['expansion'].items()
                 for family, params in methods.items()]
display(pd.DataFrame(selected_rows)[['Backend','Family','strategy','count','depth','rrf_constant','original_weight','ndcg_at_10']].round(4))
display(pd.concat([pd.read_json(RESULTS / f'optimizer_trials_{seed}.json').assign(seed=seed)
                   for seed in protocol['dspy_optimizer']['seeds']], ignore_index=True).groupby('seed')['ndcg_at_10'].agg(['count','min','max']).round(4))""")
    md("""## 4. What the metrics mean

All methods use the same published page labels and report the same metrics at **1, 5, 10 and 20**. These cutoffs control the number of final returned pages. They differ from candidate depth, which is the number of pages contributed by each query before fusion.

* **Hit@k:** at least one judged relevant page appears in the top k.
* **Recall@k / judged-page coverage:** the fraction of judged relevant pages retrieved. It does not measure the fraction of required answer facts.
* **Precision@k:** relevant pages retrieved divided by k.
* **MRR@k:** reciprocal rank of the first relevant page, with zero if absent.
* **nDCG@k:** ranking quality using the dataset's linear grades 1 and 2, discounted by rank and normalized by the ideal ordering.
* **All-judged-page hit@k:** every judged relevant page was returned. Some pages can be alternative sources, so this can be stricter than answer sufficiency.
* **Complete-answer-page hit@k:** at least one grade-2 page was returned, evaluated only for questions with a grade-2 label. This uses the author's page judgment, rather than newly validated span/fact annotations.

No reader is run, so these are retrieval results rather than answer correctness or groundedness scores. Unjudged pages are treated as nonrelevant for scoring; the annotations may not list every usable page.""")
    code("""test = summary[(summary.scope == 'split_language') & (summary['group'] == 'test/english')].copy()
headline = test[test.k == 10].set_index('method')[['queries','hit','recall','precision','mrr','ndcg','all_judged_hit','complete_page_hit','complete_page_eligible_queries']]
headline.insert(1, 'hit_at_5', test[test.k == 5].set_index('method')['hit'])
display(headline.sort_index().round(4))""")
    md("""## 5. Cutoff sensitivity and uncertainty

The first table compares matched cutoffs across all methods. Deeper retrieval can improve page coverage while returning more irrelevant pages. It does not automatically improve the final answer.

The second table gives paired differences from each backend's original-question baseline at k=10, with 2,000 bootstrap resamples of relevance-signature groups. These are pointwise descriptive intervals. They do not correct for multiple comparisons or establish performance on new reports.""")
    code("""display(test.pivot(index='method', columns='k', values='hit').sort_index().round(4).rename_axis(columns='Hit cutoff'))
display(test.pivot(index='method', columns='k', values='recall').sort_index().round(4).rename_axis(columns='Recall cutoff'))
paired = pd.read_csv(RESULTS / 'paired_comparisons.csv')
display(paired[(paired.k == 10) & (paired.metric == 'ndcg')][['backend','family','difference','paired_group_bootstrap_lower_95','paired_group_bootstrap_upper_95','wins','losses','ties']].round(4))""")
    md("""## 6. Does prompt optimization itself help?

Each main-table pipeline selects its own fusion parameters on validation. Its improvement can reflect both the query wording and those parameters. To isolate the instruction more closely, the table below compares initial and optimized DSPy queries using **the optimized pipeline's identical retrieval/fusion parameters** for both. The initial prompt still retains its independently tuned row in the main comparison.

The supplementary previous-setting controls also preserve BM25's former k1=1.5, b=0.75 and the former 50/50 OCR/image fusion, so changing those defaults is not confused with query expansion.""")
    code("""matched = pd.read_csv(RESULTS / 'matched_dspy_control.csv')
display(matched[['backend','family','queries','hit_at_10','recall_at_10','ndcg_at_10']].round(4))
controls = pd.read_csv(RESULTS / 'previous_setting_controls.csv')
test_ids = set(splits[(splits.split == 'test') & (splits.language == 'english')].query_id)
display(controls[(controls.query_id.isin(test_ids)) & (controls.k == 10)].groupby('method')[['hit','recall','ndcg']].mean().round(4))""")
    md("""## 7. Full-dataset and subgroup results

All 1,854 variants were processed; none of the final evaluation rows are sample estimates. The main test conclusion uses English because that is the development objective. The full English and multilingual tables below are descriptive aggregates that also contain development questions. They must not be presented as independent held-out confirmations.

Language-specific changes show how querying an English corpus with translated questions behaves. Visual/table/text groupings derive from dataset content tags; they are not newly verified annotations.""")
    code("""all_en = summary[(summary.scope == 'language') & (summary['group'] == 'english') & (summary.k == 10)]
display(all_en.set_index('method')[['queries','hit','recall','ndcg']].sort_index().round(4))
language = summary[(summary.scope == 'split_language') & summary['group'].str.startswith('test/') & (summary.k == 10)].copy()
display(language.pivot(index='method', columns='group', values='ndcg').sort_index().round(4))
content = summary[(summary.scope == 'test_english_content') & (summary.k == 10)]
display(content.pivot(index='method', columns='group', values='ndcg').sort_index().round(4))""")
    md("""## 8. Examples where expansion helped or hurt

These are the largest held-out English nDCG@10 gain and loss for MultiQuery within each backend, selected for inspection after evaluation. They explain behavior; they are not representative-sample estimates. Check whether the rewrite preserved dates, entities and the requested financial relationship, and whether fusion promoted relevant pages or lexical distractors.""")
    code("""examples = load('qualitative_examples.json')
display(pd.DataFrame(examples)[['backend','example_type','query_id','question','ndcg_at_10_change']].round(4))
example = next(x for x in examples if x['backend'] == 'image' and x['example_type'] == 'largest_loss')
print('Question:', example['question'])
print('Five generated alternatives:')
for i, query in enumerate(example['generated_queries'], 1):
    print(f'{i}. {query}')
print('Judged pages / grades:', example['judged_pages'])
print('Original top 10:', example['original_top10'])
print('Expanded top 10:', example['expanded_top10'])""")
    md("""## 9. Runtime, API usage and validation

GPT-4o-mini generates question alternatives; local Qwen3-VL encodes them on the GPU. The page embeddings are reused, so no new page-indexing time is claimed. Candidate search uses cached vectors on the CPU, followed by RRF.

API latency, batched GPU embedding time and cached lookup/fusion time are different measurements. The recorded lookup timings are cache/order dependent and exclude generation and encoding; they are not end-to-end query latency. The API cost estimate uses saved unique successful request usage at $0.15 per million input tokens and $0.60 per million output tokens, without cache discounts. It excludes any interrupted/unrecorded calls and is not a billing statement. Alternative queries are generated in sets of five even when a pipeline uses only one, so costs do not represent a separately optimized one-rewrite deployment.

Raw API caches and credentials stay outside the upload package. The export includes the generated queries, sanitized usage, frozen settings, per-query metrics and top-20 rankings. A separate calculation checks every saved metric row from the rankings and qrels.""")
    code("""usage = pd.read_csv(RESULTS / 'api_usage.csv')
batch_records = [batch for run in manifest['query_embedding_batches'] for batch in run]
display(pd.Series({
    'Saved unique successful API requests': manifest['api_requests'],
    'Recorded prompt tokens': manifest['prompt_tokens'],
    'Recorded output tokens': manifest['completion_tokens'],
    'Recorded-usage cost estimate (USD)': round(manifest['estimated_api_usd'], 4),
    'Median API request latency (seconds)': round(manifest['api_latency_median_seconds'], 3),
    'P95 API request latency (seconds)': round(manifest['api_latency_p95_seconds'], 3),
    'New query vectors encoded (includes validation-only prompts)': sum(b['count'] for b in batch_records),
    'GPU embedding batch seconds (excludes model loading)': round(sum(b['seconds'] for b in batch_records), 2),
    'Exported final query-generation records': manifest['exported_query_generation_records'],
}))
print('Format-repair counts:', manifest['format_repair_counts'])
display(pd.Series(load('independent_audit.json')))
display(pd.Series(manifest['versions']))""")
    md("""## 10. Interpretation and next steps

On the 247 English test questions, BM25 nDCG@10 rises from **0.5037** with the original question to **0.5191** with MultiQuery and **0.5243** with the initial DSPy prompt. The optimized DSPy pipeline reaches **0.5188**, so optimization does not improve on the initial pipeline here.

For the dense backends, expansion does **not** improve test nDCG@10 over the original question. Original dense image retrieval records **0.5791**, versus **0.5719** for MultiQuery and **0.5690** for optimized DSPy. Yet image MultiQuery Hit@10 increases from **0.9069** to **0.9150**: finding some relevant page and ranking all relevant pages well are different outcomes. These figures use frozen development-selected settings.

Use the optimizer-held-out English table to decide whether expansion improves each retrieval backend. Compare optimized DSPy against both original questions and its initial prompt, including the matched-parameter control. More alternatives or a higher development score do not guarantee a test improvement.

This experiment tests retrieval of labeled pages. It does not establish answer completeness, factual correctness, visual reading accuracy, or gains from a larger context budget. A subsequent reader experiment can keep the reader and evidence budget fixed across the selected page rankings. Fine-grained evidence sufficiency still requires reviewed fact/span annotations and an explicit policy for alternative supporting sources.

Sources: [ViDoRe Finance–EN](https://huggingface.co/datasets/vidore/vidore_v3_finance_en), [Qwen3-VL-Embedding-2B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B), [LangChain MultiQueryRetriever](https://reference.langchain.com/python/langchain-classic/retrievers/multi_query/MultiQueryRetriever), [DSPy MIPROv2](https://dspy.ai/current/api/optimizers/MIPROv2/).""")
    notebook.cells = cells
    notebook.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python", "version": "3.12"}}
    nbf.write(notebook, ROOT / "ViDoRe_Query_Expansion.ipynb")


if __name__ == "__main__":
    build()
