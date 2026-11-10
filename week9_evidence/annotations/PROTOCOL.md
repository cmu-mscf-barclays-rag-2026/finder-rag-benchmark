# Finance evidence annotation protocol — November 9, 2026

`finance_pilot.jsonl` contains **10 assistant-drafted questions and 20 required
facts, not human-validated ground truth**. Run with `--allow-draft` to inspect
the prototype. Default scoring rejects draft annotations. No LLM API or paid
model call was used. Dataset answers were inspected for feasibility, but the
retrievers receive only question text; they never receive answers or qrels.

The sample is purposive, not random or held out: IDs 2, 9, 10, 13, 14, 17, 19,
22, 24, and 55. It spans six institutions and includes tables, temporal rates,
an enumeration, a comparison, alternative sources, and evidence across pages.
It is weighted toward Bank of America. Do not tune a retriever on these ten
questions and present the result as a generalization estimate.

## Rubric and review

1. Read the question and write the minimum distinct factual requirements.
   Include the entity, reporting period, units and relevant conditions.
   Use `single_fact`, `single_fact_alternatives`, `single_fact_joint_support`,
   `multi_fact`, `comparison`, or `enumeration`. Classification is manually
   drafted; dataset query types are retained separately.
2. Read the source PDF and OCR, including table headings and footnotes. Check
   every proposed fact and its scope against the source; a supplied answer or
   relevant-page label alone is not sufficient. Record missing/corrupt OCR as
   a text-representation limitation. Do not repair source text silently.
3. For each fact, list alternative `support_sets`. **Any one support set**
   suffices for the fact; **every span inside that set** must be delivered.
   **All required facts** must be covered for annotation-relative sufficiency.
   An alternative page may cover several facts. Several redundant pages earn
   no additional credit for an already covered fact.
4. Store exact character offsets and quotes in each source page's unmodified
   `markdown`. For tables include headers/units as well as the value. A span
   split across pages must include both pieces. The validator checks offsets
   and quotes, but does not decide whether the content semantically supports
   the requirement.
5. Search beyond qrel pages for plausible alternative support, especially in
   retrieved contexts. Review without knowing which retriever produced the
   page. Annotate newly found valid alternatives and rerun **all** methods
   with the same annotation version. An uncovered fact means *no annotated
   support delivered*, not proof that the answer is impossible.
6. Have an independent human review the question decomposition, sufficient
   support sets and rejected alternatives. Resolve disagreements; retain a
   review log. Only then set `status` to `reviewed` and record a reviewer.
   Estimate agreement on fact coverage before expanding the pilot.

## Definitions

- **Fact recall:** number of covered required facts / total required facts.
- **Sufficiency:** all required facts have at least one complete support set.
  This is a rubric-dependent evidence measure, not generated-answer accuracy.
- **Evidence efficiency:** covered facts per 1,000 delivered content tokens.
  Also report fact recall and tokens; fact counts depend on decomposition.
- **Marginal gain:** newly covered facts after each unique delivered page.
  Zero gain means no *new annotated fact*, not automatically irrelevance.
- **Minimum tested sufficient budget:** smallest budget in the registered grid
  that yields sufficient delivery for that ranking/policy. Blank means no
  tested budget succeeded. This is not a globally minimum sufficient context.
  Whole-page skip-to-fit packing can be nonmonotonic across budgets.

Question 10 illustrates why arithmetic evidence and management's explanation
must be separate requirements. Question 13 deliberately requires the preceding
page to establish who the trading policy covers. Question 22 requires details
across multiple pages, including conditions omitted by the dataset answer.
Review these choices carefully; they can materially change the metric.

## Initial issues for review

| Query/page | Observation | Treatment |
|---|---|---|
| Query 0, pages 10/346 | Full-year repurchases of $18.841B cover two authorizations. The new $30B program starts July 1; page 346 reports $19.326B remaining. Comparing full-year spend with the new authorization can mix periods. | Excluded from scored pilot; verify intended scope against PDF. |
| Query 11 | Question names Bank of America; its sole qrel page 1242 belongs to Citigroup. | Entity mismatch flag; excluded from pilot. |
| Query 39 | Query itself contains an `Answer:` clause. | Answer leakage flag; excluded from pilot. Full benchmark baseline retains published queries unchanged. |
| Page 2687, query 9 | OCR corrupts the floating-rate spread; page 2766 provides the readable contractual clause. | Do not treat all relevant pages as equivalent sufficient text. |
| Query 19 | Table has both Deposits net income ($7,515M) and total Consumer Banking ($10,759M). | Preserve column headers and entity scope. |
| Query 22 | Supporting pages 2779/2782 are absent from qrels; TT reset spread has duplicated OCR text. | Add explicit draft spans; require visual review. |
| Query 24 | Some qrels are parent-company-only statements; multiple consolidated pages also supply the answer. | Credit known consolidated alternatives; review omitted alternatives. |

Raw source PDFs are available at
[`vidore/vidore_v3_finance_en`](https://huggingface.co/datasets/vidore/vidore_v3_finance_en/tree/7f432c176d82e27546501ad8064a713ac3071809/pdfs).
`page_number_in_doc` is zero based; use PDF page number plus one for review.
