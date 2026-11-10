# Finance pilot review packet

**Status: draft; no independent human review has been recorded.**

Use the [protocol](PROTOCOL.md) and edit `finance_pilot.jsonl` after review.
PDF links point to the pinned source revision. PDF page numbers below are one based;
corpus IDs are independent of page numbers. Exact quotes and character offsets are in
[the annotation file](finance_pilot.jsonl). Each bullet is a required fact; support sets
separated by OR are alternatives, while pages joined by + are jointly required.

## Query 2: How does Citigroup manage its interest rate and foreign exchange risks in its banking book and what are the potential impacts on its financial metrics?

Classification: `multi_fact`. Status: **draft**.

- **risk_cycle:** Citi identifies, measures, monitors, controls and reports risks through its risk management cycle. Support: [corpus 1707, PDF p. 70](https://huggingface.co/datasets/vidore/vidore_v3_finance_en/resolve/7f432c176d82e27546501ad8064a713ac3071809/pdfs/citigroup_2024.pdf#page=70).
- **governance:** Banking-book interest rate risk follows the Non-Trading Market Risk Policy; Treasury manages it and reviews it with ALCO. Support: [corpus 1057, PDF p. 104](https://huggingface.co/datasets/vidore/vidore_v3_finance_en/resolve/7f432c176d82e27546501ad8064a713ac3071809/pdfs/citigroup_2024.pdf#page=104).
- **metrics:** Non-trading interest-rate and FX changes can affect net interest income, economic value of equity and AOCI. Support: [corpus 1707, PDF p. 70](https://huggingface.co/datasets/vidore/vidore_v3_finance_en/resolve/7f432c176d82e27546501ad8064a713ac3071809/pdfs/citigroup_2024.pdf#page=70).

Review note: Broad question: reviewer should check that this three-part rubric is complete, especially FX management.

Reviewer: ______  Decision: ______  Corrections / additional alternatives: ______

## Query 9: Bank of America preferred stock MM dividend rate

Classification: `multi_fact`. Status: **draft**.

- **fixed_rate:** Series MM pays 4.300% until, but excluding, January 28, 2025. Support: [corpus 2766, PDF p. 215](https://huggingface.co/datasets/vidore/vidore_v3_finance_en/resolve/7f432c176d82e27546501ad8064a713ac3071809/pdfs/bank_of_america_2024.pdf#page=215).
- **floating_rate:** From January 28, 2025 the rate is Adjusted Three-Month Term SOFR plus 2.664%. Support: [corpus 2766, PDF p. 215](https://huggingface.co/datasets/vidore/vidore_v3_finance_en/resolve/7f432c176d82e27546501ad8064a713ac3071809/pdfs/bank_of_america_2024.pdf#page=215).

Review note: Page 2687 has corrupted OCR for the SOFR spread; do not infer the correct spread from that page text.

Reviewer: ______  Decision: ______  Corrections / additional alternatives: ______

## Query 10: What is the relationship between Morgan Stanley's revenue growth and its expense efficiency ratio in 2024?

Classification: `multi_fact`. Status: **draft**.

- **revenue:** Morgan Stanley revenue rose from $54.1B in 2023 to $61.8B in 2024 (14%). Support: [corpus 2193, PDF p. 27](https://huggingface.co/datasets/vidore/vidore_v3_finance_en/resolve/7f432c176d82e27546501ad8064a713ac3071809/pdfs/morgan_stanley_2024.pdf#page=27) OR [corpus 2258, PDF p. 79](https://huggingface.co/datasets/vidore/vidore_v3_finance_en/resolve/7f432c176d82e27546501ad8064a713ac3071809/pdfs/morgan_stanley_2024.pdf#page=79).
- **efficiency:** The expense efficiency ratio improved from approximately 77% to 71%. Support: [corpus 2193, PDF p. 27](https://huggingface.co/datasets/vidore/vidore_v3_finance_en/resolve/7f432c176d82e27546501ad8064a713ac3071809/pdfs/morgan_stanley_2024.pdf#page=27) OR [corpus 2258, PDF p. 79](https://huggingface.co/datasets/vidore/vidore_v3_finance_en/resolve/7f432c176d82e27546501ad8064a713ac3071809/pdfs/morgan_stanley_2024.pdf#page=79).
- **explanation:** The report attributes the ratio improvement to higher revenues and expense discipline. Support: [corpus 2193, PDF p. 27](https://huggingface.co/datasets/vidore/vidore_v3_finance_en/resolve/7f432c176d82e27546501ad8064a713ac3071809/pdfs/morgan_stanley_2024.pdf#page=27).

Review note: Table alternative requires division: expenses/revenue rounds to 71% and 77%; causal explanation is supported explicitly only on page 2193.

Reviewer: ______  Decision: ______  Corrections / additional alternatives: ______

## Query 13: How many quarterly blackout periods are there for executives, directors, and notified employees at Wells Fargo?

Classification: `single_fact_joint_support`. Status: **draft**.

- **quarterly_periods:** Four quarterly freeze periods apply to Section 16 insiders and other notified employees. Support: [corpus 2551, PDF p. 341](https://huggingface.co/datasets/vidore/vidore_v3_finance_en/resolve/7f432c176d82e27546501ad8064a713ac3071809/pdfs/wells_fargo_2024.pdf#page=341) + [corpus 2552, PDF p. 342](https://huggingface.co/datasets/vidore/vidore_v3_finance_en/resolve/7f432c176d82e27546501ad8064a713ac3071809/pdfs/wells_fargo_2024.pdf#page=342).

Review note: Count March/June/September/December. Page 2551 establishes who is covered; it is outside the original page qrels.

Reviewer: ______  Decision: ______  Corrections / additional alternatives: ______

## Query 14: Summarize the main business segments of Goldman Sachs as reported in their 2024 Form 10-K

Classification: `enumeration`. Status: **draft**.

- **segment_1:** Goldman Sachs business segment: Global Banking & Markets. Support: [corpus 576, PDF p. 224](https://huggingface.co/datasets/vidore/vidore_v3_finance_en/resolve/7f432c176d82e27546501ad8064a713ac3071809/pdfs/goldman_sachs_2024.pdf#page=224) OR [corpus 771, PDF p. 5](https://huggingface.co/datasets/vidore/vidore_v3_finance_en/resolve/7f432c176d82e27546501ad8064a713ac3071809/pdfs/goldman_sachs_2024.pdf#page=5) OR [corpus 577, PDF p. 225](https://huggingface.co/datasets/vidore/vidore_v3_finance_en/resolve/7f432c176d82e27546501ad8064a713ac3071809/pdfs/goldman_sachs_2024.pdf#page=225) OR [corpus 1023, PDF p. 75](https://huggingface.co/datasets/vidore/vidore_v3_finance_en/resolve/7f432c176d82e27546501ad8064a713ac3071809/pdfs/goldman_sachs_2024.pdf#page=75).
- **segment_2:** Goldman Sachs business segment: Asset & Wealth Management. Support: [corpus 576, PDF p. 224](https://huggingface.co/datasets/vidore/vidore_v3_finance_en/resolve/7f432c176d82e27546501ad8064a713ac3071809/pdfs/goldman_sachs_2024.pdf#page=224) OR [corpus 771, PDF p. 5](https://huggingface.co/datasets/vidore/vidore_v3_finance_en/resolve/7f432c176d82e27546501ad8064a713ac3071809/pdfs/goldman_sachs_2024.pdf#page=5) OR [corpus 577, PDF p. 225](https://huggingface.co/datasets/vidore/vidore_v3_finance_en/resolve/7f432c176d82e27546501ad8064a713ac3071809/pdfs/goldman_sachs_2024.pdf#page=225) OR [corpus 1023, PDF p. 75](https://huggingface.co/datasets/vidore/vidore_v3_finance_en/resolve/7f432c176d82e27546501ad8064a713ac3071809/pdfs/goldman_sachs_2024.pdf#page=75).
- **segment_3:** Goldman Sachs business segment: Platform Solutions. Support: [corpus 576, PDF p. 224](https://huggingface.co/datasets/vidore/vidore_v3_finance_en/resolve/7f432c176d82e27546501ad8064a713ac3071809/pdfs/goldman_sachs_2024.pdf#page=224) OR [corpus 771, PDF p. 5](https://huggingface.co/datasets/vidore/vidore_v3_finance_en/resolve/7f432c176d82e27546501ad8064a713ac3071809/pdfs/goldman_sachs_2024.pdf#page=5) OR [corpus 577, PDF p. 225](https://huggingface.co/datasets/vidore/vidore_v3_finance_en/resolve/7f432c176d82e27546501ad8064a713ac3071809/pdfs/goldman_sachs_2024.pdf#page=225) OR [corpus 1023, PDF p. 75](https://huggingface.co/datasets/vidore/vidore_v3_finance_en/resolve/7f432c176d82e27546501ad8064a713ac3071809/pdfs/goldman_sachs_2024.pdf#page=75).

Review note: Check completeness and alternative sources.

Reviewer: ______  Decision: ______  Corrections / additional alternatives: ______

## Query 17: Bank of America Series 5 Preferred Stock redemption value per share

Classification: `single_fact`. Status: **draft**.

- **redemption_value:** Series 5 redemption price is $30,000 per share plus declared and unpaid dividends, excluding undeclared dividends. Support: [corpus 2790, PDF p. 237](https://huggingface.co/datasets/vidore/vidore_v3_finance_en/resolve/7f432c176d82e27546501ad8064a713ac3071809/pdfs/bank_of_america_2024.pdf#page=237).

Review note: The summary table lists liquidation preference; use the explicit redemption clause for this requirement.

Reviewer: ______  Decision: ______  Corrections / additional alternatives: ______

## Query 19: What was the net income reported by Bank of America in its Consumer Banking segment for 2024?

Classification: `single_fact`. Status: **draft**.

- **consumer_income:** BofA total Consumer Banking 2024 net income was $10,759 million. Support: [corpus 2874, PDF p. 39](https://huggingface.co/datasets/vidore/vidore_v3_finance_en/resolve/7f432c176d82e27546501ad8064a713ac3071809/pdfs/bank_of_america_2024.pdf#page=39).

Review note: Keep table headers: $7,515M is the Deposits subsegment, not total Consumer Banking.

Reviewer: ______  Decision: ______  Corrections / additional alternatives: ______

## Query 22: Bank of America preferred stock series ss vs tt redemption terms and dividend rates

Classification: `comparison`. Status: **draft**.

- **ss_rate:** Series SS is 4.750% non-cumulative preferred stock. Support: [corpus 2736, PDF p. 189](https://huggingface.co/datasets/vidore/vidore_v3_finance_en/resolve/7f432c176d82e27546501ad8064a713ac3071809/pdfs/bank_of_america_2024.pdf#page=189).
- **tt_rate:** Series TT starts at 6.125% and resets from April 27, 2027 to the five-year Treasury rate plus 3.231%. Support: [corpus 2780, PDF p. 228](https://huggingface.co/datasets/vidore/vidore_v3_finance_en/resolve/7f432c176d82e27546501ad8064a713ac3071809/pdfs/bank_of_america_2024.pdf#page=228).
- **ss_redemption:** SS is callable from February 17, 2027 at $25,000 plus applicable dividends; capital-treatment early redemption and no holder option apply. Support: [corpus 2779, PDF p. 227](https://huggingface.co/datasets/vidore/vidore_v3_finance_en/resolve/7f432c176d82e27546501ad8064a713ac3071809/pdfs/bank_of_america_2024.pdf#page=227).
- **tt_redemption:** TT is callable from April 27, 2027 at $25,000 plus applicable dividends; capital-treatment early redemption and no holder option apply. Support: [corpus 2781, PDF p. 229](https://huggingface.co/datasets/vidore/vidore_v3_finance_en/resolve/7f432c176d82e27546501ad8064a713ac3071809/pdfs/bank_of_america_2024.pdf#page=229) + [corpus 2782, PDF p. 230](https://huggingface.co/datasets/vidore/vidore_v3_finance_en/resolve/7f432c176d82e27546501ad8064a713ac3071809/pdfs/bank_of_america_2024.pdf#page=230).

Review note: OCR duplicates 3. in TT spread; inspect PDF before approval. Pages 2779 and 2782 are supporting pages absent from original qrels. Rubric includes reset and exceptional redemption terms omitted by the dataset answer.

Reviewer: ______  Decision: ______  Corrections / additional alternatives: ______

## Query 24: What was the total net income of Bank of America in 2024?

Classification: `single_fact_alternatives`. Status: **draft**.

- **net_income:** BofA 2024 total net income was $27,132 million. Support: [corpus 2689, PDF p. 146](https://huggingface.co/datasets/vidore/vidore_v3_finance_en/resolve/7f432c176d82e27546501ad8064a713ac3071809/pdfs/bank_of_america_2024.pdf#page=146) OR [corpus 2869, PDF p. 34](https://huggingface.co/datasets/vidore/vidore_v3_finance_en/resolve/7f432c176d82e27546501ad8064a713ac3071809/pdfs/bank_of_america_2024.pdf#page=34) OR [corpus 2934, PDF p. 93](https://huggingface.co/datasets/vidore/vidore_v3_finance_en/resolve/7f432c176d82e27546501ad8064a713ac3071809/pdfs/bank_of_america_2024.pdf#page=93).

Review note: Alternative consolidated sources. Parent-only statements on 2715/2717 are not independently labeled consolidated support despite matching net income.

Reviewer: ______  Decision: ______  Corrections / additional alternatives: ______

## Query 55: Which specific energy sector index is directly referenced in the design of JPMorgan Chase's senior unsecured debt instruments?

Classification: `single_fact`. Status: **draft**.

- **index_name:** JPMorgan Financial Alerian ETNs link to the Alerian MLP Index, Bloomberg AMZ. Support: [corpus 275, PDF p. 347](https://huggingface.co/datasets/vidore/vidore_v3_finance_en/resolve/7f432c176d82e27546501ad8064a713ac3071809/pdfs/jpmorgan_chase_2024.pdf#page=347).

Review note: Check completeness and alternative sources.

Reviewer: ______  Decision: ______  Corrections / additional alternatives: ______
