"""Materialize explicitly drafted facts against the pinned source text.

This is an authored pilot, NOT an automatic conversion of qrels to facts.
All outputs remain draft until an independent reviewer checks the source PDFs.
"""
from pathlib import Path
import argparse

from week9_evidence.vidore import load_dataset, write_jsonl
from week9_evidence.evidence import validate_annotations


def build(folder):
    pages, queries, qrels, manifest = load_dataset(folder)

    def span(cid, first, last=None):
        text = pages[cid]['text']
        start = text.index(first)
        end = start + len(first) if last is None else text.index(last, start) + len(last)
        return {'page_id': cid, 'start': start, 'end': end, 'quote': text[start:end]}

    def fact(fid, requirement, *support_sets):
        return {'fact_id': fid, 'requirement': requirement, 'support_sets': list(support_sets)}

    records = []

    def add(qid, kind, facts, note=''):
        records.append({'query_id': qid, 'question': queries[qid]['query'],
            'question_type': kind, 'source_query_types': queries[qid]['query_types'],
            'content_types': queries[qid]['content_type'], 'status': 'draft',
            'annotator': 'Codex source-grounded draft; not human validated', 'reviewer': None,
            'dataset_revision': manifest['revision'], 'facts': facts, 'review_notes': note,
            'annotation_scope': 'Known supporting spans; unannotated alternatives are not proven insufficient.'})

    add('2', 'multi_fact', [
        fact('risk_cycle', 'Citi identifies, measures, monitors, controls and reports risks through its risk management cycle.',
             [span('1707', 'Risk Management-the end-to-end', 'anterprise-wide level;')]),
        fact('governance', 'Banking-book interest rate risk follows the Non-Trading Market Risk Policy; Treasury manages it and reviews it with ALCO.',
             [span('1057', 'Corporate Treasury is responsible', 'reviewing it with the ALCO.'),
              span('1057', 'Banking Book Interest Rate Risk', "approval by Citigroup's Board of Directors.")]),
        fact('metrics', 'Non-trading interest-rate and FX changes can affect net interest income, economic value of equity and AOCI.',
             [span('1707', 'Market risk of non-trading portfolios is', 'economic value of equity or AOCI.')])],
        'Broad question: reviewer should check that this three-part rubric is complete, especially FX management.')

    add('9', 'multi_fact', [
        fact('fixed_rate', 'Series MM pays 4.300% until, but excluding, January 28, 2025.',
             [span('2766', 'Dividends on each share of Series MM', '(the "fixed rate period")')]),
        fact('floating_rate', 'From January 28, 2025 the rate is Adjusted Three-Month Term SOFR plus 2.664%.',
             [span('2766', 'Dividends on each share of Series MM', '(the "floating rate period").')])],
        'Page 2687 has corrupted OCR for the SOFR spread; do not infer the correct spread from that page text.')

    add('10', 'multi_fact', [
        fact('revenue', 'Morgan Stanley revenue rose from $54.1B in 2023 to $61.8B in 2024 (14%).',
             [span('2193', 'We reported net revenues of $61.8', '$54.1 billion in 2023.')],
             [span('2258', '| In millions, except per share data', '| Net revenues | 61,761 | 54,143 | 53,668 |')]),
        fact('efficiency', 'The expense efficiency ratio improved from 77% to 71%, reflecting higher revenue and expense discipline.',
             [span('2193', 'The Firm expense efficiency ratio', 'higher revenues and expense discipline.')],
             [span('2258', '| In millions, except per share data', '| Total non-interest expenses | 43,901 | 41,798 | 39,299 |')])],
        'Table alternative requires division: expenses/revenue rounds to 71% and 77%; causal explanation is supported explicitly only on page 2193.')
    # Keep the causal explanation as a separate required fact rather than crediting
    # a numerical table with management's explanation.
    records[-1]['facts'][1]['requirement'] = 'The expense efficiency ratio improved from approximately 77% to 71%.'
    records[-1]['facts'].append(fact('explanation', 'The report attributes the ratio improvement to higher revenues and expense discipline.',
        [span('2193', 'The Firm expense efficiency ratio', 'higher revenues and expense discipline.')]))

    add('13', 'single_fact_joint_support', [
        fact('quarterly_periods', 'Four quarterly freeze periods apply to Section 16 insiders and other notified employees.',
             [span('2551', '4. Trading Freeze Periods', 'during quarterly "freeze period".'),
              span('2552', 'The quarterly freeze period begins', 'earnings results for that fiscal quarter.')])],
        'Count March/June/September/December. Page 2551 establishes who is covered; it is outside the original page qrels.')

    segments = ['Global Banking & Markets', 'Asset & Wealth Management', 'Platform Solutions']
    segment_support = [[span('576', 'The firm manages and reports', 'Asset & Wealth Management and Platform Solutions.')],
                       [span('771', 'We manage and report our activities', 'Asset & Wealth Management and Platform Solutions.')]]
    add('14', 'enumeration', [fact(f'segment_{i}', f'Goldman Sachs business segment: {name}.',
        *segment_support, [span('577', f'| {name} |')], [span('1023', f'| {name} |')])
        for i, name in enumerate(segments, 1)])

    add('17', 'single_fact', [
        fact('redemption_value', 'Series 5 redemption price is $30,000 per share plus declared and unpaid dividends, excluding undeclared dividends.',
             [span('2790', 'Redemption. The Company may redeem the Series 5', 'without accumulation of any undeclared dividends.')])],
        'The summary table lists liquidation preference; use the explicit redemption clause for this requirement.')

    add('19', 'single_fact', [
        fact('consumer_income', 'BofA total Consumer Banking 2024 net income was $10,759 million.',
             [span('2874', '| Consumer Banking |', '| Net income | $ 7,515 $ |  | 8,683 | $ 3,244 | 2,910 | $ 10,759 | 11,593 | (7) |')])],
        'Keep table headers: $7,515M is the Deposits subsegment, not total Consumer Banking.')

    add('22', 'comparison', [
        fact('ss_rate', 'Series SS is 4.750% non-cumulative preferred stock.',
             [span('2736', '28,000 shares were designated as 4.750%', '27,463 shares of which were issued and outstanding;')]),
        fact('tt_rate', 'Series TT starts at 6.125% and resets from April 27, 2027 to the five-year Treasury rate plus 3.231%.',
             [span('2780', 'Dividends on each share of Series TT', 'plus a spread of 3. 3.231% per annum.')]),
        fact('ss_redemption', 'SS is callable from February 17, 2027 at $25,000 plus applicable dividends; capital-treatment early redemption and no holder option apply.',
             [span('2779', 'Redemption. The Company may redeem the Series SS', 'do not have any optional redemption rights.')]),
        fact('tt_redemption', 'TT is callable from April 27, 2027 at $25,000 plus applicable dividends; capital-treatment early redemption and no holder option apply.',
             [span('2781', 'Redemption. The Company may redeem the Series TT', 'at a redemption price equal to $25,000 per share, plus'),
              span('2782', 'any accrued and unpaid dividends', 'do not have any optional redemption rights.')])],
        'OCR duplicates 3. in TT spread; inspect PDF before approval. Pages 2779 and 2782 are supporting pages absent from original qrels. Rubric includes reset and exceptional redemption terms omitted by the dataset answer.')

    add('24', 'single_fact_alternatives', [
        fact('net_income', 'BofA 2024 total net income was $27,132 million.',
             [span('2689', '| NOTE 15', '| Earnings per common share Net income |  | 27,132 | $ | 26,515 | $ | 27,528 |')],
             [span('2869', pages['2869']['text'][:60], '| Net income | 27,132 |  | 26,515 |  | 27,528 |')],
             [span('2934', '| Bank of America Corporation and Subsidiaries', '| Net income | $ 27,132 $ | 26,515 | $ 27,528 |')])],
        'Alternative consolidated sources. Parent-only statements on 2715/2717 are not independently labeled consolidated support despite matching net income.')

    add('55', 'single_fact', [
        fact('index_name', 'JPMorgan Financial Alerian ETNs link to the Alerian MLP Index, Bloomberg AMZ.',
             [span('275', 'The return on the Alerian ETNs is linked', '(Bloomberg ticker: AMZ)')])])

    validate_annotations(records, pages, queries, allow_draft=True)
    return records


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Refusing to overwrite annotations; choose a new output path')
    write_jsonl(args.output, build(args.dataset))
