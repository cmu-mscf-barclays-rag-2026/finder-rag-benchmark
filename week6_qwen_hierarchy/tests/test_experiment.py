"""Regression checks for evidence provenance, budgets, and development-only selection."""
import csv
import json
from pathlib import Path
import unittest

from tokenizers import Tokenizer, models, pre_tokenizers
from week6_qwen_hierarchy.qwen_hierarchy import (
    HERE, REFERENCE, CachedTokenizer, choose_window, evaluate, exact_mcnemar_p, load_reference,
    paired_comparisons,
    selected_sibling_context, sibling_candidates, validate_config,
)
from week5_hierarchy.hierarchy_experiment import ROOT, load_data
from hierarchy.handoff import evaluate_run
from hierarchy.core import SEPARATOR


def fixture():
    tokenizer = Tokenizer(models.WordLevel({'[UNK]': 0, 'aa': 1, 'bb': 2, 'cc': 3, 'dd': 4}, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    children = {str(i): {'id': str(i), 'parent_id': 'p', 'text': text}
                for i, text in enumerate(['aa aa aa aa', 'bb bb', 'cc cc', 'dd dd'])}
    text, spans = '', []
    for pid, child in children.items():
        text += SEPARATOR if text else ''
        start = len(text)
        text += child['text']
        spans.append({'child_id': pid, 'start_char': start, 'end_char': len(text)})
    return children, {'p': {'id': 'p', 'text': text, 'child_spans': spans}}, CachedTokenizer(tokenizer)


class ContextTests(unittest.TestCase):
    def test_exact_paired_test_and_comparison_direction(self):
        self.assertEqual(exact_mcnemar_p(0, 0), 1)
        self.assertEqual(exact_mcnemar_p(4, 0), .125)
        details = []
        for qid, child, parent in [('a', 1, 0), ('b', 0, 1), ('c', 0, 1), ('d', 1, 1)]:
            base = {'embedding': 'Qwen3-0.6B', 'embedding_window': 4096, 'method': 'dense',
                    'split': 'test', 'context_budget': 2048, 'k': 10, 'query_id': qid}
            details += [{**base, 'mode': 'child', 'hit': child}, {**base, 'mode': 'parent', 'hit': parent}]
        rows = paired_comparisons(details, 4096)
        row = next(r for r in rows if r['family'] == 'hierarchy_vs_child')
        self.assertEqual((row['left_hit_count'], row['right_hit_count']), (3, 2))
        self.assertEqual((row['left_wins'], row['left_losses'], row['ties']), (2, 1, 1))
        self.assertEqual(row['hit_rate_difference'], .25)

    def test_seed_is_delivered_before_parent_prefix(self):
        children, parents, tokenizer = fixture()
        result = selected_sibling_context(['2'], children, parents, tokenizer, 1, 4)
        self.assertEqual(result['fully_included_child_ids'], ['2', '1'])
        self.assertEqual(result['text'], 'cc cc' + SEPARATOR + 'bb bb')
        self.assertEqual(result['returned_content_tokens'], 4)
        self.assertNotIn('0', result['selected_child_ids'])

    def test_neighbors_deduplicate_and_do_not_backfill_seeds(self):
        children, parents, _ = fixture()
        self.assertEqual(sibling_candidates(['2', '1', '3'], children, parents, 2), ['2', '1', '3', '0'])
        self.assertEqual(sibling_candidates(['0', '3'], children, parents, 1), ['0', '1'])

    def test_oversized_chunk_skipped_without_exceeding_budget(self):
        children, parents, tokenizer = fixture()
        result = selected_sibling_context(['0'], children, parents, tokenizer, 1, 2)
        self.assertEqual(result['fully_included_child_ids'], ['1'])
        self.assertEqual(result['returned_content_tokens'], 2)
        empty = selected_sibling_context(['0'], children, parents, tokenizer, 1, 0)
        self.assertEqual(empty['text'], '')
        self.assertEqual(empty['fully_included_child_ids'], [])

    def test_gold_evidence_coverage_and_budget_match_week5_evaluator(self):
        children, parents, tokenizer = fixture()
        query = {'id': 'q', 'split': 'test', 'gold_child_ids': ['1', '2']}
        predictions = {'q': {'child': ['1', '2', '0', '3'], 'parent': ['p']}}
        config = {'ks': [1, 2], 'context_budgets': [None, 0, 3, 6],
                  'modes': ['child', 'parent', 'child_to_parent']}
        details, _ = evaluate([query], predictions, children, parents, tokenizer, config, {})
        for mode in config['modes']:
            for budget in config['context_budgets']:
                old, _ = evaluate_run([query], {'q': predictions['q']['parent' if mode == 'parent' else 'child']},
                    children, parents, tokenizer, mode=mode, ks=config['ks'], token_budget=budget)
                new = [r for r in details if r['mode'] == mode and r['context_budget'] == budget]
                for before, after in zip(old, new):
                    self.assertEqual(before['gold_child_hit_in_context'], after['hit'])
                    self.assertEqual(before['gold_body_token_coverage'], after['gold_text_coverage'])
                    self.assertEqual(before['returned_content_tokens'], after['context_tokens'])
                    if not after['context_tokens']:
                        self.assertEqual(after['coverage_efficiency'], 0)

    def test_selects_only_development_and_breaks_ties_toward_shorter_window(self):
        config = json.loads((HERE / 'config/experiment.json').read_text())
        base = {**config['selection'], 'hit_rate': .4, 'gold_text_coverage': .5,
                'mean_coverage_efficiency': .001, 'embedding_window': 4096}
        rows = [base, {**base, 'embedding_window': 2048},
                {**base, 'split': 'test', 'embedding_window': 16384, 'hit_rate': 1.0}]
        self.assertEqual(choose_window(rows, config), 2048)
        validate_config(config)
        config['bm25_weight'] = .5
        with self.assertRaises(ValueError):
            validate_config(config)


class SavedReferenceTests(unittest.TestCase):
    def test_reference_ids_and_split_are_valid(self):
        children, parents, queries, _ = load_data(ROOT / 'week5_hierarchy/kevin_hierarchy')
        test = [q for q in queries if q['split'] == 'test']
        self.assertEqual(set(load_reference(test, children, parents)), {'bm25', 'dense', 'hybrid'})

    def test_saved_reference_reproduces_all_week5_primary_metrics(self):
        output = HERE / 'results/reference/hierarchy_comparison.csv'
        if not output.exists():
            self.skipTest('Run --reference-only to enable the saved-result regression.')
        with (REFERENCE / 'summary.csv').open() as stream:
            old = list(csv.DictReader(stream))
        with output.open() as stream:
            new = list(csv.DictReader(stream))
        count = 0
        for row in new:
            if row['mode'] == 'child_to_selected_siblings':
                continue
            method = 'hybrid_w0.75' if row['method'] == 'hybrid' else row['method']
            matches = [r for r in old if r['method'] == method and r['mode'] == row['mode']
                       and r['k'] == row['k'] and r['token_budget'] == row['context_budget']]
            self.assertEqual(len(matches), 1)
            before = matches[0]
            for previous, current in [('gold_context_hit_count', 'hit_count'),
                                      ('gold_body_token_coverage', 'gold_text_coverage'),
                                      ('returned_content_tokens', 'mean_context_tokens'),
                                      ('context_tokens_p95', 'p95_context_tokens')]:
                self.assertAlmostEqual(float(before[previous]), float(row[current]), places=10)
            count += 1
        self.assertEqual(count, 36)


if __name__ == '__main__':
    unittest.main()
