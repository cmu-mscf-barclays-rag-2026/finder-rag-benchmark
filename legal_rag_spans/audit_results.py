"""Recheck saved pilot scores with character sets, independent of interval-union code."""
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

import span_bench as s


def audit(root=s.ROOT):
    docs, annotations, tokenizer = s.load(root)
    by_id = {a['question_id']: a for a in annotations}
    rankings = s.read_json(root / 'results/rankings.json')
    rows = list(csv.DictReader((root / 'results/per_query.csv').open(encoding='utf-8')))
    checks = 0
    for row in rows:
        annotation = by_id[row['question_id']]
        ranked = rankings[row['mode']][row['question_id']][:int(row['k'])]
        fragments = []
        for item in ranked:
            f = dict(item)
            doc = docs[f['source_document']]
            assert f['source_sha256'] == s.text_hash(doc['text'])
            f['text'] = doc['text'][f['start_char']:f['end_char']]
            fragments.append(f)
        # Independently pack prefixes using actual standalone token counts.
        budget = None if row['budget'] == 'unbounded' else int(row['budget'])
        seen = defaultdict(set)
        used = 0
        for f in fragments:
            encoded = tokenizer.encode(f['text'], add_special_tokens=False)
            room = len(encoded.ids) if budget is None else budget - used
            if room <= 0:
                break
            stop = len(f['text'])
            if len(encoded.ids) > room:
                take = room
                stop = encoded.offsets[take-1][1]
                while len(tokenizer.encode(f['text'][:stop], add_special_tokens=False).ids) > room:
                    take -= 1
                    if take == 0:
                        stop = 0
                        break
                    stop = encoded.offsets[take-1][1]
            used += len(tokenizer.encode(f['text'][:stop], add_special_tokens=False).ids)
            seen[f['source_document']].update(range(f['start_char'], f['start_char'] + stop))
            if stop < len(f['text']):
                break
        gold = defaultdict(set)
        flags = []
        for span in annotation['spans']:
            positions = set(range(span['start_char'], span['end_char']))
            gold[span['source_document']].update(positions)
            flags.append(positions <= seen[span['source_document']])
        expected_chars = sum(len(pos & seen[doc]) for doc, pos in gold.items()) / sum(map(len, gold.values()))
        assert math.isclose(float(row['evidence_char_coverage']), expected_chars, abs_tol=1e-12)
        assert float(row['all_spans_included']) == int(all(flags))
        assert math.isclose(float(row['complete_span_recall']), sum(flags)/len(flags), abs_tol=1e-12)
        # Reconstruct contiguous gold runs without using the evaluator's merging function.
        total = included = 0
        for doc_id, positions in gold.items():
            ordered = sorted(positions)
            runs = []
            for pos in ordered:
                if runs and pos == runs[-1][1]:
                    runs[-1][1] += 1
                else:
                    runs.append([pos, pos+1])
            for start, end in runs:
                for a, b in tokenizer.encode(docs[doc_id]['text'][start:end], add_special_tokens=False).offsets:
                    if b > a:
                        total += 1
                        included += set(range(start+a, start+b)) <= seen[doc_id]
        assert math.isclose(float(row['evidence_token_coverage']), included/total, abs_tol=1e-12)
        assert int(row['returned_tokens']) == used
        assert budget is None or used <= budget
        checks += 1
    summaries = list(csv.DictReader((root / 'results/comparison.csv').open(encoding='utf-8')))
    for summary in summaries:
        group = [r for r in rows if all(r[k] == summary[k] for k in ['mode', 'k', 'budget'])]
        assert len(group) == 20
        for metric in ['all_spans_included', 'complete_span_recall', 'evidence_char_coverage', 'evidence_token_coverage', 'whole_gold_child_included', 'returned_tokens']:
            assert math.isclose(float(summary[metric]), sum(float(r[metric]) for r in group)/len(group), abs_tol=1e-12)
    manifest = s.read_json(root / 'results/run_manifest.json')
    assert manifest['implementation_sha256'] == hashlib.sha256((root / 'span_bench.py').read_bytes()).hexdigest()
    assert manifest['annotation_sha256'] == hashlib.sha256((root / 'data/annotations.json').read_bytes()).hexdigest()
    assert manifest['config'] == s.read_json(root / 'config.json')
    result = {'query_setting_rows_independently_checked': checks, 'summary_rows_checked': len(summaries),
              'character_union_token_coverage_and_budget_checks': 'passed',
              'human_semantic_review': 'not performed; cannot be replaced by these tests'}
    s.write_json(root / 'results/audit.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    audit()
