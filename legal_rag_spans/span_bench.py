"""Source-coordinate evaluation and a fixed BM25 pilot; no answer generation."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import time

import numpy as np
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def text_hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def merge_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if start < 0 or end <= start:
            raise ValueError('Intervals must be nonempty [start, end) ranges')
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    return merged


def covered_length(start, end, ranges):
    return sum(max(0, min(end, b) - max(start, a)) for a, b in merge_intervals(ranges))


def validate_fragment(fragment, documents):
    doc = documents[fragment['source_document']]
    start, end = fragment['start_char'], fragment['end_char']
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(doc['text']):
        raise ValueError('Invalid source offsets')
    if fragment['source_sha256'] != text_hash(doc['text']):
        raise ValueError('Source version mismatch')
    if fragment['text'] != doc['text'][start:end]:
        raise ValueError('Returned text does not match its claimed coordinates')


def evaluate(annotation, fragments, documents, tokenizer):
    """Score unioned, actually returned source text, not retrieved chunk IDs."""
    returned = defaultdict(list)
    for fragment in fragments:
        validate_fragment(fragment, documents)
        returned[fragment['source_document']].append((fragment['start_char'], fragment['end_char']))
    gold = defaultdict(list)
    complete = []
    for span in annotation['spans']:
        validate_fragment(span, documents)
        doc_id, start, end = span['source_document'], span['start_char'], span['end_char']
        gold[doc_id].append((start, end))
        complete.append(covered_length(start, end, returned[doc_id]) == end - start)
    if not complete:
        raise ValueError('A scored annotation must contain at least one evidence span')
    total_chars = hit_chars = total_tokens = hit_tokens = 0
    for doc_id, intervals in gold.items():
        for start, end in merge_intervals(intervals):
            total_chars += end - start
            hit_chars += covered_length(start, end, returned[doc_id])
            offsets = tokenizer.encode(documents[doc_id]['text'][start:end], add_special_tokens=False).offsets
            for a, b in offsets:
                if b > a:
                    total_tokens += 1
                    hit_tokens += covered_length(start + a, start + b, returned[doc_id]) == b - a
    return {
        'any_span_overlap': int(hit_chars > 0),
        'any_complete_span': int(any(complete)),
        'all_spans_included': int(all(complete)),
        'complete_span_recall': sum(complete) / len(complete),
        'evidence_char_coverage': hit_chars / total_chars,
        'evidence_token_coverage': hit_tokens / total_tokens if total_tokens else 0.0,
    }


def load(root=ROOT):
    manifest = read_json(root / 'data/manifest.json')
    for relative, expected in manifest['files_sha256'].items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
            raise ValueError('Packaged input changed: ' + relative)
    with gzip.open(root / 'data/sources.jsonl.gz', 'rt', encoding='utf-8') as handle:
        rows = [json.loads(line) for line in handle]
    documents = {row['id']: row for row in rows}
    if len(documents) != len(rows):
        raise ValueError('Duplicate source IDs')
    annotations = read_json(root / 'data/annotations.json')
    if len({a['question_id'] for a in annotations}) != len(annotations):
        raise ValueError('Duplicate question IDs')
    tokenizer = Tokenizer.from_file(str(root / 'assets/context_tokenizer.json'))
    # The serialized model tokenizer has inference padding/truncation settings.
    # Evidence counting and coordinate chunking must always see the complete text.
    tokenizer.no_truncation()
    tokenizer.no_padding()
    for annotation in annotations:
        if annotation['split'] != 'dev':
            raise ValueError('This pilot must use development questions only')
        for span in annotation['spans']:
            validate_fragment(span, documents)
    return documents, annotations, tokenizer


def fragment(doc, start, end, unit_id):
    return {'id': unit_id, 'source_document': doc['id'], 'source_sha256': text_hash(doc['text']),
            'start_char': start, 'end_char': end, 'text': doc['text'][start:end]}


def make_units(documents, mode, tokenizer, size=128, overlap=32):
    """Grandchildren tile each child with overlapping WordPiece windows."""
    if not 0 <= overlap < size:
        raise ValueError('Require 0 <= overlap < window size')
    if mode not in {'parent', 'child', 'grandchild'}:
        raise ValueError('Unknown corpus mode')
    units = []
    for doc in documents.values():
        if mode == 'parent':
            units.append(fragment(doc, 0, len(doc['text']), doc['id']))
            continue
        for child in doc['child_spans']:
            start, end = child['start_char'], child['end_char']
            if mode == 'child':
                units.append(fragment(doc, start, end, child['child_id']))
                continue
            offsets = tokenizer.encode(doc['text'][start:end], add_special_tokens=False).offsets
            if not offsets:
                units.append(fragment(doc, start, end, child['child_id'] + '-g0'))
                continue
            index = 0
            while index < len(offsets):
                stop = min(index + size, len(offsets))
                a = start if index == 0 else start + offsets[index][0]
                # Keep inter-token whitespace so adjacent slices can reconstruct a span.
                b = end if stop == len(offsets) else start + offsets[stop][0]
                # A slice starting inside a word can tokenize differently on its own.
                while len(tokenizer.encode(doc['text'][a:b], add_special_tokens=False).ids) > size:
                    stop -= 1
                    if stop <= index:
                        raise ValueError('Unable to create a bounded token slice')
                    b = start + offsets[stop][0]
                units.append(fragment(doc, a, b, child['child_id'] + f'-g{index}'))
                if stop == len(offsets):
                    break
                index = max(index + 1, stop - overlap)
    if len({u['id'] for u in units}) != len(units):
        raise ValueError('Duplicate retrieval unit IDs')
    return units


def apply_budget(units, tokenizer, budget):
    """Prefix packing in rank order. Overlapping text still consumes budget."""
    if budget is not None and (type(budget) is not int or budget < 0):
        raise ValueError('Budget must be a nonnegative integer or None')
    selected, used = [], 0
    for unit in units:
        offsets = tokenizer.encode(unit['text'], add_special_tokens=False).offsets
        remaining = len(offsets) if budget is None else budget - used
        if remaining <= 0:
            break
        result = dict(unit)
        if len(offsets) > remaining:
            take = remaining
            end = offsets[take - 1][1]
            while len(tokenizer.encode(unit['text'][:end], add_special_tokens=False).ids) > remaining:
                take -= 1
                if take == 0:
                    return selected, used
                end = offsets[take - 1][1]
            result['text'] = unit['text'][:end]
            result['end_char'] = unit['start_char'] + end
            selected.append(result)
            used += len(tokenizer.encode(result['text'], add_special_tokens=False).ids)
            break
        selected.append(result)
        used += len(offsets)
    return selected, used


def terms(text):
    return re.findall(r'\w+', text.casefold())


class BM25:
    """BM25 with positive Robertson IDF, k1=1.5 and b=0.75; no tuning."""
    def __init__(self, texts, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        counts = [Counter(terms(text)) for text in texts]
        self.lengths = np.array([sum(c.values()) for c in counts], dtype=float)
        self.n = len(counts)
        self.avg = float(self.lengths.mean()) if self.n else 0
        self.postings = defaultdict(list)
        for index, count in enumerate(counts):
            for term, frequency in count.items():
                self.postings[term].append((index, frequency))

    def scores(self, query):
        scores = np.zeros(self.n)
        for term, qtf in Counter(terms(query)).items():
            matches = self.postings.get(term, [])
            if not matches:
                continue
            ids = np.array([x[0] for x in matches])
            tf = np.array([x[1] for x in matches], dtype=float)
            idf = math.log(1 + (self.n - len(matches) + .5) / (len(matches) + .5))
            norm = self.k1 * (1 - self.b + self.b * self.lengths[ids] / self.avg)
            scores[ids] += qtf * idf * tf * (self.k1 + 1) / (tf + norm)
        return scores

    def rank(self, query, k):
        return np.argsort(-self.scores(query), kind='stable')[:k].tolist()


def export_csv(path, rows):
    with Path(path).open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def boundary_control(documents, annotations, tokenizer):
    annotation = annotations[0]
    span = annotation['spans'][0]
    doc = documents[span['source_document']]
    start, end = span['start_char'], span['end_char']
    middle = (start + end) // 2
    cases = {'one_piece': [fragment(doc, start, end, 'whole')],
             'two_pieces': [fragment(doc, start, middle, 'left'), fragment(doc, middle, end, 'right')],
             'duplicate': [fragment(doc, start, end, 'whole')] * 2,
             'missing_tail': [fragment(doc, start, middle, 'left')]}
    return [{'case': name, **evaluate(annotation, parts, documents, tokenizer)} for name, parts in cases.items()]


def run(root=ROOT):
    documents, annotations, tokenizer = load(root)
    config = read_json(root / 'config.json')
    output = root / 'results'
    output.mkdir(exist_ok=True)
    rows, rankings, profiles = [], {}, []
    for mode in ['parent', 'child', 'grandchild']:
        units = make_units(documents, mode, tokenizer, config['grandchild_tokens'], config['grandchild_overlap'])
        lengths = [len(tokenizer.encode(u['text'], add_special_tokens=False).ids) for u in units]
        begin = time.perf_counter()
        retriever = BM25([u['text'] for u in units], config['bm25_k1'], config['bm25_b'])
        profiles.append({'mode': mode, 'units': len(units), 'mean_tokens': float(np.mean(lengths)),
                         'min_tokens': min(lengths), 'max_tokens': max(lengths),
                         'index_seconds': time.perf_counter() - begin})
        rankings[mode] = {}
        for annotation in annotations:
            # The question alone enters retrieval. Answers and annotations enter evaluation only.
            ids = retriever.rank(annotation['question'], max(config['ks']))
            ranked = [units[i] for i in ids]
            rankings[mode][annotation['question_id']] = [
                {key: value for key, value in unit.items() if key != 'text'} for unit in ranked]
            gold_child = annotation['original_gold_child_id']
            child_span = next(s for s in documents[annotation['spans'][0]['source_document']]['child_spans'] if s['child_id'] == gold_child)
            original = dict(annotation)
            original['spans'] = [fragment(documents[annotation['spans'][0]['source_document']], child_span['start_char'], child_span['end_char'], gold_child)]
            for k in config['ks']:
                for budget in config['budgets']:
                    returned, count = apply_budget(ranked[:k], tokenizer, budget)
                    metrics = evaluate(annotation, returned, documents, tokenizer)
                    legacy = evaluate(original, returned, documents, tokenizer)
                    rows.append({'mode': mode, 'question_id': annotation['question_id'], 'k': k,
                                 'budget': 'unbounded' if budget is None else budget,
                                 **metrics, 'whole_gold_child_included': legacy['all_spans_included'],
                                 'returned_tokens': count, 'returned_units': len(returned),
                                 'annotation_status': annotation['review_status']})
    metric_keys = list(evaluate(annotations[0], [], documents, tokenizer)) + ['whole_gold_child_included', 'returned_tokens', 'returned_units']
    groups = defaultdict(list)
    for row in rows:
        groups[(row['mode'], row['k'], row['budget'])].append(row)
    summary = []
    for (mode, k, budget), group in groups.items():
        summary.append({'mode': mode, 'k': k, 'budget': budget, 'queries': len(group),
                        **{key: float(np.mean([r[key] for r in group])) for key in metric_keys}})
    export_csv(output / 'per_query.csv', rows)
    export_csv(output / 'comparison.csv', summary)
    export_csv(output / 'corpus_profile.csv', profiles)
    write_json(output / 'rankings.json', rankings)
    write_json(output / 'boundary_controls.json', boundary_control(documents, annotations, tokenizer))
    write_json(output / 'run_manifest.json', {
        'scope': 'Actual CPU BM25 retrieval on all source units; 20 development queries; draft span labels',
        'config': config, 'source_manifest_sha256': hashlib.sha256((root / 'data/manifest.json').read_bytes()).hexdigest(),
        'implementation_sha256': hashlib.sha256((root / 'span_bench.py').read_bytes()).hexdigest(),
        'annotation_sha256': hashlib.sha256((root / 'data/annotations.json').read_bytes()).hexdigest(),
        'python': __import__('sys').version, 'numpy': np.__version__,
        'questions': len(annotations), 'sources': len(documents), 'evaluated_query_settings': len(rows),
        'human_approved_questions': sum(a['review_status'] == 'human_approved' for a in annotations),
        'no_test_questions_used': True, 'no_parameter_tuning': True})
    print(f'Completed {len(rows)} query/settings across three corpora. Labels require human review.')
    return summary


def score_external(root, prediction_path, output_path, k=5, budget=2048, approved_only=False):
    documents, annotations, tokenizer = load(root)
    predictions = read_json(prediction_path)
    selected = [a for a in annotations if not approved_only or a['review_status'] == 'human_approved']
    if not selected:
        raise ValueError('No human-approved annotations are available; do not claim a verified benchmark')
    rows = []
    for a in selected:
        if a['question_id'] not in predictions:
            raise ValueError('Missing prediction for query ' + a['question_id'])
        candidates = predictions[a['question_id']]
        for f in candidates:
            validate_fragment(f, documents)
        returned, tokens = apply_budget(candidates[:k], tokenizer, budget)
        rows.append({'question_id': a['question_id'], **evaluate(a, returned, documents, tokenizer),
                     'returned_tokens': tokens})
    export_csv(output_path, rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['run', 'validate', 'score'])
    parser.add_argument('--predictions', type=Path)
    parser.add_argument('--output', type=Path, default=Path('external_scores.csv'))
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--budget', type=int, default=2048)
    parser.add_argument('--approved-only', action='store_true')
    args = parser.parse_args()
    if args.command == 'run':
        run()
    elif args.command == 'validate':
        docs, annotations, tokenizer = load()
        print(f'{len(docs)} canonical sources; {len(annotations)} source-exact draft annotations.')
    else:
        if args.predictions is None or args.k <= 0:
            parser.error('score requires --predictions and positive --k')
        score_external(ROOT, args.predictions, args.output, args.k, args.budget, args.approved_only)
