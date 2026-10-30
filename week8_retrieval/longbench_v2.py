"""Prepare LongBench v2 for query-scoped retrieval and score answer predictions.

LongBench v2 has answer letters, not passage qrels. This tool cannot produce
Hit@k or Recall@k for its chunks without new evidence annotations.
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
from pathlib import Path


def read_examples(path):
    ids = set()
    for row in iter_json_records(path):
        needed = {'_id', 'question', 'context', 'choice_A', 'choice_B', 'choice_C', 'choice_D', 'answer'}
        if not needed <= row.keys() or row['answer'] not in ('A', 'B', 'C', 'D'):
            raise ValueError('Invalid LongBench v2 example')
        if row['_id'] in ids:
            raise ValueError('Duplicate LongBench v2 ID')
        ids.add(row['_id'])
        yield row


def iter_json_records(path):
    """Stream a JSON array or JSONL without holding LongBench's full file."""
    with path.open() as f:
        first = f.read(1)
        while first.isspace():
            first = f.read(1)
        if first != '[':
            f.seek(0)
            for line in f:
                if line.strip():
                    yield json.loads(line)
            return
        decoder = json.JSONDecoder()
        buffer = ''
        ended = False
        while True:
            if not ended:
                chunk = f.read(1 << 20)
                ended = not chunk
                buffer += chunk
            buffer = buffer.lstrip()
            while buffer.startswith(','):
                buffer = buffer[1:].lstrip()
            if buffer.startswith(']'):
                if buffer[1:].strip() or f.read().strip():
                    raise ValueError('Unexpected content after JSON array')
                return
            try:
                row, consumed = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                if ended:
                    raise ValueError('Incomplete JSON array') from None
                continue
            if not isinstance(row, dict):
                raise ValueError('Expected JSON objects')
            yield row
            buffer = buffer[consumed:]


def chunks(context, words=350, overlap=50):
    if words < 1 or overlap < 0 or overlap >= words:
        raise ValueError('Require 0 <= overlap < chunk words')
    matches = list(re.finditer(r'\S+', context))
    step = words - overlap
    for start in range(0, len(matches), step):
        end = min(start + words, len(matches))
        if end - start <= overlap and start != 0:
            break
        yield context[matches[start].start():matches[end-1].end()]
        if end == len(matches):
            break


def prepare(source, output, chunk_words=350, overlap=50, max_questions=None):
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output must be new or empty')
    if max_questions is not None and max_questions < 1:
        raise ValueError('max_questions must be positive')
    rows = read_examples(source)
    if max_questions is not None:
        rows = itertools.islice(rows, max_questions)
    output.mkdir(parents=True, exist_ok=True)
    count = 0
    with (output / 'queries.jsonl').open('w') as qfile, \
            (output / 'chunks.jsonl').open('w') as cfile, \
            (output / 'answers.jsonl').open('w') as afile:
        for row in rows:
            count += 1
            qid = str(row['_id'])
            qfile.write(json.dumps({'query_id': qid, 'question': row['question'],
                'choices': {letter: row[f'choice_{letter}'] for letter in 'ABCD'},
                'domain': row.get('domain'),
                'sub_domain': row.get('sub_domain')}) + '\n')
            afile.write(json.dumps({'query_id': qid, 'answer': row['answer']}) + '\n')
            for number, body in enumerate(chunks(row['context'], chunk_words, overlap)):
                cfile.write(json.dumps({'query_id': qid, 'chunk_id': f'{qid}:{number}', 'text': body}) + '\n')
    (output / 'manifest.json').write_text(json.dumps({'source': str(source), 'questions': count,
        'chunk_words': chunk_words, 'overlap_words': overlap,
        'evaluation': 'Answer accuracy only. No public passage relevance labels.'}, indent=2) + '\n')


def score(prepared, predictions):
    query_ids = {r['query_id'] for r in read_examples_prepared(prepared / 'queries.jsonl')}
    labels = {r['query_id']: r['answer'] for r in read_examples_prepared(prepared / 'answers.jsonl')}
    if query_ids != set(labels) or not labels:
        raise ValueError('Answer IDs differ from prepared questions')
    answers = {}
    for row in read_examples_prepared(predictions):
        qid = str(row['query_id'])
        if qid in answers or row['answer'] not in ('A', 'B', 'C', 'D'):
            raise ValueError('Duplicate or invalid prediction')
        answers[qid] = row['answer']
    if query_ids != set(answers):
        raise ValueError('Prediction IDs differ from prepared questions')
    correct = sum(answers[qid] == answer for qid, answer in labels.items())
    return {'questions': len(labels), 'correct': correct, 'accuracy': correct / len(labels)}


def read_examples_prepared(path):
    with path.open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--input', type=Path, required=True)
    prep.add_argument('--output', type=Path, required=True)
    prep.add_argument('--chunk-words', type=int, default=350)
    prep.add_argument('--overlap', type=int, default=50)
    prep.add_argument('--max-questions', type=int)
    score_cmd = sub.add_parser('score')
    score_cmd.add_argument('--prepared', type=Path, required=True)
    score_cmd.add_argument('--predictions', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.input, args.output, args.chunk_words, args.overlap, args.max_questions)
    else:
        print(json.dumps(score(args.prepared, args.predictions), indent=2))


if __name__ == '__main__':
    main()
