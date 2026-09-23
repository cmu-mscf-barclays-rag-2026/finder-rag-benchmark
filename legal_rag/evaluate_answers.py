"""Summarize fully reviewed answers; never interpret missing judgments as zero."""
import argparse
import csv
import json
import statistics
from pathlib import Path


def summarize(rows, expected):
    keys = [(r['method'], str(r['query_id'])) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate method/query judgments')
    if set(keys) != expected:
        raise ValueError('Review rows do not match generation input method/query pairs')
    for r in rows:
        if not r['generated_answer'].strip() or not r.get('reviewer', '').strip():
            raise ValueError('Every answer needs generated text and reviewer/model identity')
        if r['correctness'] not in ('0', '1') or r['groundedness'] not in ('0', '1'):
            raise ValueError('Every correctness and groundedness judgment must be explicitly 0 or 1')
        if r.get('abstained') not in ('0', '1'):
            raise ValueError('Every abstained field must be explicitly 0 or 1')
        if r['abstained'] == '1' and r['correctness'] == '1':
            raise ValueError('An abstention is not correct on this answerable-question dataset')
    result = []
    for method in sorted({r['method'] for r in rows}):
        selected = [r for r in rows if r['method'] == method]
        result.append({'method': method, 'n_queries': len(selected),
                       'correctness': statistics.mean(int(r['correctness']) for r in selected),
                       'groundedness': statistics.mean(int(r['groundedness']) for r in selected),
                       'correct_and_grounded': statistics.mean(int(r['correctness']) * int(r['groundedness']) for r in selected),
                       'abstention_rate': statistics.mean(int(r['abstained']) for r in selected)})
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reviews', required=True, type=Path)
    p.add_argument('--inputs', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    inputs = [json.loads(line) for line in args.inputs.read_text().splitlines() if line.strip()]
    with args.reviews.open() as f:
        result = summarize(list(csv.DictReader(f)), {(r['method'], str(r['query_id'])) for r in inputs})
    with args.output.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(result[0]))
        w.writeheader()
        w.writerows(result)
