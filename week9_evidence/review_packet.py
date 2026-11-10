"""Create a readable human-review packet from source-grounded fact annotations."""
import argparse
from pathlib import Path
from urllib.parse import quote

from week9_evidence.evidence import validate_annotations
from week9_evidence.vidore import DATASET, load_dataset, read_jsonl


def write_packet(dataset, annotations, output):
    pages, queries, _, manifest = load_dataset(dataset)
    records = read_jsonl(annotations)
    validate_annotations(records, pages, queries, allow_draft=True)
    lines = ['# Finance pilot review packet', '',
        '**Status: draft; no independent human review has been recorded.**', '',
        'Use the [protocol](PROTOCOL.md) and edit `finance_pilot.jsonl` after review.',
        'PDF links point to the pinned source revision. PDF page numbers below are one based;',
        'corpus IDs are independent of page numbers. Exact quotes and character offsets are in',
        '[the annotation file](finance_pilot.jsonl). Each bullet is a required fact; support sets',
        'separated by OR are alternatives, while pages joined by + are jointly required.', '']
    for record in records:
        lines += [f'## Query {record["query_id"]}: {record["question"]}', '',
            f'Classification: `{record["question_type"]}`. Status: **{record["status"]}**.', '']
        for fact in record['facts']:
            support = []
            for spans in fact['support_sets']:
                links = []
                for cid in dict.fromkeys(s['page_id'] for s in spans):
                    page = pages[cid]
                    url = (f'https://huggingface.co/datasets/{DATASET}/resolve/{manifest["revision"]}'
                           f'/pdfs/{quote(page["document"])}.pdf#page={page["position"] + 1}')
                    links.append(f'[corpus {cid}, PDF p. {page["position"] + 1}]({url})')
                support.append(' + '.join(links))
            lines.append(f'- **{fact["fact_id"]}:** {fact["requirement"]} Support: ' + ' OR '.join(support) + '.')
        lines += ['', f'Review note: {record["review_notes"] or "Check completeness and alternative sources."}', '',
            'Reviewer: ______  Decision: ______  Corrections / additional alternatives: ______', '']
    output.write_text('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    write_packet(args.dataset, args.annotations, args.output)
