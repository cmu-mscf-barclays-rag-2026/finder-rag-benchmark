"""Audit the text representation and draft labels without treating qrels as facts."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics

from week6_qwen_hierarchy.qwen_hierarchy import evaluation_tokenizer
from week9_evidence.run import ROOT, TOKENIZER
from week9_evidence.vidore import digest, load_dataset, read_jsonl


def audit(dataset, annotations, output):
    pages, queries, qrels, manifest = load_dataset(dataset)
    tokenizer, _ = evaluation_tokenizer({'evaluation_tokenizer': TOKENIZER}, ROOT / '.cache/models', True)
    lengths = {cid: len(tokenizer.encode(page['text']).ids) for cid, page in pages.items()}
    records = read_jsonl(annotations)
    supports_outside_qrels = []
    for record in records:
        for cid in sorted({s['page_id'] for f in record['facts'] for group in f['support_sets'] for s in group}):
            if cid not in qrels[record['query_id']]:
                supports_outside_qrels.append({'query_id': record['query_id'], 'page_id': cid})
    result = {'dataset_revision': manifest['revision'], 'queries': len(queries), 'pages': len(pages),
        'reports': dict(Counter(p['document'] for p in pages.values())),
        'empty_text_pages': [cid for cid, p in pages.items() if not p['text'].strip()],
        'positive_qrels_on_empty_text': [{'query_id': qid, 'page_id': cid} for qid, rel in qrels.items()
                                       for cid in rel if not pages[cid]['text'].strip()],
        'minilm_document_input': {'mean_tokens_including_special': statistics.mean(lengths.values()) + 2,
            'max_tokens_including_special': max(lengths.values()) + 2,
            'pages_truncated_at_256_including_special': sum(n + 2 > 256 for n in lengths.values())},
        'minilm_max_query_tokens_including_special': max(len(tokenizer.encode(q['query']).ids) + 2 for q in queries.values()),
        'query_types_multi_label': dict(Counter(t for q in queries.values() for t in q['query_types'])),
        'content_types_multi_label': dict(Counter(t for q in queries.values() for t in q['content_type'])),
        'draft_queries': len(records), 'draft_facts': sum(len(r['facts']) for r in records),
        'draft_support_pages_absent_from_qrels': supports_outside_qrels,
        'annotations_sha256': digest(annotations), 'source_sha256': digest(Path(__file__))}
    output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit(args.dataset, args.annotations, args.output)
