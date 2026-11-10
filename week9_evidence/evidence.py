"""Fact support is AND across facts, OR across support sets, AND within a set.

A support set contains exact source spans, potentially spanning several pages.
Coverage describes these annotations, not automatically verified answerability.
"""
from __future__ import annotations


def validate_annotations(annotations, pages, queries, allow_draft=False):
    seen = set()
    for annotation in annotations:
        qid = annotation['query_id']
        if qid in seen or qid not in queries:
            raise ValueError('Duplicate or unknown annotation query ID')
        seen.add(qid)
        status = annotation.get('status')
        if status not in ('draft', 'reviewed'):
            raise ValueError('Annotation status must be draft or reviewed')
        if status != 'reviewed' and not allow_draft:
            raise ValueError('Draft annotations require explicit --allow-draft')
        if status == 'reviewed' and not annotation.get('reviewer'):
            raise ValueError('Reviewed annotation requires a reviewer')
        facts = annotation['facts']
        if not facts or len({fact['fact_id'] for fact in facts}) != len(facts):
            raise ValueError('Facts must be nonempty with unique IDs')
        for fact in facts:
            if not fact.get('requirement') or not fact.get('support_sets'):
                raise ValueError('Every fact requires a description and grounded support')
            for support in fact['support_sets']:
                if not support:
                    raise ValueError('Empty support set would grant unconditional credit')
                for span in support:
                    cid, start, end = span['page_id'], span['start'], span['end']
                    if cid not in pages or type(start) is not int or type(end) is not int:
                        raise ValueError('Invalid support span')
                    text = pages[cid]['text']
                    if not 0 <= start < end <= len(text) or text[start:end] != span['quote']:
                        raise ValueError('Support quote/offset mismatch')
    if not seen:
        raise ValueError('No annotations')


def covered_facts(annotation, visible):
    """visible maps page ID to delivered [start, end) character intervals."""
    def contains(span):
        # Adjacent/overlapping pieces may jointly cover a span, but a gap may not.
        cursor = span['start']
        for start, end in sorted(visible.get(span['page_id'], [])):
            if start > cursor:
                break
            cursor = max(cursor, end)
            if cursor >= span['end']:
                return True
        return False
    return {fact['fact_id'] for fact in annotation['facts']
            if any(all(contains(span) for span in support) for support in fact['support_sets'])}


def fact_metrics(annotation, visible, tokens):
    found = covered_facts(annotation, visible)
    total = len(annotation['facts'])
    return {'fact_count': total, 'covered_fact_count': len(found),
            'fact_recall': len(found) / total, 'sufficient': int(len(found) == total),
            'facts_per_1k_tokens': 1000 * len(found) / tokens if tokens else 0.0,
            'covered_fact_ids': sorted(found)}


def marginal_gains(annotation, delivered, pages, count_tokens):
    visible, previous, rows, text = {}, set(), [], ''
    for cid in dict.fromkeys(delivered):
        text += ('\n\n' if text else '') + pages[cid]['text']
        visible[cid] = [(0, len(pages[cid]['text']))]
        found = covered_facts(annotation, visible)
        rows.append({'page_id': cid, 'new_fact_ids': sorted(found - previous),
                     'new_facts': len(found - previous), 'cumulative_facts': len(found),
                     'cumulative_tokens': count_tokens(text),
                     'sufficient': int(len(found) == len(annotation['facts']))})
        previous = found
    return rows
