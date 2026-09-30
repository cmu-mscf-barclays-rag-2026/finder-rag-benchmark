"""Part 2 context assembly and evaluation in the original child evidence space."""
from statistics import mean

from .core import SEPARATOR


def assemble_context(ranking, mode, children, parents, tokenizer, k, token_budget=None):
    """Materialize child, parent, or child-to-parent results with exact provenance.

    Expansion uses the first k child hits, deduplicates parents in first-hit order,
    and does not backfill. Token budgets apply to the concatenated body text only.
    Evidence metrics refer to this returned text, not a downstream rewritten prompt.
    """
    if type(k) is not int or k < 1:
        raise ValueError('k must be a positive integer.')
    if token_budget is not None and (type(token_budget) is not int or token_budget < 0):
        raise ValueError('Token budget must be a nonnegative integer or None.')
    if not isinstance(ranking, list) or any(not isinstance(pid, str) for pid in ranking):
        raise ValueError('Ranking must be an ordered list of string IDs.')
    if len(ranking) != len(set(ranking)):
        raise ValueError('Duplicate ranking IDs are invalid.')
    if mode not in ('child', 'parent', 'child_to_parent'):
        raise ValueError('Unknown retrieval mode.')
    lookup = parents if mode == 'parent' else children
    if any(pid not in lookup for pid in ranking):
        raise ValueError('A ranked ID does not belong to the chosen corpus.')
    seeds = ranking[:k]
    if mode == 'child_to_parent':
        unit_ids = list(dict.fromkeys(children[pid]['parent_id'] for pid in seeds))
    else:
        unit_ids = seeds
    selected_lookup = children if mode == 'child' else parents
    text, spans = '', {}
    for uid in unit_ids:
        if text:
            text += SEPARATOR
        origin = len(text)
        unit = selected_lookup[uid]
        text += unit['text']
        if mode == 'child':
            spans[uid] = (origin, len(text))
        else:
            for span in unit['child_spans']:
                spans[span['child_id']] = (origin + span['start_char'], origin + span['end_char'])
    encoded = tokenizer.encode(text, add_special_tokens=False)
    full_tokens = len(encoded.ids)
    retained = text
    if token_budget is not None and full_tokens > token_budget:
        if token_budget == 0:
            retained = ''
        else:
            # Include whitespace before the first omitted token, but never part of it.
            cut = encoded.offsets[token_budget][0]
            retained = text[:cut]
            # Re-encoding a wordpiece prefix can change its tokenization.
            while len(tokenizer.encode(retained, add_special_tokens=False).ids) > token_budget:
                cut -= 1
                retained = text[:cut]
    visible_chars = {}
    fully_included = []
    for pid, (start, end) in spans.items():
        count = max(0, min(len(retained), end) - start)
        visible_chars[pid] = count
        if count == end - start:
            fully_included.append(pid)
    return {'seed_ids': seeds, 'unit_ids': unit_ids, 'text': retained,
            'selected_child_ids': list(spans), 'fully_included_child_ids': fully_included,
            'visible_child_characters': visible_chars,
            'unbudgeted_content_tokens': full_tokens,
            'returned_content_tokens': len(tokenizer.encode(retained, add_special_tokens=False).ids),
            'returned_tokens_with_specials': len(tokenizer.encode(retained, add_special_tokens=True).ids),
            'truncated': len(retained) < len(text)}


def evaluate_run(queries, predictions, children, parents, tokenizer,
                 mode='child', ks=(1, 5, 10), token_budget=None):
    """Return per-query/aggregate diagnostics; never compare parent IDs to child IDs."""
    qids = [q['id'] for q in queries]
    if not qids or len(set(qids)) != len(qids) or set(predictions) != set(qids):
        raise ValueError('Predictions must exactly match unique evaluation query IDs.')
    if not ks or len(set(ks)) != len(ks):
        raise ValueError('Supply distinct k values.')
    rows = []
    for query in queries:
        gold = query['gold_child_ids']
        if not isinstance(gold, list) or not gold or any(pid not in children for pid in gold):
            raise ValueError('Invalid original gold child labels.')
        gold = set(gold)
        for k in ks:
            context = assemble_context(predictions[query['id']], mode, children, parents,
                                       tokenizer, k, token_budget)
            gold_units = ({children[pid]['parent_id'] for pid in gold} if mode == 'parent' else gold)
            unit_hits = len(gold_units & set(context['seed_ids']))
            full = len(gold & set(context['fully_included_child_ids']))
            tokens_seen = tokens_total = 0
            for pid in gold:
                offsets = tokenizer.encode(children[pid]['text'], add_special_tokens=False).offsets
                visible = context['visible_child_characters'].get(pid, 0)
                tokens_seen += sum(end <= visible for start, end in offsets)
                tokens_total += len(offsets)
            rows.append({'query_id': query['id'], 'mode': mode, 'k': k,
                         'token_budget': token_budget,
                         'ranked_unit_hit_at_k': float(unit_hits > 0),
                         'ranked_unit_recall_at_k': unit_hits / len(gold_units),
                         'gold_child_in_selected_units': float(bool(gold & set(context['selected_child_ids']))),
                         'gold_child_hit_in_context': float(full > 0),
                         'gold_child_coverage_in_context': full / len(gold),
                         'all_gold_children_in_context': float(full == len(gold)),
                         'gold_body_token_coverage': tokens_seen / tokens_total if tokens_total else 0.0,
                         'returned_units': len(context['unit_ids']),
                         'unbudgeted_content_tokens': context['unbudgeted_content_tokens'],
                         'returned_content_tokens': context['returned_content_tokens'],
                         'context_truncated': float(context['truncated'])})
    metrics = [key for key in rows[0] if key not in ('query_id', 'mode', 'k', 'token_budget')]
    summaries = [{'mode': mode, 'k': k, 'token_budget': token_budget, 'n_queries': len(queries),
                  **{metric: mean(row[metric] for row in rows if row['k'] == k) for metric in metrics}}
                 for k in ks]
    return rows, summaries
