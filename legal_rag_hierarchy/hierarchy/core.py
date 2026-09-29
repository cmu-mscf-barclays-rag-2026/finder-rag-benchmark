"""Build ID-derived parent units while preserving source text exactly."""
import re
from collections import defaultdict

ID_PATTERN = re.compile(r'^(?P<section>.+)-c(?P<c>\d+)-s(?P<s>\d+)$')
SEPARATOR = '\n\n'


def parse_id(pid):
    match = ID_PATTERN.fullmatch(pid)
    if not match or int(match['c']) < 1 or int(match['s']) < 1:
        raise ValueError(f'Invalid hierarchy ID: {pid}')
    return {'section_id': match['section'], 'parent_id': pid.rsplit('-s', 1)[0],
            'c_index': int(match['c']), 's_index': int(match['s'])}


def natural_key(value):
    return tuple((0, int(part)) if part.isdigit() else (1, part)
                 for part in re.split(r'(\d+)', value))


def suffix_prefix_length(left, right):
    """Longest exact suffix/prefix overlap, computed in linear time."""
    combined = list(right) + [object()] + list(left)
    lengths = [0] * len(combined)
    for i in range(1, len(combined)):
        j = lengths[i - 1]
        while j and combined[i] != combined[j]:
            j = lengths[j - 1]
        if combined[i] == combined[j]:
            j += 1
        lengths[i] = j
    return lengths[-1] if lengths else 0


def build_hierarchy(records):
    if not records:
        raise ValueError('Empty source corpus.')
    groups, seen = defaultdict(list), set()
    children = []
    for row in records:
        if row['id'] in seen:
            raise ValueError(f"Duplicate child ID: {row['id']}")
        seen.add(row['id'])
        if not isinstance(row['text'], str) or not row['text'].strip():
            raise ValueError('Child text must be nonempty.')
        child = {**row, **parse_id(row['id'])}
        groups[child['parent_id']].append(child)
        children.append(child)
    parents, mapping, audit, boundaries = [], [], [], []
    for pid in sorted(groups, key=natural_key):
        source_order = groups[pid]
        ordered = sorted(source_order, key=lambda row: row['s_index'])
        indices = [row['s_index'] for row in ordered]
        if len(indices) != len(set(indices)):
            raise ValueError(f'Duplicate child sequence within {pid}')
        titles = sorted({row.get('title') or '' for row in ordered})
        if len(titles) != 1:
            raise ValueError(f'Conflicting titles in parent {pid}: inspect grouping before merging.')
        missing = sorted(set(range(1, max(indices) + 1)) - set(indices))
        # Gaps could conceal missing text; do not silently reconstruct a parent.
        if missing:
            raise ValueError(f'Missing child sequence in {pid}: {missing}')
        text, spans, footnotes = '', [], defaultdict(list)
        for row in ordered:
            if text:
                text += SEPARATOR
            start = len(text)
            text += row['text']
            span = {'child_id': row['id'], 'start_char': start, 'end_char': len(text)}
            spans.append(span)
            mapping.append({'child_id': row['id'], 'parent_id': pid,
                            'section_id': row['section_id'], 's_index': row['s_index'],
                            'start_char': start, 'end_char': len(text)})
            if row.get('footnotes'):
                footnotes[row['footnotes']].append(row['id'])
        parents.append({'id': pid, 'section_id': ordered[0]['section_id'],
                        'title': titles[0], 'text': text,
                        'child_ids': [row['id'] for row in ordered], 'child_spans': spans,
                        'footnote_variants': [{'text': note, 'child_ids': ids}
                                              for note, ids in footnotes.items()]})
        audit.append({'parent_id': pid, 'section_id': ordered[0]['section_id'],
                      'n_children': len(ordered), 'first_s': min(indices), 'last_s': max(indices),
                      'missing_s_count': len(missing),
                      'source_order_was_numeric': indices == [r['s_index'] for r in source_order],
                      'unique_titles': len(titles), 'footnote_variants': len(footnotes),
                      'duplicate_child_text_count': len(ordered) - len({r['text'] for r in ordered})})
        for left, right in zip(ordered, ordered[1:]):
            raw_overlap = suffix_prefix_length(left['text'], right['text'])
            # Casefolded whitespace tokens detect exact lexical overlap despite case/spacing.
            word_overlap = suffix_prefix_length(left['text'].casefold().split(),
                                                 right['text'].casefold().split())
            boundaries.append({'parent_id': pid, 'left_id': left['id'], 'right_id': right['id'],
                               'exact_overlap_characters': raw_overlap,
                               'normalized_overlap_words': word_overlap,
                               'review_flag': raw_overlap >= 30 or word_overlap >= 5,
                               'identical_text': left['text'] == right['text'],
                               'same_nonempty_footnotes': bool(left.get('footnotes')) and
                                  left.get('footnotes') == right.get('footnotes')})
    children.sort(key=lambda row: natural_key(row['id']))
    return children, parents, mapping, audit, boundaries
