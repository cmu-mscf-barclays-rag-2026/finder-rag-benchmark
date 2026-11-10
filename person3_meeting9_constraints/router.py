"""Conservative company routing using only literal question text and fixed aliases."""
import re
import unicodedata

ALIASES = {
    'bank_of_america_2024': ['Bank of America'],
    'citigroup_2024': ['Citigroup', 'Citi'],
    'goldman_sachs_2024': ['Goldman Sachs'],
    'jpmorgan_chase_2024': ['JPMorgan', 'JPMorgan Chase', 'JPMorganChase', 'J.P. Morgan'],
    'morgan_stanley_2024': ['Morgan Stanley'],
    'wells_fargo_2024': ['Wells Fargo'],
}
# Abstain conservatively; do not infer a mentioned bank's semantic scope.
AMBIGUITY = re.compile(r'\b(?:not|no|except|excluding|exclude|without|other\s+than|rather\s+than|instead\s+of)\b', re.I)
OPEN_COMPARISON = re.compile(r'\b(?:(?:other|another)\s+banks?|peers?|competitors?|industry)\b', re.I)
QUOTES = re.compile(r'"[^"\n]*"|“[^”\n]*”|‘[^’\n]*’|(?<!\w)\x27[^\x27\n]*\x27(?!\w)')

def route(question):
    """Return fixed doc IDs, or [] for an unchanged unfiltered search.

    This function has no access to qrels, answers, source pages or retrieval scores.
    """
    question = unicodedata.normalize('NFKC', question)
    hits = []
    for doc_id, aliases in ALIASES.items():
        for alias in aliases:
            # Only normalize spacing, hyphens and periods inside the fixed name.
            compact = re.sub(r'[\s.\-]+', '', alias)
            pattern = r'(?<!\w)' + r'[\s.\-]*'.join(re.escape(c) for c in compact) + r'(?!\w)'
            for match in re.finditer(pattern, question, re.I):
                hits.append({'doc_id': doc_id, 'alias': alias, 'literal': match.group(), 'start': match.start(), 'end': match.end()})
    # Keep longest overlapping alias per document for readable audit records.
    hits = sorted(hits, key=lambda h: (h['start'], -(h['end']-h['start']), h['doc_id']))
    kept = []
    for hit in hits:
        if not any(k['doc_id'] == hit['doc_id'] and k['start'] <= hit['start'] and k['end'] >= hit['end'] for k in kept):
            kept.append(hit)
    reason = 'no_explicit_alias' if not kept else 'explicit_alias'
    if kept and AMBIGUITY.search(question): reason = 'abstain_negation_or_exclusion_marker'
    if kept and any(a.start() < hit['end'] and hit['start'] < a.end() for a in QUOTES.finditer(question) for hit in kept):
        reason = 'abstain_quoted_company_mention'
    if len({hit['doc_id'] for hit in kept}) == 1 and OPEN_COMPARISON.search(question):
        reason = 'abstain_open_comparison'
    docs = sorted({hit['doc_id'] for hit in kept}) if reason == 'explicit_alias' else []
    return {'route_doc_ids': docs, 'reason': reason, 'literal_matches': kept}
