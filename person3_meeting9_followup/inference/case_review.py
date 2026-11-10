"""Serialize a bounded second-AI semantic review; verify delivered-source anchors.

No model calls, rescoring, aggregate metric calculation, or old-result writes.
The semantic notes are authored by the assistant; this script only preserves and
checks their provenance. Rerunning it does not conduct a new AI judgment.
"""
from pathlib import Path
from datetime import datetime, timezone
import csv, hashlib, json, re

ROOT=Path(__file__).resolve().parents[2]
E=ROOT/'results/person3_meeting9_followup/evidence_v1'
OUT=ROOT.parent/'outputs/meeting9_followup_20261109/case_review'
METHODS=['bm25','question_bank_rrf_w050','focusedmix_bank_rrf_w050','no_context']
Q='question_bank_rrf_w050';F='focusedmix_bank_rrf_w050'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):
    with Path(p).open() as f:return list(csv.DictReader(f))
files=['context_records.json','reference_answers_judge_only.json','successful_api_outputs.json',
       'judge/successful_api_outputs.json','judge/alias_mapping.json','per_query_metrics.csv',
       'source_quote_verification_failures.csv','completion_status.json']
before={name:sha(E/name) for name in files}
status=read(E/'completion_status.json')
assert status['reader_saved']==80 and status['valid_joint_judges']==20 and status['human_reviewed']==0
contexts={(r['query_id'],r['method']):r for r in read(E/'context_records.json')}
answers={(r['query_id'],r['style']):r for r in read(E/'successful_api_outputs.json')}
references={r['query_id']:r for r in read(E/'reference_answers_judge_only.json')}
judges={r['query_id']:r for r in read(E/'judge/successful_api_outputs.json')}
original={(r['query_id'],r['method']):r for r in rows(E/'per_query_metrics.csv')}
aliases={(r['query_id'],r['blind_label']):r['judged_label'] for r in read(E/'judge/alias_mapping.json')}
failures=rows(E/'source_quote_verification_failures.csv')
hash_key=lambda q:hashlib.sha256(('meeting9-case-review-v1|'+q).encode()).hexdigest()
selected=sorted({r['query_id'] for r in failures},key=hash_key)[:6]
assert selected==['199','236','178','235','228','218']

def delivered(q,method,pid):
    c=contexts[(q,method)];assert hashlib.sha256(c['context'].encode()).hexdigest()==c['context_sha256']
    span=next(s for s in c['delivered_spans'] if s['passage_id']==pid)
    header=re.search(r'(?m)^\[P'+re.escape(pid)+r'\] [^\n]*\n',c['context']);assert header
    text=c['context'][header.end():header.end()+span['char_end']-span['char_start']]
    assert len(text)==span['char_end']-span['char_start']
    if not span['truncated']:assert hashlib.sha256(text.encode()).hexdigest()==span['source_sha256']
    return c,span,header.end(),text

def source(q,method,pid,quote):
    c,s,base,text=delivered(q,method,pid);start=text.index(quote)
    return {'kind':'delivered_source','method':method,'source_id':'P'+pid,'doc_id':s['doc_id'],
      'quote':quote,'source_char_start':s['char_start']+start,'source_char_end':s['char_start']+start+len(quote),
      'context_char_start':base+start,'context_char_end':base+start+len(quote),
      'delivered_source_char_start':s['char_start'],'delivered_source_char_end':s['char_end'],
      'context_sha256':c['context_sha256']}
def answer(q,method,quote):
    text=answers[(q,method)]['text'];start=text.index(quote)
    return {'kind':'reader_answer','method':method,'quote':quote,'answer_char_start':start,'answer_char_end':start+len(quote)}
def judge(q,quote):
    text=judges[q]['text'];start=text.index(quote)
    return {'kind':'original_v1_joint_judge','quote':quote,'judge_output_char_start':start,'judge_output_char_end':start+len(quote)}
def finding(severity,category,methods,assessment,evidence,confidence='high'):
    return {'severity':severity,'category':category,'methods':methods,'assessment':assessment,
      'review_confidence':confidence,'evidence':evidence,'changes_original_score':False}

findings={
 '199':[
  finding('medium','coverage_and_scope_qualifiers',METHODS[:3],
   'The answers list the core blackout bans, but omit the delivered Rule10b5-1 exception and the reference’s pre-clearance/MNPI qualification. The BM25 answer nevertheless received full coverage. This is a completeness/scope concern, not proof that its four listed bans are false; the question’s blackout-only scope makes the exact coverage obligation debatable.',
   [source('199','bm25','2854','transactions pursuant to duly adopted Rule 10b5-1 Trading Plans.'),
    source('199',Q,'2854','Designated Insiders must pre-clear transactions in Company Securities'),
    source('199',F,'2854','must never trade in Company Securities when they are in possession of MNPI')],confidence='medium'),
  finding('low','mechanical_quote_format',METHODS[:3],
   'The judge surrounds copied snippets with quotation marks that are not present around those phrases in the delivered source. Mechanical failure therefore does not by itself demonstrate unsupported reader content.',
   [source('199','bm25','2854','purchasing or selling Company Securities;'),source('199',F,'2854','effecting cashless exercises of options.')])],
 '236':[
  finding('medium','missing_deconsolidation_consequences',[Q,F],
   'The shared hybrid answer substitutes a footnote saying amounts are excluded from the exposure table for the requested deconsolidation consequences. It omits the net-investment write-off and does not clearly say the CTA loss is recognized through earnings. The relevant P1081 consequence paragraph is delivered to BM25 but lies beyond both hybrids’ P1081 prefix. The original judge already partly flags this limitation.',
   [source('236',Q,'1080','are excluded from the above table.'),source('236',F,'1080','Citi has separately described these amounts in "Deconsolidation Risk" below.'),
    source('236','bm25','1081','write off the net investment'),source('236','bm25','1081','recognize a CTA loss of approximately $1.6 billion')]),
  finding('low','mechanical_quote_noncontiguous',['bm25'],
   'The judge’s short quote removes an injected OCR heading between “remained” and “unchanged”. The failure is not evidence that the $1.6 billion statement is absent. No separate material reader error is identified here; the source also has OCR currency-glyph noise.',
   [source('236','bm25','1081',"Citi's Russia-related exposures remained Deconsolidation Risk unchanged at $1.6 billion")])],
 '178':[
  finding('high','own_context_grounding_judge_inconsistency',[Q,F],
   'Both hybrids appropriately state that their own supplied excerpts do not establish the requested Wealth Management percentages. Their delivered pages are P2203 and P2214[0:495]; P2214 is an Investment Management table with8%, not the Wealth Management18%/9% pair. P2211 containing18%/9% is delivered only to BM25. The v1 joint judge gives the hybrids G0/T0 with “Misses provided percentages”. This conflicts with candidate-own-context evaluation and strongly suggests cross-candidate evidence contamination or an abstention-rubric error; it cannot prove the judge’s internal reasoning. Reference-consistency C0/V0 for a non-answer is a separate issue and is not disputed.',
   [source('178',Q,'2214','| Revenues Asset management and related fees | 5,627 | $ 5,231 | $ 5,332 | 8 % | (2)% |'),
    source('178','bm25','2211','|  | Revenues Asset management |  | $ 16,501 | $ 14,019 | $ 13,872 | 18 % | 1 % |'),
    source('178','bm25','2211','|  | Transactionall |  | 3,864 | 3,556 | 2,473 | 9 % | 44 %'),
    answer('178',Q,'a total sum of those percentages cannot be established from the supplied excerpts'),judge('178','Misses provided percentages')]),
  finding('low','mechanical_quote_ellipsis',['bm25'],
   'The judge uses an ellipsis rather than a verbatim contiguous quote. BM25’s18%+9%=27% answer has both percentages in its own delivered P2211 table; no numeric/unit/year defect is found in that answer.',
   [source('178','bm25','2211','| $ 13,872 | 18 % | 1 % |')])],
 '235':[
  finding('low','reference_scope_ambiguity',[Q,F],
   'The hybrid answers accurately read the delivered2024/2023 Level2/Level3 table amounts. They give a company-specific classification example rather than the reference’s conceptual explanation of all three hierarchy levels and valuation inputs. Full coverage is therefore scope-sensitive and merits human review, but no definite numeric/unit/year error is identified; this review does not force the reference wording onto a vague question.',
   [source('235',Q,'2050','| Non-investment grade |  |  | 71 | 50 | 121 |'),source('235',F,'2050','| Non-investment grade |  |  | 10 | 56 | 66 |')],confidence='medium'),
  finding('low','mechanical_quote_table_compression',[Q,F],
   'The judge collapses empty table cells in its source quote. Its simplified string is not literally contiguous, although the cited table supports the reported71/50/121 values.',
   [source('235',Q,'2050','| Non-investment grade |  |  | 71 | 50 | 121 |')])],
 '228':[
  finding('high','reader_internal_contradiction_and_measure_switch',[Q],
   'The question hybrid begins “Yes” but finishes “no,2024 was lower”. It switches to cash collateral netted against derivative payables ($60.8bn/$70.0bn), whereas the reference targets the collateral-posted table ($15,204m/$14,673m). Its delivered P134 prefix ends before that target table. The contradiction is definite; the quantity mismatch is relative to the released reference and should have been qualified rather than answered confidently. The original judge partly flags the conflict.',
   [answer('228',Q,'Yes.'),answer('228',Q,'So for posted cash collateral on derivative payables: no, 2024 was lower'),
    source('228',Q,'133','Net derivatives payable included cash collateral netted of $60.8 billion and $70.0 billion'),
    source('228',F,'134','Collateral posted\n15,204\n14,673')]),
  finding('low','mechanical_quote_whitespace',['bm25',F],
   'The short judge quote replaces table newlines with spaces. BM25 and Focused both receive the complete P134 table, including its millions unit and2024/2023 headers; their comparison is supported despite the failed literal-quote check.',
   [source('228',F,'134','(in millions)\nDecember 31, 2024\nDecember 31, 2023'),source('228','bm25','134','Collateral posted\n15,204\n14,673')])],
 '218':[
  finding('medium','citation_rubric_consistency',['bm25',F],
   'All three fractions occur together in each candidate’s own P2795. BM25 and Focused each place one P2795 citation at the end of the same three-item list, yet the judge gives BM25 T1 and Focused T2. No material source gap explaining that distinction is apparent. This is a citation-rating consistency concern for human adjudication, not a new replacement score.',
   [source('218','bm25','2795','Holders of depositary receipts are entitled to a fraction of a vote per depositary share'),
    source('218',F,'2795','1/25th in the case of the Unlisted Legacy Bank of America Depositary Shares'),
    source('218',F,'2795','1/1,200th in the case of the Legacy ML Depositary Shares')],confidence='medium'),
  finding('low','mechanical_quote_ellipsis_and_wrapping',METHODS[:3],
   'The judge’s quoted string includes ellipsis and wrapper quotes instead of copied contiguous text. The underlying voting fractions are present; no fractional arithmetic or year/unit error is identified.',
   [source('218',Q,'2795','1/1,200th in the case of the Legacy ML Depositary Shares')])]
}

cases=[]
for q in selected:
    candidates=[]
    for method in METHODS:
        c=contexts[(q,method)];a=answers[(q,method)];o=original[(q,method)]
        assert o['human_status']=='pending'
        candidates.append({'method':method,'blind_label':c['blind_label'],'judged_alias':aliases[(q,c['blind_label'])],
          'answer':a['text'],'answer_sha256':hashlib.sha256(a['text'].encode()).hexdigest(),'reader_key':a['key'],
          'context_sha256':c['context_sha256'],'delivered_spans':c['delivered_spans'],
          'original_v1_ai_values_unchanged':{name:o[name] for name in ['ai_reference_consistency','ai_coverage','ai_grounding','ai_citation','ai_reason','ai_source_quote','ai_source_quote_mechanically_verified','human_status']}})
    cases.append({'query_id':q,'selection_sha256':hash_key(q),'question':references[q]['question'],
      'released_reference':references[q]['reference_answer'],'reference_not_human_validated':True,
      'selection_reason':'Existing nonempty AI source quote failed literal verification; all six unique failure questions included',
      'original_v1_joint_judge_output':judges[q]['text'],'candidates':candidates,'findings':findings[q]})

# The key judge-risk contrast depends on which prefix each candidate actually saw.
for method in [Q,F]:assert [s['passage_id'] for s in contexts[('178',method)]['delivered_spans']]==['2203','2214']
assert next(s for s in contexts[('178',Q)]['delivered_spans'] if s['passage_id']=='2214')['char_end']==495
assert next(s for s in contexts[('228',Q)]['delivered_spans'] if s['passage_id']=='134')['char_end']==1326
assert before=={name:sha(E/name) for name in files}
result={'review_type':'Bounded second-AI semantic case review; NOT human review or a new metric',
  'created_utc':datetime.now(timezone.utc).isoformat(),'source_version':'v1 joint judge retained unchanged',
  'selection':{'rule':'All unique existing mechanical quote-failure questions; SHA256(meeting9-case-review-v1|query_id) order; cap6','query_ids':selected},
  'severity_definition':{'high':'Material answer contradiction or candidate-own-context judgment concern','medium':'Missing material qualifications/facts or unresolved rubric inconsistency','low':'Formatting-only literal-quote failure or interpretation/coverage ambiguity'},
  'limits':['Nonrepresentative failure-focused cases, not aggregate rescoring.','Reviewer sees method labels and v1 ratings; not blinded human adjudication.',
    'Only actual delivered prefixes, released references and saved answers/judgments were used; no unseen full pages or image evidence.',
    'Cross-candidate borrowing is a risk inferred from an own-context mismatch, not proof of hidden reasoning.',
    'No original score, reference, answer or human-status field was edited; human reviewed remains0.'],
  'offset_convention':'Zero-based Unicode character indices, end exclusive; source offsets refer to original OCR prefixes described in delivered_spans, context offsets include headers.',
  'api_calls':0,'new_aggregate_scores':False,'human_reviewed':0,'original_source_hashes_unchanged':before,
  'review_code_sha256':sha(__file__),'cases':cases}
OUT.mkdir(parents=True,exist_ok=True);(OUT/'cases.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
lines=['# Bounded semantic case review (v1)','','Second AI review of the six fixed mechanical-quote-failure questions. This is **not human review, aggregate rescoring or a second accuracy estimate**. All original ratings and human-status fields remain unchanged; human-reviewed0. No API calls. Source strings and offsets are mechanically checked against each candidate’s actual delivered prefix.','',
 'Severity is a case-review label: high=material contradiction/own-context judgment concern; medium=coverage or rubric concern; low=formatting or interpretation ambiguity. It does not replace rubric values.','']
for case in cases:
    q=case['query_id'];first=case['findings'][0];lines += [f"## q{q} — {first['severity']}",'',case['question'],'',first['assessment'],'']
    for e in first['evidence']:
        if e['kind']=='delivered_source':lines.append(f"- `{e['method']}` {e['source_id']} source chars[{e['source_char_start']}:{e['source_char_end']}): “{e['quote']}”")
    lines += ['',case['findings'][1]['assessment'],'']
lines += ['The cases are failure-focused and not representative. No new rates, totals of semantic successes/failures, p-values or replacement grades are computed. q178 indicates an own-context grading problem; it does not reveal the model’s internal reasoning. q235’s numerical example is supported and is not treated as a fabricated-number error.','',
 'The complete saved answers, original v1 AI fields, short delivered-source anchors, context hashes and exact source/answer/judge offsets are in [cases.json](cases.json). No full unseen pages were consulted. Any later isolated judging design would close the other-candidate-context access path; whether it fixes judgments requires a separate audit.']
(OUT/'README.md').write_text('\n'.join(lines)+'\n')
print('Saved bounded6-question v1 second-AI case review. All source anchors verified; original files unchanged; no new scores/API calls.')
