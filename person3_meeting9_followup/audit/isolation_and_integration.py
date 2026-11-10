"""Audit only isolated-v2 repair and final integration; no old audit rerun."""
from pathlib import Path
from datetime import datetime, timezone
import ast
import hashlib
import json
import re
import sqlite3
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT/'results/person3_meeting9_followup'
V2 = RESULT/'evidence_isolated_v2'
V1 = RESULT/'evidence_v1'
OUT = RESULT/'audit_v1'
EXPORT = ROOT.parent/'outputs/meeting9_followup_20261109'
METHODS = ['bm25', 'question_bank_rrf_w050', 'focusedmix_bank_rrf_w050', 'no_context']
METRICS = ['ai_reference_consistency', 'ai_coverage', 'ai_grounding', 'ai_citation',
           'ai_complete_reference_consistent', 'ai_supported_reference_consistent']


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv(path):
    return pd.read_csv(path, dtype={'query_id': str}, keep_default_na=False, float_precision='round_trip')


def close(left, right):
    assert np.allclose(left, right, rtol=0, atol=1e-12)


def verify_frame(expected, path, keys):
    actual = csv(path)
    assert set(expected.columns).issubset(actual.columns), path.name
    paired = expected.merge(actual, on=keys, suffixes=('_want', '_got'), validate='one_to_one')
    assert len(paired) == len(expected) == len(actual), path.name
    if not len(paired):
        return 0
    for column in expected.columns:
        if column not in keys:
            left, right = paired[column+'_want'], paired[column+'_got']
            if pd.api.types.is_numeric_dtype(left):
                close(left, right)
            else:
                assert left.equals(right), (path.name, column)
    return len(actual)


def identity(job):
    request = dict(model='gpt-5.4-2026-03-05', prompt=job['prompt'], max_output_tokens=512,
                   temperature=0, reasoning='none', store=False)
    return hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()


def repair():
    preserved = [p for p in V2.iterdir() if p.is_file() and p.suffix not in ('.sqlite', '.lock')]
    snapshot = {str(p): sha(p) for p in preserved}
    receipt, protocol = read(V2/'freeze_receipt.json'), read(V2/'protocol.json')
    pinned = {
        'protocol.json': '38325fe5d191b3fbc4d8eb91ffd709c4ab6e51e9fb4b8cacd026f664f7fa4efc',
        'pure_abstention_cases.json': 'a2e3f8a1aa03422792717353abe8764f0ce7aaca103d308407cdc8971ab39d83',
        'judge_calls.jsonl': '5e698ee37921ce74e969a1b240f4fc128b7647542c08e22cb8858e71da42b8c8',
        'implementation_hashes.json': '9c185a40286a73f3ea10b053f84ffc164fe1c9d20a945320092c40e4034becb1'}
    for name, digest in pinned.items():
        assert sha(V2/name) == digest
    assert receipt['utc'] == '2026-11-10T01:44:12.661570+00:00'
    for name, digest in receipt['artifact_hashes'].items():
        assert sha(V2/name) == digest
    for manifest in ['source_hashes.json', 'implementation_hashes.json']:
        for path, digest in read(V2/manifest).items():
            assert sha(ROOT/path) == digest, path
    assert receipt['no_v2_outputs_existed'] is True
    assert protocol['human_reviewed'] == 0 and 'Post-review repair' in protocol['status']
    assert protocol['max_output_tokens'] == 512 and protocol['temperature'] == 0 and protocol['reasoning'] == 'none'
    approval = read(V2/'paid_approval.json')
    assert approval['approved'] is True and approval['unique_calls_upper'] == 66
    assert datetime.fromisoformat(receipt['utc']) < datetime.fromisoformat(approval['utc'])
    for name in ['protocol.json', 'implementation_hashes.json', 'judge_calls.jsonl']:
        assert approval['bound_hashes'][name] == sha(V2/name)
    source = ast.parse((ROOT/'person3_meeting9_followup/evidence_isolated/run.py').read_text())
    rules = ast.literal_eval(next(node.value for node in source.body if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == 'RULES' for t in node.targets)))
    inputs = read(V2/'inputs.json')
    contexts = {(r['query_id'], r['method']): r for r in read(V1/'context_records.json')}
    answers = {(r['query_id'], r['style']): r for r in read(V1/'successful_api_outputs.json')}
    refs = {r['query_id']: r for r in read(V1/'reference_answers_judge_only.json')}
    original_selection = read(V1/'selection.json')['queries']
    assert len(inputs) == 80 and {r['query_id'] for r in inputs} == {q['query_id'] for q in original_selection}
    assert all({r['method'] for r in inputs if r['query_id'] == q['query_id']} == set(METHODS) for q in original_selection)
    jobs = [json.loads(line) for line in (V2/'judge_calls.jsonl').read_text().splitlines() if line.strip()]
    queue = {(j['query_id'], j['style']): j for j in jobs}
    unique_jobs = {identity(j): j for j in jobs}
    assert len(queue) == 80 and len(unique_jobs) == 66
    for item in inputs:
        key = (item['query_id'], item['method'])
        old, answer = contexts[key], answers[key]
        assert all(item[k] == old[k] for k in ['query_id', 'method', 'question', 'context', 'delivered_spans', 'used_wordpiece_tokens'])
        assert item['answer'] == answer['text'] and item['original_reader_key'] == answer['key']
        assert item['reference_answer'] == refs[key[0]]['reference_answer'] and item['original_qrels'] == refs[key[0]]['original_qrels']
        # Exact reconstruction proves no second answer/context, method label,
        # qrels or previous judge score enters the actual API input.
        prompt = (rules+'\nQUESTION:\n'+item['question']+'\nRELEASED REFERENCE (comparison only):\n'+item['reference_answer']
                  +'\nCANDIDATE ANSWER:\n'+item['answer']+'\nOWN REPORT EXCERPTS:\n'+item['context'])
        job = queue[key]
        assert job['prompt'] == prompt and job['max_output_tokens'] == 512 and job['model'] == 'gpt-5.4-2026-03-05'
        assert item['judge_key'] == identity(job)
    pure = read(V2/'pure_abstention_cases.json')['cases']
    pure_set = {(r['query_id'], r['method'], r['answer_sha256']) for r in pure}
    assert len(pure_set) == 20 and {r['query_id'] for r in pure} == {q['query_id'] for q in original_selection}
    for case in pure:
        item = next(r for r in inputs if r['query_id'] == case['query_id'] and r['method'] == case['method'])
        assert case['method'] == 'no_context' and not item['delivered_spans']
        assert case['answer_sha256'] == hashlib.sha256(item['answer'].encode()).hexdigest()
        assert case['judge_key'] == item['judge_key']
    saved = read(V2/'successful_api_outputs.json')
    lookup = {}
    for row in saved:
        assert row['key'] == identity(queue[(row['query_id'], row['style'])])
        assert row['prompt'] == queue[(row['query_id'], row['style'])]['prompt']
        assert row['model_returned'] == 'gpt-5.4-2026-03-05' and row['status'] == 'completed'
        assert row['request_concurrency'] == 1
        if row['key'] in lookup:
            assert all(row[k] == lookup[row['key']][k] for k in ['text', 'usage', 'response_id', 'status'])
        lookup[row['key']] = row
    assert len(saved) == 80 and len(lookup) == 66
    pages = {str(p['corpus_id']): p for p in map(json.loads, (ROOT/'results/person3_meeting9_transfer/vidore_finance_v1/data/corpus.jsonl').read_text().splitlines())}
    expected, invalid = [], []
    for item in inputs:
        output = lookup[item['judge_key']]
        try:
            r = json.loads(output['text'])
            assert set(r) == {'C', 'V', 'G', 'T', 'answer_status', 'reference_issue', 'sourcequotes', 'reason'}
            assert all(type(r[k]) is int and r[k] in (0, 1, 2) for k in 'CVGT')
            assert r['answer_status'] in ('answer', 'partial', 'abstain') and type(r['reference_issue']) is bool
            assert isinstance(r['reason'], str) and isinstance(r['sourcequotes'], list) and len(r['sourcequotes']) <= 2
            assert all(set(q) == {'source_id', 'quote'} and all(isinstance(v, str) for v in q.values()) for q in r['sourcequotes'])
        except Exception:
            invalid.append(dict(query_id=item['query_id'], method=item['method'], key=item['judge_key']))
            continue
        ids = {s['passage_id'] for s in item['delivered_spans']}
        cites = re.findall(r'\[P(\d+)\]', item['answer'])
        valid_ids = bool(cites) and all(pid in ids for pid in cites)
        raw = [r[k] for k in 'CVGT']
        guarded, reasons = raw.copy(), []
        exact = (item['query_id'], item['method'], hashlib.sha256(item['answer'].encode()).hexdigest()) in pure_set
        if exact:
            guarded = [0, 0, 2, 0]
            if guarded != raw:
                reasons.append('frozen_exact_pure_abstention_0_0_2_0')
        elif r['answer_status'] == 'abstain':
            guarded = [0, 0, raw[2], 0]
            if guarded != raw:
                reasons.append('AI_classified_abstention_C0_V0_T0_preserve_G')
        if not valid_ids:
            if guarded[3] != 0:
                reasons.append('missing_or_invalid_citation_T0')
            guarded[3] = 0
        verified = []
        for quote in r['sourcequotes']:
            pid = quote['source_id'].removeprefix('P')
            verified.append(bool(quote['quote']) and any(span['passage_id'] == pid
                and quote['quote'] in pages[pid]['markdown'][span['char_start']:span['char_end']] for span in item['delivered_spans']))
        quote_valid = bool(verified) and all(verified)
        row = dict(query_id=item['query_id'], method=item['method'], judge_key=item['judge_key'], answer_status=r['answer_status'],
            frozen_exact_pure_abstention=exact, guard_abstention_classification_source='frozen_AI_whole_answer_SHA_list' if exact else ('AI_judge_answer_status' if r['answer_status'] == 'abstain' else 'none'),
            reference_issue_flag=r['reference_issue'], reader_has_citation=bool(cites), reader_citation_ids_valid=valid_ids,
            ai_source_quote_mechanically_verified=quote_valid, sourcequotes_count=len(verified), sourcequotes_verified_count=sum(verified),
            rule_guard_applied=guarded != raw, rule_guard_reasons='|'.join(reasons), ai_reason=r['reason'],
            ai_sourcequotes=json.dumps(r['sourcequotes'], ensure_ascii=False), human_status='pending')
        for prefix, values in [('raw_', raw), ('rule_guard_', guarded), ('', guarded)]:
            row.update({prefix+metric: value/2 for metric, value in zip(METRICS[:4], values)})
            row[prefix+METRICS[4]] = int(values[:2] == [2, 2])
            row[prefix+METRICS[5]] = int(values == [2, 2, 2, 2])
        row['ai_strict_supported_reference_consistent'] = int(guarded == [2, 2, 2, 2] and quote_valid and valid_ids)
        expected.append(row)
    frame = pd.DataFrame(expected)
    verify_frame(frame, V2/'per_query_metrics.csv', ['query_id', 'method'])
    summary = frame.groupby('method').agg(n=('query_id', 'size'), **{m: (m, 'mean') for m in METRICS}).reset_index()
    verify_frame(summary, V2/'summary.csv', ['method'])
    raw_metrics = ['raw_'+m for m in METRICS]
    raw_summary = frame.groupby('method').agg(n=('query_id', 'size'), **{m: (m, 'mean') for m in raw_metrics}).reset_index()
    verify_frame(raw_summary, V2/'raw_summary.csv', ['method'])
    mechanical_metrics = ['reader_has_citation', 'reader_citation_ids_valid', 'ai_source_quote_mechanically_verified', 'ai_strict_supported_reference_consistent', 'rule_guard_applied']
    mechanical = frame.groupby('method').agg(n=('query_id', 'size'), **{m: (m, 'mean') for m in mechanical_metrics}).reset_index()
    verify_frame(mechanical, V2/'mechanical_diagnostics.csv', ['method'])
    verify_frame(frame[frame.rule_guard_applied], V2/'raw_rubric_violations.csv', ['query_id', 'method'])
    quote_failures = frame[(frame.sourcequotes_count > 0) & ~frame.ai_source_quote_mechanically_verified]
    verify_frame(quote_failures, V2/'source_quote_verification_failures.csv', ['query_id', 'method'])
    paired = []
    for method, control in [(METHODS[1], METHODS[0]), (METHODS[2], METHODS[0]), (METHODS[2], METHODS[1])]+[(m, 'no_context') for m in METHODS[:3]]:
        delta = frame[frame.method == method].set_index('query_id')[METRICS]-frame[frame.method == control].set_index('query_id')[METRICS]
        for metric in METRICS:
            d = delta[metric]
            paired.append(dict(method=method, control=control, metric=metric, n=len(d), mean_delta=d.mean(),
                               wins=int((d > 0).sum()), losses=int((d < 0).sum()), ties=int((d == 0).sum())))
    verify_frame(pd.DataFrame(paired), V2/'paired_comparisons.csv', ['method', 'control', 'metric'])
    no_context = frame[frame.method == 'no_context']
    assert np.array_equal(no_context[METRICS[:4]].to_numpy(), np.tile([0, 0, 1, 0], (20, 1)))
    q178 = frame[(frame.query_id == '178') & frame.method.isin(METHODS[1:3])]
    assert len(q178) == 2
    q178_acceptance = True
    q178_details = []
    for row in q178.to_dict('records'):
        item = next(i for i in inputs if i['query_id'] == '178' and i['method'] == row['method'])
        # OCR writes the percentage as "8 %" in the Investment Management row.
        # Whitespace normalization is required to check the declared fact.
        source8 = any(s['passage_id'] == '2214' and re.search(
            r'Asset management and related fees[^\n]*\|\s*8\s*%',
            pages['2214']['markdown'][s['char_start']:s['char_end']]) is not None for s in item['delivered_spans'])
        scores = [row[m]*2 for m in METRICS[:4]]
        passed = scores == [0, 0, 2, 0] and row['answer_status'] == 'abstain' and source8 and not row['frozen_exact_pure_abstention']
        q178_acceptance &= passed
        q178_details.append(dict(method=row['method'], guarded_CVGT=scores, raw_CVGT=[row['raw_'+m]*2 for m in METRICS[:4]],
                                 answer_status=row['answer_status'], judge_key=row['judge_key'], reason=row['ai_reason'],
                                 explanatory_8percent_in_own_P2214_prefix=source8, accepted=passed))
    status = read(V2/'completion_status.json')
    assert status['invalid_rows'] == invalid and status['human_reviewed'] == 0
    assert len(frame) == 80 and not invalid and status['engineering_complete'] is True
    # Producer's gate checks only abstention status; independent acceptance also
    # requires own-fact G2 and the specific 8% source evidence, declared pre-result.
    human = csv(V2/'human_review_template.csv')
    assert len(human) == 80 and human.status.eq('pending').all()
    assert human.reviewer_id.eq('').all() and human.reviewed_at.eq('').all()
    assert all(human[column].eq('').all() for column in human if column.startswith('human_'))
    usage = read(V2/'usage.json')
    api_cost = sum((r['usage']['input_tokens']*2.5+r['usage']['output_tokens']*15)/1e6 for r in lookup.values())
    assert usage['new_reader_calls'] == 0 and usage['judge_unique_calls'] == 66 and usage['judge_logical_jobs'] == 80
    close(usage['new_cost_estimate_usd'], api_cost)
    assert usage['input_tokens'] == sum(r['usage']['input_tokens'] for r in lookup.values())
    assert usage['output_tokens'] == sum(r['usage']['output_tokens'] for r in lookup.values())
    db = sqlite3.connect(f'file:{V2 / "api_checkpoint.sqlite"}?mode=ro', uri=True)
    ledger = db.execute('SELECT key,state,reserve,cost,payload FROM calls').fetchall()
    db.close()
    assert len(ledger) == 66 and {r[1] for r in ledger} == {'saved'}
    for key, state, reserve, cost, payload in ledger:
        r = lookup[key]
        close(cost, (r['usage']['input_tokens']*2.5+r['usage']['output_tokens']*15)/1e6)
        assert json.loads(payload)['text'] == r['text'] and cost <= reserve and r['usage']['output_tokens'] <= 512
    close(sum(r[3] for r in ledger), api_cost)
    prior = 2.4225025
    for path in [ROOT/'results/person3_meeting9_workflow/v1/api_checkpoint.sqlite', V1/'api_checkpoint.sqlite']:
        db = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
        prior += db.execute('SELECT coalesce(sum(coalesce(cost,reserve)),0) FROM calls').fetchone()[0]
        db.close()
    close(usage['dynamic_prior_base_usd'], prior)
    close(usage['cumulative_estimate_usd'], prior+api_cost)
    assert prior+api_cost <= 7
    preflight = read(V2/'preflight.json')
    reserve = sum(((len(j['prompt'].encode())+256)*2.5+512*15)/1e6 for j in unique_jobs.values())
    close(preflight['reserve_usd'], reserve)
    close(preflight['cumulative_worst_usd'], prior+reserve)
    assert preflight['cumulative_worst_usd'] <= 7
    assert snapshot == {str(p): sha(p) for p in preserved}
    return {'engineering_checks_passed': True, 'isolated_prompt_reconstructions': 80, 'unique_paid_judges': 66,
            'new_reader_calls': 0, 'new_embeddings': 0, 'strict_q178_repair_acceptance_passed': bool(q178_acceptance),
            'q178_checks': q178_details, 'no_context_guarded_0_0_2_0_cases': 20,
            'raw_guard_changes': int(frame.rule_guard_applied.sum()), 'quote_mechanical_failure_rows': len(quote_failures),
            'human_reviewed': 0, 'actual_isolated_judge_cost_usd': api_cost, 'cumulative_estimate_usd': prior+api_cost,
            'raw_scores_preserved_and_every_guard_recomputed': True, 'pure_abstention_SHA_cases': 20,
            'other_AI_abstention_did_not_receive_G_upgrade': True,
            'pinned_pre_result_hashes_verified': pinned, 'freeze_utc': receipt['utc'],
            'inference_limit': 'Post-review AI-rater repair, not independent confirmation or semantic/human validation.'}, summary, mechanical


def integration(repaired, summary, mechanical):
    completion = read(EXPORT/'completion.json')
    for name, digest in completion['integrated_output_hashes'].items():
        assert sha(EXPORT/name) == digest, name
    assert completion['script_sha256'] == sha(ROOT/'person3_meeting9_followup/build_report.py')
    assert completion['human_reviewed'] == 0 and completion['human_validated_accuracy_complete'] is False
    assert completion['task_3_reliable_answer_accuracy_complete'] is False
    assert 'both rejected' in completion['automated_answer_raters']
    # Root reader table contains direct mechanical checks, reconstructed here
    # from original parquet grades and saved delivered spans/reader strings.
    grades = {}
    for r in pd.read_parquet(ROOT/'results/person3_meeting9_transfer/vidore_finance_v1/data/qrels.parquet').itertuples(index=False):
        key = (str(r.query_id), str(r.corpus_id))
        grades[key] = max(int(r.score), grades.get(key, 0))
    contexts = {(r['query_id'], r['method']): r for r in read(V1/'context_records.json')}
    rows = []
    for answer in read(V1/'successful_api_outputs.json'):
        qid, method = answer['query_id'], answer['style']
        ctx = contexts[qid, method]
        ids = {span['passage_id'] for span in ctx['delivered_spans']}
        cited = re.findall(r'\[P(\d+)\]', answer['text'])
        rows.append(dict(query_id=qid, method=method,
            relevant_page_prefix_present=any(grades.get((qid, pid), 0) > 0 for pid in ids),
            reader_has_page_citation=bool(cited),
            all_cited_ids_in_own_delivered_context=bool(cited) and set(cited).issubset(ids),
            starts_insufficient_evidence=bool(re.match(r'^\s*insufficient evidence\b', answer['text'], re.I)),
            used_wordpiece_tokens=ctx['used_wordpiece_tokens'], delivered_page_prefixes=len(ids), human_status='pending'))
    direct = pd.DataFrame(rows)
    assert len(direct) == 80
    verify_frame(direct, EXPORT/'reader20_mechanical_per_query.csv', ['query_id', 'method'])
    fields = ['relevant_page_prefix_present', 'reader_has_page_citation', 'all_cited_ids_in_own_delivered_context', 'starts_insufficient_evidence']
    expected = direct.groupby('method').agg(n=('query_id', 'size'), **{field: (field, 'sum') for field in fields}).reset_index()
    verify_frame(expected, EXPORT/'reader20_mechanical_summary.csv', ['method'])
    assert expected.set_index('method').loc[METHODS, 'relevant_page_prefix_present'].tolist() == [15, 15, 15, 0]
    assert not (EXPORT/'reader20_isolated_ai_summary.csv').exists()
    cells = csv(RESULT/'inference_v1/two_by_two_summary.csv')
    cells = cells[(cells.cohort == 'remaining249') & (cells.k == 10)]
    verify_frame(cells, EXPORT/'retrieval249_filter_control.csv', ['cohort', 'filter', 'representation', 'method', 'k'])
    tests = csv(RESULT/'inference_v1/primary_tests.csv')
    verify_frame(tests, EXPORT/'retrieval249_complete_test_family.csv', ['id'])
    # Copy consistency only: this auditor produced costs, which another agent audits.
    costs = csv(RESULT/'costs_v1/quality_cost_table.csv')
    verify_frame(costs[costs.cohort == 'remaining249'], EXPORT/'retrieval249_cost_latency.csv', ['cohort', 'method'])
    funds = read(EXPORT/'api_budget.json')
    close(funds['cumulative_estimate_including_reserves_usd'], repaired['cumulative_estimate_usd'])
    assert funds['billing_statement'] is False and funds['guard_usd'] == 7
    preservation = read(EXPORT/'historical_preservation.json')
    assert preservation['passed'] is True
    assert all(not c['missing'] and not c['changed'] for c in preservation['checks'])
    assert preservation['checks'][0]['files'] == 565
    prior_audit = read(EXPORT/'audit/audit.json')
    assert prior_audit['semantic_rubric_reliability'] == 'failed_counterexample'
    assert prior_audit['evidence']['semantic_findings'][0]['count'] == 16
    report = (EXPORT/'README.md').read_text()
    required = ['rejected rater calibration evidence', 'The isolated repair also failed', 'Human-reviewed0/pending',
        'No valid final-answer accuracy or superiority conclusion', 'exploratory question-level inference',
        'does not correct earlier adaptive experiments', 'not measured end-to-end latency', 'not a billing statement',
        '4 from initial60,16 from remaining249', 'not answer ratings', 'human must review the80 blinded',
        'one unique bad judgment affecting two logical method rows', 'task3\'s reliable human answer-quality assessment remains incomplete']
    for phrase in required:
        assert phrase in report, phrase
    labels = dict(zip(METHODS, ['BM25', 'Question Hybrid .50', 'Focused HyDE Hybrid .50', 'No-context control']))
    for row in expected.itertuples():
        assert f'| {labels[row.method]} | {row.n} | {row.relevant_page_prefix_present}/{row.n} | {row.all_cited_ids_in_own_delivered_context}/{row.n} | {row.starts_insufficient_evidence}/{row.n} |' in report
    assert 'Retrieval gains cannot be assumed to translate into better final answers' in report
    rejected = (EXPORT/'evidence_isolated/README.md').read_text()
    assert rejected.startswith('# FAILED RATER CRITERION')
    assert 'Do not use v2 summary.csv as a primary answer-quality or accuracy result' in rejected
    for rel in ['cost_audit/audit_summary.json', 'integration_cost_audit/audit_summary.json']:
        external = read(EXPORT/rel)
        assert external['passed'] is True
    reader_cost_check = read(EXPORT/'integration_cost_audit/audit_summary.json')['integrated_comparison']
    assert reader_cost_check['passed'] is True
    for name, digest in reader_cost_check['input_hashes'].items():
        assert sha(EXPORT/name) == digest, name
    # The retained cost cross-audit predates editorial changes in the builder;
    # its exact integrated cost artifact hashes still match this final report.
    return {'static_integration_checks_passed': True, 'primary_reader_table_uses_AI_scores': False,
            'primary_reader_mechanical_rows_independently_recomputed': 80,
            'primary_reader_metric_is_answer_accuracy': False, 'all_requirements_met': False,
            'task3_reliable_answer_quality_complete': False, 'both_AI_raters_rejected_and_human_pending': True,
            'retrieval_tables_match_previously_independently_audited_inference': True,
            'cost_table_checked_only_for_copy_consistency': True, 'reader_cost_join_independent_audit': 'Another agent, not this auditor',
            'semantic_and_adaptive_inference_caveats_preserved': True,
            'root_output_hashes_verified': completion['integrated_output_hashes'],
            'overview_visual_review': {'reviewed': True, 'sha256': sha(EXPORT/'overview.png'),
                'result': 'All visible labels match retained tables; right panel is 15/15/15/0 relevant-page-prefix counts, explicitly not evidence sufficiency; no AI accuracy panel or clipping.'}}


def main():
    repaired, summary, mechanical = repair()
    integrated = integration(repaired, summary, mechanical)
    result = {'utc': datetime.now(timezone.utc).isoformat(), 'audit_script_sha256': sha(Path(__file__)),
        'engineering_checks_passed': True, 'all_requirements_met': False,
        'scientific_answer_measurement_accepted': False,
        'strict_q178_repair_acceptance_passed': repaired['strict_q178_repair_acceptance_passed'],
        'audit_api_calls': 0, 'new_encoder_calls': 0, 'old_audit_or_profile_repeated': False,
        'repair': repaired, 'integration': integrated,
        'v1_high_failures_preserved': True, 'semantic_human_validation_status': 'Not established; human review pending'}
    if not repaired['strict_q178_repair_acceptance_passed']:
        result['high_severity_finding'] = 'Frozen strict q178 repair acceptance failed. Preserve raw results and refer to human; do not revise prompts until passing.'
    text = json.dumps(result, ensure_ascii=False, indent=2)+'\n'
    for destination in [OUT/'isolation_and_integration_audit.json', EXPORT/'audit/isolation_and_integration_audit.json']:
        destination.write_text(text)
    print(text)


if __name__ == '__main__':
    main()
