"""Audit actual delivered contexts and all completed budget-sweep predictions."""
import csv
import hashlib
import math
import statistics
from pathlib import Path
from retrieval import ROOT, read, save, signature
from run_longbench import counter, chunks, parse_letter
from run_longbench_budgets import OUTPUT, METHODS, load_rows, label

def read_csv(path):
    with path.open(encoding='utf-8') as f:
        return list(csv.DictReader(f))

def audit():
    base = read(ROOT/'config.json')
    config = read(ROOT/'longbench_budget_config.json')
    source = {q['_id']:q for q in load_rows()}
    contexts = read(OUTPUT/'contexts.json')
    original = read(ROOT/'results/longbench/contexts.json')
    tok = counter()
    context_checks = 0
    for qid,methods in contexts.items():
        text = source[qid]['context']
        units = chunks(text,tok,base['longbench_chunk_tokens'],base['longbench_overlap'])
        for method,result in methods.items():
            assert result['ranking']==original[qid][method]['ranking'][:40]
            for budget in config['budgets']:
                record = result['budgets'][label(budget)]
                parts = record['delivered']
                assert len(parts)<=40
                actual = []
                for rank,part in enumerate(parts):
                    parent = units[result['ranking'][rank]]
                    assert part['id']==parent['id']
                    assert part['start_char']==parent['start_char']
                    assert part['end_char']<=parent['end_char']
                    assert part['end_char']>part['start_char']
                    actual.append(text[part['start_char']:part['end_char']])
                assert record['context']=='\n\n'.join(actual)
                token_count = sum(len(tok.encode(t,add_special_tokens=False).ids) for t in actual)
                assert token_count==record['returned_tokens']
                if budget is None:
                    assert len(parts)==40
                    for rank,part in enumerate(parts):
                        assert part['end_char']==units[result['ranking'][rank]]['end_char']
                else:
                    assert token_count<=budget
                context_checks += 1
            assert result['budgets']['2048']['context']==original[qid][method]['context']
    preflight = read(OUTPUT/'prompt_preflight.json')
    assert len(preflight)==200
    for p in preflight:
        assert p['fits'] and p['prompt_tokens']+p['generation_allowance']<=p['model_capacity']
    preflight_by_key = {(p['id'],p['method'],p['budget']):p for p in preflight}
    answers = read(OUTPUT/'answers.json')
    assert len(answers)==200 and len({(r['id'],r['method'],r['budget']) for r in answers})==200
    for r in answers:
        q = source[r['id']]
        assert r['split']==q['pilot_split'] and r['correct_answer']==q['answer']
        assert parse_letter(r['raw_response'])==r['predicted']
        assert r['correct']==int(r['predicted']==q['answer'])
        assert r['valid_letter']==int(r['predicted'] is not None)
        assert r['prompt_tokens']==preflight_by_key[(r['id'],r['method'],r['budget'])]['prompt_tokens']
        if r['method']!='No context':
            assert r['returned_tokens']==contexts[r['id']][r['method']]['budgets'][r['budget']]['returned_tokens']
    summaries = read_csv(OUTPUT/'comparison.csv')
    assert len(summaries)==20
    for row in summaries:
        group = [a for a in answers if all(a[k]==row[k] for k in ['split','method','budget'])]
        assert len(group)==int(row['questions'])
        assert sum(r['correct'] for r in group)==int(row['correct'])
        assert math.isclose(sum(r['correct'] for r in group)/len(group),float(row['accuracy']),abs_tol=1e-12)
    identity = read(OUTPUT/'answer_identity.json')
    assert identity['config']==config
    assert identity['contexts_sha256']==hashlib.sha256((OUTPUT/'contexts.json').read_bytes()).hexdigest()
    assert identity['code_sha256']==hashlib.sha256((ROOT/'run_longbench_budgets.py').read_bytes()).hexdigest()
    assert identity['shared_code_sha256']==signature({p:(ROOT/p).read_text(encoding='utf-8') for p in ['retrieval.py','run_longbench.py']})
    paired = []
    for split in ['dev','test']:
        for method in METHODS:
            small = {a['id']:a for a in answers if a['split']==split and a['method']==method and a['budget']=='2048'}
            for budget in ['8192','all_retrieved']:
                large = {a['id']:a for a in answers if a['split']==split and a['method']==method and a['budget']==budget}
                wins = sum(large[q]['correct']>small[q]['correct'] for q in small)
                losses = sum(large[q]['correct']<small[q]['correct'] for q in small)
                paired.append({'split':split,'method':method,'budget':budget,'questions':len(small),
                               'wins_vs_2048':wins,'losses_vs_2048':losses,'ties':len(small)-wins-losses})
    with (OUTPUT/'paired_changes.csv').open('w',newline='',encoding='utf-8') as f:
        writer = csv.DictWriter(f,fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)
    timing = []
    for summary in summaries:
        times = [a['generation_seconds'] for a in answers if all(a[k]==summary[k] for k in ['split','method','budget'])]
        timing.append({'split':summary['split'],'method':summary['method'],'budget':summary['budget'],
            'mean_generation_seconds':sum(times)/len(times),'median_generation_seconds':statistics.median(times),
            'maximum_generation_seconds':max(times),'runs_above_10_seconds':sum(t>10 for t in times)})
    with (OUTPUT/'timing_summary.csv').open('w',newline='',encoding='utf-8') as f:
        writer = csv.DictWriter(f,fieldnames=list(timing[0]))
        writer.writeheader()
        writer.writerows(timing)
    legacy = {(r['id'],r['method']):r for r in read(ROOT/'results/longbench/answers.json')}
    original_comparisons = [r for r in answers if r['budget'] in ['none','2048']]
    prediction_changes = sum(r['predicted']!=legacy[(r['id'],r['method'])]['predicted'] for r in original_comparisons)
    result = {'contexts_checked':context_checks,'answers_checked':len(answers),'summary_rows_checked':len(summaries),
              'same_top40_ranking_across_budgets':True,'all_uncapped_conditions_include_all40':True,
              'reader_prompt_truncations':0,'maximum_native_prompt_tokens':max(p['prompt_tokens'] for p in preflight),
              'original_2048_contexts_identical':True,'prediction_changes_vs_original_baseline':prediction_changes,
              'status':'passed','interpretation':'Exploratory follow-up on previously inspected questions'}
    save(OUTPUT/'audit.json',result)
    print(result)

if __name__=='__main__':
    audit()
