"""Independent offline audit of the Legal MultiQuery/DSPy study.

No experiment runner, model invocation or upstream credential is imported. An
incomplete checkpoint is explicitly reported as incomplete, never as passed.
"""
from collections import Counter
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import sqlite3
import string
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.feature_extraction.text import CountVectorizer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/person3_meeting10_legal/v1'
KS = [1, 3, 5, 8, 10, 15, 20, 25]
METHODS = ['bm25_reader', 'multiquery_reader', 'dspy_unoptimized', 'dspy_optimized']
TOKEN = re.compile(r'[A-Za-z0-9_$%]+(?:[.\-/][A-Za-z0-9_$%]+)*')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def close(a, b, label='numeric mismatch', tolerance=3e-14):
    assert math.isclose(float(a), float(b), rel_tol=0, abs_tol=tolerance), (label, a, b)


def csv(path):
    return pd.read_csv(path, dtype={'query_id': str}, float_precision='round_trip')


def token_f1(prediction, reference):
    def words(text):
        lowered = text.lower().translate(str.maketrans('', '', string.punctuation))
        return re.sub(r'\b(?:a|an|the)\b', ' ', lowered).split()
    a, b = words(prediction), words(reference)
    ac, bc = Counter(a), Counter(b)
    overlap = sum(min(count, bc[word]) for word, count in ac.items())
    return float(a == b), 2 * overlap / (len(a) + len(b)) if a or b else 1.


def bootstrap(values):
    values = np.asarray(values)
    indices = np.random.default_rng(42).integers(0, len(values), (20000, len(values)))
    return np.quantile(values[indices].mean(axis=1), [.025, .975])


def verify_inputs(result):
    protocol, split = read(OUT / 'protocol.json'), read(OUT / 'selection.json')
    assert protocol['split'] == split
    for filename in ['source_hashes.json', 'implementation_hashes.json']:
        for path, digest in read(OUT / filename).items():
            assert sha(ROOT / path) == digest, f'Frozen input changed: {path}'
    for filename, digest in read(OUT / 'freeze_receipt.json')['hashes'].items():
        assert sha(OUT / filename) == digest, filename
    for package, version in protocol['versions'].items():
        assert importlib.metadata.version(package) == version, package
    qa_path, corpus_path = [ROOT / 'data/legal_rag_bench' / name for name in ['qa.jsonl', 'corpus.jsonl']]
    original_protocol = read(ROOT / 'results/person3_openai_hyde/protocol.json')
    for path in [qa_path, corpus_path]:
        assert sha(path) == original_protocol['data_hashes'][path.name], path.name
    qa = [json.loads(line) for line in qa_path.read_text().splitlines() if line.strip()]
    pages = [json.loads(line) for line in corpus_path.read_text().splitlines() if line.strip()]
    for q in qa:
        q['id'] = str(q['id'])
    qmap, pmap = {q['id']: q for q in qa}, {p['id']: p for p in pages}
    assert len(qa) == len(qmap) == 100 and len(pages) == len(pmap) == 4876
    assert all(q['relevant_passage_id'] in pmap for q in qa)
    historical = pd.read_csv(ROOT / 'results/person3_openai_hyde/evaluation_queries.csv', dtype=str).to_dict('records')
    assert all(all(old[key] == q[key] for key in ['id', 'question', 'answer', 'relevant_passage_id'])
               for old, q in zip(historical, qa)) and len(historical) == 100
    groups = [split[name] for name in ['train_ids', 'val_ids', 'test_ids']]
    assert [len(x) for x in groups] == [10, 10, 80]
    assert set.union(*(set(x) for x in groups)) == set(qmap)
    assert all(len(x) == len(set(x)) for x in groups)
    assert set(split['dev_ids']) == set(groups[0]) | set(groups[1])
    for i, a in enumerate(groups):
        for b in groups[:i]:
            assert not set(a) & set(b)
            assert not {qmap[q]['relevant_passage_id'] for q in a} & {qmap[q]['relevant_passage_id'] for q in b}
    # Reproduce the specified split, including group ordering and greedy rule.
    by_passage = {}
    for q in qa:
        by_passage.setdefault(q['relevant_passage_id'], []).append(q['id'])
    order = np.random.default_rng(42).permutation(sorted(by_passage)).tolist()
    dev, selected = [], []
    for pid in order:
        if len(dev) + len(by_passage[pid]) <= 20:
            dev.extend(by_passage[pid]); selected.append(pid)
        if len(dev) == 20:
            break
    train = []
    for pid in selected:
        if len(train) + len(by_passage[pid]) <= 10:
            train.extend(by_passage[pid])
        if len(train) == 10:
            break
    assert train == split['train_ids'] and dev == split['dev_ids']
    assert [q for q in dev if q not in train] == split['val_ids']
    assert [q['id'] for q in qa if q['id'] not in dev] == split['test_ids']
    result['checks']['inputs'] = dict(frozen_hashes=True, original_corpus_and_qa_byte_hashes_preserved=True,
         train=10, validation=10, test=80, gold_passages_disjoint=True, split_replayed=True)
    return protocol, split, qa, pages


class IndependentBM25:
    def __init__(self, pages):
        self.ids = [p['id'] for p in pages]
        self.index = {p: i for i, p in enumerate(self.ids)}
        texts = [(p['title'] + ' ' + p['text']).strip() for p in pages]
        self.vectorizer = CountVectorizer(lowercase=True, tokenizer=lambda text: TOKEN.findall(text.lower()),
                                         token_pattern=None, dtype=np.float32)
        counts = self.vectorizer.fit_transform(texts).tocsr().astype(np.float32)
        lengths = np.asarray(counts.sum(axis=1)).ravel()
        dfs = np.asarray((counts > 0).sum(axis=0)).ravel()
        inverse = np.log1p((len(texts) - dfs + .5) / (dfs + .5)).astype(np.float32)
        row_ids = np.repeat(np.arange(len(texts)), np.diff(counts.indptr))
        terms = counts.data.copy()
        counts.data = terms * 2.2 / (terms + 1.2 * (.25 + .75 * lengths[row_ids] / float(lengths.mean()))) * inverse[counts.indices]
        self.weights = counts
        self.cache = {}

    def rank(self, text):
        if text not in self.cache:
            query = self.vectorizer.transform([text]).tocsr()
            query.data.fill(1)
            scores = (query @ self.weights.T).toarray().ravel()
            self.cache[text] = np.argsort(-scores, kind='stable')[:100]
        return self.cache[text]

    def pids(self, text, k):
        return [self.ids[i] for i in self.rank(text)[:k]]


def verify_baseline(result, qa, pages):
    bm = IndependentBM25(pages)
    current = np.load(OUT / 'baseline_rankings.npz')
    historical = np.load(ROOT / 'results/person3_openai_hyde/saved_rankings.npz')
    assert current['query_ids'].tolist() == [q['id'] for q in qa]
    assert current['passage_ids'].tolist() == bm.ids
    gold = [bm.index[q['relevant_passage_id']] for q in qa]
    assert gold == current['gold'].tolist() == historical['gold'].tolist()
    replayed = np.stack([bm.rank(q['question']) for q in qa])
    assert np.array_equal(replayed, current['bm25']) and np.array_equal(replayed, historical['bm25'])
    assert sum(q['relevant_passage_id'] in bm.pids(q['question'], 5) for q in qa) == 31
    result['checks']['baseline'] = dict(all100_top100_exact=True, full100_hit5=.31,
                                       new_hyde_q_to_h_experiment=False)
    return bm


def verify_multiquery(result, qa, bm):
    saved, rankings = read(OUT / 'multiquery.json'), read(OUT / 'rankings.json')
    assert set(saved) <= {q['id'] for q in qa}
    if len(saved) != 100:
        return False
    assert set(rankings) == set(saved)
    for q in qa:
        item, routes = saved[q['id']], saved[q['id']]['per_query_rankings']
        queries = item['queries']
        assert item['question'] == q['question'] and queries[-1] == q['question']
        assert len(queries) == len(routes) == item['query_count']
        assert item['generated_queries'] == queries[:-1] and item['generated_query_count'] == len(queries) - 1
        assert [r['query'] for r in routes] == queries
        union = []
        for route in routes:
            assert route['passage_ids'] == bm.pids(route['query'], 25)
            assert len(route['passage_ids']) == len(set(route['passage_ids'])) == 25
            for pid in route['passage_ids']:
                if pid not in union:
                    union.append(pid)
        assert union == item['native_union_ids'] and len(union) == item['native_union_size']
        for k in KS:
            assert rankings[q['id']]['bm25'][str(k)] == bm.pids(q['question'], k)
            candidates = {pid for route in routes for pid in route['passage_ids'][:k]}
            scores = {pid: sum(1 / (60 + route['passage_ids'].index(pid) + 1)
                      for route in routes if pid in route['passage_ids'][:k]) for pid in candidates}
            expected = sorted(candidates, key=lambda pid: (-scores[pid], bm.index[pid]))[:k]
            assert expected == rankings[q['id']]['multiquery'][str(k)], (q['id'], k)
    result['checks']['multiquery'] = dict(queries=100, raw_generated_lines_preserved=True,
          per_route_bm25_recomputed=True, native_union_recomputed=True, per_k_rrf_rank_checks=800,
          actual_generated_query_count=dict(Counter(x['generated_query_count'] for x in saved.values())))
    return True


def context(ids, pages):
    full_texts = [(pid, (pages[pid]['title'] + ' ' + pages[pid]['text']).strip().split()) for pid in ids]
    cumulative, blocks, delivered = 0, [], []
    for pid, words in full_texts:
        available = max(0, 2400 - cumulative)
        count = min(len(words), available)
        if not count:
            break
        blocks.append(f'Passage {len(blocks)+1}:\n' + ' '.join(words[:count]))
        delivered.append(dict(passage_id=pid, full=count == len(words), words=count, original_words=len(words)))
        cumulative += count
        if cumulative == 2400 or count < len(words):
            break
    return dict(context='\n\n'.join(blocks), delivery=delivered, used_words=cumulative)


def verify_optimizer(result, split, qa, pages):
    folder = OUT / 'optimizer'
    status = read(folder / 'optimization_status.json')
    if status['status'] != 'complete':
        return False
    for filename, digest in status['completed_artifact_sha256'].items():
        assert sha(folder / filename) == digest
    inputs = read(folder / 'optimization_inputs.json')
    body = {k: v for k, v in inputs.items() if k != 'input_sha256'}
    # The optimizer input digest uses sorted JSON with default separators.
    digest = hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    assert digest == inputs['input_sha256'] == status['input_sha256']
    qmap, pmap = {q['id']: q for q in qa}, {p['id']: p for p in pages}
    rankings = read(OUT / 'rankings.json')
    for field, ids in [('train', split['train_ids']), ('validation', split['val_ids'])]:
        rows = inputs[field]
        assert [r['id'] for r in rows] == ids
        for row in rows:
            q = qmap[row['id']]
            assert row['question'] == q['question'] and row['answer'] == q['answer']
            assert row['context'] == context(rankings[q['id']]['bm25']['8'], pmap)['context']
    assert status['inference_inputs'] == ['context', 'question']
    assert status['max_bootstrapped_demos'] == status['max_labeled_demos'] == 0
    initial, compiled = read(folder / 'unoptimized_signature.json'), read(folder / 'compiled_signature.json')
    for signature in initial + compiled:
        assert signature['demonstration_count'] == 0
        assert signature['input_fields'] == ['context', 'question']
        assert signature['output_fields'] == ['reasoning', 'answer']
    result['checks']['optimizer'] = dict(dev_only=True, train=10, validation=10, zero_demonstrations=True,
         inference_inputs=['context', 'question'], compiled_artifacts_hash_checked=True,
         optimized_instructions=compiled[0]['instructions'])
    return True


def verify_metrics(result, protocol, split, qa, pages):
    qmap, pmap = {q['id']: q for q in qa}, {p['id']: p for p in pages}
    rankings, outputs = read(OUT / 'rankings.json'), read(OUT / 'reader_outputs.json')
    expected_keys = {m + '::' + qid for m in METHODS for qid in split['test_ids']}
    assert set(outputs) <= expected_keys
    if set(outputs) != expected_keys:
        return False
    rframe = csv(OUT / 'retrieval_per_query.csv')
    assert len(rframe) == 1600 and not rframe.duplicated(['query_id', 'method', 'k']).any()
    for row in rframe.itertuples(index=False):
        assert row.k in KS and row.method in ['bm25', 'multiquery']
        assert row.cohort == ('test80' if row.query_id in split['test_ids'] else 'dev20')
        ids = rankings[row.query_id][row.method][str(row.k)]
        gold = qmap[row.query_id]['relevant_passage_id']
        position = ids.index(gold) + 1 if gold in ids else None
        expected = [float(position is not None)] * 2 + [1 / position if position else 0., 1 / math.log2(position + 1) if position else 0.]
        for actual, value in zip([row.hit, row.recall, row.mrr, row.ndcg], expected):
            close(actual, value)
    for row in csv(OUT / 'retrieval_summary.csv').itertuples(index=False):
        subset = rframe[(rframe.method == row.method) & (rframe.k == row.k)]
        if row.cohort != 'full100':
            subset = subset[subset.cohort == row.cohort]
        assert len(subset) == row.n
        for field in ['hit', 'recall', 'mrr', 'ndcg']:
            close(getattr(row, field), subset[field].mean())
    answer_values = {}
    for key, output in outputs.items():
        method, qid = key.split('::')
        assert output['method'] == method and output['query_id'] == qid
        assert output['question'] == qmap[qid]['question'] and output['answer'].strip()
        route = 'multiquery' if method == 'multiquery_reader' else 'bm25'
        delivery = context(rankings[qid][route]['8'], pmap)
        assert all(output[field] == value for field, value in delivery.items()), key
        assert output['used_words'] <= 2400
        em, f1 = token_f1(output['answer'], qmap[qid]['answer'])
        gold = qmap[qid]['relevant_passage_id']
        answer_values[(qid, method)] = dict(em=em, token_f1=f1, empty_answer=False,
            used_words=delivery['used_words'], full_passages=sum(x['full'] for x in delivery['delivery']),
            partial_passage=sum(not x['full'] for x in delivery['delivery']),
            gold_full=any(x['full'] and x['passage_id'] == gold for x in delivery['delivery']),
            gold_touched=any(x['passage_id'] == gold for x in delivery['delivery']))
    aframe = csv(OUT / 'reader_per_query.csv')
    assert len(aframe) == 320 and not aframe.duplicated(['query_id', 'method']).any()
    assert set(zip(aframe.query_id, aframe.method)) == set(answer_values)
    for row in aframe.itertuples(index=False):
        for field, value in answer_values[(row.query_id, row.method)].items():
            close(getattr(row, field), value, field)
    for row in csv(OUT / 'reader_summary.csv').itertuples(index=False):
        subset = aframe[aframe.method == row.method]
        assert row.n == len(subset) == 80
        for field in ['em', 'token_f1', 'used_words', 'full_passages', 'partial_passage', 'gold_full', 'gold_touched']:
            close(getattr(row, field), subset[field].mean())
    result['checks']['metrics'] = dict(retrieval_rows=1600, reader_rows=320,
        reader_contexts_repacked=True, dspy_evidence_identical_to_bm25=True,
        em_and_token_f1_recomputed=True, token_f1_is_lexical_proxy_not_legal_accuracy=True)
    return True


def verify_statistics(result, split):
    retrieval = csv(OUT / 'retrieval_per_query.csv')
    pivot = retrieval[retrieval.k == 5].pivot(index='query_id', columns='method', values='hit').loc[split['test_ids']]
    delta = (pivot.multiquery - pivot.bm25).to_numpy()
    saved = read(OUT / 'primary_retrieval_test.json')
    wins, losses = int((delta > 0).sum()), int((delta < 0).sum())
    assert (saved['wins'], saved['losses'], saved['ties'], saved['n']) == (wins, losses, int((delta == 0).sum()), 80)
    close(saved['mean_delta'], delta.mean())
    close(saved['p_exact_mcnemar'], binomtest(wins, wins + losses, .5).pvalue if wins + losses else 1.)
    assert np.allclose(saved['paired_bootstrap95'], bootstrap(delta), atol=3e-14, rtol=0)
    answers = csv(OUT / 'reader_per_query.csv').pivot(index='query_id', columns='method', values='token_f1').loc[split['test_ids']]
    pairs = [('multiquery_reader', 'bm25_reader'), ('dspy_optimized', 'bm25_reader'), ('dspy_optimized', 'dspy_unoptimized')]
    matrix = np.column_stack([(answers[a] - answers[b]).to_numpy() for a, b in pairs])
    counts, rng = np.zeros(3, dtype=int), np.random.default_rng(42)
    for _ in range(100):
        signs = rng.integers(0, 2, (1000, 80), dtype=np.int8) * 2 - 1
        counts += (np.abs(signs @ matrix) >= np.abs(matrix.sum(axis=0)) - 1e-12).sum(axis=0)
    raw = (counts + 1) / 100001
    adjusted, previous = np.zeros(3), 0.
    for rank, index in enumerate(sorted(range(3), key=lambda j: raw[j])):
        previous = min(1., max(previous, (3 - rank) * raw[index])); adjusted[index] = previous
    tests = csv(OUT / 'reader_paired_tests.csv')
    assert list(zip(tests.candidate, tests.control)) == pairs
    for j, row in enumerate(tests.itertuples(index=False)):
        d = matrix[:, j]
        assert row.n == 80 and row.metric == 'reference_token_f1_lexical_proxy'
        assert (row.wins, row.losses, row.ties) == (int((d > 1e-12).sum()), int((d < -1e-12).sum()), int((abs(d) <= 1e-12).sum()))
        close(row.mean_delta, d.mean()); close(row.p_raw, raw[j]); close(row.p_holm3, adjusted[j])
        assert np.allclose(json.loads(row.paired_bootstrap95), bootstrap(d), atol=3e-14, rtol=0)
    result['checks']['statistics'] = dict(exact_mcnemar=True, reader_signflip_draws=100000,
            reader_holm_family=3, seed=42, paired_bootstrap_draws=20000, all_recomputed=True)


def verify_api(result, protocol, split, qa):
    path = OUT / 'api/api_checkpoint.sqlite'
    db = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    with db:
        calls = [dict(row) for row in db.execute('SELECT * FROM calls')]
        events = [dict(row) for row in db.execute('SELECT * FROM events ORDER BY event_id')]
        meta = dict(db.execute('SELECT key,value FROM meta'))
    db.close()
    settings = json.loads(meta['settings'])
    assert settings['model'] == 'gpt-4o-mini' and settings['max_unique_attempts'] == 600
    close(settings['input_price'], .15); close(settings['output_price'], .60)
    close(settings['prior_usd'], protocol['budget']['prior_estimate_usd'])
    total, known, reserve = 0., 0., 0.
    callmap, models, finishes, stages = {}, Counter(), Counter(), Counter()
    for row in calls:
        request = json.loads(row['request_json'])
        assert canonical(request) == row['request_json']
        assert hashlib.sha256(row['request_json'].encode()).hexdigest() == row['key']
        assert request['model'] == 'gpt-4o-mini' and request.get('n', 1) == 1 and not request.get('stream', False)
        assert ('max_tokens' in request) != ('max_completion_tokens' in request)
        cap = request.get('max_tokens', request.get('max_completion_tokens'))
        expected_reserve = ((len(row['request_json'].encode()) + 256) * .15 + cap * .60) / 1e6
        close(row['reserve'], expected_reserve)
        assert row['state'] != 'pending', 'API attempts are still in flight'
        payload = json.loads(row['response_json']) if row['response_json'] else None
        if row['state'] == 'saved':
            assert payload is not None
            usage = payload['usage']
            assert usage == json.loads(row['usage_json'])
            for field in ['prompt_tokens', 'completion_tokens']:
                assert isinstance(usage[field], int) and not isinstance(usage[field], bool) and usage[field] >= 0
            assert usage['completion_tokens'] <= cap
            cost = (usage['prompt_tokens'] * .15 + usage['completion_tokens'] * .60) / 1e6
            close(row['cost'], cost)
            models[payload['model']] += 1
            assert payload['model'].startswith('gpt-4o-mini')
            assert len(payload['choices']) == 1
            reason = payload['choices'][0]['finish_reason']
            finishes[reason] += 1; stages[(row['created_stage'], reason)] += 1
            known += cost
        else:
            assert row['state'] == 'unknown_reserved' and row['cost'] is None
            reserve += row['reserve']
        total += row['cost'] if row['cost'] is not None else row['reserve']
        callmap[row['key']] = (request, payload)
    assert len(calls) <= 600 and total <= 1 + 1e-12 and settings['prior_usd'] + total <= 7 + 1e-12
    summary = read(OUT / 'api/cost_summary.json')
    assert summary['unique_attempts'] == len(calls)
    close(summary['known_actual_cost_usd'], known); close(summary['unresolved_reserved_usd'], reserve)
    close(summary['new_cost_and_reserves_usd'], total)
    assert dict(Counter(row['state'] for row in calls)) == summary['states']
    assert sum(row['cache_hit'] for row in events) == int(meta['cache_hits']) == summary['cache_hits']
    for row in events:
        if row['identity_key'] is not None:
            assert row['identity_key'] in callmap or row['http_status'] != 200
    event_lines = [json.loads(line) for line in (OUT / 'api/request_events.jsonl').read_text().splitlines() if line.strip()]
    assert len(event_lines) == len(events)
    for line, event in zip(event_lines, events):
        assert all(line[k] == v for k, v in event.items())
    # Every generated line is reconstructed from the stored *actual* LLM output.
    generated = read(OUT / 'multiquery.json')
    template = read(OUT / 'framework_metadata.json')['multiquery_default_prompt']
    for q in qa:
        messages = [dict(role='user', content=template.format(question=q['question']))]
        matches = [(req, payload) for req, payload in callmap.values()
                   if req['messages'] == messages and payload is not None]
        assert len(matches) == 1, ('MQR request mapping', q['id'])
        text = matches[0][1]['choices'][0]['message']['content']
        assert list(filter(None, text.strip().split('\n'))) == generated[q['id']]['generated_queries']
    # Reconstruct native DSPy formatting without executing a predictor or model.
    import dspy
    from dspy.adapters.base import prompt_to_openai_messages
    programs = {}
    for name, filename in [('dspy_unoptimized', 'unoptimized_program.json'), ('dspy_optimized', 'compiled_program.json')]:
        program = dspy.ChainOfThought('context, question -> answer', rationale_field_type=str)
        program.load(str(OUT / 'optimizer' / filename))
        programs[name] = program
    outputs = read(OUT / 'reader_outputs.json')
    reader_finishes = Counter()
    truncated = []
    for identity, output in outputs.items():
        method, qid = identity.split('::')
        if method in programs:
            signature = programs[method].predict.signature
            formatted = dspy.XMLAdapter().format(signature, [], dict(context=output['context'], question=output['question']))
            # In DSPy3.5 format() returns a typed Prompt with separate system
            # instructions. Use its offline SDK-boundary writer, not equality
            # between a Prompt object and the retained OpenAI message list.
            expected_messages = prompt_to_openai_messages(formatted)
        else:
            expected_messages = [dict(role='system', content=protocol['reader']['system']),
                 dict(role='user', content=protocol['reader']['template'].format(context=output['context'], question=output['question']))]
        keys = {e['identity_key'] for e in events if e['stage'] == method and e['http_status'] == 200}
        candidates = []
        for key in keys:
            request, payload = callmap[key]
            if request['messages'] != expected_messages or payload is None:
                continue
            content = payload['choices'][0]['message']['content']
            if method in programs:
                try:
                    parsed = dspy.XMLAdapter().parse(signature, content)
                except Exception:
                    continue
                if parsed['answer'] != output['answer']:
                    continue
                assert str(parsed['reasoning']) == output['reasoning']
            elif content != output['answer']:
                continue
            candidates.append(payload)
        assert candidates, ('Reader prompt/output not mapped to ledger', identity)
        reasons = {payload['choices'][0]['finish_reason'] for payload in candidates}
        assert len(reasons) == 1, identity
        reason = next(iter(reasons)); reader_finishes[(method, reason)] += 1
        if 'finish_reason' in output:
            assert output['finish_reason'] == reason
        if reason == 'length':
            truncated.append(identity)
    result['checks']['api'] = dict(unique_attempts=len(calls), known_actual_cost_usd=known,
        unresolved_reserved_usd=reserve, spend_plus_reserves_usd=total,
        cumulative_estimate_including_reserves_usd=settings['prior_usd'] + total,
        cache_hits=int(meta['cache_hits']), upstream_model_counts=dict(models),
        finish_reason_counts=dict(finishes),
        finish_reasons_by_created_stage={stage+'::'+str(reason): count for (stage, reason), count in stages.items()},
        reader_finish_reason_counts={method+'::'+str(reason): count for (method, reason), count in reader_finishes.items()},
        reader_truncated_outputs=truncated,
        request_response_line_parser_checked=True, reader_prompts_reconstructed_without_references=True,
        actual_reader_outputs_mapped_to_api_ledger=320, prices_recomputed=True)
    if truncated:
        result['limitations'].append(f'{len(truncated)} reader outputs ended with length; IDs retained in checks.api.reader_truncated_outputs. Scores include these outputs.')
    if reserve:
        result['limitations'].append('Unresolved API attempts retain their full reservation and are not imputed as successful outputs.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-partial', action='store_true', help='Exit0 on an explicitly incomplete checkpoint; passed remains false')
    args = parser.parse_args()
    result = dict(passed=False, status='incomplete', verified_at_utc=datetime.now(timezone.utc).isoformat(), checks={},
         limitations=['All100 Legal questions were previously inspected; test80 isolates only this optimizer round.',
                      'Question-level inference does not adjust shared source dependence or all historical experiments.',
                      'Answer EM/token-F1 are lexical reference proxies, not validated legal correctness.'])
    code = 2
    try:
        protocol, split, qa, pages = verify_inputs(result)
        bm = verify_baseline(result, qa, pages)
        required = ['multiquery.json', 'rankings.json', 'optimizer/optimization_status.json', 'reader_outputs.json',
                    'retrieval_per_query.csv', 'retrieval_summary.csv', 'primary_retrieval_test.json',
                    'reader_per_query.csv', 'reader_summary.csv', 'reader_paired_tests.csv', 'api/api_checkpoint.sqlite']
        missing = [name for name in required if not (OUT / name).exists()]
        if (OUT / 'multiquery.json').exists() and (OUT / 'rankings.json').exists():
            assert verify_multiquery(result, qa, bm), 'MultiQuery checkpoint has fewer than100 outputs'
        if (OUT / 'optimizer/optimization_status.json').exists():
            if not verify_optimizer(result, split, qa, pages):
                missing.append('optimizer completion')
        result['missing_or_incomplete'] = missing
        if not missing:
            if verify_metrics(result, protocol, split, qa, pages):
                verify_statistics(result, split)
                verify_api(result, protocol, split, qa)
                result.update(passed=True, status='complete')
                code = 0
            else:
                result['missing_or_incomplete'].append('320 reader outputs')
    except FileNotFoundError as error:
        result['missing_or_incomplete'] = [str(error.filename)]
    except Exception as error:
        result.update(status='failed', error_type=type(error).__name__, error=str(error))
        code = 1
    OUT.mkdir(parents=True, exist_ok=True)
    destination = OUT / 'independent_validation.json'
    temporary = destination.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(destination)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if args.allow_partial and result['status'] == 'incomplete' else code


if __name__ == '__main__':
    sys.exit(main())
