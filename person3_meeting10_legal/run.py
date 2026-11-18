"""Legal MultiQueryRetriever and DSPy benchmark, isolated from prior experiments."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import argparse
import concurrent.futures as cf
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import re
import shutil
import string
import threading
import time

os.environ.setdefault('LANGSMITH_TRACING', 'false')
os.environ.setdefault('LANGCHAIN_TRACING_V2', 'false')
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/person3_meeting10_legal/v1'
EXPORT = ROOT.parent / 'outputs/meeting10_20261117/legal'
DOC = ROOT / 'presentation/meeting10_legal_20261117'
os.environ.setdefault('DSPY_CACHEDIR', str(OUT / 'dspy_cache'))
os.environ.setdefault('MPLCONFIGDIR', str(OUT / 'matplotlib_cache'))
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

MODEL = 'gpt-4o-mini'
KS = [1, 3, 5, 8, 10, 15, 20, 25]
DEPTH = max(KS)
READER_K, WORD_BUDGET = 8, 2400
QA_PATH = ROOT / 'data/legal_rag_bench/qa.jsonl'
CORPUS_PATH = ROOT / 'data/legal_rag_bench/corpus.jsonl'
TOKEN_RE = re.compile(r'[A-Za-z0-9_$%]+(?:[.\-/][A-Za-z0-9_$%]+)*')
READER_SYSTEM = ('Answer the question using only the supplied legal passages. '
                 'If the passages do not support an answer, say that the evidence is insufficient. '
                 'Give a concise answer of at most 80 words, without a heading.')
READER_TEMPLATE = 'Legal passages:\n{context}\n\nQuestion: {question}\nAnswer:'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def frozen(path, value):
    if Path(path).exists():
        assert read(path) == value, f'Frozen artifact changed: {path}'
    else:
        dump(path, value)


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


class BM25:
    """Exact prior BM25 algebra, binary query counts, stable corpus-index ties."""
    def __init__(self, pages):
        self.pages = pages
        self.pids = [p['id'] for p in pages]
        texts = [(p['title'] + ' ' + p['text']).strip() for p in pages]
        self.vec = CountVectorizer(lowercase=True, tokenizer=tokenize, token_pattern=None, dtype=np.float32)
        self.mat = self.vec.fit_transform(texts).tocsr().astype(np.float32)
        lengths = np.asarray(self.mat.sum(axis=1)).ravel()
        average = float(lengths.mean())
        df = np.asarray((self.mat > 0).sum(axis=0)).ravel()
        idf = np.log1p((len(texts) - df + .5) / (df + .5)).astype(np.float32)
        ids = np.repeat(np.arange(len(texts)), np.diff(self.mat.indptr))
        tf = self.mat.data
        self.mat.data = tf * 2.2 / (tf + 1.2 * (.25 + .75 * lengths[ids] / average)) * idf[self.mat.indices]

    def indices(self, question, depth=100):
        query = self.vec.transform([question]).tocsr()
        query.data[:] = 1
        scores = (query @ self.mat.T).toarray().ravel()
        return np.argsort(-scores, kind='stable')[:depth]

    def search(self, question, depth):
        return [dict(passage_id=self.pids[i], text=(self.pages[i]['title']+' '+self.pages[i]['text']).strip())
                for i in self.indices(question, depth)]


def data():
    qa = [json.loads(x) for x in QA_PATH.read_text().splitlines() if x.strip()]
    pages = [json.loads(x) for x in CORPUS_PATH.read_text().splitlines() if x.strip()]
    for q in qa:
        q['id'] = str(q['id'])
    assert len(qa) == 100 and len(pages) == 4876
    assert len({q['id'] for q in qa}) == 100 and len({p['id'] for p in pages}) == 4876
    return qa, pages


def choose_groups(qa):
    groups = {}
    for q in qa:
        groups.setdefault(q['relevant_passage_id'], []).append(q['id'])
    order = np.random.default_rng(42).permutation(sorted(groups)).tolist()
    dev = []
    selected = set()
    for pid in order:
        if len(dev) + len(groups[pid]) <= 20:
            dev.extend(groups[pid])
            selected.add(pid)
        if len(dev) == 20:
            break
    train = []
    train_groups = set()
    for pid in order:
        if pid in selected and len(train) + len(groups[pid]) <= 10:
            train.extend(groups[pid])
            train_groups.add(pid)
        if len(train) == 10:
            break
    val = [qid for qid in dev if qid not in train]
    test = [q['id'] for q in qa if q['id'] not in dev]
    assert len(train) == len(val) == 10 and len(test) == 80
    mapping = {q['id']: q for q in qa}
    passage_sets = [{mapping[q]['relevant_passage_id'] for q in ids} for ids in [train, val, test]]
    assert all(not passage_sets[i] & passage_sets[j] for i in range(3) for j in range(i))
    return dict(seed=42, grouping='relevant_passage_id', algorithm='Permute sorted gold-passage groups; greedily fill dev20 then train10; rest val10/test80',
                train_ids=train, val_ids=val, test_ids=test, dev_ids=dev,
                caveat='All100 previously inspected; test80 excludes only this round optimizer data, not historical exposure')


def check():
    for filename in ['source_hashes.json', 'implementation_hashes.json']:
        for path, digest in read(OUT / filename).items():
            assert sha(ROOT / path) == digest, f'Changed frozen input: {path}'
    for path, digest in read(OUT / 'freeze_receipt.json')['hashes'].items():
        assert sha(OUT / path) == digest, path


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    qa, pages = data()
    split = choose_groups(qa)
    versions = {p: importlib.metadata.version(p) for p in ['langchain-classic', 'langchain-core', 'langchain-openai', 'dspy', 'optuna', 'openai', 'numpy', 'pandas', 'scikit-learn']}
    assert versions['langchain-classic'] == '1.0.8' and versions['dspy'] == '3.5.0'
    from methods import framework_metadata
    metadata = framework_metadata()
    protocol = dict(date='2026-11-17', dataset='Legal RAG Bench', revision='db0b31dc6d195ce9916897e1ac5e4e6209736c8a',
        n=100, corpus_passages=4876, split=split, model=MODEL, versions=versions,
        primary_retrieval=dict(metric='Hit@5', cohort='test80', contrast='MultiQueryRetriever+BM25 minus original-question BM25'),
        secondary_retrieval=dict(ks=KS, metrics=['hit', 'recall', 'mrr', 'ndcg'], single_gold=True, dev20_and_full100='descriptive'),
        multiquery=dict(framework='Native MultiQueryRetriever.from_llm', default_prompt=metadata['multiquery_default_prompt'],
            include_original=True, requested_alternatives=3, actual_line_parser='Native behavior retained; all generated lines reported, no outcome-based cleanup',
            each_route_depth=DEPTH, per_k='Truncate every per-query list to k, then equal-weight RRF constant60 to k',
            distinct_document_union='Native framework union saved separately; RRF is our explicit adaptation matching teammate',
            temperature=0, max_output_tokens=256),
        dspy=dict(strategy='ChainOfThought reader over fixed original-question BM25; NOT a new retriever',
            optimizer='MIPROv2 auto light', seed=42, train_n=10, validation_n=10,
            max_bootstrapped_demos=0, max_labeled_demos=0, metric='Normalized reference-answer token-F1, lexical proxy only',
            controls=['BM25 fixed concise reader', 'same ChainOfThought before optimization'],
            test_n=80, test_reference_answers_in_prompt=False, evidence_identical_to_bm25=True,
            signature=metadata['dspy_signature']),
        reader=dict(depth=READER_K, word_budget=WORD_BUDGET, temperature=0, max_output_tokens=512,
            system=READER_SYSTEM, template=READER_TEMPLATE, context='Original title+text passages in rank order, last prefix truncated at word budget; then stop',
            overhead='Budget counts document title+text only, not labels/question/instructions',
            metric='Normalized exact match and token-F1; NOT validated legal correctness/accuracy',
            delivery_metrics=['full_passages', 'partial_passage', 'used_words', 'gold_full', 'gold_touched']),
        statistics=dict(primary='Paired exact McNemar for test80 MQR vs BM25 Hit@5',
            reader='Three paired token-F1 contrasts vs BM25 reader and unoptimized CoT; sign-flip100000 draws seed42; Holm3',
            intervals='Paired question bootstrap95% intervals,20000 draws seed42, descriptive; no source clustering',
            caveat='Repeatedly inspected Legal benchmark. Group split isolates current optimization only. No global historical multiplicity adjustment.'),
        budget=dict(prior_estimate_usd=4.570655, cumulative_guard_usd=7, new_guard_usd=1, unique_attempt_guard=600,
            prices_per_million=dict(input=.15, output=.60), pricing_checked='2026-11-17',
            source='https://developers.openai.com/api/docs/models/gpt-4o-mini',
            no_automatic_retries=True, unknown_attempts_remain_reserved=True, successful_exact_payload_cache=True),
        cross_dataset_reporting='Long-form schema retains dataset, stage changed, test n, metric name, matched baseline and delta. Legal F1 must not be labeled MC accuracy.',
        scope='New framework study only. Task1 Q-to-H result is explained from saved evidence without a new HyDE experiment.')
    frozen(OUT / 'selection.json', split)
    frozen(OUT / 'protocol.json', protocol)
    frozen(OUT / 'framework_metadata.json', metadata)
    sources = [QA_PATH, CORPUS_PATH, ROOT / 'results/person3_openai_hyde/saved_rankings.npz',
               ROOT / 'results/person3_openai_hyde/evaluation_queries.csv',
               ROOT / 'results/person3_offline_followup/step1_v1/legal_sparse_tests.csv']
    frozen(OUT / 'source_hashes.json', {str(p.relative_to(ROOT)): sha(p) for p in sources})
    scripts = [Path(__file__).resolve()] + [Path(__file__).with_name(p).resolve() for p in ['api.py', 'methods.py']]
    frozen(OUT / 'implementation_hashes.json', {str(p.relative_to(ROOT)): sha(p) for p in scripts})
    bm = BM25(pages)
    original = np.load(ROOT / 'results/person3_openai_hyde/saved_rankings.npz')
    evaluation = pd.read_csv(ROOT / 'results/person3_openai_hyde/evaluation_queries.csv', dtype=str)
    assert evaluation.id.tolist() == [q['id'] for q in qa]
    ranks = np.stack([bm.indices(q['question'], 100) for q in qa])
    assert np.array_equal(ranks, original['bm25']), 'Prior BM25 replay differs; no API calls permitted'
    gold = np.array([bm.pids.index(q['relevant_passage_id']) for q in qa])
    assert np.array_equal(gold, original['gold']) and (ranks[:, :5] == gold[:, None]).any(axis=1).sum() == 31
    np.savez_compressed(OUT / 'baseline_rankings.npz', query_ids=np.array([q['id'] for q in qa]),
                        passage_ids=np.array(bm.pids), gold=gold, bm25=ranks)
    dump(OUT / 'baseline_replay.json', dict(passed=True, exact_rank_checks=100, depth=100,
         full100_hit5=.31, api_calls=0, source_hashes_unchanged=True))
    if not (OUT / 'freeze_receipt.json').exists():
        assert not (OUT / 'multiquery.json').exists() and not (OUT / 'optimizer').exists()
        names = ['protocol.json', 'selection.json', 'framework_metadata.json', 'source_hashes.json', 'implementation_hashes.json']
        dump(OUT / 'freeze_receipt.json', dict(utc=datetime.now(timezone.utc).isoformat(),
            hashes={n: sha(OUT / n) for n in names}, outcomes_not_computed=True))
    print('Frozen protocol:10 train/10 validation/80 test, same4876 passages; all100 BM25 rankings replay exactly.', flush=True)


def rrf(routes, k, pids):
    scores = {}
    indices = {pid: i for i, pid in enumerate(pids)}
    for ids in routes:
        for rank, pid in enumerate(ids[:k], 1):
            scores[pid] = scores.get(pid, 0.) + 1 / (60 + rank)
    return sorted(scores, key=lambda pid: (-scores[pid], indices[pid]))[:k]


def multiquery(proxy):
    from methods import build_multiquery
    check()
    qa, pages = data()
    bm = BM25(pages)
    url = proxy.start()
    local = threading.local()
    path = OUT / 'multiquery.json'
    saved = read(path) if path.exists() else {}
    proxy.set_stage('multiquery_generation')
    def work(q):
        if not hasattr(local, 'runner'):
            local.runner = build_multiquery(url, bm.search, depth=DEPTH)
        start = time.perf_counter()
        result = local.runner.invoke(q['question'])
        result['elapsed_seconds'] = time.perf_counter() - start
        result['question'] = q['question']
        return result
    def save(q, result):
        saved[q['id']] = result
        dump(path, saved)
        print(f'MultiQuery checkpoint {len(saved)}/100', flush=True)
    windowed([q for q in qa if q['id'] not in saved], work, save)


def windowed(items, work, save, concurrency=4):
    """Bound concurrency and drain/save in-flight successes before halting."""
    pending = iter(items)
    failures = []
    with cf.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {}
        def submit():
            try:
                item = next(pending)
            except StopIteration:
                return
            futures[executor.submit(work, item)] = item
        for _ in range(concurrency):
            submit()
        while futures:
            done, _ = cf.wait(futures, return_when=cf.FIRST_COMPLETED)
            for future in done:
                item = futures.pop(future)
                try:
                    result = future.result()
                    save(item, result)
                except Exception as error:
                    failures.append(error)
            if not failures:
                for _ in done:
                    submit()
    if failures:
        raise failures[0]


def build_rankings():
    qa, pages = data()
    base = np.load(OUT / 'baseline_rankings.npz')
    pids = base['passage_ids'].tolist()
    saved = read(OUT / 'multiquery.json')
    assert set(saved) == {q['id'] for q in qa}
    result = {}
    for i, q in enumerate(qa):
        item = saved[q['id']]
        routes = item['per_query_rankings']
        # Adapter returns records with exact query and its ordered passage IDs.
        lists = [r['passage_ids'] for r in routes]
        assert item['queries'][-1] == q['question'] and len(lists) == len(item['queries'])
        result[q['id']] = {'bm25': {str(k): [pids[j] for j in base['bm25'][i, :k]] for k in KS},
                           'multiquery': {str(k): rrf(lists, k, pids) for k in KS}}
    dump(OUT / 'rankings.json', result)
    return result


def pack(ids, page_map):
    remaining = WORD_BUDGET
    pieces, delivery = [], []
    for pid in ids:
        text = (page_map[pid]['title'] + ' ' + page_map[pid]['text']).strip()
        words = text.split()
        amount = min(remaining, len(words))
        if amount == 0:
            break
        pieces.append(f'Passage {len(pieces)+1}:\n' + ' '.join(words[:amount]))
        delivery.append(dict(passage_id=pid, full=amount == len(words), words=amount, original_words=len(words)))
        remaining -= amount
        if amount < len(words) or remaining == 0:
            break
    return dict(context='\n\n'.join(pieces), delivery=delivery, used_words=WORD_BUDGET-remaining)


def normalize_answer(text):
    text = str(text).lower().translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    return ' '.join(text.split())


def score_answer(prediction, reference):
    pred, ref = normalize_answer(prediction).split(), normalize_answer(reference).split()
    common = sum((Counter(pred) & Counter(ref)).values())
    f1 = 2 * common / (len(pred) + len(ref)) if pred or ref else 1.
    return dict(em=float(pred == ref), token_f1=f1)


def dev_rows():
    qa, pages = data()
    mapping = {q['id']: q for q in qa}
    page_map = {p['id']: p for p in pages}
    ranks = read(OUT / 'rankings.json')
    split = read(OUT / 'selection.json')
    def rows(ids):
        return [dict(id=qid, question=mapping[qid]['question'], answer=mapping[qid]['answer'],
                     context=pack(ranks[qid]['bm25'][str(READER_K)], page_map)['context']) for qid in ids]
    return rows(split['train_ids']), rows(split['val_ids'])


def optimize(proxy):
    check()
    from methods import configure_dspy, optimize as compile_program
    configure_dspy(proxy.start())
    proxy.set_stage('dspy_optimizer_setup')
    train, val = dev_rows()
    def metric(example, prediction, trace=None):
        return score_answer(prediction.answer, example.answer)['token_f1']
    compile_program(train, val, metric, OUT / 'optimizer')
    print('DSPy optimizer checkpoint saved; test references never used in optimization.', flush=True)


def readers(proxy):
    check()
    from methods import configure_dspy, unoptimized_program, load_optimized, predict
    from openai import OpenAI
    url = proxy.start()
    lm = configure_dspy(url)
    programs = dict(dspy_unoptimized=unoptimized_program(), dspy_optimized=load_optimized(OUT / 'optimizer'))
    qa, pages = data()
    qmap, pmap = {q['id']: q for q in qa}, {p['id']: p for p in pages}
    ids = read(OUT / 'selection.json')['test_ids']
    ranks = read(OUT / 'rankings.json')
    client = OpenAI(api_key='local-budget-proxy', base_url=url, max_retries=0, timeout=150)
    outpath = OUT / 'reader_outputs.json'
    saved = read(outpath) if outpath.exists() else {}
    for method in ['bm25_reader', 'multiquery_reader', 'dspy_unoptimized', 'dspy_optimized']:
        proxy.set_stage(method)
        def work(qid):
            q = qmap[qid]
            route = 'multiquery' if method == 'multiquery_reader' else 'bm25'
            delivery = pack(ranks[qid][route][str(READER_K)], pmap)
            start = time.perf_counter()
            if method in programs:
                import dspy
                with dspy.context(lm=lm):
                    output = predict(programs[method], delivery['context'], q['question'])
            else:
                response = client.chat.completions.create(model=MODEL, temperature=0, max_tokens=512,
                    messages=[dict(role='system', content=READER_SYSTEM),
                              dict(role='user', content=READER_TEMPLATE.format(context=delivery['context'], question=q['question']))])
                output = dict(answer=response.choices[0].message.content or '', reasoning='',
                              finish_reason=response.choices[0].finish_reason)
            assert output['answer'].strip(), f'Empty answer retained in API ledger: {method}::{qid}'
            return dict(method=method, query_id=qid, question=q['question'],
                **delivery, **output, elapsed_seconds=time.perf_counter()-start)
        def save(qid, result):
            saved[method+'::'+qid] = result
            dump(outpath, saved)
            print(f'Reader checkpoint {method}: {sum(v["method"]==method for v in saved.values())}/80', flush=True)
        windowed([qid for qid in ids if method+'::'+qid not in saved], work, save)


def retrieval_metric(ids, gold):
    rank = ids.index(gold) + 1 if gold in ids else None
    return dict(hit=float(rank is not None), recall=float(rank is not None),
                mrr=1 / rank if rank else 0., ndcg=1 / math.log2(rank+1) if rank else 0.)


def paired_ci(differences):
    arr = np.asarray(differences, dtype=float)
    rng = np.random.default_rng(42)
    draws = arr[rng.integers(0, len(arr), size=(20000, len(arr)))].mean(axis=1)
    return [float(v) for v in np.quantile(draws, [.025, .975])]


def evaluate():
    from scipy.stats import binomtest
    check()
    qa, _ = data()
    ranks = read(OUT / 'rankings.json')
    split = read(OUT / 'selection.json')
    rows = []
    for q in qa:
        for method in ['bm25', 'multiquery']:
            for k in KS:
                rows.append(dict(query_id=q['id'], cohort='test80' if q['id'] in split['test_ids'] else 'dev20',
                    method=method, k=k, **retrieval_metric(ranks[q['id']][method][str(k)], q['relevant_passage_id'])))
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / 'retrieval_per_query.csv', index=False)
    summary = pd.concat([frame, frame.assign(cohort='full100')]).groupby(['cohort','method','k'], sort=False).agg(
        n=('query_id','size'), **{m:(m,'mean') for m in ['hit','recall','mrr','ndcg']}).reset_index()
    summary.to_csv(OUT / 'retrieval_summary.csv', index=False)
    t = frame[(frame.cohort=='test80') & (frame.k==5)].pivot(index='query_id', columns='method', values='hit').loc[split['test_ids']]
    d = (t.multiquery-t.bm25).to_numpy()
    wins, losses = int((d>0).sum()), int((d<0).sum())
    dump(OUT / 'primary_retrieval_test.json', dict(n=80, metric='Hit@5', candidate='MultiQueryRetriever+BM25',
        control='BM25', mean_delta=float(d.mean()), wins=wins, losses=losses, ties=int((d==0).sum()),
        p_exact_mcnemar=float(binomtest(wins,wins+losses,.5).pvalue) if wins+losses else 1.,
        paired_bootstrap95=paired_ci(d), exploratory=True, global_historical_multiplicity_adjustment=False))
    qmap = {q['id']: q for q in qa}
    outputs = read(OUT / 'reader_outputs.json')
    assert len(outputs)==320 and all(sum(x['method']==m for x in outputs.values())==80 for m in
        ['bm25_reader','multiquery_reader','dspy_unoptimized','dspy_optimized'])
    answer_rows = []
    for output in outputs.values():
        q = qmap[output['query_id']]
        delivery = output['delivery']
        gold = q['relevant_passage_id']
        answer_rows.append(dict(query_id=q['id'], method=output['method'],
            **score_answer(output['answer'],q['answer']), empty_answer=not bool(output['answer'].strip()),
            used_words=output['used_words'], full_passages=sum(p['full'] for p in delivery),
            partial_passage=sum(not p['full'] for p in delivery),
            gold_full=any(p['passage_id']==gold and p['full'] for p in delivery),
            gold_touched=any(p['passage_id']==gold for p in delivery), elapsed_seconds=output['elapsed_seconds']))
    answers=pd.DataFrame(answer_rows)
    answers.to_csv(OUT/'reader_per_query.csv', index=False)
    answer_summary=answers.groupby('method',sort=False).agg(n=('query_id','size'),
        **{m:(m,'mean') for m in ['em','token_f1','used_words','full_passages','partial_passage','gold_full','gold_touched']}).reset_index()
    answer_summary.to_csv(OUT/'reader_summary.csv',index=False)
    pivot=answers.pivot(index='query_id',columns='method',values='token_f1').loc[split['test_ids']]
    comparisons=[('multiquery_reader','bm25_reader'),('dspy_optimized','bm25_reader'),('dspy_optimized','dspy_unoptimized')]
    matrix=np.column_stack([(pivot[a]-pivot[b]).to_numpy() for a,b in comparisons])
    observed=abs(matrix.sum(axis=0));counts=np.zeros(3,dtype=int);rng=np.random.default_rng(42)
    for _ in range(100):
        signs=rng.integers(0,2,(1000,80),dtype=np.int8)*2-1
        counts+=(abs(signs@matrix)>=observed-1e-12).sum(axis=0)
    raw=(counts+1)/100001;order=np.argsort(raw,kind='stable');adjusted=np.empty(3)
    adjusted[order]=np.minimum(1,np.maximum.accumulate((3-np.arange(3))*raw[order]))
    tests=[]
    for j,(a,b) in enumerate(comparisons):
        delta=matrix[:,j]
        tests.append(dict(candidate=a,control=b,n=80,metric='reference_token_f1_lexical_proxy',mean_delta=float(delta.mean()),
            wins=int((delta>1e-12).sum()),losses=int((delta< -1e-12).sum()),ties=int((abs(delta)<=1e-12).sum()),
            p_raw=float(raw[j]),p_holm3=float(adjusted[j]),paired_bootstrap95=json.dumps(paired_ci(delta))))
    pd.DataFrame(tests).to_csv(OUT/'reader_paired_tests.csv',index=False)
    print(summary[(summary.cohort=='test80') & (summary.k.isin([5,8,20]))].to_string(index=False))
    print(answer_summary.to_string(index=False))


def run_paid(stage):
    from api import BudgetProxy
    proxy=BudgetProxy(OUT/'api')
    try:
        proxy.access_check()
        if stage=='multiquery':
            multiquery(proxy)
            build_rankings()
        elif stage=='optimize':
            optimize(proxy)
        elif stage=='readers':
            readers(proxy)
    finally:
        dump(OUT/'api_summary.json',proxy.summary())
        proxy.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','multiquery','optimize','readers','evaluate'])
    args=parser.parse_args()
    if args.stage in ['multiquery','optimize','readers']:
        run_paid(args.stage)
    else:
        globals()[args.stage]()
