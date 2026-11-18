"""Matched FinDER single-query, LangChain MultiQueryRetriever, and DSPy trials.

Use the Week 3 pooled-reference corpus and fixed dev/test split. All methods
search the same tuned BM25 index, retrieve 20 passages per formulation and use
RRF to rank the union. DSPy bootstraps two query rewrites on dev questions only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import statistics
import time
from typing import Any
import importlib.metadata

from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_core.documents import Document
from langchain_core.language_models.llms import LLM
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

from week3_bm25.finder_bm25.bm25 import BM25Retriever
from week3_bm25.finder_bm25.data import load_bundle, select_queries, digest, write_json, write_jsonl
from week3_bm25.finder_bm25.evaluation import evaluate_rag

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'week3_bm25/data/team_shared'
SELECTED = ROOT / 'week3_bm25/results/b_bm25_common/selected_config.json'
QWEN_MODEL = 'Qwen/Qwen3-0.6B'
QWEN_REVISION = 'c1899de289a04d12100db370d81485cdf75e47ca'
GPT_MODEL = 'gpt-4o-mini'
PER_QUERY_DEPTH = 20
EVAL_KS = (5, 10)
PROMPT = PromptTemplate.from_template(
    'Write exactly two alternative search queries for finding financial evidence. '
    'Preserve all company names, dates, quantities, and the requested relationship. '
    'Use different useful wording in each query. Output only two plain lines, one query per line.\n'
    'Question: {question}\nQueries:')


def load_key():
    key = os.environ.get('OPENAI_API_KEY')
    if key:
        return key
    path = ROOT / 'week10_followup/.env'
    if path.exists():
        for line in path.read_text().splitlines():
            if line.startswith('OPENAI_API_KEY='):
                return line.partition('=')[2].strip().strip('"\'')
    raise ValueError('OPENAI_API_KEY missing; set it in the environment or week10_followup/.env')


def normalize_queries(values, original):
    result, seen = [], {original.strip().casefold()}
    for value in values:
        value = re.sub(r'^\s*(?:[-*]|\d+[.)])\s*', '', str(value)).strip().strip('"')
        if value and value.casefold() not in seen:
            seen.add(value.casefold())
            result.append(value)
    return result[:2]


class TwoLineParser(BaseOutputParser[list[str]]):
    def parse(self, text: str) -> list[str]:
        lines = [re.sub(r'^\s*(?:[-*]|\d+[.)])\s*', '', line).strip()
                 for line in text.splitlines() if line.strip()]
        return lines[:2]


class LocalQwen:
    def __init__(self, device='mps'):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        self.torch = torch
        self.model = AutoModelForCausalLM.from_pretrained(
            QWEN_MODEL, revision=QWEN_REVISION, cache_dir=str(ROOT / '.cache/models'),
            local_files_only=True, torch_dtype=torch.float32,
            attn_implementation='sdpa').to(device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            QWEN_MODEL, revision=QWEN_REVISION, cache_dir=str(ROOT / '.cache/models'),
            local_files_only=True)
        self.device = device
        self.usage = []

    def generate(self, messages, max_tokens=160):
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        encoded = self.tokenizer(text, return_tensors='pt').to(self.device)
        self.torch.manual_seed(42)
        with self.torch.inference_mode():
            tokens = self.model.generate(**encoded, max_new_tokens=max_tokens,
                                         do_sample=True, temperature=0.7, top_p=0.8, top_k=20,
                                         pad_token_id=self.tokenizer.eos_token_id)
        generated = tokens[0, encoded.input_ids.shape[1]:]
        answer = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        record = {'input_tokens': int(encoded.input_ids.shape[1]),
                  'output_tokens': int(len(generated))}
        self.usage.append(record)
        return answer, record


class QwenLangChainLLM(LLM):
    backend: Any

    @property
    def _llm_type(self) -> str:
        return 'local_qwen3_0_6b'

    def _call(self, prompt: str, stop=None, run_manager=None, **kwargs) -> str:
        result, _ = self.backend.generate([{'role': 'user', 'content': prompt}])
        return result


class OpenAILangChainLLM(LLM):
    backend: Any
    model_name: str = GPT_MODEL
    usage: list[dict] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return 'openai_gpt_4o_mini'

    def _call(self, prompt: str, stop=None, run_manager=None, **kwargs) -> str:
        response = self.backend.chat.completions.create(
            model=self.model_name, messages=[{'role': 'user', 'content': prompt}],
            temperature=0.7, max_tokens=160)
        self.usage.append({'input_tokens': response.usage.prompt_tokens,
                           'output_tokens': response.usage.completion_tokens})
        return response.choices[0].message.content or ''


class LoggedBM25Retriever(BaseRetriever):
    index: Any
    texts: dict[str, str]
    depth: int = PER_QUERY_DEPTH
    history: list[dict] = Field(default_factory=list)

    def _get_relevant_documents(self, query: str, *, run_manager) -> list[Document]:
        hits = self.index.search(query, self.depth)
        self.history.append({'query': query, 'doc_ids': [h['doc_id'] for h in hits]})
        return [Document(page_content=self.texts[h['doc_id']], metadata={'doc_id': h['doc_id']})
                for h in hits]


def rrf(searches, candidates=None):
    """Equal-weight RRF with stable ID ties; one query preserves its BM25 order."""
    scores = {}
    for search in searches:
        for rank, cid in enumerate(search['doc_ids'], 1):
            if candidates is None or cid in candidates:
                scores[cid] = scores.get(cid, 0.0) + 1 / (60 + rank)
    ids = sorted(scores, key=lambda cid: (-scores[cid], cid))
    return ids, [scores[cid] for cid in ids]


def make_dspy_lm(generator, backend):
    import dspy
    if generator == 'gpt':
        import certifi
        os.environ.setdefault('SSL_CERT_FILE', certifi.where())
        return dspy.LM('openai/' + GPT_MODEL, api_key=load_key(),
                       temperature=0.7, max_tokens=160, cache=False)
    from dspy.lm15 import Message, Response, Usage

    class QwenEngine:
        def complete(self, request):
            messages = []
            if request.system:
                messages.append({'role': 'system', 'content': str(request.system)})
            messages.extend({'role': message.role, 'content': message.text}
                            for message in request.messages)
            answer, usage = backend.generate(messages,
                                             max_tokens=request.config.max_tokens or 160)
            return Response(id=None, model=request.model, message=Message.assistant(answer),
                            finish_reason='stop', usage=Usage(
                                input_tokens=usage['input_tokens'],
                                output_tokens=usage['output_tokens'],
                                total_tokens=sum(usage.values())))

    return dspy.BaseLM('local/' + QWEN_MODEL, engine=QwenEngine(),
                       temperature=0.7, max_tokens=160, cache=False)


def dspy_program(index, bundle, dev, lm):
    import dspy
    from week3_bm25.finder_bm25.metrics import recall_at_k

    class Rewrite(dspy.Signature):
        """Write two distinct financial evidence search queries. Preserve entities, dates, quantities and the requested relationship."""
        question: str = dspy.InputField()
        query_1: str = dspy.OutputField(desc='First standalone search query')
        query_2: str = dspy.OutputField(desc='Second standalone search query')

    program = dspy.Predict(Rewrite)

    def metric(example, prediction, trace=None):
        alternatives = normalize_queries([prediction.query_1, prediction.query_2], example.question)
        searches = [{'doc_ids': [h['doc_id'] for h in index.search(q, PER_QUERY_DEPTH)]}
                    for q in [example.question, *alternatives]]
        ids, _ = rrf(searches)
        return recall_at_k(ids[:5], bundle['qrels'][example.query_id], 5)

    trainset = [dspy.Example(question=q['text'], query_id=q['query_id']).with_inputs('question')
                for q in dev]
    with dspy.context(lm=lm, adapter=dspy.ChatAdapter(use_native_function_calling=False)):
        optimizer = dspy.BootstrapFewShot(metric=metric, max_bootstrapped_demos=2,
                                         max_labeled_demos=0, max_rounds=1)
        compiled = optimizer.compile(program, trainset=trainset)
    return compiled, trainset


def usage_totals(records):
    return {'model_calls': sum(r.get('model_calls', 1) for r in records),
            'input_tokens': sum(r.get('input_tokens') or 0 for r in records),
            'output_tokens': sum(r.get('output_tokens') or 0 for r in records)}


def dspy_usage(lm, start):
    records = []
    for row in lm.history[start:]:
        usage = row.get('usage') if isinstance(row, dict) else None
        if usage:
            records.append({'input_tokens': usage.get('input_tokens', usage.get('prompt_tokens', 0)),
                            'output_tokens': usage.get('output_tokens', usage.get('completion_tokens', 0))})
    return usage_totals(records)


def evaluate_rows(bundle, rows, method, output, config):
    for k in EVAL_KS:
        run = {'schema_version': 1, 'method': method,
               'bundle_fingerprint': bundle['manifest']['fingerprint'], 'partition': 'test',
               'query_ids': [r['query_id'] for r in rows], 'top_k': k,
               'config': config, 'build_latency_ms': None,
               'latency_scope': 'query_generation_plus_all_retrievals_and_fusion',
               'timing_protocol': 'sequential_one_measurement_per_query',
               'environment': {'python': platform.python_version()},
               'results': [{'query_id': r['query_id'], 'doc_ids': r['doc_ids'][:k],
                            'scores': r['scores'][:k], 'latency_ms': r['latency_ms']}
                           for r in rows]}
        result = evaluate_rag(bundle, run)
        write_json(output / f'run_k{k}.json', run)
        write_json(output / f'evaluation_k{k}.json', result)


def run(args):
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Output must be new or empty')
    bundle = load_bundle(BUNDLE)
    selected = json.loads(SELECTED.read_text())
    if selected['bundle_fingerprint'] != bundle['manifest']['fingerprint']:
        raise ValueError('FinDER bundle differs from tuned BM25 configuration')
    dev = select_queries(bundle, 'dev', args.dev_limit)
    test = select_queries(bundle, 'test', args.test_limit)
    index = BM25Retriever(bundle['corpus'], k1=selected['k1'], b=selected['b'])
    texts = {r['doc_id']: r['text'] for r in bundle['corpus']}
    args.output.mkdir(parents=True)

    baseline = []
    for q in test:
        started = time.perf_counter()
        hits = index.search(q['text'], PER_QUERY_DEPTH)
        baseline.append({'query_id': q['query_id'], 'doc_ids': [h['doc_id'] for h in hits],
                         'scores': [h['score'] for h in hits],
                         'latency_ms': 1000 * (time.perf_counter() - started)})
    baseline_dir = args.output / 'single_query'
    baseline_dir.mkdir()
    evaluate_rows(bundle, baseline, 'single_query_bm25', baseline_dir,
                  {'k1': selected['k1'], 'b': selected['b'], 'depth': PER_QUERY_DEPTH})

    if args.generator == 'qwen':
        backend = LocalQwen(args.device)
        langchain_lm = QwenLangChainLLM(backend=backend)
    else:
        from openai import OpenAI
        langchain_lm = OpenAILangChainLLM(backend=OpenAI(api_key=load_key(), max_retries=2))
        backend = None
    wrapper = LoggedBM25Retriever(index=index, texts=texts)
    chain = PROMPT | langchain_lm | TwoLineParser()
    multi = MultiQueryRetriever(retriever=wrapper, llm_chain=chain,
                                include_original=True, verbose=False)
    multi_rows, traces = [], []
    for pos, q in enumerate(test, 1):
        wrapper.history.clear()
        usage_before = len(backend.usage if backend else langchain_lm.usage)
        started = time.perf_counter()
        documents = multi.invoke(q['text'])
        elapsed = 1000 * (time.perf_counter() - started)
        candidate_ids = {d.metadata['doc_id'] for d in documents}
        ids, scores = rrf(wrapper.history, candidate_ids)
        usages = (backend.usage if backend else langchain_lm.usage)[usage_before:]
        multi_rows.append({'query_id': q['query_id'], 'doc_ids': ids,
                           'scores': scores, 'latency_ms': elapsed})
        traces.append({'query_id': q['query_id'], 'generated_queries': [h['query'] for h in wrapper.history[:-1]],
                       'searches': wrapper.history.copy(), 'candidate_count': len(candidate_ids),
                       'latency_ms': elapsed, **usage_totals(usages)})
        if pos % 20 == 0:
            print(f'MultiQuery {args.generator}: {pos}/{len(test)}', flush=True)
    multi_dir = args.output / 'multiquery'
    multi_dir.mkdir()
    evaluate_rows(bundle, multi_rows, f'multiquery_{args.generator}', multi_dir,
                  {'alternatives': 2, 'original_included': True, 'fusion': 'RRF k=60',
                   'depth_per_query': PER_QUERY_DEPTH, 'prompt': PROMPT.template})
    write_jsonl(multi_dir / 'traces.jsonl', traces)

    lm = make_dspy_lm(args.generator, backend)
    started = time.perf_counter()
    start_history = len(lm.history)
    program, _ = dspy_program(index, bundle, dev, lm)
    optimization_seconds = time.perf_counter() - started
    training_usage = dspy_usage(lm, start_history)
    dspy_rows, dspy_traces = [], []
    import dspy
    for pos, q in enumerate(test, 1):
        started = time.perf_counter()
        before = len(lm.history)
        with dspy.context(lm=lm, adapter=dspy.ChatAdapter(use_native_function_calling=False)):
            prediction = program(question=q['text'])
        alternatives = normalize_queries([prediction.query_1, prediction.query_2], q['text'])
        searches = [{'query': query, 'doc_ids': [h['doc_id'] for h in index.search(query, PER_QUERY_DEPTH)]}
                    for query in [q['text'], *alternatives]]
        ids, scores = rrf(searches)
        elapsed = 1000 * (time.perf_counter() - started)
        dspy_rows.append({'query_id': q['query_id'], 'doc_ids': ids,
                          'scores': scores, 'latency_ms': elapsed})
        dspy_traces.append({'query_id': q['query_id'], 'generated_queries': alternatives,
                            'searches': searches, 'latency_ms': elapsed,
                            **dspy_usage(lm, before)})
        if pos % 20 == 0:
            print(f'DSPy {args.generator}: {pos}/{len(test)}', flush=True)
    dspy_dir = args.output / 'dspy'
    dspy_dir.mkdir()
    evaluate_rows(bundle, dspy_rows, f'dspy_{args.generator}', dspy_dir,
                  {'alternatives': 2, 'original_included': True, 'fusion': 'RRF k=60',
                   'depth_per_query': PER_QUERY_DEPTH, 'optimizer': 'BootstrapFewShot',
                   'max_bootstrapped_demos': 2, 'dev_examples': len(dev)})
    write_jsonl(dspy_dir / 'traces.jsonl', dspy_traces)
    (dspy_dir / 'optimized_prompt.json').write_text(json.dumps(program.dump_state(), indent=2) + '\n')

    manifest = {'dataset': 'FinDER', 'bundle_fingerprint': bundle['manifest']['fingerprint'],
                'corpus_scope': bundle['manifest']['corpus_scope'], 'corpus_passages': len(bundle['corpus']),
                'generator': QWEN_MODEL if args.generator == 'qwen' else GPT_MODEL,
                'generator_revision': QWEN_REVISION if args.generator == 'qwen' else GPT_MODEL,
                'retriever': 'tuned BM25', 'k1': selected['k1'], 'b': selected['b'],
                'dev_query_ids_sha256': digest([q['query_id'] for q in dev]),
                'test_query_ids_sha256': digest([q['query_id'] for q in test]),
                'dev_count': len(dev), 'test_count': len(test), 'depth_per_query': PER_QUERY_DEPTH,
                'fusion': 'equal-weight RRF with denominator 60+rank, stable ID ties',
                'dspy_optimization_seconds': optimization_seconds,
                'dspy_optimization_usage': training_usage,
                'multiquery_usage': usage_totals(traces),
                'dspy_test_usage': usage_totals(dspy_traces),
                'packages': {p: importlib.metadata.version(p) for p in
                             ('langchain-classic', 'langchain-core', 'dspy', 'transformers', 'torch')},
                'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    write_json(args.output / 'manifest.json', manifest)
    print(f'Completed {args.generator}: {len(test)} held-out FinDER questions -> {args.output}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generator', choices=('qwen', 'gpt'), required=True)
    parser.add_argument('--device', choices=('mps', 'cpu'), default='mps')
    parser.add_argument('--test-limit', type=int, default=200)
    parser.add_argument('--dev-limit', type=int, default=16)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.test_limit < 1 or args.dev_limit < 1:
        parser.error('Query limits must be positive')
    run(args)


if __name__ == '__main__':
    main()
