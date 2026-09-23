"""Measure synchronous, single-query retrieval calls using a shared protocol."""

import random
import time
from collections.abc import Mapping, Set
from statistics import mean, median


def benchmark_retrievers(retrievers, queries, k=5, warmup=3, repeats=3, seed=42,
                         synchronize=None):
    """Return raw timings, summaries, and first-repeat rankings by method.

    Each callable accepts (question_text, k) and returns ordered original passage IDs.
    Build models/indexes before calling. Include encoding, search, refinement/fusion,
    and evidence lookup inside the callable. Optional synchronization hooks fence GPU
    work before/after each call. Exceptions abort the run, never silently drop queries.
    Warm-ups precede randomized interleaving of method/query/repeat combinations.
    """
    if not retrievers or not queries:
        raise ValueError("Supply retrievers and queries.")
    for name, value, lower in (("k", k, 1), ("warmup", warmup, 0), ("repeats", repeats, 1)):
        if type(value) is not int or value < lower:
            raise ValueError(f"Invalid {name}.")
    if len({q['id'] for q in queries}) != len(queries):
        raise ValueError("Duplicate query IDs.")
    if any(not isinstance(q['id'], str) or not q['id'] or
           not isinstance(q['question'], str) or not q['question'].strip() for q in queries):
        raise ValueError("Queries require nonempty string IDs and question text.")
    hooks = synchronize or {}
    if set(hooks) - set(retrievers):
        raise ValueError("Synchronization hook has no matching retriever.")
    predictions = {name: {} for name in retrievers}
    raw = []

    def invoke(name, query):
        hook = hooks.get(name, lambda: None)
        hook()
        start = time.perf_counter()
        result = retrievers[name](query['question'], k)
        if isinstance(result, (str, bytes, Mapping, Set)):
            raise ValueError("Return an ordered sequence of passage IDs, not a string, mapping, or set.")
        ranking = list(result)
        hook()
        elapsed = (time.perf_counter() - start) * 1000
        if any(not isinstance(pid, str) for pid in ranking):
            raise ValueError("Return string passage IDs.")
        if len(ranking) > k or len(ranking) != len(set(ranking)):
            raise ValueError("Return at most k distinct passage IDs.")
        return ranking, elapsed

    for name in retrievers:
        for i in range(warmup):
            invoke(name, queries[i % len(queries)])
    tasks = [(name, i, repeat) for repeat in range(repeats)
             for i in range(len(queries)) for name in retrievers]
    random.Random(seed).shuffle(tasks)
    for name, i, repeat in tasks:
        query = queries[i]
        ranking, elapsed = invoke(name, query)
        if repeat == 0:
            predictions[name][query['id']] = ranking
        raw.append({'method': name, 'query_id': query['id'], 'repeat': repeat,
                    'k': k, 'latency_ms': elapsed, 'n_returned': len(ranking)})
    summaries = []
    for name in retrievers:
        values = sorted(row['latency_ms'] for row in raw if row['method'] == name)
        position = .95 * (len(values) - 1)
        lo = int(position)
        hi = min(lo + 1, len(values) - 1)
        summaries.append({'method': name, 'k': k, 'n_queries': len(queries),
                          'measurements': len(values), 'mean_ms': mean(values),
                          'median_ms': median(values),
                          'p95_ms': values[lo] + (values[hi] - values[lo]) * (position - lo),
                          'warmup_calls_per_method': warmup, 'repeats': repeats, 'seed': seed,
                          'protocol': 'single_query_interleaved',
                          'synchronization_hook': name in hooks})
    return raw, summaries, predictions
