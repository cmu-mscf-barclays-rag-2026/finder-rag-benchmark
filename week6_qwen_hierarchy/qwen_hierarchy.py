"""Controlled Qwen window sweep over the frozen Week 5 hierarchy.

Run from the repository root: python -m week6_qwen_hierarchy.qwen_hierarchy --help
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from functools import lru_cache
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import statistics
import sys
import time

from week5_hierarchy.hierarchy_experiment import ROOT, dense_search, load_data
from week4_legal_rag.run import fuse
from finder_bm25.bm25 import BM25Retriever, TOKENIZER_VERSION
from hierarchy.handoff import assemble_context
from hierarchy.core import SEPARATOR
from hierarchy.io import digest, read_jsonl, write_csv, write_json, write_jsonl

HERE = Path(__file__).resolve().parent
REFERENCE = ROOT / 'week5_hierarchy/results/comparison'


class CachedTokenizer:
    """Memoize full, untruncated measurement encodings; never an encoder tokenizer."""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @lru_cache(maxsize=16000)
    def encode(self, text, add_special_tokens=False):
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)


def evaluation_tokenizer(config, cache, offline):
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer
    spec = config['evaluation_tokenizer']
    path = hf_hub_download(spec['model'], 'tokenizer.json', revision=spec['revision'],
                            cache_dir=str(cache), local_files_only=offline)
    tokenizer = Tokenizer.from_file(path)
    tokenizer.no_truncation()
    tokenizer.no_padding()
    return CachedTokenizer(tokenizer), digest(path)


def sibling_candidates(ranking, children, parents, k):
    """Seed, previous sibling, next sibling, in seed rank order; no backfill seeds."""
    ordered = []
    for seed in ranking[:k]:
        siblings = [s['child_id'] for s in parents[children[seed]['parent_id']]['child_spans']]
        i = siblings.index(seed)
        ordered.append(seed)
        if i:
            ordered.append(siblings[i - 1])
        if i + 1 < len(siblings):
            ordered.append(siblings[i + 1])
    return list(dict.fromkeys(ordered))


def selected_sibling_context(ranking, children, parents, tokenizer, k, budget):
    candidates = sibling_candidates(ranking, children, parents, k)
    kept, text = [], ''
    for pid in candidates:
        proposed = text + (SEPARATOR if text else '') + children[pid]['text']
        if budget is None or len(tokenizer.encode(proposed).ids) <= budget:
            kept.append(pid)
            text = proposed
    context = assemble_context(kept, 'child', children, parents, tokenizer,
                               max(1, len(kept)), None)
    context['seed_ids'] = ranking[:k]
    context['selected_child_ids'] = candidates
    context['unbudgeted_content_tokens'] = len(tokenizer.encode(
        SEPARATOR.join(children[pid]['text'] for pid in candidates)).ids)
    context['truncated'] = len(kept) < len(candidates)
    return context


def evaluate(queries, rankings, children, parents, tokenizer, config, metadata):
    """Evidence inclusion in exact returned text; the same ruler for both models."""
    if set(rankings) != {q['id'] for q in queries}:
        raise ValueError('Rankings must exactly match evaluation query IDs.')
    details, summaries = [], []
    for mode in config['modes']:
        for budget in config['context_budgets']:
            for k in config['ks']:
                group = []
                for q in queries:
                    ranking = rankings[q['id']]['parent' if mode == 'parent' else 'child']
                    if len(ranking) != len(set(ranking)):
                        raise ValueError('Duplicate ranking IDs.')
                    if mode == 'child_to_selected_siblings':
                        context = selected_sibling_context(ranking, children, parents, tokenizer, k, budget)
                    else:
                        context = assemble_context(ranking, mode, children, parents, tokenizer, k, budget)
                    gold = set(q['gold_child_ids'])
                    full = gold & set(context['fully_included_child_ids'])
                    seen = total = 0
                    for pid in gold:
                        offsets = tokenizer.encode(children[pid]['text']).offsets
                        visible = context['visible_child_characters'].get(pid, 0)
                        seen += sum(end <= visible for _, end in offsets)
                        total += len(offsets)
                    tokens = context['returned_content_tokens']
                    coverage = seen / total if total else 0.0
                    row = {**metadata, 'split': q['split'], 'query_id': q['id'],
                           'mode': mode, 'context_budget': budget, 'k': k,
                           'hit': int(bool(full)), 'gold_text_coverage': coverage,
                           'context_tokens': tokens,
                           'coverage_efficiency': coverage / tokens if tokens else 0.0,
                           'seed_ids': json.dumps(context['seed_ids']),
                           'returned_unit_ids': json.dumps(context['unit_ids']),
                           'fully_included_child_ids': json.dumps(context['fully_included_child_ids'])}
                    if budget is not None and tokens > budget:
                        raise AssertionError('Context exceeds the declared budget.')
                    group.append(row)
                import numpy as np
                summaries.append({**metadata, 'split': queries[0]['split'], 'mode': mode,
                    'context_budget': budget, 'k': k, 'n_queries': len(group),
                    'hit_count': sum(r['hit'] for r in group),
                    'hit_rate': statistics.mean(r['hit'] for r in group),
                    'gold_text_coverage': statistics.mean(r['gold_text_coverage'] for r in group),
                    'mean_context_tokens': statistics.mean(r['context_tokens'] for r in group),
                    'p95_context_tokens': float(np.percentile([r['context_tokens'] for r in group], 95)),
                    'mean_coverage_efficiency': statistics.mean(r['coverage_efficiency'] for r in group)})
                details.extend(group)
    return details, summaries


def load_reference(queries, children, parents):
    """Use only saved MiniLM/BM25 test rankings; verify split and ID integrity."""
    split = json.loads((REFERENCE / 'split.json').read_text())
    if split['test'] != [q['id'] for q in queries]:
        raise ValueError('Week 5 reference split mismatch.')
    result = {}
    for method, source in [('bm25', 'bm25'), ('dense', 'dense'), ('hybrid', 'hybrid_w0.75')]:
        rankings = {q['id']: {} for q in queries}
        for unit, lookup in [('child', children), ('parent', parents)]:
            rows = read_jsonl(REFERENCE / 'rankings' / f'{source}_{unit}.jsonl')
            if len(rows) != len(queries) or {r['query_id'] for r in rows} != set(rankings):
                raise ValueError('Reference rankings do not match the frozen test split.')
            for row in rows:
                ids = row['passage_ids']
                if len(set(ids)) != len(ids) or any(pid not in lookup for pid in ids):
                    raise ValueError('Invalid reference ranking.')
                rankings[row['query_id']][unit] = ids
        result[method] = rankings
    return result


class QwenEncoder:
    """Last-token pooling, L2 normalization, and resumable input-level caches."""
    def __init__(self, config, args):
        import numpy as np
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.np, self.torch = np, torch
        self.args, self.config = args, config
        self.tokenizer = AutoTokenizer.from_pretrained(config['model'], revision=config['revision'],
            cache_dir=str(args.model_cache), local_files_only=args.offline, padding_side='left')
        self.tokenizer.truncation_side = 'right'
        self.model = None  # Load weights only if vectors actually need computing.
        self.loader = AutoModel
        self.packages = {p: importlib.metadata.version(p) for p in
                         ('torch', 'transformers', 'numpy', 'tokenizers')}
        self.identity = {'model': config['model'], 'revision': config['revision'],
            'pooling': 'last_nonpadding_token', 'normalize': True, 'dtype': args.dtype,
            'device': args.device, 'attention': 'sdpa', 'packages': self.packages}
        self.cache = args.embedding_cache
        self.cache.mkdir(parents=True, exist_ok=True)
        self.audit = []

    def encode(self, texts, window, label, prompt=''):
        np, torch = self.np, self.torch
        inputs = [prompt + t for t in texts]
        # No instruction on documents, no chat template, no title/footnote additions.
        full = self.tokenizer(inputs, add_special_tokens=True, truncation=False)['input_ids']
        self.audit.append({'unit': label, 'embedding_window': window, 'units': len(texts),
            'max_input_tokens': max(map(len, full)),
            'mean_input_tokens': statistics.mean(map(len, full)),
            'truncated_units': sum(len(ids) > window for ids in full)})
        if label == 'query' and any(len(ids) > window for ids in full):
            raise ValueError('The fixed query window truncates questions; revise protocol before running.')
        clipped = [ids[:window] for ids in full]
        salt = json.dumps(self.identity, sort_keys=True)
        keys = [hashlib.sha256((salt + json.dumps(ids)).encode()).hexdigest() for ids in clipped]
        unique = dict(zip(keys, clipped))
        vectors = {}
        pending = []
        for key, ids in unique.items():
            path = self.cache / f'{key}.npy'
            if path.exists():
                vector = np.load(path, allow_pickle=False)
                if vector.shape != (1024,) or not np.isfinite(vector).all() or not np.isclose(np.linalg.norm(vector), 1, atol=1e-4):
                    raise ValueError(f'Invalid cached embedding: {path}')
                vectors[key] = vector
            else:
                pending.append((key, ids))
        print(f'{label} window={window}: {len(texts)} inputs; {len(pending)} uncached unique inputs', flush=True)
        if pending and self.model is None:
            self.model = self.loader.from_pretrained(self.config['model'], revision=self.config['revision'],
                cache_dir=str(self.args.model_cache), local_files_only=self.args.offline,
                torch_dtype=getattr(torch, self.args.dtype), attn_implementation='sdpa').to(self.args.device).eval()
        # Sort by length to limit padding; constrain total padded tokens per batch.
        pending.sort(key=lambda item: (len(item[1]), item[0]))
        i, started = 0, time.monotonic()
        while i < len(pending):
            end = i + 1
            while end < len(pending) and end - i < self.args.batch_size:
                if (end - i + 1) * len(pending[end][1]) > self.args.batch_tokens:
                    break
                end += 1
            batch = pending[i:end]
            encoded = self.tokenizer.pad({'input_ids': [ids for _, ids in batch]},
                                         padding=True, return_tensors='pt').to(self.args.device)
            with torch.inference_mode():
                output = self.model(**encoded).last_hidden_state[:, -1]
                output = torch.nn.functional.normalize(output.float(), p=2, dim=1).cpu().numpy()
            for (key, _), vector in zip(batch, output):
                if not np.isfinite(vector).all() or not np.isclose(np.linalg.norm(vector), 1, atol=1e-4):
                    raise ValueError('Encoder produced an invalid embedding.')
                temporary = self.cache / f'{key}.tmp.npy'
                np.save(temporary, vector)
                temporary.replace(self.cache / f'{key}.npy')
                vectors[key] = vector
            i = end
            print(f'  {label}: {i}/{len(pending)} newly encoded ({time.monotonic() - started:.0f}s)', flush=True)
        return np.stack([vectors[key] for key in keys])


def rankings_for(queries, units, query_vectors, indexes, sparse, config):
    result = {method: {q['id']: {} for q in queries} for method in ('dense', 'hybrid')}
    for unit, records in units.items():
        for i, q in enumerate(queries):
            dense = dense_search(query_vectors[i], indexes[unit], records)
            hybrid = fuse(sparse[unit][q['id']], dense, config['bm25_weight'], config['rrf_constant'])
            for method, hits in [('dense', dense), ('hybrid', hybrid)]:
                result[method][q['id']][unit] = [h['doc_id'] for h in hits]
    return result


def choose_window(summaries, config):
    selection = config['selection']
    candidates = [r for r in summaries if all(r[key] == value for key, value in selection.items())]
    if not candidates:
        raise ValueError('No development rows satisfy the registered selection rule.')
    return max(candidates, key=lambda r: (r['hit_rate'], r['gold_text_coverage'],
                                         r['mean_coverage_efficiency'], -r['embedding_window']))['embedding_window']


def exact_mcnemar_p(wins, losses):
    """Two-sided exact sign test over discordant paired binary outcomes."""
    import math
    discordant = wins + losses
    if not discordant:
        return 1.0
    tail = sum(math.comb(discordant, i) for i in range(min(wins, losses) + 1)) / 2 ** discordant
    return min(1.0, 2 * tail)


def paired_comparisons(details, selected=None):
    """Registered test-set contrasts; positive difference favors the left side."""
    test = [r for r in details if r['split'] == 'test' and r['k'] == 10]
    index = {(r['embedding'], r['embedding_window'], r['method'], r['mode'],
              r['context_budget'], r['query_id']): r for r in test}
    rows = []

    def add(family, left, right):
        left_rows = {key[-1]: row for key, row in index.items() if key[:-1] == left}
        right_rows = {key[-1]: row for key, row in index.items() if key[:-1] == right}
        if not left_rows or set(left_rows) != set(right_rows):
            return
        wins = sum(left_rows[q]['hit'] > right_rows[q]['hit'] for q in left_rows)
        losses = sum(left_rows[q]['hit'] < right_rows[q]['hit'] for q in left_rows)
        left_hits = sum(left_rows[q]['hit'] for q in left_rows)
        right_hits = sum(right_rows[q]['hit'] for q in right_rows)
        rows.append({'family': family,
            'left_embedding': left[0], 'left_window': left[1], 'left_method': left[2], 'left_mode': left[3],
            'right_embedding': right[0], 'right_window': right[1], 'right_method': right[2], 'right_mode': right[3],
            'context_budget': left[4], 'k': 10, 'n_queries': len(left_rows),
            'left_hit_count': left_hits, 'right_hit_count': right_hits,
            'hit_rate_difference': (left_hits - right_hits) / len(left_rows),
            'left_wins': wins, 'left_losses': losses, 'ties': len(left_rows) - wins - losses,
            'exact_mcnemar_p': exact_mcnemar_p(wins, losses)})

    embeddings = sorted({(r['embedding'], r['embedding_window']) for r in test})
    for embedding, window in embeddings:
        if embedding == 'none':
            methods = ('bm25',)
        else:
            methods = ('dense', 'hybrid')
        for method in methods:
            for budget in (None, 2048):
                child = (embedding, window, method, 'child', budget)
                for mode in ('parent', 'child_to_parent', 'child_to_selected_siblings'):
                    add('hierarchy_vs_child', (embedding, window, method, mode, budget), child)
    if selected is not None:
        for method in ('dense', 'hybrid'):
            for mode in ('child', 'parent', 'child_to_parent', 'child_to_selected_siblings'):
                for budget in (None, 2048):
                    add('qwen_vs_minilm', ('Qwen3-0.6B', selected, method, mode, budget),
                        ('MiniLM', 256, method, mode, budget))
        for mode in ('child', 'parent', 'child_to_parent', 'child_to_selected_siblings'):
            for budget in (None, 2048):
                add('qwen_dense_vs_hybrid', ('Qwen3-0.6B', selected, 'dense', mode, budget),
                    ('Qwen3-0.6B', selected, 'hybrid', mode, budget))
    return rows


def export(output, details, summaries, config, selected, reference_only):
    write_csv(output / 'per_query.csv', details)
    development = [r for r in summaries if r['split'] == 'dev']
    if development:
        write_csv(output / 'window_sweep.csv', development)
    else:
        with (output / 'window_sweep.csv').open('w', newline='') as stream:
            csv.DictWriter(stream, fieldnames=list(summaries[0])).writeheader()
    comparison = [r for r in summaries if r['split'] == 'test']
    write_csv(output / 'hierarchy_comparison.csv', comparison)
    paired = paired_comparisons(details, selected)
    write_csv(output / 'paired_comparisons.csv', paired)
    # Keep concrete wins AND losses against full-parent expansion (hybrid, budget 2048, k=10).
    lookup = {(r['embedding'], r['embedding_window'], r['query_id'], r['mode']): r
              for r in details if r['split'] == 'test' and r['method'] == 'hybrid'
              and r['context_budget'] == 2048 and r['k'] == 10}
    examples = []
    for key, row in lookup.items():
        if key[-1] != 'child_to_selected_siblings':
            continue
        parent = lookup.get((*key[:-1], 'child_to_parent'))
        if parent and row['hit'] != parent['hit']:
            examples.append({'embedding': key[0], 'embedding_window': key[1], 'query_id': key[2],
                'sibling_hit': row['hit'], 'full_parent_hit': parent['hit'],
                'sibling_coverage': row['gold_text_coverage'], 'full_parent_coverage': parent['gold_text_coverage'],
                'sibling_context_tokens': row['context_tokens'], 'full_parent_context_tokens': parent['context_tokens'],
                'seed_ids': row['seed_ids'], 'sibling_returned_unit_ids': row['returned_unit_ids']})
    if examples:
        write_csv(output / 'examples.csv', examples)
    else:
        (output / 'examples.csv').write_text('embedding,embedding_window,query_id,sibling_hit,full_parent_hit\n')
    lines = ['# Week 6: Qwen hierarchy experiment', '',
        '**Reference-only run; Qwen results are pending.**' if reference_only else
        f'Qwen embedding window selected on development questions: **{selected} tokens**.', '',
        'Evidence inclusion uses the frozen Week 5 tokenizer and 80 test questions. '
        'Context budgets are body-only; model input windows use Qwen tokens.', '']
    if not reference_only:
        summary_index = {(r['embedding'], r['embedding_window'], r['method'], r['mode'],
                          r['context_budget'], r['k']): r for r in comparison}
        def hits(embedding, window, method, mode, budget):
            return summary_index[(embedding, window, method, mode, budget, 10)]['hit_count']
        paired_index = {(r['family'], r['left_embedding'], r['left_window'], r['left_method'],
                         r['left_mode'], r['right_embedding'], r['right_window'], r['right_method'],
                         r['right_mode'], r['context_budget']): r for r in paired}
        def hierarchy_pair(method, mode, budget):
            return paired_index[('hierarchy_vs_child', 'Qwen3-0.6B', selected, method, mode,
                                 'Qwen3-0.6B', selected, method, 'child', budget)]
        qwen_budget = {mode: hits('Qwen3-0.6B', selected, 'dense', mode, 2048)
                       for mode in config['modes']}
        qwen_unbounded = {mode: hits('Qwen3-0.6B', selected, 'dense', mode, None)
                          for mode in config['modes']}
        hierarchy_pairs = {mode: hierarchy_pair('dense', mode, 2048)
                           for mode in config['modes'] if mode != 'child'}
        window_rows = {r['embedding_window']: r for r in development if r['method'] == 'hybrid'
                       and r['mode'] == 'parent' and r['context_budget'] == 2048 and r['k'] == 10}
        windows = ', '.join(f'{window // 1024}k: {window_rows[window]["hit_count"]}/20'
                            for window in sorted(window_rows))
        lines += ['## Answer', '',
            'Qwen substantially improves dense evidence retrieval, but the hierarchy advantage still does not '
            f'survive a 2,048-token delivery budget. At Hit@10, Qwen dense child retrieval reaches '
            f'**{qwen_budget["child"]}/80**, compared with **{qwen_budget["parent"]}/80** for direct parents, '
            f'**{qwen_budget["child_to_parent"]}/80** for full-parent expansion, and '
            f'**{qwen_budget["child_to_selected_siblings"]}/80** for selected siblings. Unlimited context '
            f'reverses the ordering: child **{qwen_unbounded["child"]}/80**, parent '
            f'**{qwen_unbounded["parent"]}/80**, and full-parent expansion '
            f'**{qwen_unbounded["child_to_parent"]}/80**.', '',
            'The selected-sibling policy recovers most of the budgeted child baseline and is less damaging than '
            f'returning full parents, but it does not improve on child retrieval. Its paired comparison against '
            f'children is {hierarchy_pairs["child_to_selected_siblings"]["left_wins"]} wins, '
            f'{hierarchy_pairs["child_to_selected_siblings"]["left_losses"]} losses, and '
            f'{hierarchy_pairs["child_to_selected_siblings"]["ties"]} ties. Direct parents have '
            f'{hierarchy_pairs["parent"]["left_wins"]} wins and '
            f'{hierarchy_pairs["parent"]["left_losses"]} losses; full-parent expansion has '
            f'{hierarchy_pairs["child_to_parent"]["left_wins"]} wins and '
            f'{hierarchy_pairs["child_to_parent"]["left_losses"]} losses.', '',
            f'The development sweep plateaus after 4k on its registered criterion ({windows}). The result supports '
            '4k over 2k on these 20 development questions, while 8k and 16k add no measured benefit. Because every '
            'child fits at 2k, this sweep changes only parent representations.', '',
            f'Qwen dense also exceeds MiniLM dense under the shared budget: child '
            f'{qwen_budget["child"]}/80 versus {hits("MiniLM", 256, "dense", "child", 2048)}/80 and parent '
            f'{qwen_budget["parent"]}/80 versus {hits("MiniLM", 256, "dense", "parent", 2048)}/80. The inherited '
            f'hybrid weight is weaker than Qwen dense (budgeted child '
            f'{hits("Qwen3-0.6B", selected, "hybrid", "child", 2048)}/80 and parent '
            f'{hits("Qwen3-0.6B", selected, "hybrid", "parent", 2048)}/80), so the Week 5 fusion setting should '
            'not be treated as Qwen-optimal.', '',
            'The paired tests in `paired_comparisons.csv` are exploratory, two-sided exact McNemar tests without '
            'multiple-comparison adjustment. The small test set supports the direction and size of these observed '
            'differences more strongly than broad claims about other legal corpora.', '',
            '## Complete test results', '']
    lines += [
        '| Embedding | Window | Method | Mode | Budget | Hit@5 | Hit@10 | Coverage@10 | Mean tokens | P95 tokens | Coverage / 1k tokens |',
        '|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|']
    grouped = defaultdict(dict)
    for row in comparison:
        grouped[tuple(row[k] for k in ('embedding', 'embedding_window', 'method', 'mode', 'context_budget'))][row['k']] = row
    for key, rows in grouped.items():
        if 5 not in rows or 10 not in rows:
            continue
        r = rows[10]
        lines.append('| ' + ' | '.join(str(x) if x is not None else 'unbounded' for x in key) +
            f' | {rows[5]["hit_count"]}/{r["n_queries"]} | {r["hit_count"]}/{r["n_queries"]}'
            f' | {r["gold_text_coverage"]:.1%} | {r["mean_context_tokens"]:.0f} | {r["p95_context_tokens"]:.0f}'
            f' | {1000*r["mean_coverage_efficiency"]:.4f} |')
    lines += ['', 'Coverage efficiency is the mean of per-query coverage fractions divided by returned tokens '
              '(zero for empty contexts), scaled by 1,000 in this table.', '',
              'MiniLM rows reuse saved rankings; BM25 is independent of the embedding. ' +
              ('No Qwen window sweep ran; window_sweep.csv has headers only.' if reference_only else
               'Qwen development results are in window_sweep.csv; only the selected window is evaluated on test.'), '',
              'A MiniLM–Qwen difference mixes model capacity, training, tokenizer, instruction and window effects. '
              'Only the within-Qwen window sweep holds the encoder fixed. '
              'Whole-sibling selection changes delivery policy; compare it within each encoder.', '',
              'The split is query-held-out, not parent-disjoint (three gold parents cross dev/test). '
              'No HyDE, answer generation, correctness judging, or latency benchmark is included.']
    (output / 'findings.md').write_text('\n'.join(lines) + '\n')


def validate_config(config):
    frozen = json.loads((REFERENCE / 'selected.json').read_text())
    for key in ('bm25_weight', 'rrf_constant', 'candidate_depth'):
        if config[key] != frozen[key]:
            raise ValueError(f'{key} must remain frozen at the Week 5 value.')
    if config['bm25'] != {key: frozen[key] for key in ('k1', 'b')}:
        raise ValueError('BM25 parameters must remain frozen.')
    windows = config['embedding_windows']
    if not windows or windows != sorted(set(windows)) or any(type(w) is not int or not 1 <= w <= 32768 for w in windows):
        raise ValueError('Embedding windows must be distinct ascending integers in [1, 32768].')
    if config['ks'] != [5, 10] or config['context_budgets'] != [None, 2048]:
        raise ValueError('Keep the registered cutoffs and context budgets fixed.')
    if config['evaluation_tokenizer'] != {
            'model': 'sentence-transformers/all-MiniLM-L6-v2',
            'revision': '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'}:
        raise ValueError('Keep the measurement tokenizer frozen for reference comparability.')
    if config['selection'] != {'split': 'dev', 'method': 'hybrid', 'mode': 'parent', 'context_budget': 2048, 'k': 10}:
        raise ValueError('Window selection must use the registered development-only criterion.')
    if config['modes'] != ['child', 'parent', 'child_to_parent', 'child_to_selected_siblings']:
        raise ValueError('Keep the registered hierarchy modes fixed.')
    if config['model'] != 'Qwen/Qwen3-Embedding-0.6B' or not re.fullmatch(r'[0-9a-f]{40}', config['revision']):
        raise ValueError('Use the Qwen3-Embedding-0.6B model with a pinned commit revision.')
    if type(config['query_window']) is not int or not 1 <= config['query_window'] <= 32768:
        raise ValueError('Invalid fixed query window.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=HERE / 'config/experiment.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference-only', action='store_true', help='Re-evaluate saved BM25/MiniLM rankings, including siblings; no Qwen download.')
    parser.add_argument('--include-32k', action='store_true')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda', 'mps'])
    parser.add_argument('--dtype', default='float32', choices=['float32', 'float16', 'bfloat16'])
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--batch-tokens', type=int, default=4096)
    parser.add_argument('--model-cache', type=Path, default=ROOT / '.cache/models')
    parser.add_argument('--embedding-cache', type=Path, default=ROOT / '.cache/week6_embeddings')
    args = parser.parse_args()
    if min(args.threads, args.batch_size, args.batch_tokens) < 1:
        parser.error('Thread and batch limits must be positive.')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output must be new or empty; cached embeddings can resume into a new output directory.')
    if not args.reference_only:
        import torch
        available = {'cpu': True, 'cuda': torch.cuda.is_available(), 'mps': torch.backends.mps.is_available()}
        if not available[args.device]:
            parser.error(f'Requested device {args.device} is unavailable in this execution environment.')
    config = json.loads(args.config.read_text())
    if args.include_32k:
        config['embedding_windows'] = sorted(set(config['embedding_windows'] + [config['optional_embedding_window']]))
    validate_config(config)
    children, parents, queries, build = load_data(ROOT / 'week5_hierarchy/kevin_hierarchy')
    dev, test = [[q for q in queries if q['split'] == s] for s in ('dev', 'test')]
    if (len(dev), len(test)) != (20, 80):
        raise ValueError('Unexpected frozen split.')
    tokenizer, tokenizer_hash = evaluation_tokenizer(config, args.model_cache, args.offline)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / 'experiment.json', config)
    write_json(args.output / 'split.json', {s: [q['id'] for q in queries if q['split'] == s] for s in ('dev', 'test')})
    manifest = {'status': 'running', 'reference_only': args.reference_only, 'config': config,
        'measurement_tokenizer_sha256': tokenizer_hash, 'input_sha256': build['outputs_sha256'],
        'reference_sha256': {str(p.relative_to(ROOT)): digest(p) for p in sorted((REFERENCE / 'rankings').glob('*.jsonl'))},
        'source_sha256': {str(p.relative_to(ROOT)): digest(p) for p in [Path(__file__),
            ROOT / 'week5_hierarchy/hierarchy_experiment.py', ROOT / 'week4_legal_rag/run.py',
            ROOT / 'week3_bm25/finder_bm25/bm25.py', ROOT / 'week5_hierarchy/kevin_hierarchy/hierarchy/handoff.py',
            ROOT / 'week5_hierarchy/kevin_hierarchy/hierarchy/core.py']},
        'python': sys.version, 'platform': platform.platform(), 'device': args.device, 'dtype': args.dtype,
        'batch_size': args.batch_size, 'batch_tokens': args.batch_tokens, 'threads': args.threads,
        'overlapping_dev_test_gold_parents': sorted(set(g for q in dev for g in q['gold_parent_ids']) &
                                                   set(g for q in test for g in q['gold_parent_ids']))}
    write_json(args.output / 'manifest.json', manifest)
    details, summaries = [], []
    reference = load_reference(test, children, parents)
    for method, rankings in reference.items():
        print(f'Evaluating reference {method}', flush=True)
        rows, totals = evaluate(test, rankings, children, parents, tokenizer, config,
            {'embedding': 'none' if method == 'bm25' else 'MiniLM',
             'embedding_window': 0 if method == 'bm25' else 256, 'method': method})
        details.extend(rows)
        summaries.extend(totals)
    selected = None
    if not args.reference_only:
        import torch
        from threadpoolctl import threadpool_limits
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        threadpool_limits(limits=args.threads)
        encoder = QwenEncoder(config, args)
        prompt = f'Instruct: {config["query_instruction"]}\nQuery: '
        query_vectors = encoder.encode([q['question'] for q in dev], config['query_window'], 'query', prompt)
        units = {name: sorted(records.values(), key=lambda r: r['id']) for name, records in [('child', children), ('parent', parents)]}
        sparse = {}
        for unit, records in units.items():
            retriever = BM25Retriever([{'doc_id': r['id'], 'text': r['text']} for r in records], **config['bm25'])
            sparse[unit] = {q['id']: retriever.search(q['question'], config['candidate_depth']) for q in queries}
            # Guard against drifting BM25/tokenization or corpus ordering.
            if any([h['doc_id'] for h in sparse[unit][q['id']]] != reference['bm25'][q['id']][unit] for q in test):
                raise ValueError('Current BM25 differs from the frozen reference.')
        for window in config['embedding_windows']:
            indexes = {unit: encoder.encode([r['text'] for r in records], window, unit) for unit, records in units.items()}
            methods = rankings_for(dev, units, query_vectors, indexes, sparse, config)
            for method, rankings in methods.items():
                write_jsonl(args.output / 'rankings' / f'dev_qwen_{window}_{method}.jsonl',
                    [{'query_id': qid, **ranking} for qid, ranking in rankings.items()])
                rows, totals = evaluate(dev, rankings, children, parents, tokenizer, config,
                    {'embedding': 'Qwen3-0.6B', 'embedding_window': window, 'method': method})
                details.extend(rows)
                summaries.extend(totals)
            write_csv(args.output / 'window_sweep.csv', [r for r in summaries if r['split'] == 'dev'])
            write_csv(args.output / 'index_audit.csv', encoder.audit)
        selected = choose_window(summaries, config)
        write_json(args.output / 'selected.json', {'embedding_window': selected,
            'criterion': config['selection'], 'tie_break': 'coverage, mean coverage efficiency, smaller window'})
        # Selection is saved before Qwen test queries are encoded or ranked.
        indexes = {unit: encoder.encode([r['text'] for r in records], selected, unit) for unit, records in units.items()}
        test_vectors = encoder.encode([q['question'] for q in test], config['query_window'], 'query', prompt)
        for method, rankings in rankings_for(test, units, test_vectors, indexes, sparse, config).items():
            write_jsonl(args.output / 'rankings' / f'test_qwen_{selected}_{method}.jsonl',
                [{'query_id': qid, **ranking} for qid, ranking in rankings.items()])
            rows, totals = evaluate(test, rankings, children, parents, tokenizer, config,
                {'embedding': 'Qwen3-0.6B', 'embedding_window': selected, 'method': method})
            details.extend(rows)
            summaries.extend(totals)
        write_csv(args.output / 'index_audit.csv', encoder.audit)
        manifest['encoder'] = encoder.identity
    export(args.output, details, summaries, config, selected, args.reference_only)
    manifest['status'] = 'reference_complete_qwen_pending' if args.reference_only else 'complete'
    manifest['bm25_tokenizer'] = TOKENIZER_VERSION
    manifest['output_sha256'] = {p.name: digest(p) for p in args.output.glob('*.csv')}
    write_json(args.output / 'manifest.json', manifest)
    print(f'Wrote {args.output / "findings.md"}', flush=True)


if __name__ == '__main__':
    main()
