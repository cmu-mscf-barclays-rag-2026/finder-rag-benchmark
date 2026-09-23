"""Audit frozen configurations at Top-100 without selecting new parameters."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

from run import (BM25Retriever, MODEL, MODEL_REVISION, HASHES, make_chunks,
                 read_jsonl, write_json, write_jsonl, write_csv, fuse)


def first_rank(ranking, gold, lookup):
    return next((i for i, h in enumerate(ranking, 1) if lookup[h['doc_id']]['parent_id'] == gold), None)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True, type=Path)
    p.add_argument('--baseline', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--model-cache', type=Path)
    p.add_argument('--embedding-cache', type=Path)
    p.add_argument('--offline', action='store_true')
    a = p.parse_args()
    if a.output.exists() and any(a.output.iterdir()):
        p.error('Use a new/empty output directory')
    for name, digest in HASHES.items():
        assert hashlib.sha256((a.data/name).read_bytes()).hexdigest() == digest
    baseline_manifest = json.loads((a.baseline/'manifest.json').read_text())
    for name, version in baseline_manifest['packages'].items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f'Use the baseline environment: {name}=={version}')
    import numpy as np
    import torch
    from threadpoolctl import threadpool_limits
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    threadpool_limits(limits=4)
    model = SentenceTransformer(MODEL, revision=MODEL_REVISION, device='cpu',
                               cache_folder=str(a.model_cache) if a.model_cache else None,
                               local_files_only=a.offline)
    assert model.max_seq_length == baseline_manifest['model_max_seq_length']
    corpus = read_jsonl(a.data/'corpus.jsonl')
    qa = read_jsonl(a.data/'qa.jsonl')
    split = json.loads((a.baseline/'split.json').read_text())
    test = [q for q in qa if q['id'] in split['test']]
    selected = json.loads((a.baseline/'selected.json').read_text())
    previous = {(r['method'], r['query_id']): r['hits'] for r in read_jsonl(a.baseline/'retrievals.jsonl')}
    indexes = {}
    for size in sorted({c['chunk_size'] for c in selected.values()}):
        print('Preparing frozen chunk size', size, flush=True)
        chunks = make_chunks(corpus, model.tokenizer, size)
        lookup = {c['doc_id']: c for c in chunks}
        vectors = None
        if any(c['chunk_size'] == size and m != 'bm25' for m,c in selected.items()):
            key = hashlib.sha256(json.dumps({
                'model': MODEL, 'revision': MODEL_REVISION, 'max_length': model.max_seq_length,
                'packages': {n: importlib.metadata.version(n) for n in ('torch','sentence-transformers','transformers')},
                'texts': [c['text'] for c in chunks]}, sort_keys=True).encode()).hexdigest()
            cached = a.embedding_cache/(key+'.npy') if a.embedding_cache else None
            vectors = np.load(cached, allow_pickle=False) if cached and cached.exists() else model.encode(
                [c['text'] for c in chunks], normalize_embeddings=True, batch_size=32, show_progress_bar=True)
            assert len(vectors) == len(chunks)
        indexes[size] = chunks,lookup,vectors
    per_query, saved = [], []
    for method,cfg in selected.items():
        chunks,lookup,vectors = indexes[cfg['chunk_size']]
        bm = BM25Retriever(chunks,k1=cfg['k1'],b=cfg['b']) if method != 'dense' else None
        print('Retrieving',method,len(test),'queries',flush=True)
        for q in test:
            lexical = bm.search(q['question'],100) if bm else []
            dense = []
            if method != 'bm25':
                vec = model.encode([q['question']], normalize_embeddings=True,show_progress_bar=False)[0]
                scores = vectors @ vec
                order = np.argsort(-scores,kind='stable')[:100]
                dense = [{'doc_id':chunks[i]['doc_id'],'score':float(scores[i])} for i in order]
            ranking = lexical if method == 'bm25' else dense if method == 'dense' else fuse(lexical,dense,cfg['bm25_weight'])
            old = previous[method,q['id']]
            assert [h['doc_id'] for h in ranking[:20]] == [h['doc_id'] for h in old], (method,q['id'],'baseline rank mismatch')
            assert all(abs(h['score']-v['score'])<1e-5 for h,v in zip(ranking,old))
            gold = q['relevant_passage_id']
            rank = first_rank(ranking,gold,lookup)
            pool = {h['doc_id'] for h in lexical+dense} if method == 'hybrid' else {h['doc_id'] for h in ranking}
            pool_hit = any(lookup[i]['parent_id'] == gold for i in pool)
            row = {'method':method,'query_id':q['id'],'question':q['question'],
                   'gold_parent_id':gold,'first_gold_rank':rank or '',
                   'hit5':int(rank is not None and rank<=5),'hit20':int(rank is not None and rank<=20),
                   'hit100':int(rank is not None),'candidate_pool_size':len(pool),'candidate_pool_hit':int(pool_hit),
                   'category':'top5_hit' if rank is not None and rank<=5 else 'rank_6_to_100' if rank else 'absent_from_top100',
                   'hybrid_bm25_component_rank':first_rank(lexical,gold,lookup) or '' if method=='hybrid' else '',
                   'hybrid_dense_component_rank':first_rank(dense,gold,lookup) or '' if method=='hybrid' else '',
                   'fusion_dropped_gold_from_pool':int(method=='hybrid' and pool_hit and rank is None)}
            per_query.append(row)
            def compact(items):
                return [{**h,'parent_id':lookup[h['doc_id']]['parent_id']} for h in items]
            saved.append({'method':method,'query_id':q['id'],'gold_parent_id':gold,
                          'ranking':compact(ranking), 'bm25_candidates':compact(lexical) if method=='hybrid' else [],
                          'dense_candidates':compact(dense) if method=='hybrid' else []})
    summary = []
    for method in selected:
        rows = [r for r in per_query if r['method']==method]
        n = len(rows)
        hits = sum(r['hit100'] for r in rows)
        top5 = sum(r['hit5'] for r in rows)
        summary.append({'method':method,'n_queries':n,'top5_hits':top5,'rank_6_to_100':hits-top5,
                        'absent_from_top100':n-hits,'top100_hits':hits,'recall_at_100':hits/n,
                        'pool_hits':sum(r['candidate_pool_hit'] for r in rows),
                        'fusion_dropped_gold_from_pool':sum(r['fusion_dropped_gold_from_pool'] for r in rows)})
    # Independent audit directly from persisted-style rankings and parent IDs.
    for row,record in zip(per_query,saved):
        gold = row['gold_parent_id']
        matches = [i+1 for i,h in enumerate(record['ranking']) if h['parent_id']==gold]
        assert row['first_gold_rank'] == (min(matches) if matches else '')
        assert row['hit100'] == int(bool(matches))
        assert len({h['doc_id'] for h in record['ranking']}) == len(record['ranking'])
    a.output.mkdir(parents=True,exist_ok=True)
    write_csv(a.output/'summary.csv',summary)
    write_csv(a.output/'per_query.csv',per_query)
    write_jsonl(a.output/'rankings.jsonl',saved)
    write_json(a.output/'manifest.json',{'baseline_manifest_sha256':hashlib.sha256((a.baseline/'manifest.json').read_bytes()).hexdigest(),
               'dataset_sha256':HASHES,'selected':selected,'test_query_ids':split['test'],
               'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'top20_baseline_matches':len(saved),'validation':'all ranked IDs and scores match baseline Top-20; all first-gold ranks independently checked',
               'protocol':'Frozen baseline; no retuning; K counts chunks; hybrid pool is union of Top-100 per component, then RRF truncated to 100'})
    lines=['# Top-100 candidate-recall diagnostic','','Same 80 held-out questions and frozen configurations as the initial run. No retuning.',
           '', '| Method | Top-5 hits | Gold at ranks 6–100 | Gold absent from Top-100 | Recall@100 |',
           '|---|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f'| {r["method"]} | {r["top5_hits"]} | {r["rank_6_to_100"]} | {r["absent_from_top100"]} | {r["recall_at_100"]:.2%} |')
    h = next(r for r in summary if r['method']=='hybrid')
    lines += ['',f'Hybrid pre-fusion union (up to 200 chunks): gold present for {h["pool_hits"]}/80 questions. Fusion drops it from the final Top-100 for {h["fusion_dropped_gold_from_pool"]} questions.',
              '', '## Interpretation','',
              '- Ranks 6–100: an ideal reranker could potentially recover these parent hits at Top-5 using this candidate set.',
              '- Absent from Top-100: a reranker restricted to that list cannot recover the labeled passage. It might still rank below 100; this is not proof it is never retrievable.',
              '- Hybrid uses its own selected BM25 settings and chunk size; its BM25 component differs from the standalone BM25 baseline.',
              '- K counts actual chunks, not distinct parents. A gold-parent hit does not prove a chunk contains sufficient answer evidence.',
              '- Recall@100 is a parent-hit ceiling for reranking these candidates, not predicted answer accuracy or an attainable guarantee.',
              '- All 240 Top-20 rankings match the original run exactly by ID, with scores checked within 1e-5.',
              '- Per-query failure categories and complete Top-100 rankings are saved alongside this report. This is a descriptive test-set diagnostic, not new model selection.', '']
    (a.output/'report.md').write_text('\n'.join(lines))
    print('\n'.join(lines))

if __name__=='__main__':
    main()
