"""Shared BM25, normalized Qwen vectors, ranking fusion and retrieval metrics."""
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
import numpy as np

ROOT = Path(__file__).resolve().parent

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)

def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

def terms(text):
    return re.findall(r'\w+', text.casefold())

class BM25:
    # Reuses the positive-IDF BM25 baseline from the Legal RAG span pilot.
    def __init__(self, texts, k1=1.5, b=.75):
        self.k1, self.b = k1, b
        counts = [Counter(terms(t)) for t in texts]
        self.n = len(counts)
        self.lengths = np.array([sum(c.values()) for c in counts], dtype=float)
        self.avg = max(float(self.lengths.mean()), 1.)
        self.postings = defaultdict(list)
        for index, count in enumerate(counts):
            for term, tf in count.items():
                self.postings[term].append((index, tf))

    def scores(self, query):
        scores = np.zeros(self.n)
        for term, qtf in Counter(terms(query)).items():
            pairs = self.postings.get(term, [])
            if not pairs:
                continue
            ids, tf = np.array(pairs).T
            idf = math.log(1 + (self.n-len(pairs)+.5)/(len(pairs)+.5))
            norm = self.k1 * (1-self.b+self.b*self.lengths[ids]/self.avg)
            scores[ids] += qtf*idf*tf*(self.k1+1)/(tf+norm)
        return scores

def ranked(scores, k):
    return np.argsort(-np.asarray(scores), kind='stable')[:k].tolist()

def fuse(sparse, dense, bm25_weight=.5, constant=60):
    # Weighted reciprocal rank fusion avoids mixing incomparable raw scores.
    scores = defaultdict(float)
    for ids, weight in [(sparse, bm25_weight), (dense, 1-bm25_weight)]:
        for rank, identifier in enumerate(ids, 1):
            if weight:
                scores[identifier] += weight / (constant+rank)
    return sorted(scores, key=lambda i: (-scores[i], str(i)))

def metrics(ids, qrels, k):
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate ranked IDs')
    gold = {str(i): r for i,r in qrels.items() if r > 0}
    if not gold:
        raise ValueError('No judged relevant items')
    relevant = [gold.get(str(i), 0) for i in ids[:k]]
    hits = sum(r > 0 for r in relevant)
    dcg = sum((2**r-1)/math.log2(j+2) for j,r in enumerate(relevant))
    ideal = sum((2**r-1)/math.log2(j+2) for j,r in enumerate(sorted(gold.values(), reverse=True)[:k]))
    return {'hit': int(hits > 0), 'recall': hits/len(gold), 'precision': hits/k,
            'mrr': next((1/(j+1) for j,r in enumerate(relevant) if r > 0), 0.),
            'ndcg': dcg/ideal, 'all_judged_relevant_retrieved': int(hits == len(gold))}

class QwenEncoder:
    def __init__(self, config):
        import torch
        from transformers import AutoModel, AutoTokenizer
        torch.set_num_threads(config['cpu_threads'])
        torch.manual_seed(42)
        self.torch, self.config = torch, config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        cache = os.environ.get('RAG_MODEL_CACHE', str(ROOT / '.cache/models'))
        self.tokenizer = AutoTokenizer.from_pretrained(config['embedding_model'], revision=config['embedding_revision'], cache_dir=cache, padding_side='left')
        self.model = AutoModel.from_pretrained(config['embedding_model'], revision=config['embedding_revision'], cache_dir=cache,
            torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32, attn_implementation='sdpa').to(self.device).eval()

    def sync(self):
        if self.device == 'cuda':
            self.torch.cuda.synchronize()

    def encode(self, texts, instruction=None):
        if instruction:
            texts = [f'Instruct: {instruction}\nQuery:{t}' for t in texts]
        batch = self.tokenizer(texts, padding=True, truncation=True, max_length=self.config['embedding_window'], return_tensors='pt').to(self.device)
        with self.torch.inference_mode():
            last = self.model(**batch).last_hidden_state[:, -1]
            vectors = self.torch.nn.functional.normalize(last.float(), p=2, dim=1)
        return vectors.cpu().numpy()

    def matrix(self, texts, name):
        cache = ROOT / '.cache/embeddings' / name
        cache.mkdir(parents=True, exist_ok=True)
        key = signature({'texts': texts, 'model': self.config['embedding_model'], 'revision': self.config['embedding_revision'],
                         'window': self.config['embedding_window'], 'pooling': 'last-left-padding-float16-v1'})
        metadata = cache / 'identity.json'
        if metadata.exists() and read(metadata)['signature'] != key:
            raise ValueError('Embedding cache configuration changed; use a new cache directory')
        save(metadata, {'signature': key})
        lengths = [len(self.tokenizer.encode(t, add_special_tokens=True)) for t in texts]
        blocks, seconds = [], 0.
        for start in range(0, len(texts), 512):
            path = cache / f'{start:06d}.npz'
            n = min(512, len(texts)-start)
            if path.exists():
                with np.load(path) as saved:
                    vectors, duration = saved['vectors'].copy(), float(saved['seconds'])
            else:
                ordered = sorted(range(n), key=lambda j: lengths[start+j])
                vectors = np.empty((n, 1024), dtype=np.float32)
                batch_size = self.config['batch_size']
                self.sync()
                begin = time.perf_counter()
                for position in range(0, n, batch_size):
                    ids = ordered[position:position+batch_size]
                    vectors[ids] = self.encode([texts[start+j] for j in ids])
                self.sync()
                duration = time.perf_counter()-begin
                np.savez(path, vectors=vectors, seconds=duration)
            blocks.append(vectors)
            seconds += duration
            print(f'{name}: {start+n}/{len(texts)} documents; measured encoding {seconds:.1f}s', flush=True)
        matrix = np.concatenate(blocks)
        assert np.allclose(np.linalg.norm(matrix, axis=1), 1, atol=1e-3)
        return matrix, {'documents': len(texts), 'encoding_seconds': seconds,
                        'input_truncation_rate': float(np.mean(np.array(lengths) > self.config['embedding_window'])),
                        'mean_input_tokens': float(np.mean(lengths)), 'max_input_tokens': max(lengths), 'device': self.device}
