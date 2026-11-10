"""Independent single-input CPU cross-check of saved Qwen MPS embeddings."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from week9_evidence.run import QWEN, ROOT
from week9_evidence.qwen import FinanceQwenEncoder
from week9_evidence.vidore import digest, load_dataset


def smoke(dataset, vectors, device):
    pages, queries, _, _ = load_dataset(dataset)
    args = SimpleNamespace(model_cache=ROOT / '.cache/models',
        embedding_cache=ROOT / '.cache/week9_embeddings', offline=True,
        device=device, dtype='float32', batch_size=8, batch_tokens=4096)
    encoder = FinanceQwenEncoder(QWEN, args)
    pids, qids = ['10', '2552', '2934'], ['0', '13', '24']
    pv = encoder.encode([pages[c]['text'] for c in pids], 256, 'page')
    qv = encoder.encode([queries[q]['query'] for q in qids], 2048, 'query',
                        prompt=f'Instruct: {QWEN["query_instruction"]}\nQuery: ')
    np.savez_compressed(vectors, page_ids=np.array(pids), query_ids=np.array(qids), pages=pv, queries=qv)


def check(dataset, vectors, output):
    pages, queries, _, _ = load_dataset(dataset)
    saved = np.load(vectors)
    tokenizer = AutoTokenizer.from_pretrained(QWEN['model'], revision=QWEN['revision'],
        cache_dir=str(ROOT / '.cache/models'), local_files_only=True)
    model = AutoModel.from_pretrained(QWEN['model'], revision=QWEN['revision'],
        cache_dir=str(ROOT / '.cache/models'), local_files_only=True,
        torch_dtype=torch.float32, attn_implementation='sdpa').eval()
    torch.set_num_threads(4)
    checks = []
    for kind, ids in [('page', ['10', '2552', '2934', '275', '2193', '2766']),
                      ('query', ['0', '13', '24', '55', '10', '9'])]:
        saved_ids = list(saved['page_ids' if kind == 'page' else 'query_ids'])
        for uid in ids:
            if uid not in saved_ids:
                continue  # The smoke file has three pages/queries; full runs have all six.
            text = pages[uid]['text'] if kind == 'page' else (
                f'Instruct: {QWEN["query_instruction"]}\nQuery: ' + queries[uid]['query'])
            batch = tokenizer(text, truncation=True, max_length=256 if kind == 'page' else 2048,
                              return_tensors='pt')
            with torch.inference_mode():
                hidden = model(**batch, use_cache=False).last_hidden_state
                vec = torch.nn.functional.normalize(hidden[:, -1].float(), p=2, dim=1)[0].numpy()
            prior = saved['pages' if kind == 'page' else 'queries'][saved_ids.index(uid)]
            checks.append({'kind': kind, 'id': uid, 'cpu_vs_saved_cosine': float(vec @ prior),
                           'max_absolute_difference': float(np.max(np.abs(vec - prior)))})
    if len(checks) < 6:
        raise ValueError('Too few reference samples in saved vectors')
    minimum = min(r['cpu_vs_saved_cosine'] for r in checks)
    result = {'checks': checks, 'minimum_cosine': minimum, 'passed': minimum > 0.999,
        'scope': f'{len(checks)} independent CPU single-input encodings versus cached MPS batched vectors; not a full-corpus device equivalence proof.',
        'vectors_sha256': digest(vectors), 'source_sha256': digest(Path(__file__))}
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)
    if not result['passed']:
        raise ValueError('CPU/MPS mismatch: investigate before interpreting Qwen retrieval')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--vectors', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--create-smoke', action='store_true')
    parser.add_argument('--device', default='mps', choices=['cpu', 'mps'])
    args = parser.parse_args()
    if args.create_smoke:
        smoke(args.dataset, args.vectors, args.device)
    check(args.dataset, args.vectors, args.output)
