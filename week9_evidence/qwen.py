"""Qwen encoding with tokenizer-managed truncation that preserves special tokens.

The frozen Week 6 encoder slices an already-tokenized list. That drops EOS
when a page is too long, which changes the last-token embedding. Keep its
historical outputs untouched and use a separate preprocessing/cache identity.
"""
import hashlib
import json
import statistics
import time

from week6_qwen_hierarchy.qwen_hierarchy import QwenEncoder


def tokenize_inputs(tokenizer, inputs, window):
    full = tokenizer(inputs, add_special_tokens=True, truncation=False)['input_ids']
    clipped = tokenizer(inputs, add_special_tokens=True, truncation=True,
                        max_length=window)['input_ids']
    # This checkpoint appends <|endoftext|>; its chat eos_token_id is a
    # different special token. Preserve the tokenizer's actual suffix.
    if any(not ids or len(ids) > window or ids[-1] != original[-1]
           or ids[-1] not in tokenizer.all_special_ids
           for original, ids in zip(full, clipped)):
        raise ValueError('Qwen inputs must retain their final special token')
    return full, clipped


class FinanceQwenEncoder(QwenEncoder):
    def __init__(self, config, args):
        super().__init__(config, args)
        self.identity['tokenization'] = 'tokenizer_truncation_preserve_special_tokens_v1'
        self.identity['use_cache'] = False

    def encode(self, texts, window, label, prompt=''):
        np, torch = self.np, self.torch
        inputs = [prompt + text for text in texts]
        full, clipped = tokenize_inputs(self.tokenizer, inputs, window)
        if label == 'query' and any(len(ids) > window for ids in full):
            raise ValueError('Question would be truncated')
        self.audit.append({'unit': label, 'embedding_window': window, 'units': len(inputs),
            'max_input_tokens': max(map(len, full)), 'mean_input_tokens': statistics.mean(map(len, full)),
            'truncated_units': sum(len(ids) > window for ids in full), 'all_inputs_end_in_eos': True})
        salt = json.dumps(self.identity, sort_keys=True)
        keys = [hashlib.sha256((salt + json.dumps(ids)).encode()).hexdigest() for ids in clipped]
        vectors, pending = {}, []
        for key, ids in dict(zip(keys, clipped)).items():
            path = self.cache / f'{key}.npy'
            if path.exists():
                vector = np.load(path, allow_pickle=False)
                self.validate(vector)
                vectors[key] = vector
            else:
                pending.append((key, ids))
        self.audit[-1]['cached_unique_inputs'] = len(vectors)
        self.audit[-1]['new_unique_inputs'] = len(pending)
        print(f'{label} window={window}, EOS preserved: {len(pending)} uncached inputs', flush=True)
        if pending and self.model is None:
            self.model = self.loader.from_pretrained(self.config['model'], revision=self.config['revision'],
                cache_dir=str(self.args.model_cache), local_files_only=self.args.offline,
                torch_dtype=getattr(torch, self.args.dtype), attn_implementation='sdpa').to(self.args.device).eval()
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
                output = self.model(**encoded, use_cache=False).last_hidden_state[:, -1]
                output = torch.nn.functional.normalize(output.float(), p=2, dim=1).cpu().numpy()
            for (key, _), vector in zip(batch, output):
                self.validate(vector)
                temporary = self.cache / f'{key}.tmp.npy'
                np.save(temporary, vector)
                temporary.replace(self.cache / f'{key}.npy')
                vectors[key] = vector
            i = end
            print(f'  {label}: {i}/{len(pending)} ({time.monotonic() - started:.0f}s)', flush=True)
        return np.stack([vectors[key] for key in keys])

    def validate(self, vector):
        np = self.np
        if vector.shape != (1024,) or not np.isfinite(vector).all() or not np.isclose(np.linalg.norm(vector), 1, atol=1e-4):
            raise ValueError('Invalid normalized Qwen embedding')
