"""Pinned source/tokenizer I/O; generated files use UTF-8."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_REVISION = 'db0b31dc6d195ce9916897e1ac5e4e6209736c8a'
TOKENIZER_REVISION = '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
SOURCE_HASHES = {
    'corpus.jsonl': '3a3565bc5429f6cead90548e81f87352b449927e4be1cdd804c28d766bb9c246',
    'qa.jsonl': 'e3b869a4e293d081ec5f5b39c2058c8d27b36f611aa9f8275eb1877a7c8b38b0',
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError('Cannot write a CSV with an unknown schema.')
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fetch(url, target):
    import requests
    response = requests.get(url, timeout=(15, 120))
    response.raise_for_status()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(response.content)


def prepare_source(directory=None):
    directory = Path(directory) if directory else ROOT / '.cache/source'
    for filename, expected in SOURCE_HASHES.items():
        path = directory / filename
        if not path.exists():
            fetch(f'https://huggingface.co/datasets/isaacus/legal-rag-bench/resolve/{DATA_REVISION}/{filename}', path)
        if digest(path) != expected:
            raise ValueError(f'Pinned dataset checksum mismatch: {filename}')
    return read_jsonl(directory / 'corpus.jsonl'), read_jsonl(directory / 'qa.jsonl')


def load_tokenizer(directory=None):
    from tokenizers import Tokenizer
    directory = Path(directory) if directory else ROOT / '.cache/tokenizer'
    for filename in ('tokenizer.json', 'sentence_bert_config.json', 'tokenizer_config.json'):
        if not (directory / filename).exists():
            fetch('https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/'
                  f'{TOKENIZER_REVISION}/{filename}', directory / filename)
    tokenizer = Tokenizer.from_file(str(directory / 'tokenizer.json'))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    metadata = {'model': 'sentence-transformers/all-MiniLM-L6-v2',
                'revision': TOKENIZER_REVISION, 'sha256': digest(directory / 'tokenizer.json'),
                'sentence_transformer_config': json.loads((directory / 'sentence_bert_config.json').read_text()),
                'policy': 'WordPiece; no truncation; content and special-token-inclusive counts separately'}
    return tokenizer, metadata
