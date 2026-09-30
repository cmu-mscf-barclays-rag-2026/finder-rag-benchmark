"""Download and verify the two pinned benchmark files using Python's TLS defaults."""
import argparse
import hashlib
import requests
from pathlib import Path
from run import DATA_REVISION, HASHES

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name, digest in HASHES.items():
        path = args.output / name
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == digest:
            continue
        url = f'https://huggingface.co/datasets/isaacus/legal-rag-bench/resolve/{DATA_REVISION}/{name}'
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        data = response.content
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f'Checksum mismatch: {name}')
        path.write_bytes(data)
        print(path)
