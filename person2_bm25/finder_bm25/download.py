"""Fetch the public, pinned FinDER Parquet snapshot; no API keys required."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from urllib.request import Request, urlopen

from .data import DATASET_ID

REVISION = "c4c1b6454aef7f0bb1c37235c7f52ce644642da0"
FILENAME = "train-00000-of-00001.parquet"
SHA256 = "213395beb38a075dcc7b23f8e213ccbd2768df0f5c615479c33e10ce8cc267f1"
URL = f"https://huggingface.co/datasets/{DATASET_ID}/resolve/{REVISION}/data/{FILENAME}"


def fetch_records(cache_dir: Path) -> tuple[list[dict], dict]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError("Install the download dependency: python -m pip install -r requirements.txt") from exc
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / FILENAME
    if not path.exists():
        temporary = path.with_suffix(".download")
        try:
            with urlopen(Request(URL, headers={"User-Agent": "finder-person2-bm25/1.0"}), timeout=120) as response:
                with temporary.open("wb") as target:
                    shutil.copyfileobj(response, target)
            if hashlib.sha256(temporary.read_bytes()).hexdigest() != SHA256:
                raise ValueError("Downloaded FinDER file failed its SHA-256 check")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    if hashlib.sha256(path.read_bytes()).hexdigest() != SHA256:
        raise ValueError(f"Cached file checksum mismatch: {path}. Move it aside and download again.")
    return parquet.read_table(path).to_pylist(), {
        "kind": "huggingface", "dataset_id": DATASET_ID, "revision": REVISION,
        "split": "train", "url": URL, "sha256": SHA256,
    }
