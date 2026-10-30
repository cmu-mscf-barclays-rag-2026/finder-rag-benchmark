"""Restore the pinned local tokenizer and float32 PyTorch weights."""
from pathlib import Path
import hashlib,os
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
os.environ['DO_NOT_TRACK']='1'
from huggingface_hub import snapshot_download
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
 cache=ROOT.parent/'model_cache';cache.mkdir(exist_ok=True)
 snapshot=Path(snapshot_download('sentence-transformers/all-MiniLM-L6-v2',
   revision='1110a243fdf4706b3f48f1d95db1a4f5529b4d41',cache_dir=cache,
   allow_patterns=['config.json','model.safetensors','tokenizer.json','tokenizer_config.json','special_tokens_map.json','vocab.txt']))
 print('Pinned PyTorch weights restored; SHA256',hashlib.sha256((snapshot/'model.safetensors').read_bytes()).hexdigest())
