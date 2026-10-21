"""Download exact data/model snapshots used by run.py."""
import sys,urllib.request
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run import DATA,MODELS,EMBED,EMBED_REV,SOURCE_HASHES,sha256
from huggingface_hub import snapshot_download
DATA.mkdir(parents=True,exist_ok=True)
for name,digest in SOURCE_HASHES.items():
 p=DATA/name
 if not p.exists():urllib.request.urlretrieve('https://huggingface.co/datasets/isaacus/legal-rag-bench/resolve/db0b31dc6d195ce9916897e1ac5e4e6209736c8a/'+name,p)
 assert sha256(p)==digest,name
for name,rev in [(EMBED,EMBED_REV),*MODELS.values()]:
 snapshot_download(name,revision=rev,allow_patterns=['*.json','*.safetensors','*.txt','*.model','1_Pooling/*'],ignore_patterns=['onnx/*','openvino/*'])
