"""Fetch pinned ViDoRe metadata and OCR only; never download or encode images."""
from pathlib import Path
import concurrent.futures,hashlib,json,time,urllib.parse,urllib.request,io
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/person3_meeting9_transfer/vidore_finance_v1'
DATA=OUT/'data';REPO='vidore/vidore_v3_finance_en';REV='7f432c176d82e27546501ad8064a713ac3071809'
def sha(b):return hashlib.sha256(b).hexdigest()
def dump(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2));t.replace(p)
def get(url):
 with urllib.request.urlopen(url,timeout=90) as r:return r.read()
def main():
 DATA.mkdir(parents=True,exist_ok=True)
 meta=json.loads(get('https://huggingface.co/api/datasets/'+REPO));assert meta['sha']==REV,'Upstream changed: do not silently mix versions'
 dump(DATA/'repository_metadata.json',meta)
 for name in ['queries','qrels','documents_metadata']:
  path=DATA/(name+'.parquet')
  if not path.exists():
   b=get(f'https://huggingface.co/datasets/{REPO}/resolve/{REV}/{name}/test-00000-of-00001.parquet');path.write_bytes(b)
  print(name,len(pd.read_parquet(path)),flush=True)
 blockdir=DATA/'ocr_blocks';blockdir.mkdir(exist_ok=True)
 def block(offset):
  path=blockdir/f'{offset:04d}.json'
  if path.exists():return json.loads(path.read_text())
  rows=[];rawbytes=[]
  for part in [offset,offset+50]:
   if part>=2942:continue
   url='https://datasets-server.huggingface.co/rows?'+urllib.parse.urlencode(dict(dataset=REPO,config='corpus',split='test',offset=part,length=50))
   raw=get(url);rawbytes.append(raw);x=json.loads(raw);assert x['num_rows_total']==2942 and not x['partial']
   for r in x['rows']:
    assert not r['truncated_cells'],f'OCR truncated: {r["row_idx"]}'
    a=r['row'];assert '/--/'+REV+'/--/' in a['image']['src'],'Cached dataset revision mismatch'
    rows.append({k:v for k,v in a.items() if k!='image'}|{'image_available':True,'image_width':a['image']['width'],'image_height':a['image']['height']})
  raw=b''.join(rawbytes)
  result={'offset':offset,'upstream_revision':REV,'raw_http_response_sha256':sha(raw),'rows':rows}
  dump(path,result);print('OCR',offset,'+',len(rows),flush=True);return result
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:blocks=list(ex.map(block,range(0,2942,100)))
 corpus=[r for block in blocks for r in block['rows']];assert len(corpus)==2942 and [x['corpus_id'] for x in corpus]==list(range(2942))
 target=DATA/'corpus.jsonl';target.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in corpus))
 assert json.loads(get('https://huggingface.co/api/datasets/'+REPO))['sha']==REV
 manifest={'dataset':REPO,'revision':REV,'corpus_n':2942,'retrieval_unit':'entire page OCR, dense first256 encoder tokens; BM25 full markdown','image_parquet_downloaded':False,'download_method':'Official datasets-server nontruncated OCR cells; cached image URL revision verified per row; repository HEAD checked before and after. Metadata parquet resolved at pinned SHA.','files':{str(p.relative_to(OUT)):sha(p.read_bytes()) for p in [target,DATA/'queries.parquet',DATA/'qrels.parquet',DATA/'documents_metadata.parquet']},'sources':['https://huggingface.co/datasets/'+REPO,'https://docs.mteb.org/overview/available_benchmarks/']}
 dump(DATA/'manifest.json',manifest);print('Downloaded full2942page OCR corpus',flush=True)
if __name__=='__main__':main()
