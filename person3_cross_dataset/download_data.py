"""Download pinned public benchmark files, with hash verification and atomic writes."""
from pathlib import Path
import concurrent.futures,hashlib,json,urllib.request
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/external_benchmarks'
FILES=[
 ('fiqa_corpus.parquet','BeIR/fiqa','979c07a7cb5ccc6ca009792241fa1250b98055dd','corpus/corpus-00000-of-00001.parquet','b36bb07dab3bfad488190818828e749a52faf5ba391c1d51dac8b06064576f99'),
 ('fiqa_queries.parquet','BeIR/fiqa','979c07a7cb5ccc6ca009792241fa1250b98055dd','queries/queries-00000-of-00001.parquet','e9b3863724ffd538d8efff19684ef8ae495bdbddd96b35e81bdc39e3e22860d6'),
 ('fiqa_test.tsv','BeIR/fiqa-qrels','252958f2d646e22cab6d0c72dd3f0d5de6d0655a','test.tsv','6adc2a640dcdd22bb8b3858f89107adef2a7c3db20a63550dfa7a0f71e379e44'),
 ('longbench2.json','zai-org/LongBench-v2','2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9','data.json','15d61c22d92c96900b3c4948b6aeea218d3214b676a65df48e7b8555604c7fe2')]
def fetch(row):
    name,repo,rev,remote,expected=row;p=DATA/name
    if not p.exists():
        temp=p.with_suffix(p.suffix+'.tmp')
        with urllib.request.urlopen(f'https://huggingface.co/datasets/{repo}/resolve/{rev}/{remote}',timeout=180) as src,temp.open('wb') as out:
            for block in iter(lambda:src.read(1024*1024),b''):out.write(block)
        temp.replace(p)
    digest=hashlib.sha256(p.read_bytes()).hexdigest();assert digest==expected,name
    return dict(name=name,repo=repo,revision=rev,remote=remote,sha256=digest,bytes=p.stat().st_size)
if __name__=='__main__':
    DATA.mkdir(parents=True,exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:rows=list(ex.map(fetch,FILES))
    (DATA/'sources.json').write_text(json.dumps(rows,indent=2));print('Four pinned source files verified.')
