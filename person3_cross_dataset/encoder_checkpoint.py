"""Persist costly offline encoder work in bounded, credential-free checkpoint parts."""
from pathlib import Path
import argparse,hashlib,json,sqlite3,zipfile
ROOT=Path(__file__).resolve().parents[1]
DB=ROOT/'data/external_benchmarks/cache_v1/chunk_embeddings.sqlite'
REV='1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
N=100000;WIDTH=32+384*4
def export(part):
    c=sqlite3.connect(DB);count=c.execute('select count(*) from embeddings').fetchone()[0]
    offset=(part-1)*N;needed=min(N,626520-offset,count-offset)
    assert part>=1 and needed>0,(count,offset,needed)
    out=ROOT.parent/'exports'/f'LongBench2_Encoder_Checkpoint_Part{part:02d}.zip';tmp=out.with_suffix('.tmp.zip')
    digest=hashlib.sha256();n=0
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        with z.open('vectors.bin','w',force_zip64=True) as f:
            for key,vector in c.execute('select key,vector from embeddings order by rowid limit ? offset ?',(needed,offset)):
                assert len(vector)==1536
                block=bytes.fromhex(key)+vector;digest.update(block);f.write(block);n+=1
        assert n==needed
        z.writestr('manifest.json',json.dumps(dict(part=part,count=n,model='sentence-transformers/all-MiniLM-L6-v2',revision=REV,dtype='float32',dimensions=384,record_bytes=WIDTH,normalized=True,sha256=digest.hexdigest())))
    tmp.replace(out);print(json.dumps(dict(path=str(out),bytes=out.stat().st_size,part=part,count=n)),flush=True)
def restore(path):
    DB.parent.mkdir(parents=True,exist_ok=True);c=sqlite3.connect(DB)
    c.execute('create table if not exists embeddings (key text primary key,vector blob)')
    with zipfile.ZipFile(path) as z:
        m=json.loads(z.read('manifest.json'));assert m['revision']==REV and m['record_bytes']==WIDTH
        digest=hashlib.sha256();n=0;batch=[]
        with z.open('vectors.bin') as f:
            while block:=f.read(WIDTH):
                assert len(block)==WIDTH;digest.update(block);batch.append((block[:32].hex(),block[32:]));n+=1
                if len(batch)==1000:c.executemany('insert or ignore into embeddings values (?,?)',batch);c.commit();batch=[]
        c.executemany('insert or ignore into embeddings values (?,?)',batch);c.commit()
        assert n==m['count'] and digest.hexdigest()==m['sha256']
    print('Restored',n,'exact float32 vectors',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--part',type=int);g.add_argument('--restore',type=Path);a=p.parse_args()
    export(a.part) if a.part else restore(a.restore)
