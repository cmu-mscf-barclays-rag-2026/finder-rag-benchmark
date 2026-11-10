"""Package completed results and source; exclude secrets, SQLite and bulky vectors."""
from pathlib import Path
import hashlib,json,re,zipfile
from api import ROOT
BASE=ROOT.parent;REPORT=BASE/'outputs/meeting9_20261109'

def main():
    assert json.loads((REPORT/'completion_status.json').read_text())['computational_stages_complete']
    folders=[REPORT]
    folders += [ROOT/x for x in ['person3_meeting9_workflow','person3_meeting9_ablation',
        'person3_meeting9_transfer','person3_meeting9_fullpage','person3_meeting9_evidence']]
    folders += sorted((ROOT/'results').glob('person3_meeting9_*'))
    files={}
    allowed={'.md','.json','.jsonl','.csv','.npz','.parquet','.py','.png','.svg','.txt'}
    for folder in folders:
        for p in folder.rglob('*'):
            if not p.is_file() or p.suffix not in allowed:continue
            parts=p.relative_to(folder).parts
            if any(x.startswith('.') or x in ['__pycache__','vectors','chunk_vectors'] for x in parts):continue
            if p.name in ['PACKAGE_MANIFEST.json','package_verification.json']:continue
            files[str(p.relative_to(BASE))]=p
    secret=ROOT/'.env.local'
    values=dict(line.split('=',1) for line in secret.read_text().splitlines() if '=' in line)
    key=values['OPENAI_API_KEY'].strip().strip('\"\'').encode()
    assert len(key)>20
    pattern=re.compile(rb'(?<![A-Za-z0-9_-])sk-(?:proj-)?[A-Za-z0-9_-]{24,}')
    entries=[]
    for name,p in sorted(files.items()):
        data=p.read_bytes()
        assert key not in data and not pattern.search(data),'Credential scan failed: '+name
        assert not re.search(rb'^-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----$',data,re.M),'Private key detected: '+name
        entries.append(dict(path=name,size=len(data),sha256=hashlib.sha256(data).hexdigest()))
    manifest=dict(purpose='Reviewable results/source overlay for the existing repository; not a standalone model/environment bundle.',
        files=len(entries),uncompressed_bytes=sum(x['size'] for x in entries),entries=entries,
        excludes=['credentials','encrypted key packages','SQLite checkpoints','API logs','model weights','embedding block caches','.git','.venv'],
        successful_generations_retained=True,secret_scan='passed',
        reproduction='Start with outputs/meeting9_20261109/README.md; cached reproduction does not call the API.')
    dest=BASE/'outputs/Meeting9_Findings_20261109.zip'
    with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name,p in sorted(files.items()):z.write(p,name)
        z.writestr('PACKAGE_MANIFEST.json',json.dumps(manifest,indent=2))
    with zipfile.ZipFile(dest) as z:assert z.testzip() is None
    public=dict(archive=str(dest),files=len(entries),archive_bytes=dest.stat().st_size,
        sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),secret_scan='passed',zip_crc='passed')
    (REPORT/'package_verification.json').write_text(json.dumps(public,indent=2))
    print(json.dumps(public))

if __name__=='__main__':main()
