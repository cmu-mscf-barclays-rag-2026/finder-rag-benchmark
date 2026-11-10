"""Explicit one-time investigated recovery; keep all timeout reservations charged."""
import fcntl,json,sqlite3,time
from api import OUT,ROOT,MODEL,ORG,dump

def main():
    # This is an explicit recovery action, never invoked by the ordinary runner.
    with (OUT/'.runner.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=OUT/'timeout_recovery_1.json'
        assert not path.exists(),'One-time recovery already recorded'
        log=(ROOT.parent/'work/meeting9_transfer_api.log').read_text()
        assert log.count('HALTED: APITimeoutError no automatic retry')==4
        db=sqlite3.connect(OUT/'api_checkpoint.sqlite')
        rows=db.execute("SELECT key,state,reserve,cost,payload,created_at FROM calls WHERE state!='saved'").fetchall()
        assert len(rows)==4 and all(r[1]=='failed_or_unknown' and r[3] is None and r[4] is None for r in rows)
        from openai import OpenAI
        v=dict(line.split('=',1) for line in (ROOT/'.env.local').read_text().splitlines() if '=' in line)
        client=OpenAI(api_key=v['OPENAI_API_KEY'].strip().strip('\"\''),organization=ORG,max_retries=0,timeout=25)
        model=client.models.retrieve(MODEL)
        assert model.id==MODEL
        audit=dict(investigation='Four concurrent Responses APITimeoutError; no response payload returned. Read-only model access recovered.',
            unknown_billing=True,conservative_retained_reservations_usd=sum(r[2] for r in rows),
            authorization='User requested autonomous completion within existing USD7 cumulative guard; no budget/model/sample expansion.',
            action='Retain original timed-out attempts and full reservations under archived keys; permit one deliberate retry as a new charged attempt.',
            records=[dict(original_key=r[0],archive_key=r[0]+'::unknown_attempt1',reserve_usd=r[2],created_at=r[5]) for r in rows],
            next_concurrency=1,automatic_sdk_retries=0,checked_at_unix=time.time())
        dump(path,audit)
        db.execute('BEGIN IMMEDIATE')
        for r in rows:db.execute("UPDATE calls SET key=?,state='unknown_reserved_acknowledged' WHERE key=?",(r[0]+'::unknown_attempt1',r[0]))
        db.commit()
        print('Investigated4timeouts; retained full USD',audit['conservative_retained_reservations_usd'],'as possible spend. No paid calls made.')

if __name__=='__main__':main()
