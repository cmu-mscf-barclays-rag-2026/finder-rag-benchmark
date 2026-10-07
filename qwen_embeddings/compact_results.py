"""Restore the saved GPU records without replacing existing experiment results."""
from pathlib import Path, PurePosixPath
import zipfile

def restore_records():
    root = Path(__file__).resolve().parent
    with zipfile.ZipFile(root / 'results/run_records.zip') as archive:
        entries = []
        for entry in archive.infolist():
            rel = PurePosixPath(entry.filename)
            if rel.is_absolute() or '..' in rel.parts or len(rel.parts) != 4 or rel.parts[:2] not in [('results', 'dev'), ('results', 'test')] or rel.suffix != '.json':
                raise ValueError('Unexpected archive path: ' + entry.filename)
            dest = root.joinpath(*rel.parts)
            if not dest.resolve().is_relative_to(root.resolve()):
                raise ValueError('Archive path escapes package')
            payload = archive.read(entry)
            if dest.exists() and dest.read_bytes() != payload:
                raise ValueError('Existing results differ; use a clean package: ' + str(dest))
            entries.append((dest, payload))
        for dest, payload in entries:
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(payload)

if __name__ == '__main__':
    restore_records()
