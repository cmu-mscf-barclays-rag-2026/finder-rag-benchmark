"""Command-line entry point for the compact distribution."""
from pathlib import Path
import runpy
from compact_results import restore_records

if __name__ == '__main__':
    restore_records()
    runpy.run_path(str(Path(__file__).with_name('experiment.py')), run_name='__main__')
