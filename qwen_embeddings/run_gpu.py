"""Reproduce the full registered experiment on one CUDA GPU."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

# Fix CPU search threading before NumPy is imported by the experiment module.
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test-only', action='store_true', help='Require and use an existing frozen selection')
    args = parser.parse_args()
    from compact_results import restore_records
    restore_records()
    import experiment as e
    config = e.read_json(ROOT / 'config.json')
    def run(*arguments):
        subprocess.run([sys.executable, '-u', str(ROOT / 'experiment.py'), *arguments], cwd=ROOT, check=True)
    if not args.test_only:
        for model in ['minilm', 'qwen']:
            run('smoke', '--model', model, '--device', 'cuda')
        for model, windows in [('minilm', [256]), ('qwen', config['qwen_windows'])]:
            for corpus in ['child', 'parent']:
                for window in windows:
                    run('dev', '--model', model, '--corpus', corpus, '--window', str(window),
                        '--device', 'cuda', '--batch-size', '16' if model == 'minilm' else '4')
        run('select')
    selection = e.read_json(ROOT / 'results/selection.json')
    if selection['experiment_signature'] != e.identity(config):
        raise ValueError('Frozen selection no longer matches the experiment')
    for path, expected in selection['dev_hashes'].items():
        if e.digest(ROOT / path) != expected:
            raise ValueError('Development results changed after selection')
    for entry in selection['runs']:
        e.execute(entry['model'], entry['corpus'], entry['window'], 'test', config, 'cuda',
                  16 if entry['model'] == 'minilm' else 4)
    run('report')


if __name__ == '__main__':
    main()
