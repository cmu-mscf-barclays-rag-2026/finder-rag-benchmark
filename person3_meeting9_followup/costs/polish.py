"""Presentation-only chart label spacing; never repeats profiling."""
from pathlib import Path
import hashlib
import json
import os
import shutil

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/person3_meeting9_followup/costs_v1'
EXPORT = ROOT.parent/'outputs/meeting9_followup_20261109/costs'
os.environ['MPLCONFIGDIR'] = str(OUT/'matplotlib_cache')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

table_path = OUT/'quality_cost_table.csv'
table_before = sha(table_path)
data = pd.read_csv(table_path)
methods = ['bm25', 'question_bank_rrf_w050', 'focusedmix_bank_rrf_w050']
labels = ['Constrained BM25', 'Question hybrid .50', 'Focused hybrid .50']
data = data[data.cohort == 'remaining249'].set_index('method').loc[methods]
fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5))
for method, label, color in zip(methods, labels, ['#bc873c', '#458cb1', '#245a81']):
    row = data.loc[method]
    axes[0].scatter(row.mean_estimated_generation_usd_uncached, row.ndcg10, color=color, s=65)
    axes[0].annotate(label, (row.mean_estimated_generation_usd_uncached, row.ndcg10),
                     xytext=(7, 7), textcoords='offset points', fontsize=8)
axes[0].set(xlabel='Estimated generation USD / uncached query', ylabel='nDCG@10',
            title='Quality and generation cost', ylim=(.56, .62))
axes[0].set_xlim(-.00015, float(data.mean_estimated_generation_usd_uncached.max())*1.8)
x = np.arange(3)
axes[1].bar(x-.18, data.retrieval_total_seconds_p50*1000, .36, label='Measured cached-vector retrieval')
axes[1].bar(x+.18, data.modeled_uncached_query_seconds_p50*1000, .36, label='Modeled uncached query')
axes[1].set_yscale('log')
axes[1].set_xticks(x, ['BM25', 'Question hybrid', 'Focused hybrid'])
axes[1].set(ylabel='Milliseconds, log scale', title='Warm service: generation included\nonly in modeled duration')
axes[1].legend(frameon=False, fontsize=7)
for ax in axes:
    ax.grid(axis='y', alpha=.15)
    ax.set_axisbelow(True)
fig.suptitle('ViDoRe Finance: all remaining249, fixed methods')
fig.tight_layout(rect=(0, .08, 1, .94))
fig.text(.5, .012, 'One CPU thread; warm model/index. Modeled sums include historical concurrency2 API duration; not measured E2E. Reader excluded.', ha='center', fontsize=8)
fig.savefig(OUT/'quality_cost_latency.png', dpi=180)
plt.close(fig)
readme = OUT/'README.md'
text = readme.read_text().replace('batch1,20-query warmup,309 queries', 'batch 1, 20-query warmup, 309 queries')
text = text.replace('mixed and96 legacy', 'mixed and 96 legacy').replace('all778 prior', 'all 778 prior')
text = text.replace('used8 threads', 'used 8 threads').replace('All old input/model-cache hashes', 'All original input/model-cache hashes')
text = text.replace('`.venv/bin/python person3_meeting9_followup/costs/run.py`.',
                    '`.venv/bin/python person3_meeting9_followup/costs/run.py`, then `.venv/bin/python person3_meeting9_followup/costs/polish.py` for final label spacing.')
readme.write_text(text)
assert sha(table_path) == table_before
manifest = {'presentation_only': True, 'profile_and_metric_files_modified': False,
            'code_sha256': sha(Path(__file__)), 'quality_cost_table_sha256': table_before,
            'figure_sha256': sha(OUT/'quality_cost_latency.png'), 'readme_sha256': sha(readme)}
(OUT/'presentation_hashes.json').write_text(json.dumps(manifest, indent=2)+'\n')
for filename in ['README.md', 'quality_cost_latency.png', 'presentation_hashes.json']:
    shutil.copy2(OUT/filename, EXPORT/filename)
print('Presentation polished; retained timings and table unchanged.')
