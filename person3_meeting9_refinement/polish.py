"""Presentation-only corrections; frozen evaluation code and metrics stay unchanged."""
from pathlib import Path
import hashlib, json, os, shutil
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/person3_meeting9_refinement/v1'
EXPORT = ROOT.parent/'outputs/meeting9_refinement_20261109'
os.environ.setdefault('MPLCONFIGDIR', str(OUT/'matplotlib_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

summary = pd.read_csv(OUT/'summary.csv')
choice = json.loads((OUT/'development_choice.json').read_text())['candidate']
name = choice['name']
names = ['bm25', 'question_dense', 'focused160_mix_dense'] + [n for n in summary[summary.cohort == 'remaining249'].method.unique() if n not in ['bm25','question_dense','focused160_mix_dense']]
labels = {'bm25': 'BM25', 'question_dense': 'Question dense', 'focused160_mix_dense': 'Focused mix dense'}
for method in names:
    if '_rrf_w' in method:
        labels[method] = ('Question' if method.startswith('question') else 'Focused mix')+' RRF\n'+f"dense weight {int(method[-3:])/100:.2f}"
fig, axes = plt.subplots(1, 2, figsize=(10.5,4.5), sharey=True)
for ax, cohort in zip(axes, ['remaining249','full309']):
    data = summary[(summary.cohort == cohort) & (summary.k == 10)].set_index('method')
    values = data.loc[names, 'ndcg'].to_numpy()
    bars = ax.bar(np.arange(len(names)), values, color=['#c58b44' if n == 'bm25' else '#347da5' if n == name else '#9aa6b1' for n in names])
    for bar, value in zip(bars, values):
        ax.text(bar.get_x()+bar.get_width()/2, value+.006, f'{value:.3f}', ha='center', fontsize=8)
    ax.set_xticks(np.arange(len(names)), [labels[n] for n in names], rotation=17, ha='right', fontsize=8)
    ax.set_title(cohort); ax.set_ylim(0,.63); ax.grid(axis='y', alpha=.15); ax.set_axisbelow(True)
axes[0].set_ylabel('nDCG@10')
fig.suptitle('Offline rank fusion: previously inspected ViDoRe Finance')
fig.tight_layout(rect=[0,.055,1,.94])
fig.text(.5,.015,'Exploratory descriptive results; no new holdout or p-values. Selected on initial60.',ha='center',fontsize=9)
fig.savefig(OUT/'rrf_refinement.png',dpi=180); plt.close(fig)

path = OUT/'README.md'; text = path.read_text()
for cohort in ['remaining249','full309']:
    data = summary[(summary.cohort == cohort) & (summary.k == 10)].set_index('method')
    old = f"{cohort}, selected minus BM25: **{data.loc[name,'ndcg']-data.loc['bm25','ndcg']:+.4f}**"
    new = f"{cohort}, selected minus BM25: **{data.loc[name,'ndcg']-data.loc['bm25','ndcg']:+.6f}**"
    text = text.replace(old,new)
text = text.replace('The275 single-source queries are grouped by bank;34 multi-source queries', 'The 275 single-source queries are grouped by bank; 34 multi-source queries')
text = text.replace('No reselection or expanded grid.', 'The selected configuration uses no HyDE; its question-hybrid comparator is identical. On remaining249 it essentially matches BM25 nDCG@10 while reducing Hit@10 and Recall@10. No reselection or expanded grid.')
text = text.replace('then `evaluate`. `validate`', 'then `evaluate`; run `.venv/bin/python person3_meeting9_refinement/polish.py` for final labels/precision. `validate`')
path.write_text(text)
manifest = {'presentation_only': True, 'changes': ['Correct chart weight label to0.50', 'Show tiny BM25 difference at six decimals', 'Clarify selected question hybrid and benchmark limits'],
            'evaluation_code_and_raw_metrics_modified': False,
            'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(OUT/'presentation_hashes.json').write_text(json.dumps(manifest,indent=2)+'\n')
for name in ['README.md','rrf_refinement.png','presentation_hashes.json']:
    shutil.copy2(OUT/name,EXPORT/name)
print(f'Polished report/figure; {len(text.split())} words; raw metrics unchanged.')
