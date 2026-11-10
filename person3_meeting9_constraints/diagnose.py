"""Development60 diagnosis only; never computes filtered ranks or scores."""
from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd
from router import ALIASES, route

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/person3_meeting9_constraints/v1'
EXPORT = ROOT.parent/'outputs/meeting9_constraints_20261109'
D = ROOT/'results/person3_meeting9_fullpage/v1'
S = ROOT/'results/person3_meeting9_transfer/vidore_finance_full309_summary_v1'
CORPUS = ROOT/'results/person3_meeting9_transfer/vidore_finance_v1/data/corpus.jsonl'
OUT.mkdir(parents=True,exist_ok=True); EXPORT.mkdir(parents=True,exist_ok=True)
read = lambda p: json.loads(p.read_text())
qs = read(S/'selection.json')['queries'][:60]
qrels = read(S/'qrels.json')
pages = [json.loads(line) for line in CORPUS.read_text().splitlines() if line.strip()]
sources = {str(p['corpus_id']):p['doc_id'] for p in pages}
dense, sparse = np.load(D/'rankings.npz'), np.load(S/'rankings.npz')
assert dense['query_ids'][:60].tolist() == sparse['query_ids'][:60].tolist() == [q['query_id'] for q in qs]
pids = dense['passage_ids'].tolist(); assert pids == sparse['passage_ids'].tolist()
rankings = {m:dense[m][:60] for m in ['question_dense','focused160_mix_dense','short_30_50_mix_dense']}
rankings['bm25'] = sparse['bm25'][:60]
routes, errors = [], []
for i,q in enumerate(qs):
    routing = route(q['question'])
    # Routing is complete before labels are inspected; labels only describe failure modes.
    rel = qrels[q['query_id']]
    gold_sources = sorted({sources[p] for p,g in rel.items() if g > 0})
    docs = routing['route_doc_ids']
    routes.append({'query_id':q['query_id'],'question':q['question'],**routing,
        'diagnostic_gold_sources':gold_sources})
    for method,arr in rankings.items():
        top = [pids[int(p)] for p in arr[i,:10]]; top_sources = [sources[p] for p in top]
        top1_relevant = rel.get(top[0],0)>0; hit10 = any(rel.get(p,0)>0 for p in top)
        errors.append({'query_id':q['query_id'],'method':method,'routed':bool(docs),
          'top1_relevant':top1_relevant,'top1_gold_source_mismatch':not top1_relevant and top_sources[0] not in gold_sources,
          'top1_not_labeled_relevant_same_source':not top1_relevant and top_sources[0] in gold_sources,
          'hit10':hit10,'miss10_no_gold_source_page':not hit10 and not any(s in gold_sources for s in top_sources),
          'miss10_with_gold_source_pages':not hit10 and any(s in gold_sources for s in top_sources),
          'top1_outside_literal_route':bool(docs) and top_sources[0] not in docs,
          'top10_pages_outside_literal_route':sum(s not in docs for s in top_sources) if docs else 0,
          'top10_gold_source_pages':sum(s in gold_sources for s in top_sources),
          'top1_source':top_sources[0],'top1_page_id':top[0], 'route_doc_ids':'|'.join(docs),
          'diagnostic_gold_sources':'|'.join(gold_sources)})
frame = pd.DataFrame(errors); frame.to_csv(OUT/'development60_error_diagnosis.csv',index=False)
(OUT/'development60_routes.json').write_text(json.dumps(routes,ensure_ascii=False,indent=2)+'\n')
cols = ['top1_relevant','top1_gold_source_mismatch','top1_not_labeled_relevant_same_source','hit10','miss10_no_gold_source_page','miss10_with_gold_source_pages','top1_outside_literal_route','top10_pages_outside_literal_route']
summary = frame.groupby('method',sort=False).agg(n=('query_id','size'),routed_queries=('routed','sum'),**{c:(c,'sum') for c in cols}).reset_index()
summary.to_csv(OUT/'development60_diagnostic_summary.csv',index=False)
route_summary = {'queries':60,'routed_queries':sum(bool(x['route_doc_ids']) for x in routes),
  'multiple_literal_companies':sum(len(x['route_doc_ids'])>1 for x in routes),
  'abstain_or_unnamed':{reason:sum(x['reason']==reason for x in routes) for reason in sorted({x['reason'] for x in routes})},
  'filtered_rankings_or_outcomes_computed':False,'api_calls':0,'aliases':ALIASES}
(OUT/'diagnosis_status.json').write_text(json.dumps(route_summary,indent=2)+'\n')
paths = [Path(__file__),Path(__file__).with_name('router.py'),D/'rankings.npz',S/'rankings.npz',S/'qrels.json',S/'selection.json',CORPUS]
(OUT/'diagnosis_input_code_hashes.json').write_text(json.dumps({str(p.resolve().relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2)+'\n')
for path in OUT.iterdir():
    if path.is_file(): shutil_destination = EXPORT/path.name; shutil_destination.write_bytes(path.read_bytes())
print(json.dumps(route_summary,indent=2));print(summary.to_string(index=False))
