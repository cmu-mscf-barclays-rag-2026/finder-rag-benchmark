"""Create compact review packets; does not assign semantic quality scores."""
import json,re,sys
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
P=Path(__file__).resolve().parents[1]/'results/person3_stronger_hyde'
answers=json.loads((P/'final_answers.json').read_text())
chosen=[int(x) for x in sys.argv[1:]]
for a in answers.values():
 if chosen and a['query_id'] not in chosen:continue
 print('\nID',str(a['query_id'])+':'+a['method'],'\nQ',a['question'],'\nREFERENCE',a['reference_answer'],'\nANSWER',a['text'])
 # Full contexts remain available in final_answers.json; excerpts only aid navigation.
 paragraphs=[x.strip() for x in a['context'].split('\n') if len(x.strip())>30]
 if paragraphs:
  v=TfidfVectorizer().fit_transform([a['text'],*paragraphs]);s=(v[1:]@v[0].T).toarray().ravel();order=s.argsort()[::-1][:3]
  for j in order:print('CONTEXT EXCERPT',paragraphs[j][:1300])
 print('GOLD TOKENS',a['gold_tokens_in_context'])
