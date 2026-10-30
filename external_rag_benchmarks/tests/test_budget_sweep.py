"""Check the distinction between body budgets, retrieved depth and exact prompts."""
import unittest
from retrieval import ROOT, read
from run_longbench import counter
from run_longbench_budgets import delivered, make_prompt, label, jobs

class BudgetTests(unittest.TestCase):
    def test_uncapped_returns_every_selected_chunk(self):
        tok = counter()
        units = [{'id':str(i),'text':'one two three','start_char':i*20,'end_char':i*20+13} for i in range(40)]
        out,tokens = delivered(units,tok,None)
        self.assertEqual(len(out),40)
        self.assertEqual(tokens,120)
        self.assertEqual(out,units)

    def test_small_budget_stops_on_prefix(self):
        tok = counter()
        units = [{'id':'a','text':'one two three','start_char':100,'end_char':113},
                 {'id':'b','text':'four five six','start_char':200,'end_char':213}]
        out,tokens = delivered(units,tok,4)
        self.assertEqual(tokens,4)
        self.assertEqual(out[1]['text'],'four')
        self.assertEqual(out[1]['end_char'],204)

    def test_budget_is_separate_from_top_k(self):
        c = read(ROOT/'longbench_budget_config.json')
        self.assertEqual(c['top_k'],40)
        self.assertEqual(c['budgets'],[2048,8192,None])
        self.assertEqual(label(None),'all_retrieved')

    def test_gold_answer_not_in_prompt(self):
        q = {'question':'Question?','answer':'HIDDEN',**{'choice_'+a:'Option '+a for a in 'ABCD'}}
        prompt = make_prompt(q,'Source text')
        self.assertNotIn('HIDDEN',str(prompt))
        self.assertIn('Source text',prompt[1]['content'])

    def test_no_budget_never_changes_input_chunks(self):
        tok = counter()
        unit = {'text':'one two three','start_char':0,'end_char':13}
        original = dict(unit)
        delivered([unit],tok,1)
        self.assertEqual(unit,original)

if __name__=='__main__':
    unittest.main()
