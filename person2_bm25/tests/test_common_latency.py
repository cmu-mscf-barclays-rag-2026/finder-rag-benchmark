"""Check equal timing coverage and warmup exclusion without model downloads."""
import importlib.util
from pathlib import Path
import unittest
PATH=Path(__file__).resolve().parents[2]/'scripts/common_latency.py'
spec=importlib.util.spec_from_file_location('common_latency',PATH)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TimingProtocolTests(unittest.TestCase):
    def test_same_queries_repeats_and_untimed_warmups(self):
        calls={'a':[],'b':[]}
        def method(name):
            def search(query):
                calls[name].append(query)
                return ['evidence']
            return search
        tick=iter(range(100))
        queries=[{'query_id':str(i),'text':f'question {i}'} for i in range(4)]
        rows=module.measure({name:method(name) for name in calls},queries,repeats=3,warmup=2,timer=lambda:next(tick))
        self.assertEqual(len(rows),24)
        self.assertTrue(all(row['latency_ms']==1000 for row in rows))
        for name in calls:
            self.assertEqual(calls[name][:2],['question 0','question 1'])
            self.assertEqual(len(calls[name]),14)
            self.assertEqual({(row['query_id'],row['repeat']) for row in rows if row['method_id']==name},
                             {(str(i),repeat) for i in range(4) for repeat in range(3)})
        self.assertNotEqual([row['method_id'] for row in rows],sorted(row['method_id'] for row in rows))

    def test_rejects_invalid_protocol_and_excess_results(self):
        with self.assertRaises(ValueError): module.measure({},[{'text':'q'}])
        with self.assertRaises(ValueError): module.measure({'a':lambda q:[]},[])
        with self.assertRaises(ValueError):
            module.measure({'a':lambda q:['x']*6},[{'query_id':'q','text':'q'}],warmup=0)


if __name__=='__main__': unittest.main()
