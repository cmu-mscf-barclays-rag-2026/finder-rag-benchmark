"""Small deterministic controls, not real-model retrieval results."""
import unittest
from unittest.mock import patch
import tempfile
from pathlib import Path
import experiment as e


class TestExperiment(unittest.TestCase):
    def test_metrics(self):
        r = e.ranking_metrics(["x", "a", "b"], ["a", "b"], 3)
        self.assertEqual(r["hit"], 1)
        self.assertEqual(r["recall"], 1)
        self.assertEqual(r["precision"], 2 / 3)
        self.assertEqual(r["mrr"], .5)

    def test_miss(self):
        self.assertEqual(e.ranking_metrics(["x"], ["a"], 5)["hit"], 0)

    def test_duplicate_rejected(self):
        with self.assertRaises(ValueError):
            e.ranking_metrics(["x", "x"], ["a"], 5)

    def test_stable_rank(self):
        import numpy as np
        self.assertEqual(e.rank(np.array([1., 0.]), np.array([[1., 0.], [1., 0.], [0., 1.]]), 3), [0, 1, 2])

    def test_data_and_spans(self):
        data = e.load_data()
        self.assertEqual(len(data["children"]), 4876)
        self.assertEqual(len(data["parents"]), 2785)
        self.assertEqual(sum(q["split"] == "test" for q in data["queries"]), 80)
        children = {r["id"]: r for r in data["children"]}
        for p in data["parents"]:
            for s in p["child_spans"]:
                self.assertEqual(p["text"][s["start_char"]:s["end_char"]], children[s["child_id"]]["text"])

    def test_budget_and_mapping(self):
        from _hierarchy.handoff import evaluate_run
        d = e.load_data()
        q = d["queries"][0]
        args = ([q], {q["id"]: q["gold_parent_ids"]}, {r["id"]: r for r in d["children"]},
                {r["id"]: r for r in d["parents"]}, e.context_tokenizer())
        full, _ = evaluate_run(*args, mode="parent", ks=(5,), token_budget=None)
        empty, _ = evaluate_run(*args, mode="parent", ks=(5,), token_budget=0)
        self.assertEqual(full[0]["gold_child_coverage_in_context"], 1)
        self.assertEqual(empty[0]["gold_child_coverage_in_context"], 0)
        self.assertEqual(empty[0]["ranked_unit_hit_at_k"], 1)

    def test_selection_dev_only(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(e, "ROOT", Path(temp)), patch.object(e, "identity", return_value="sig"):
            config = {"qwen_windows": [512, 2048]}
            for corpus in ("child", "parent"):
                for model, window, coverage in [("minilm", 256, .1), ("qwen", 512, .2), ("qwen", 2048, .3)]:
                    folder = Path(temp) / "results/dev" / e.run_id(model, corpus, window)
                    e.write_json(folder / "run.json", {"experiment_signature": "sig"})
                    e.write_json(folder / "summary.json", [{"k": 5, "gold_body_token_coverage": coverage, "unit_hit": .5}])
            e.select(config)
            selection = e.read_json(Path(temp) / "results/selection.json")
            self.assertEqual(len(selection["runs"]), 6)
            self.assertTrue(any(r["window"] == 2048 for r in selection["runs"]))
            self.assertFalse((Path(temp) / "results/test").exists())

    def test_complete_runner_with_fake_encoder(self):
        """Exercise storage, latency, evaluation and reporting without claiming model quality."""
        import numpy as np
        tok = e.context_tokenizer()
        class FakeEncoder:
            device = "cpu"
            def __init__(self, *args):
                self.tokenizer = lambda text, **kw: {"input_ids": text.split()}
            def query_text(self, text):
                return text
            def sync(self):
                pass
            def encode(self, texts, window, query=False):
                return np.array([[1., 0.] if 'jury' in t else [0., 1.] for t in texts], dtype=np.float32)
        d = {"children": [{"id": "a", "parent_id": "p", "text": "jury law"},
                           {"id": "b", "parent_id": "r", "text": "ocean water"}],
             "parents": [], "queries": [{"id": "1", "question": "jury", "split": "dev",
                                            "gold_child_ids": ["a"], "gold_parent_ids": ["p"]}]}
        cfg = {"models": {"minilm": {"query_window": 256}}, "ks": [1, 5, 10],
               "context_budget": 2048, "latency_repeats": 2}
        with tempfile.TemporaryDirectory() as temp, patch.object(e, "ROOT", Path(temp)), \
             patch.object(e, "load_data", return_value=d), patch.object(e, "identity", return_value="fake"), \
             patch.object(e, "Encoder", FakeEncoder), patch.object(e, "context_tokenizer", return_value=tok), \
             patch.object(e, "hardware", return_value={"device": "fake"}):
            e.execute("minilm", "child", 256, "dev", cfg, "cpu", 1)
            path = Path(temp) / "results/dev/minilm_child_w256"
            self.assertEqual(e.read_json(path / "summary.json")[0]["unit_hit"], 1)
            self.assertEqual(len(e.read_json(path / "latency_raw.json")), 2)
            e.report()
            self.assertTrue((Path(temp) / "results/comparison.csv").exists())


if __name__ == "__main__":
    unittest.main()
