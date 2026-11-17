import math

import pytest

from evaluate_fiqa_graph import metrics
from evaluate_longbench_graph import chunk, extract_answer, truncate_middle


def test_chunks_cover_the_text_exactly_and_keep_markup():
    text = "# Title\n\n" + " ".join(f"w{i}" for i in range(7)) + "\n## Next\nend"
    pieces = chunk(text, words=3)
    assert "".join(pieces) == text
    assert pieces[0].startswith("# Title") and [len(p.split()) for p in pieces][:-1] == [3] * (len(pieces) - 1)


def test_truncate_middle_keeps_both_ends_within_budget():
    text = " ".join(f"w{i}" for i in range(100))
    short = truncate_middle(text, words=10)
    assert short.startswith("w0 w1 w2 w3 w4") and short.endswith("w95 w96 w97 w98 w99")
    assert len(short.replace("[...]", "").split()) == 10
    assert truncate_middle("a b c", words=10) == "a b c"


def test_official_answer_extraction():
    assert extract_answer("The correct answer is (B) because ...") == "B"
    assert extract_answer("**The correct answer is C**") == "C"
    assert extract_answer("I think it is (A).") is None


def test_fiqa_metrics_handle_several_relevant_passages():
    values = metrics(["x", "a", "y", "b"], {"a", "b", "c"})
    ideal = 1 + 1 / math.log2(3) + 1 / math.log2(4)
    assert values["ndcg@10"] == pytest.approx((1 / math.log2(3) + 1 / math.log2(5)) / ideal)
    assert values["mrr@10"] == 0.5
    assert values["recall@10"] == pytest.approx(2 / 3)


def test_openai_client_respects_the_call_cap(tmp_path):
    from openai_llm import BudgetExceeded, CachedOpenAIChat

    llm = CachedOpenAIChat("gpt-4o-mini", tmp_path / "cache.jsonl", max_new_calls=0)
    with pytest.raises(BudgetExceeded):
        llm([{"role": "user", "content": "hi"}])


def test_finance_split_keeps_gold_groups_together_and_scores_one_gold():
    from finance_data import FinanceQuestion, metrics as finance_metrics, split_questions

    questions = {f"q{i}": FinanceQuestion(f"q{i}", "text", frozenset({f"p{i // 3}"})) for i in range(30)}
    dev, test = split_questions(questions)
    assert dev and test and not dev & test and len(dev) >= 6
    for group in {q.group for q in questions.values()}:
        members = {qid for qid, q in questions.items() if q.group == group}
        assert members <= dev or members <= test
    values = finance_metrics(["x", "p0"], questions["q0"])
    assert values["gold_rank"] == 2 and values["recall@1"] == 0.0 and values["mrr@10"] == 0.5
    assert values["ndcg@10"] == pytest.approx(1 / math.log2(3))


def test_openai_cache_keys_stay_compatible_with_existing_caches(tmp_path):
    import hashlib
    import json

    from openai_llm import CachedOpenAIChat

    llm = CachedOpenAIChat("gpt-4o-mini", tmp_path / "cache.jsonl", max_new_calls=0)
    messages = [{"role": "user", "content": "hi"}]
    old = hashlib.sha256(json.dumps(["gpt-4o-mini", 96, messages], ensure_ascii=False).encode()).hexdigest()
    assert llm._cache_key(messages, 96, 0.0) == old
    assert llm._cache_key(messages, 96, 1.0) != old


def test_longbench_split_is_fixed_and_disjoint():
    from evaluate_longbench_methods import split

    samples = [{"_id": f"id{i}"} for i in range(503)]
    dev, test = split(samples)
    assert (len(dev), len(test)) == (101, 402)
    assert not {s["_id"] for s in dev} & {s["_id"] for s in test}
    assert [s["_id"] for s in split(list(reversed(samples)))[0]] == [s["_id"] for s in reversed(dev)]


def test_rrf_cuts_the_multiquery_union_to_the_budget():
    from langchain_core.documents import Document

    from evaluate_longbench_methods import rrf

    def docs(*ids):
        return [Document(page_content=i, metadata={"chunk_id": i}) for i in ids]

    fused = rrf([docs("a", "b", "c"), docs("b", "d"), docs("e", "b", "a")], top_n=3)
    assert [d.metadata["chunk_id"] for d in fused] == ["b", "a", "e"]


def test_dspy_engine_sends_requests_through_the_cached_client():
    import dspy

    from evaluate_longbench_methods import CachedEngine

    class FakeChat:
        def __call__(self, messages, label="", *, max_tokens=None, temperature=0.0):
            self.seen = (messages, label, max_tokens, temperature)
            return {"text": "pong", "prompt_tokens": 3, "completion_tokens": 1, "model": "fake"}

    chat = FakeChat()
    lm = dspy.BaseLM(model="openai/gpt-4o-mini", engine=CachedEngine(chat), cache=False,
                     temperature=0.0, max_tokens=50, num_retries=0)
    assert lm("ping") == ["pong"]
    messages, label, max_tokens, temperature = chat.seen
    assert messages[-1] == {"role": "user", "content": "ping"} and label == "dspy"
    assert (max_tokens, temperature) == (50, 0.0)
