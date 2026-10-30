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
