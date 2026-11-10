"""Independently reconcile saved per-query results with rankings and official labels."""
from collections import Counter, defaultdict
import csv
import json
from math import isclose, log2
from pathlib import Path
from statistics import mean


RESULTS = Path(__file__).resolve().parent / "results"
METRICS = (
    "hit", "recall", "precision", "mrr", "ndcg", "all_judged_hit", "complete_page_hit",
)


def read_csv(name):
    with (RESULTS / name).open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def read_json(name):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def audit():
    manifest = read_json("run_manifest.json")
    labels = read_json("qrels.json")
    rankings = read_json("rankings.json")
    details = read_csv("per_query.csv")
    methods = set(manifest["methods"])
    languages = dict.fromkeys(manifest["languages"], 309)
    assert len(methods) == 4 and len(languages) == 6
    assert manifest["pages"] == 2942 and manifest["documents"] == 6
    assert set(rankings) == set(labels) and len(rankings) == 1854
    assert sum(len(label) for label in labels.values()) == 8766
    for qid, by_method in rankings.items():
        assert set(by_method) == methods
        for pages in by_method.values():
            assert len(pages) == len(set(pages)) == 10
            assert all(isinstance(pid, int) and 0 <= pid < 2942 for pid in pages)

    seen = set()
    groups = defaultdict(list)
    q_lang = {}
    for row in details:
        qid, method, k = row["query_id"], row["method"], int(row["k"])
        assert k in (1, 5, 10) and (qid, method, k) not in seen
        seen.add((qid, method, k))
        q_lang.setdefault(qid, row["language"])
        assert q_lang[qid] == row["language"]
        expected = {int(pid): grade for pid, grade in labels[qid].items()}
        returned = rankings[qid][method][:k]
        hit_positions = [pos for pos, page in enumerate(returned, 1) if page in expected]
        full = {pid for pid, grade in expected.items() if grade == 2}
        gains = [1 / log2(i + 2) for i in range(k)]
        discounted = sum(expected[page] * gains[pos - 1] for pos, page in enumerate(returned, 1)
                         if page in expected)
        ideal = sum(a * b for a, b in zip(sorted(expected.values(), reverse=True), gains))
        recomputed = (
            bool(hit_positions), len(hit_positions) / len(expected), len(hit_positions) / k,
            1 / hit_positions[0] if hit_positions else 0, discounted / ideal,
            set(expected).issubset(returned), bool(full.intersection(returned)) if full else None,
        )
        assert len(expected) == int(row["positive_pages"])
        assert len(full) == int(row["full_answer_pages"])
        for field, value in zip(METRICS, recomputed):
            if value is None:
                assert row[field] == ""
            else:
                assert isclose(float(row[field]), value, abs_tol=1e-12)
        groups[("all_variants", "all", method, k)].append(recomputed)
        groups[("language", row["language"], method, k)].append(recomputed)
        if row["language"] == "english":
            groups[("english_content", row["content_group"], method, k)].append(recomputed)
    assert len(seen) == len(rankings) * len(methods) * 3
    assert Counter(q_lang.values()) == languages
    summaries = read_csv("summary.csv")
    assert len(summaries) == len(groups)
    for summary in summaries:
        key = summary["scope"], summary["group"], summary["method"], int(summary["k"])
        rows = groups.pop(key)
        assert len(rows) == int(summary["queries"])
        assert sum(row[-1] is not None for row in rows) == int(summary["complete_page_eligible_queries"])
        for field, col in zip(METRICS, zip(*rows)):
            values = [v for v in col if v is not None]
            if values:
                assert isclose(float(summary[field]), mean(values), abs_tol=1e-12)
            else:
                assert summary[field] == ""
    assert not groups
    timings = read_csv("query_times.csv")
    assert {(r["query_id"], r["method"]) for r in timings} == {
        (qid, method) for qid in rankings for method in methods
    } and len(timings) == 1854 * 4
    for row in timings:
        encode, rank, total = (float(row[name]) for name in (
            "encoder_ms", "rank_ms", "estimated_total_ms",
        ))
        assert encode >= 0 and rank >= 0 and isclose(total, encode + rank, abs_tol=1e-12)
    result = {
        "passed": True, "pages": manifest["pages"], "query_variants": len(rankings),
        "metric_rows_checked": len(details), "summary_rows_checked": len(summaries),
        "timing_rows_checked": len(timings),
        "checks": "Coverage, ranking IDs, qrels, per-query values, grouped means and timing arithmetic.",
    }
    (RESULTS / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
