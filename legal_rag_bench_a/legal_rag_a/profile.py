"""Compute descriptive statistics and a reproducible team split from all rows."""

import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from statistics import mean

from .data import ROOT, REVISION, parent_section, sha256, word_count


def distribution(values):
    values = sorted(values)

    def percentile(p):
        position = (len(values) - 1) * p
        low, high = math.floor(position), math.ceil(position)
        return values[low] + (values[high] - values[low]) * (position - low)

    return {"count": len(values), "min": min(values), "p25": percentile(.25),
            "median": percentile(.5), "p75": percentile(.75), "p95": percentile(.95),
            "max": max(values), "mean": mean(values), "total": sum(values)}


def team_split(queries, config):
    """Keep queries sharing a gold passage together; allocate 20% of groups to dev."""
    groups = defaultdict(list)
    for query in queries:
        groups[tuple(sorted(query["gold_ids"]))].append(query["id"])
    ordered = sorted(groups, key=lambda group: hashlib.sha256(
        (config["split_seed"] + "|" + "|".join(group)).encode()).hexdigest())
    n_dev = max(1, round(len(ordered) * config["team_dev_fraction"]))
    dev_groups = set(ordered[:n_dev])
    return [{"query_id": q["id"], "split": "dev" if tuple(sorted(q["gold_ids"]))
             in dev_groups else "test", "split_id": config["team_split_id"]} for q in queries]


def profile(corpus, queries, data_dir, config):
    source_rows = [json.loads(line) for line in (data_dir / "qa.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    schema = {}
    for name, rows in (("corpus", corpus), ("qa", source_rows)):
        fields = sorted(set().union(*(row.keys() for row in rows)))
        schema[name] = {field: {"types": sorted({type(row.get(field)).__name__ for row in rows}),
                               "missing": sum(field not in row for row in rows),
                               "null": sum(row.get(field) is None for row in rows)}
                        for field in fields}
    lengths = []
    groups = defaultdict(list)
    for passage in corpus:
        section = parent_section(passage["id"])
        groups[section].append(passage)
        lengths.append({"passage_id": passage["id"], "section_id": section,
                        "text_characters": len(passage["text"]),
                        "text_words": word_count(passage["text"]),
                        "footnote_words": word_count(passage.get("footnotes") or ""),
                        "title_words": word_count(passage.get("title") or "")})
    section_rows = [{"section_id": section, "n_passages": len(rows),
                     "text_words": sum(word_count(row["text"]) for row in rows),
                     "text_characters": sum(len(row["text"]) for row in rows)}
                    for section, rows in groups.items()]
    qlengths = [{"query_id": q["id"], "question_words": word_count(q["question"]),
                 "answer_words": word_count(q["answer"]), "n_gold": len(q["gold_ids"])}
                for q in queries]
    counts = Counter(len(q["gold_ids"]) for q in queries)
    split = team_split(queries, config)
    summary = {
        "dataset": config["dataset"], "revision": REVISION,
        "computed_utc": datetime.now(timezone.utc).isoformat(),
        "n_passages": len(corpus), "n_queries": len(queries),
        "n_id_derived_sections": len(groups),
        "n_unique_gold_passages": len({pid for q in queries for pid in q["gold_ids"]}),
        "duplicate_text_rows": len(corpus) - len({p["text"] for p in corpus}),
        "n_passages_with_footnotes": sum(bool(p.get("footnotes")) for p in corpus),
        "single_gold_queries": counts[1],
        "multi_gold_queries": sum(n for g, n in counts.items() if g > 1),
        "single_gold_fraction": counts[1] / len(queries),
        "multi_gold_fraction": sum(n for g, n in counts.items() if g > 1) / len(queries),
        "gold_count_distribution": dict(counts),
        "length_unit": "Whitespace-delimited words; characters use Python len(str).",
        "lengths": {
            "passage_text_words": distribution([row["text_words"] for row in lengths]),
            "passage_text_characters": distribution([row["text_characters"] for row in lengths]),
            "id_derived_section_words": distribution([row["text_words"] for row in section_rows]),
            "question_words": distribution([row["question_words"] for row in qlengths]),
            "answer_words": distribution([row["answer_words"] for row in qlengths]),
        },
        "team_split_counts": dict(Counter(row["split"] for row in split)),
        "official_split": "test (all questions); team split is an exploratory convention",
        "schema": schema,
        "source_sha256": {name: sha256(data_dir / name) for name in ("corpus.jsonl", "qa.jsonl")},
    }
    return summary, lengths, section_rows, qlengths, split
