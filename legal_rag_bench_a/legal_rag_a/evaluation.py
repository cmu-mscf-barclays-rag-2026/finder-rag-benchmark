"""Binary ID-based retrieval metrics; no LLM or embedding dependency."""

import math
from statistics import mean


def validate_ids(values, label):
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{label} must be a collection of IDs, not a string.")
    values = list(values)
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError(f"{label} must contain nonempty string IDs.")
    return values


def query_metrics(ranking, gold_ids, k):
    """A missing result occupies an unfilled slot; duplicate IDs are invalid."""
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be a positive integer.")
    if isinstance(ranking, (str, bytes)) or isinstance(gold_ids, (str, bytes)):
        raise ValueError("Expected sequences of passage IDs, not a string.")
    if not isinstance(ranking, (list, tuple)):
        raise ValueError("Ranking must be an ordered list or tuple.")
    ranking = validate_ids(ranking, "Ranking")
    gold = set(validate_ids(gold_ids, "Gold evidence"))
    if not gold:
        raise ValueError("Unlabeled questions require a separate evaluation policy.")
    if len(ranking) != len(set(ranking)):
        raise ValueError("Duplicate result IDs would inflate evidence counts.")
    top = ranking[:k]
    hits = [int(pid in gold) for pid in top]
    count = sum(hits)
    recall = count / len(gold)
    dcg = sum(h / math.log2(i + 2) for i, h in enumerate(hits))
    ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(gold))))
    return {
        "precision_at_k": count / k,
        "precision_returned": count / len(top) if top else 0.0,
        "recall_at_k": recall,
        "evidence_coverage_at_k": recall,
        "all_gold_included_at_k": float(count == len(gold)),
        "hit_rate_at_k": float(count > 0),
        "mrr_at_k": next((1 / (i + 1) for i, h in enumerate(hits) if h), 0.0),
        "ndcg_at_k": dcg / ideal,
        "n_returned": len(top),
        "empty_rate": float(not top),
    }


def evaluate(queries, predictions, corpus_ids, ks=(1, 3, 5, 10)):
    """Require one explicit ranking per query. Return per-query and macro rows.

    Input predictions: {query_id: [passage_id, ...]} in decreasing rank order.
    Empty lists are valid abstentions; absent queries and foreign IDs are errors.
    """
    ids = validate_ids([q["id"] for q in queries], "Query IDs")
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Evaluation requires unique, nonempty query records.")
    if set(predictions) != set(ids):
        raise ValueError("Prediction query IDs must exactly match evaluation query IDs.")
    corpus_ids = set(validate_ids(corpus_ids, "Corpus"))
    ks = tuple(ks)
    if not ks or len(ks) != len(set(ks)):
        raise ValueError("Supply unique evaluation k values.")
    details = []
    for query in queries:
        gold = set(validate_ids(query["gold_ids"], "Gold evidence"))
        if not gold.issubset(corpus_ids):
            raise ValueError("Gold IDs are absent from the corpus.")
        ranking = predictions[query["id"]]
        if not isinstance(ranking, list):
            raise ValueError("Each prediction must be a ranked list of passage IDs.")
        if not set(ranking).issubset(corpus_ids):
            raise ValueError(f"Unknown passage ID for query {query['id']}.")
        for k in ks:
            details.append({"query_id": query["id"], "k": k, "n_gold": len(gold),
                            **query_metrics(ranking, gold, k)})
    metric_names = [key for key in details[0] if key not in ("query_id", "k", "n_gold")]
    aggregates = []
    for k in ks:
        rows = [row for row in details if row["k"] == k]
        aggregates.append({"k": k, "n_queries": len(rows),
                           **{name: mean(row[name] for row in rows) for name in metric_names}})
    return details, aggregates
