"""Link checked facts to OCR offsets and the publisher's page regions."""

import argparse
from pathlib import Path

import pyarrow.parquet as pq

from audit_results import read_csv, read_json
from run_benchmark import DATA_FILES, ROOT, inputs, save_json


# These selected occurrences do not exhaust all possible sources.
EXAMPLES = {
    45: [
        ("Services 2023 net income was USD 4,635 million.", 1141, "4,635", "Net income"),
        ("Markets 2023 net income was USD 3,871 million.", 1175, "3,871", "Net Income (loss)"),
    ],
    111: [
        ("Citi Investments Bahamas Ltd.", 1982, "Citi Investments Bahamas Ltd.", "Bahamas"),
        ("Citi Overseas Holdings Bahamas Limited", 1982, "Citi Overseas Holdings Bahamas Limited", "Bahamas"),
        ("Citi Overseas Investments Bahamas Inc.", 1982, "Citi Overseas Investments Bahamas Inc.", "Bahamas"),
        ("Citigroup Global Markets Holdings Bahamas Limited", 1983,
         "Citigroup Global Markets Holdings Bahamas Limited", "Bahamas"),
    ],
}


def build(data):
    queries, pages, labels, _ = inputs(data)
    queries = {q["query_id"]: q for q in queries}
    pages = {p["corpus_id"]: p for p in pages}
    rankings = read_json("rankings.json")
    boxes = {
        (row["query_id"], row["corpus_id"]): row["bounding_boxes"]
        for row in pq.read_table(data / DATA_FILES[1].replace("/", "-")).to_pylist()
        if row["query_id"] in EXAMPLES
    }
    records = []
    for qid, required in EXAMPLES.items():
        facts = []
        for description, pid, substring, row_anchor in required:
            page = pages[pid]
            text = page["markdown"]
            matching = [line for line in text.splitlines() if substring in line and row_anchor in line]
            if len(matching) != 1:
                raise ValueError(f"Ambiguous fact anchor for {qid}/{pid}.")
            start = text.index(matching[0]) + matching[0].index(substring)
            end = start + len(substring)
            assert text[start:end] == substring
            facts.append({
                "fact": description, "page_id": pid, "document": page["doc_id"],
                "page_index_in_doc": page["page_number_in_doc"],
                "ocr_start": start, "ocr_end_exclusive": end, "ocr_text": substring,
                "publisher_query_page_regions": boxes[qid, pid],
            })
        scores = []
        for method, pages_ranked in rankings[str(qid)].items():
            for k in (1, 5, 10):
                returned = set(pages_ranked[:k])
                present = [fact["fact"] for fact in facts if fact["page_id"] in returned]
                scores.append({
                    "method": method, "k": k, "page_hit": bool(labels[qid].keys() & returned),
                    "facts_with_located_page": len(present), "required_facts": len(facts),
                    "fact_page_coverage": len(present) / len(facts), "covered_facts": present,
                    "retrieved_pages": pages_ranked[:k],
                })
        records.append({
            "query_id": qid, "question": queries[qid]["query"],
            "publisher_labeled_pages": labels[qid], "facts": facts, "retrieval": scores,
        })
    english_at_10 = {
        (int(row["query_id"]), row["method"]): float(row["ndcg"])
        for row in read_csv("per_query.csv") if row["language"] == "english" and row["k"] == "10"
    }
    deltas = sorted(
        (english_at_10[qid, "dense_image"] - english_at_10[qid, "dense_ocr"], qid)
        for qid, query in queries.items() if query["language"] == "english"
    )
    shifts = []
    for delta, qid in (deltas[0], deltas[-1]):
        shifts.append({
            "query_id": qid, "question": queries[qid]["query"],
            "content_tags": queries[qid]["content_type"],
            "ocr_ndcg_at_10": english_at_10[qid, "dense_ocr"],
            "image_ndcg_at_10": english_at_10[qid, "dense_image"],
            "image_minus_ocr_ndcg_at_10": delta,
            "ocr_top_five": rankings[str(qid)]["dense_ocr"][:5],
            "image_top_five": rankings[str(qid)]["dense_image"][:5],
            "publisher_labeled_pages": labels[qid],
        })
    save_json(ROOT / "results/evidence_examples.json", {
        "scope": "Two source-checked worked examples without independent human review or exhaustive alternative locations.",
        "metric": "A fact is credited when its mapped source page is retrieved. "
                  "This locates evidence in the source; it does not show whether an encoder "
                  "or answer model read the text or image region. Publisher boxes are query/page-level.",
        "examples": records,
        "rank_shifts": shifts,
        "rank_shift_scope": "English queries with the largest observed nDCG changes in each direction.",
    })
    print("Verified OCR offsets and official page regions for two worked examples.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / ".cache/data")
    build(parser.parse_args().data)
