"""Build a feasibility brief and executed, offline showcase notebook."""

from contextlib import redirect_stdout
import csv
from io import StringIO
import json
from pathlib import Path
import statistics
import zipfile

import nbformat as nbf

from audit_results import audit, read_csv, read_json


ROOT = Path(__file__).resolve().parent
LABELS = {
    "bm25_ocr": "BM25 OCR", "dense_ocr": "Qwen OCR",
    "dense_image": "Qwen image", "rrf_ocr_image": "Text + image RRF",
}


def fmt(value):
    return f"{float(value):.4f}" if value != "" else "n/a"


def paired():
    english = [r for r in read_csv("per_query.csv") if r["language"] == "english"]
    table = {(r["query_id"], r["method"], r["k"]): r for r in english}
    ids = sorted({r["query_id"] for r in english}, key=int)
    comparisons = []
    for baseline, candidate in (
        ("dense_ocr", "dense_image"), ("dense_ocr", "rrf_ocr_image"),
        ("dense_image", "rrf_ocr_image"),
    ):
        for metric, k in (("hit", "5"), ("recall", "5"), ("ndcg", "10")):
            delta = [float(table[qid, candidate, k][metric]) -
                     float(table[qid, baseline, k][metric]) for qid in ids]
            comparisons.append({
                "baseline": LABELS[baseline], "candidate": LABELS[candidate], "metric": f"{metric}@{k}",
                "mean_delta": statistics.mean(delta), "wins": sum(v > 1e-12 for v in delta),
                "losses": sum(v < -1e-12 for v in delta), "ties": sum(abs(v) <= 1e-12 for v in delta),
                "queries": len(ids),
            })
    with (ROOT / "results/paired_english.csv").open("w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)


def brief():
    m = read_json("run_manifest.json")
    score_rows = {(r["method"], r["k"]): r for r in read_csv("summary.csv")
                  if r["scope"] == "language" and r["group"] == "english"}
    timings = {r["method"]: r for r in read_csv("timing_summary.csv") if r["scope"] == "english"}
    best = max(LABELS, key=lambda method: float(score_rows[method, "10"]["ndcg"]))
    best_image = float(score_rows["dense_image", "10"]["ndcg"])
    best_text = float(score_rows["dense_ocr", "10"]["ndcg"])
    comparison = f"{best_image - best_text:+.4f}"
    visual_hits = next(r for r in read_csv("paired_english.csv")
                       if r["baseline"] == LABELS["dense_ocr"] and r["candidate"] == LABELS["dense_image"]
                       and r["metric"] == "hit@5")
    visual_group = [r for r in read_csv("summary.csv")
                    if r["scope"] == "english_content" and r["group"] == "chart_or_other_visual"
                    and r["k"] == "10"]
    chart_scores = {r["method"]: r for r in visual_group}
    chart_sentence = (
        f"Among the {chart_scores['dense_image']['queries']} English questions tagged for charts, "
        "images or infographics, "
        f"OCR and image nDCG@10 were {fmt(chart_scores['dense_ocr']['ndcg'])} and "
        f"{fmt(chart_scores['dense_image']['ndcg'])}, respectively. "
        if chart_scores else ""
    )
    n = score_rows["dense_ocr", "5"]["complete_page_eligible_queries"]
    table = [
        "| Method | Hit@5 | Recall@5 | nDCG@10 | Complete-answer-page Hit@5 | Estimated median query ms |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method, label in LABELS.items():
        row5, row10 = score_rows[method, "5"], score_rows[method, "10"]
        latency = float(timings[method]["median_estimated_total_ms"])
        table.append(f"| {label} | {fmt(row5['hit'])} | {fmt(row5['recall'])} | {fmt(row10['ndcg'])}"
                     f" | {fmt(row5['complete_page_hit'])} | {latency:.2f} |")
    builds = []
    for kind in ("ocr", "image"):
        emb = m["embeddings"][kind]
        builds.append(f"- {kind.upper()}: {emb['encode_seconds']:.1f} s encoding, "
                      f"{emb['matrix_seconds']:.2f} s matrix assembly, "
                      f"{emb['matrix_bytes'] / 2**20:.1f} MiB of float32 vectors.")
    builds.append(f"- BM25: {m['bm25_build_seconds']:.2f} s to build postings, "
                  f"{m['bm25_numeric_postings_bytes'] / 2**20:.1f} MiB in numeric postings "
                  "before counting the vocabulary and Python objects.")
    result = (
        "# Feasibility assessment: ViDoRe V3 Finance–EN\n\n"
        "ViDoRe supports a matched comparison of page images and OCR text for financial documents. "
        "The public corpus pairs both representations for each page and provides graded relevance and annotated regions. "
        "The run covers all 2,942 pages in six bank 10-Ks and all 1,854 query variants. "
        "The 309 original English questions form the primary result. Translations are summarized by language.\n\n"
        "## Measured English results\n\n"
        + "\n".join(table) + "\n\n"
        f"**{LABELS[best]}** had the highest English nDCG@10 in this run. "
        f"The image-minus-OCR difference using the same Qwen checkpoint was {comparison}. "
        f"Complete-answer-page Hit@5 uses the {n} English questions with a grade-2 page. "
        f"Image retrieval added {visual_hits['wins']} top-five page hits missed by OCR and "
        f"lost {visual_hits['losses']} hits that OCR found. "
        + chart_sentence + "\n\n"
        "## Compute\n\n"
        + "\n".join(builds) + "\n\n"
        f"The run retained {len(m['empty_ocr_pages'])} pages with empty OCR in the shared candidate pool. "
        "The encoding time starts with prepared input records and includes model preprocessing and inference. "
        "The query total sums measured encoder and ranking components. It is an estimate for local sequential execution.\n\n"
        "## Evidence evaluation and recommendation\n\n"
        "Use page metrics for the full benchmark and add a separate fact-coverage layer for a reviewed set of questions. "
        "The two worked examples link specific facts to OCR offsets and preserve the publisher's query/page regions. "
        "For example, a question comparing two segments can score a page hit when only one of the two mapped "
        "fact pages is retrieved. Grade-1 and grade-2 labels score page relevance. They do not enumerate all facts "
        "required across pages or verify what text or pixels an answer model ultimately used.\n\n"
        f"For a first English ranking baseline under this configuration, use {LABELS[best]} as the measured "
        "reference by nDCG@10. Before generalizing to Barclays documents, check additional report types and "
        "inspect where each input format adds evidence. The corpus contains six reports from one sector and year. "
        "The visual path caps page images at 786,432 pixels and OCR encoding uses a 4,096-token input limit. "
        "Both input limits can remove detail before ranking. "
        "Timing describes one laptop setup, and the comparisons do not test answer generation.\n"
    )
    (ROOT / "FEASIBILITY.md").write_text(result, encoding="utf-8")


def notebook():
    cells = []

    def md(s):
        cells.append(nbf.v4.new_markdown_cell(s))

    def code(s):
        cells.append(nbf.v4.new_code_cell(s))

    md("# ViDoRe Finance–EN\n\n"
       "**Question:** Does representing the same bank 10-K page as OCR text or as an image change which evidence we find?\n\n"
       "The saved results cover the entire Finance–EN corpus and all six query languages. "
       "This notebook reads saved outputs and runs without an API key or GPU.")
    code("""import csv
import json
from html import escape
from pathlib import Path
from IPython.display import HTML

results = Path("results")

def read_csv(name):
    with (results / name).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))

def read_json(name):
    return json.loads((results / name).read_text(encoding="utf-8"))

def show(columns, rows):
    head = "".join(f"<th>{escape(str(cell))}</th>" for cell in columns)
    body = "".join("<tr>" + "".join(f"<td>{escape(str(cell))}</td>" for cell in row) + "</tr>" for row in rows)
    display(HTML(f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"))

def score(value):
    return f"{float(value):.4f}" if value != "" else "n/a"

labels = {
    "bm25_ocr": "BM25 OCR", "dense_ocr": "Qwen OCR",
    "dense_image": "Qwen image", "rrf_ocr_image": "Text + image RRF",
}
manifest = read_json("run_manifest.json")
summary = read_csv("summary.csv")
show(["Source reports", "Pages", "Query variants", "Relevance judgments", "Empty OCR pages"], [
    [manifest["documents"], manifest["pages"], manifest["queries"], manifest["qrels"],
     len(manifest["empty_ocr_pages"])]
])
show(["Language", "Questions"], sorted(manifest["languages"].items()))""")
    md("## Methods\n\n"
       "- **BM25 OCR** ranks pages by the extracted words in the query.\n"
       "- **Qwen OCR** compares the question embedding with embeddings of OCR page text.\n"
       "- **Qwen image** uses the same Qwen checkpoint to embed page images. Both dense runs share the same question vector.\n"
       "- **Text + image RRF** combines the two dense top-100 lists with equal weights and rank constant 60.\n\n"
       "All methods search the same page pool. Dense search scores every page by exact cosine similarity. "
       "The page representations are cached and neither labels nor reference answers enter the retrieval query.")
    md("## English retrieval results\n\n"
       "Hit checks for any relevant page. Recall counts what fraction of the labeled pages is retrieved. "
       "Graded nDCG gives greater credit to fully relevant pages and to earlier ranks. "
       "Complete-answer-page Hit@5 is evaluated only where the publisher marked at least one page as containing a complete answer.")
    code("""english = {(r["method"], r["k"]): r for r in summary if r["scope"] == "language" and r["group"] == "english"}
show(["Method", "Hit@5", "Recall@5", "nDCG@10", "Grade-2 hit@5", "Eligible questions"], [
    [label, score(english[method, "5"]["hit"]), score(english[method, "5"]["recall"]),
     score(english[method, "10"]["ndcg"]), score(english[method, "5"]["complete_page_hit"]),
     english[method, "5"]["complete_page_eligible_queries"]]
    for method, label in labels.items()
])
paired = read_csv("paired_english.csv")
show(["Baseline", "Candidate", "Metric", "Mean change", "Wins", "Losses", "Ties"], [
    [r["baseline"], r["candidate"], r["metric"], score(r["mean_delta"]), r["wins"], r["losses"], r["ties"]]
    for r in paired
])""")
    md("## Language and content breakdown\n\n"
       "Each language column has 309 questions. Translations share the underlying tasks, so they are not independent extra questions. "
       "The English content groups use publisher tags and prioritize chart, image or infographic tags above table tags.")
    code("""def grouped_ndcg(scope):
    rows = [r for r in summary if r["scope"] == scope and r["k"] == "10"]
    groups = sorted({r["group"] for r in rows})
    by_key = {(r["group"], r["method"]): r for r in rows}
    show(["Group", "Questions", *labels.values()], [
        [group, by_key[group, "bm25_ocr"]["queries"],
         *(score(by_key[group, method]["ndcg"]) for method in labels)]
        for group in groups
    ])

grouped_ndcg("all_variants")
grouped_ndcg("language")
grouped_ndcg("english_content")""")
    md("## Evidence coverage walkthrough\n\n"
       "These worked examples mark where specific facts appear. A fact is counted when its mapped page is in the ranking. "
       "A relevant page can support only part of the answer. OCR offsets and the official query/page bounding boxes "
       "are saved for inspection. The two examples illustrate the method and are distinct from full benchmark scores.")
    code("""for example in read_json("evidence_examples.json")["examples"]:
    display(HTML(f"<p><b>Query {example['query_id']}</b> — {escape(example['question'])}</p>"))
    show(["Fact", "Page ID", "OCR character interval"], [
        [f["fact"], f["page_id"], f"[{f['ocr_start']}, {f['ocr_end_exclusive']})"]
        for f in example["facts"]
    ])
    show(["Method", "Cutoff", "Page hit", "Facts whose mapped page was retrieved"], [
        [labels[r["method"]], r["k"], r["page_hit"], f"{r['facts_with_located_page']}/{r['required_facts']}"]
        for r in example["retrieval"] if r["k"] in (1, 5)
    ])""")
    md("## Where the two page representations rank differently\n\n"
       "These are the largest observed English nDCG@10 changes between the two dense runs. "
       "Use them for error inspection alongside the paired summary.")
    code("""shifts = read_json("evidence_examples.json")["rank_shifts"]
show(["Query", "OCR nDCG@10", "Image nDCG@10", "OCR top 5 page IDs", "Image top 5 page IDs"], [
    [r["question"], score(r["ocr_ndcg_at_10"]), score(r["image_ndcg_at_10"]),
     r["ocr_top_five"], r["image_top_five"]]
    for r in shifts
])""")
    md("## Cost and verification\n\n"
       "Index build cost is measured separately from query latency. Query totals combine GPU encoding and CPU ranking; "
       "fusion uses a single query encoding. Timing is specific to the recorded laptop run.")
    code("""timing = [r for r in read_csv("timing_summary.csv") if r["scope"] == "english"]
show(["Method", "Median encoder ms", "Median rank ms", "Estimated median total ms", "Estimated p95 ms"], [
    [labels[r["method"]], *(score(r[field]) for field in ("median_encode_ms", "median_rank_ms",
       "median_estimated_total_ms", "p95_estimated_total_ms"))]
    for r in timing
])
show(["Modality", "Pages encoded", "Encoding seconds", "Matrix assembly seconds", "Vector MiB"], [
    [key, e["rows"], f"{e['encode_seconds']:.1f}", f"{e['matrix_seconds']:.2f}", f"{e['matrix_bytes']/2**20:.1f}"]
    for key, e in manifest["embeddings"].items() if key in ("ocr", "image")
])
print(json.dumps(read_json("audit.json"), indent=2))""")
    md("See [FEASIBILITY.md](FEASIBILITY.md) for the recommendation, limits, and follow-up design. "
       "The benchmark evaluates retrieval. A shared answer reader could be added in a separate end-to-end experiment.")

    # Execute the display cells and save their outputs for offline review.
    from os import chdir

    chdir(ROOT)
    namespace = {}
    number = 0
    for cell in cells:
        if cell.cell_type != "code":
            continue
        number += 1
        output = []

        def capture(value):
            output.append(nbf.v4.new_output(
                "display_data", data={"text/html": value._repr_html_()}, metadata={},
            ))

        namespace["display"] = capture
        buf = StringIO()
        with redirect_stdout(buf):
            exec(compile(cell.source, f"<notebook cell {number}>", "exec"), namespace)
        if buf.getvalue():
            output.append(nbf.v4.new_output("stream", name="stdout", text=buf.getvalue()))
        cell.outputs = output
        cell.execution_count = number
    nb = nbf.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    })
    nbf.validate(nb)
    nbf.write(nb, ROOT / "ViDoRe_Finance.ipynb")


def package():
    root_files = sorted(path for path in ROOT.iterdir() if path.is_file())
    result_files = sorted((ROOT / "results").glob("*"))
    files = root_files + [p for p in result_files if p.is_file()]
    zip_path = ROOT.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            archive.write(path, path.relative_to(ROOT.parent).as_posix())
    with zipfile.ZipFile(zip_path) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist()) == len(files)
        assert not any(".cache/" in name or name.endswith(".safetensors") for name in archive.namelist())
    print(f"Packaged {len(files)} files in {zip_path} ({zip_path.stat().st_size:,} bytes).")


if __name__ == "__main__":
    print(audit())
    paired()
    brief()
    notebook()
    package()
