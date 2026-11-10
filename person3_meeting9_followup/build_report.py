"""Integrate frozen follow-up artifacts; offline, no model or API calls.

This script does not rerun experiments or alter component artifacts. It explicitly
keeps retrieval249 and reader20 separate, and sums latency per query before taking
quantiles. Existing reader outputs are also the inputs to the repaired v2 judge.
"""
from pathlib import Path
import csv
import hashlib
import json
import os
import re
import sqlite3

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent
RESULT = ROOT / "results/person3_meeting9_followup"
OUT = BASE / "outputs/meeting9_followup_20261109"
METHODS = ["bm25", "question_bank_rrf_w050", "focusedmix_bank_rrf_w050", "no_context"]
LABELS = dict(zip(METHODS, ["BM25", "Question Hybrid .50", "Focused HyDE Hybrid .50", "No-context control"]))


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def preservation():
    checks = []
    for name, root in [("full_results_hashes.json", ROOT), ("meeting9_followup_prior_hashes.json", BASE)]:
        expected = load(BASE / "work" / name)
        missing, changed = [], []
        for rel, digest in expected.items():
            p = root / rel
            if not p.is_file():
                missing.append(rel)
            elif sha(p) != digest:
                changed.append(rel)
        checks.append({"manifest": name, "files": len(expected), "missing": missing, "changed": changed})
    assert all(not c["missing"] and not c["changed"] for c in checks), checks
    dump(OUT / "historical_preservation.json", {"passed": True, "checks": checks})
    return checks


def budget():
    parts = []
    total = 2.4225025
    for name, path in [
        ("prior Meeting9 workflow", ROOT / "results/person3_meeting9_workflow/v1/api_checkpoint.sqlite"),
        ("reader plus rejected joint-rater v1", RESULT / "evidence_v1/api_checkpoint.sqlite"),
        ("isolated judge repair v2", RESULT / "evidence_isolated_v2/api_checkpoint.sqlite"),
    ]:
        assert path.exists(), path
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
            rows = db.execute("SELECT state,count(*),sum(coalesce(cost,reserve)) FROM calls GROUP BY state").fetchall()
        states = {r[0]: r[1] for r in rows}
        assert set(states) <= {"saved", "unknown_reserved_acknowledged"}, states
        value = sum(r[2] for r in rows)
        parts.append({"stage": name, "cost_plus_reserves_usd": value, "states": states})
        total += value
    assert total <= 7
    result = {"historical_prior_usd": 2.4225025, "parts": parts,
              "cumulative_estimate_including_reserves_usd": total, "guard_usd": 7,
              "this_round_usd": sum(x["cost_plus_reserves_usd"] for x in parts[1:]),
              "cached_discount_ignored": True, "billing_statement": False,
              "prices_per_million": {"input": 2.5, "output": 15}}
    dump(OUT / "api_budget.json", result)
    return result


def reader_cost(pd, np):
    """Logical standalone costs, not double-counted experiment billing."""
    logical = load(RESULT / "evidence_v1/successful_api_outputs.json")
    assert len(logical) == 80
    costs = pd.read_csv(RESULT / "costs_v1/per_query_quality_cost_latency.csv", dtype={"query_id": str})
    assert not costs.duplicated(["query_id", "method"]).any()
    by_key = costs.set_index(["query_id", "method"])
    rows = []
    for r in logical:
        qid, method = r["query_id"], r["style"]
        if method == "no_context":
            retrieval = encoding = historical_generation = generation_cost = 0.
        else:
            x = by_key.loc[(qid, method)]
            retrieval = float(x.retrieval_total_seconds)
            encoding = float(x.encoding_stage_sum_seconds)
            historical_generation = float(x.historical_generation_seconds)
            generation_cost = float(x.estimated_generation_usd_uncached_query)
        rcost = (r["usage"]["input_tokens"] * 2.5 + r["usage"]["output_tokens"] * 15) / 1e6
        rows.append({"query_id": qid, "method": method, "reader_key": r["key"],
                     "reader_input_tokens": r["usage"]["input_tokens"],
                     "reader_output_tokens": r["usage"]["output_tokens"],
                     "estimated_reader_usd_standalone": rcost,
                     "estimated_generation_usd_fresh": generation_cost,
                     "estimated_fresh_generation_plus_reader_usd": generation_cost + rcost,
                     "cached_vector_retrieval_seconds": retrieval,
                     "encoding_stage_sum_seconds": encoding,
                     "historical_generation_seconds": historical_generation,
                     "observed_reader_request_seconds": r["latency_seconds"],
                     "modeled_fresh_retrieval_plus_reader_seconds": retrieval + encoding + historical_generation + r["latency_seconds"],
                     "reader_request_concurrency": r["request_concurrency"],
                     "e2e_measured": False})
    df = pd.DataFrame(rows)
    assert set(df.reader_request_concurrency) == {2}
    df.to_csv(OUT / "reader20_quality_cost_latency_per_query.csv", index=False)
    aggregates = []
    for method in METHODS:
        d = df[df.method == method]
        assert len(d) == 20
        row = {"method": method, "n": len(d)}
        for col in ["estimated_reader_usd_standalone", "estimated_generation_usd_fresh", "estimated_fresh_generation_plus_reader_usd"]:
            row["mean_" + col] = d[col].mean()
        for col in ["observed_reader_request_seconds", "modeled_fresh_retrieval_plus_reader_seconds"]:
            for q in [50, 95]:
                row[col + f"_p{q}"] = np.percentile(d[col], q)
        aggregates.append(row)
    summary = pd.DataFrame(aggregates)
    summary.to_csv(OUT / "reader20_cost_latency.csv", index=False)
    unique = {r["key"]: r for r in logical}
    assert len(unique) == 66
    actual_cost = sum((r["usage"]["input_tokens"] * 2.5 + r["usage"]["output_tokens"] * 15) / 1e6 for r in unique.values())
    dump(OUT / "reader20_cost_accounting.json", {
        "logical_answers": 80, "unique_paid_reader_calls": 66,
        "actual_reader_cost_estimate_usd": actual_cost,
        "table_interpretation": "Mean standalone method cost counts each logical query once if that method were deployed alone; not summed for actual experiment billing. Judge evaluation overhead is excluded.",
        "latency_interpretation": "Per-query sum of separately observed stages, then cohort quantiles. Historical HyDE generation plus warm local profiling plus new reader request duration is modeled, never measured E2E.",
        "reader_concurrency": 2, "human_reviewed": 0})
    return summary, actual_cost


def reader_mechanics(pd):
    """Direct string/ID diagnostics, deliberately not answer-quality ratings."""
    prior = RESULT / "evidence_v1"
    inputs = {(r["query_id"], r["method"]): r for r in load(prior / "context_records.json")}
    refs = {r["query_id"]: r["original_qrels"] for r in load(prior / "reference_answers_judge_only.json")}
    rows = []
    for answer in load(prior / "successful_api_outputs.json"):
        qid, method = answer["query_id"], answer["style"]
        context = inputs[qid, method]
        ids = {s["passage_id"] for s in context["delivered_spans"]}
        cites = re.findall(r"\[P(\d+)\]", answer["text"])
        rows.append({"query_id": qid, "method": method,
                     "relevant_page_prefix_present": any(refs[qid].get(pid, 0) > 0 for pid in ids),
                     "reader_has_page_citation": bool(cites),
                     "all_cited_ids_in_own_delivered_context": bool(cites) and all(pid in ids for pid in cites),
                     "starts_insufficient_evidence": bool(re.match(r"^\s*insufficient evidence\b", answer["text"], re.I)),
                     "used_wordpiece_tokens": context["used_wordpiece_tokens"],
                     "delivered_page_prefixes": len(ids), "human_status": "pending"})
    df = pd.DataFrame(rows)
    assert len(df) == 80
    df.to_csv(OUT / "reader20_mechanical_per_query.csv", index=False)
    fields = ["relevant_page_prefix_present", "reader_has_page_citation", "all_cited_ids_in_own_delivered_context", "starts_insufficient_evidence"]
    result = df.groupby("method").agg(n=("query_id", "size"), **{f: (f, "sum") for f in fields}).loc[METHODS].reset_index()
    result.to_csv(OUT / "reader20_mechanical_summary.csv", index=False)
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(OUT / ".mplcache"))
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    checks = preservation()
    funds = budget()
    i = RESULT / "inference_v1"
    e = RESULT / "evidence_isolated_v2"
    completion = load(e / "completion_status.json")
    assert completion["engineering_complete"]
    assert not completion["critical_high_repair_checks_passed"], "Update report explicitly if the frozen rejection gate changes."
    assert completion["human_reviewed"] == 0
    cells = pd.read_csv(i / "two_by_two_summary.csv")
    cells = cells[(cells.cohort == "remaining249") & (cells.k == 10)]
    assert len(cells) == 4
    cells.to_csv(OUT / "retrieval249_filter_control.csv", index=False)
    tests = pd.read_csv(i / "primary_tests.csv")
    assert len(tests) == 10 and set(tests.n) == {249}
    tests.to_csv(OUT / "retrieval249_complete_test_family.csv", index=False)
    contrasts = pd.read_csv(i / "two_by_two_contrasts.csv")
    interaction = contrasts[(contrasts.cohort == "remaining249") & (contrasts.k == 10) &
                            (contrasts.contrast == "interaction_filter_by_hyde")].iloc[0].ndcg_delta
    # Both automated raters failed predeclared criteria. Do not promote their
    # diagnostic summaries into an accuracy table or chart.
    evidence = reader_mechanics(pd)
    assert set(evidence.n) == {20}
    quality_cost = pd.read_csv(RESULT / "costs_v1/quality_cost_table.csv")
    quality_cost = quality_cost[quality_cost.cohort == "remaining249"]
    quality_cost.to_csv(OUT / "retrieval249_cost_latency.csv", index=False)
    reader_summary, reader_fee = reader_cost(pd, np)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    palette = ["#7e8997", "#14648b"]
    x = np.arange(2)
    for j, rep in enumerate(["question", "focusedmix"]):
        vals = [float(cells[(cells["filter"] == f) & (cells.representation == rep)].iloc[0].ndcg)
                for f in ["no_bank_filter", "bank_filter"]]
        bars = axes[0].bar(x + (j - .5) * .34, vals, .34, color=palette[j], label=["Question Hybrid", "Focused HyDE Hybrid"][j])
        for bar, v in zip(bars, vals):
            axes[0].text(bar.get_x() + bar.get_width()/2, v + .012, f"{v:.3f}", ha="center", fontsize=10)
    axes[0].set_xticks(x, ["Bank filter off", "Bank filter on"])
    axes[0].set_ylim(0, .75)
    axes[0].set_ylabel("nDCG@10")
    axes[0].set_title("Retrieval: remaining 249 questions\nRRF dense weight 0.50", fontsize=11)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    values = evidence.relevant_page_prefix_present.to_numpy()
    bars = axes[1].bar(np.arange(4), values, color=["#7e8997", "#86a5b8", "#14648b", "#c7c7c7"])
    for bar, v in zip(bars, values):
        axes[1].text(bar.get_x()+bar.get_width()/2, v+.4, f"{v:.0f}/20", ha="center", fontsize=10)
    axes[1].set_xticks(np.arange(4), ["BM25", "Question\nHybrid", "Focused\nHyDE Hybrid", "No context"])
    axes[1].set_ylim(0, 23)
    axes[1].set_yticks([0, 5, 10, 15, 20])
    axes[1].set_ylabel("Questions with a relevant page prefix delivered")
    axes[1].set_title("Reader context: fixed 20-question sample\nPage-ID proxy, not evidence sufficiency", fontsize=11)
    for ax in axes:
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("HyDE follow-up: retrieval gains and evidence-level limits", fontsize=13)
    fig.tight_layout(rect=[0, .10, 1, .91])
    fig.text(.5, .035, "Different cohorts. A delivered page prefix can omit required facts. Answer accuracy is unvalidated; human reviewed: 0.", ha="center", fontsize=8)
    fig.savefig(OUT / "overview.png", dpi=190)
    fig.savefig(OUT / "overview.pdf")
    plt.close(fig)

    text = ["# Meeting 9 follow-up: tasks 1–4", "",
            "Tasks1,2 and4 are complete. Task3 reader execution and scoring attempts are saved, but both automatic raters failed acceptance checks; reliable answer-quality measurement awaits human adjudication. All new experiments use the existing viDoRe v3 Finance English data; no historical full experiment was rerun.", "",
            "| Task | Result | Evidence |", "|---|---|---|",
            "| 1. Bank-filter dependence | Focused HyDE Hybrid improves with and without the fixed bank filter. | 2×2 control below |",
            "| 2. Statistical robustness | Focused hybrids pass the frozen, exploratory Holm10 family; shared-report and adaptive-history limits remain. | Full 10-test table below |",
            "| 3. Fixed-reader/evidence |80 logical answers from66 reader calls saved. Both joint and isolated AI raters rejected; reliable answer accuracy is pending. Human review:0. | Mechanical diagnostics and blind human packet |",
            "| 4. Quality/cost/latency | Local stages profiled under common settings; historical generation and new reader durations separately retained. | Measured and modeled cost tables below |", "",
            "![Retrieval and reader-context diagnostics](overview.png)", "",
            "## 1. Does improvement depend on bank filtering?", "",
            "Remaining249, complete original qrels, RRF dense weight0.50; question–HyDE embedding mix α=0.50. The initial60 had been used in development; neither cohort is a new untouched holdout.", "",
            "| Bank filter | Query representation | nDCG@10 | Hit@10 | Recall@10 |", "|---|---|---:|---:|---:|"]
    for row in cells.itertuples():
        text.append(f"| {'On' if row.filter == 'bank_filter' else 'Off'} | {'Focused HyDE mix' if row.representation == 'focusedmix' else 'Original question'} | {row.ndcg:.4f} | {row.hit:.1%} | {row.recall:.4f} |")
    text += ["", f"HyDE gain: +0.0316 without the filter and +0.0360 with it. Descriptive interaction: **{interaction:+.4f}**; no interaction p-value or independent interaction claim. Only the missing unfiltered Focused Hybrid cell was added; the other three were reused.", "",
             "## 2. Statistical robustness", "",
             "All methods in this test family, including BM25 controls, use the same frozen bank filter. Two-sided paired sign-flip tests,100,000 shared draws, seed20261109, nDCG@10 on remaining249. Holm covers exactly the10 comparisons frozen before these new tests. This is **exploratory question-level inference**: six shared annual reports create dependence, and Holm10 does not correct earlier adaptive experiments. No clustered significance, bootstrap interval or new holdout confirmation is claimed.", "",
             "| Method | Control | Δ nDCG@10 | W/L/T | Raw p | Holm10 p |", "|---|---|---:|---:|---:|---:|"]
    for r in tests.itertuples():
        def label(m):
            return {"bm25": "BM25", "question_bank_rrf_w025": "Question Hybrid .25", "question_bank_rrf_w050": "Question Hybrid .50", "focusedmix_bank_rrf_w025": "Focused Hybrid .25", "focusedmix_bank_rrf_w050": "Focused Hybrid .50", "shortmix_bank_rrf_w025": "Short Hybrid .25", "shortmix_bank_rrf_w050": "Short Hybrid .50"}.get(m, m)
        text.append(f"| {label(r.method)} | {label(r.control)} | {r.mean_delta:+.4f} | {r.wins}/{r.losses}/{r.ties} | {r.p_raw:.5f} | {r.p_holm10:.5f} |")
    text += ["", "Leaving out each report also removes multi-source queries involving that report. Focused .50 stays above BM25 in all six descriptive exclusions (Δ+0.0282 to+0.0490), and above Question .50 (Δ+0.0289 to+0.0434). This is sensitivity evidence, not independent source-level inference. [All K=5/10/15/20/25 and source sensitivities](inference/README.md).", "",
             "## 3. Fixed reader and evidence", "",
             "Same20 questions selected by SHA256 before new outcomes, all four methods (4 from initial60,16 from remaining249). All three retrieval methods use the same frozen bank filter. Top10 real OCR prefixes share a1536 MiniLM WordPiece budget, in ranking order. Fixed reader: gpt-5.4-2026-03-05, temperature0, reasoning none, max256 output tokens. Reader never receives references, qrels or generated HyDE evidence. No-context uses the same evidence-only instruction, so it measures evidence-free response/abstention rather than unrestricted model knowledge.", "",
             "The original joint AI judge violated the missing-citation rubric and showed an own-context mismatch on q178. Its raw results are retained as **rejected rater calibration evidence**. The repair reuses every saved reader answer, with one candidate and only its own context per judge request:80 logical /66 exact-prompt unique judges. Reference answers are for consistency/coverage only. No new reader call, question selection or method tuning was done.", "",
             "**The isolated repair also failed the criterion declared before its outcomes.** On q178, the two identical hybrid answers say the requested percentages are unavailable. The isolated judge correctly describes that abstention in its reason, but labels it `answer` and gives full C/V/G/T=2/2/2/2. That is one unique bad judgment affecting two logical method rows. Its own-context access is isolated, but it still confuses justified abstention with a complete reference-consistent answer. Both raters are rejected for answer-quality reporting. Raw/guard scores are retained as failure diagnostics; no further paid prompt-tuning loop is run.", "",
             "**Human-reviewed0/pending. No valid final-answer accuracy or superiority conclusion is reported.** The table below contains direct page-ID/string checks only:", "",
             "| Method | n | Relevant page prefix delivered | All cited page IDs in own context | Starts with `Insufficient evidence` |", "|---|---:|---:|---:|---:|"]
    for r in evidence.itertuples():
        text.append(f"| {LABELS[r.method]} | {r.n} | {r.relevant_page_prefix_present}/{r.n} | {r.all_cited_ids_in_own_delivered_context}/{r.n} | {r.starts_insufficient_evidence}/{r.n} |")
    text += ["", "A positive labeled-page prefix can end before the needed fact; page presence does not establish evidence completeness. A valid page ID does not establish claim support, and an abstention prefix can precede explanatory facts. These checks are not answer ratings. No-context has no citations by design.",
             "Retrieval gains cannot be assumed to translate into better final answers. Same-model reader/judge bias, vague questions, incomplete OCR prefixes and released-reference issues remain. Three v1 reference-issue flags are retained for adjudication; flags are not exclusions. No significance test is added to this20-question answer study.", "",
             "[Rejected isolated-rater diagnostics](evidence_isolated/README.md) · [Original blind human review packet](evidence/blinded_human_review.html) · [Six bounded failure-focused AI case reviews](case_review/README.md). The case review is not a new accuracy estimate. To complete task3 scientifically, a human must review the80 blinded answer/context pairs and adjudicate reference issues; earlier Legal fact/quote labels also remain AI drafts with human review pending.", "",
             "## 4. Quality, API cost and latency", "",
             "Remaining249; all three methods use the same frozen bank filter. Warm resident CPU, one thread, batch1,20 warmups, three interleaved repeats. Query-local repeat medians are calculated before adding stages and taking cohort quantiles. No new generations were needed for retrieval/cost tasks. Costs below describe fresh generation; cached-generation incremental API cost is0.", "",
             "| Method | nDCG@10 | Fresh generation USD/query | Cached-vector retrieval p50/p95 ms | Modeled fresh query p50/p95 s |", "|---|---:|---:|---:|---:|"]
    for r in quality_cost.itertuples():
        text.append(f"| {LABELS[r.method]} | {r.ndcg10:.4f} | {r.mean_estimated_generation_usd_uncached:.6f} | {r.retrieval_total_seconds_p50*1000:.3f}/{r.retrieval_total_seconds_p95*1000:.3f} | {r.modeled_uncached_query_seconds_p50:.4f}/{r.modeled_uncached_query_seconds_p95:.4f} |")
    text += ["", "Cached-vector retrieval timings exclude encoding. Modeled fresh query totals add locally profiled encoding/retrieval to retained historical HyDE request durations; **they are not measured end-to-end latency** and exclude reader, queue and process/index loading. Generation dominates the local millisecond stages. See [timing, usage, index build/storage and quality chart](costs/README.md).", "",
             "For the fixed reader20 sample, standalone per-method mean costs and stage-sum latency are:", "",
             "| Method | Mean reader USD/query | Mean fresh generation + reader USD/query | Modeled retrieval + reader p50/p95 s |", "|---|---:|---:|---:|"]
    for r in reader_summary.itertuples():
        text.append(f"| {LABELS[r.method]} | {r.mean_estimated_reader_usd_standalone:.6f} | {r.mean_estimated_fresh_generation_plus_reader_usd:.6f} | {r.modeled_fresh_retrieval_plus_reader_seconds_p50:.3f}/{r.modeled_fresh_retrieval_plus_reader_seconds_p95:.3f} |")
    text += ["", f"These reader20 modeled totals sum per-query stages before quantiles. They combine historical generation, separate warm local timings and newly observed reader request durations (concurrency2), so they are not an end-to-end benchmark. Standalone logical method costs are not summed for experiment billing: exact duplicate reader prompts shared66 actual paid calls costing **${reader_fee:.6f}**. Judges are evaluation overhead, excluded from deployment cost tables.", "",
             f"This round's actual API estimate including evaluation: **${funds['this_round_usd']:.6f}**. Cumulative project estimate plus unresolved historical reserves: **${funds['cumulative_estimate_including_reserves_usd']:.6f}/$7**. Conservative token rates are $2.50/M input and $15/M output; cached discounts ignored. This is a software estimate, not a billing statement. [Official model/pricing reference](https://developers.openai.com/api/docs/models/gpt-5.4). [Ledger breakdown](api_budget.json).", "",
             "## Validation and limits", "",
             "Independent cross-checks reconstructed all18,540 retrieval metric rows, the complete100,000-draw sign stream/Holm10 family,2×2 controls and source sensitivities; costs reconcile778 retained historical calls,2,781 retrieval timings and1,854 encoder timings. The joint-rater failure and isolated-rater failure were independently confirmed and preserved. Engineering checks do not certify semantic accuracy; task3's reliable human answer-quality assessment remains incomplete.", "",
             f"Historical preservation: {checks[0]['files']} original and {checks[1]['files']} more-recent artifacts checked by SHA256, no missing or changed files. [Preservation receipt](historical_preservation.json) · [Original independent audit](audit/audit.json) · [Independent cost audit](cost_audit/README.md) · [Independent reader-cost integration audit](integration_cost_audit/README.md) · [Isolated repair and final integration audit](audit/isolation_and_integration_audit.json).", "",
             "Data-quality disclosure: original query39 contains an appended `Answer:` hint. It is retained in historical results, lies in initial60 and is absent from the new reader20 sample; excluding it leaves remaining249 unchanged. This marker check is not an exhaustive leakage audit. Images/visual retrieval and unrestricted full-context answering are outside tasks1–4.", "",
             "## Reproduce without new API calls", "",
             "Run from the existing repository with retained artifacts and .venv:", "",
             "```bash",
             ".venv/bin/python person3_meeting9_followup/inference/run.py validate",
             ".venv/bin/python person3_meeting9_followup/evidence/verify.py",
             ".venv/bin/python person3_meeting9_followup/evidence_isolated/run.py verify",
             ".venv/bin/python person3_meeting9_followup/build_report.py",
             "```", "",
             "The integration script reads saved artifacts only and verifies both historical manifests; it never imports an API client or calls an encoder. Component protocols, exact inputs, source/model revisions, saved outputs and durable SQLite ledgers remain under `results/person3_meeting9_followup/`. Costs can be recomputed from retained raw usage/timing tables; do not reprofile or repeat successful paid calls merely to reproduce aggregate scores. Any interrupted approved queue resumes only missing keys and halts on unresolved attempts. Credentials stay local in `.env.local` and must never be added to reports or transfer bundles.", ""]
    (OUT / "README.md").write_text("\n".join(text))
    dump(OUT / "completion.json", {
        "tasks_1_2_4_executed": True, "task_3_reader_and_isolated_ai_evaluation_executed": True,
        "task_3_reliable_answer_accuracy_complete": False,
        "automated_answer_raters": "both rejected after predeclared acceptance failures; no accuracy claim",
        "human_validated_accuracy_complete": False, "human_reviewed": 0,
        "final_independent_integrated_audit": "see audit/isolation_and_integration_audit.json when completed",
        "historical_preservation_passed": True, "new_api_calls_by_this_script": 0,
        "new_encoder_calls_by_this_script": 0, "script_sha256": sha(__file__),
        "integrated_output_hashes": {n: sha(OUT / n) for n in ["README.md", "overview.png", "overview.pdf", "retrieval249_filter_control.csv", "retrieval249_complete_test_family.csv", "reader20_mechanical_summary.csv", "reader20_mechanical_per_query.csv", "retrieval249_cost_latency.csv", "reader20_cost_latency.csv", "reader20_quality_cost_latency_per_query.csv", "api_budget.json", "historical_preservation.json"]}})
    print(json.dumps({"report": str(OUT / "README.md"), "cumulative_estimate_usd": funds["cumulative_estimate_including_reserves_usd"], "human_reviewed": 0}, indent=2))


if __name__ == "__main__":
    main()
