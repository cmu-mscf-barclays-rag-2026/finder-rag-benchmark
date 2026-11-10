"""Verify the fixed Holm10 family and rebuild charts from included CSVs.

No API, corpus, model, embedding, credential or experimental runner is loaded.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "meeting9-findings-matplotlib"))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
COLORS = {"question": "#929eaa", "focused": "#116b85", "short": "#d09442", "bm25": "#554d69"}


def table(name):
    # Python's float parser preserves the CSV round trip without relying on
    # pandas' deprecated float_precision argument.
    with (DATA / name).open(newline="") as f:
        records = list(csv.DictReader(f))
    frame = pd.DataFrame(records)
    for column in frame:
        try:
            values = [float(v) for v in frame[column]]
        except ValueError:
            continue
        frame[column] = values
    return frame


def verify_statistics():
    frame = table("paired_ndcg_differences.csv")
    tests = table("primary_tests.csv")
    assert len(frame) == 249 and len(tests) == 10
    assert frame.query_id.nunique() == 249
    matrix = frame[tests.id.tolist()].to_numpy()
    assert matrix.shape == (249, 10) and np.isfinite(matrix).all()
    assert np.allclose(matrix.mean(axis=0), tests.mean_delta, atol=1e-12, rtol=0)
    for j, r in enumerate(tests.itertuples()):
        d = matrix[:, j]
        assert (int((d > 1e-12).sum()), int((d < -1e-12).sum()), int((abs(d) <= 1e-12).sum())) == (r.wins, r.losses, r.ties)
    rng = np.random.default_rng(20261109)
    observed = abs(matrix.sum(axis=0))
    counts = np.zeros(10, dtype=np.int64)
    stream = hashlib.sha256()
    for _ in range(100):
        signs = rng.integers(0, 2, size=(1000, 249), dtype=np.int8) * 2 - 1
        stream.update(signs.tobytes())
        counts += (abs(signs @ matrix) >= observed - 1e-12).sum(axis=0)
    raw = (counts + 1) / 100001
    order = np.argsort(raw, kind="stable")
    adjusted = np.empty(10)
    adjusted[order] = np.minimum(1, np.maximum.accumulate((10 - np.arange(10)) * raw[order]))
    assert np.array_equal(counts, tests.extreme_count)
    assert np.allclose(raw, tests.p_raw, atol=1e-15, rtol=0)
    assert np.allclose(adjusted, tests.p_holm10, atol=1e-15, rtol=0)
    saved = json.loads((DATA / "signflip_audit.json").read_text())
    assert stream.hexdigest() == saved["sign_matrix_stream_sha256"]
    assert hashlib.sha256((DATA / "paired_ndcg_differences.csv").read_bytes()).hexdigest() == saved["difference_csv_sha256"]
    acceptance = json.loads((DATA / "reader_acceptance.json").read_text())
    assert not acceptance["usable_as_primary_answer_quality_results"]
    assert not acceptance["q178_abstention_rule"]["passed"]
    assert acceptance["human_reviewed"] == 0
    return {"passed": True, "paired_questions": 249, "fixed_family_size": 10,
            "draws": 100000, "seed": 20261109, "sign_stream_sha256": stream.hexdigest(),
            "reader_accuracy_claim": False, "api_calls": 0, "encoder_calls": 0}


def save(fig, out, name, bottom=.16, top=.90):
    fig.tight_layout(rect=[0, bottom, 1, top])
    fig.savefig(out / name, dpi=180, facecolor="white")
    plt.close(fig)


def hybrid_chart(out):
    d = table("hybrid_all_k.csv")
    d = d[(d.cohort == "remaining249") & (d.k == 10)].set_index("method")
    methods = [f"{q}_bank_rrf_w{w}" for q in ["question", "focusedmix", "shortmix"] for w in ["025", "050"]]
    labels = [f"{q} Hybrid  {w}" for q in ["Question", "Focused HyDE", "Short HyDE"] for w in ["0.25", "0.50"]]
    values = [d.loc[m, "ndcg"] for m in methods]
    fig, ax = plt.subplots(figsize=(8.8, 4.9))
    bars = ax.barh(labels, values, color=[COLORS[k] for k in ["question", "question", "focused", "focused", "short", "short"]])
    ax.invert_yaxis()
    for bar, value in zip(bars, values):
        ax.text(value + .004, bar.get_y() + bar.get_height()/2, f"{value:.4f}", va="center", fontsize=10)
    baseline = d.loc["bm25", "ndcg"]
    ax.axvline(baseline, color=COLORS["bm25"], linestyle="--", linewidth=1.6, label=f"BM25 {baseline:.4f}")
    ax.set(xlim=(0, .70), xlabel="nDCG@10", title="Focused HyDE + BM25 improves financial-report retrieval")
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=.12)
    ax.set_axisbelow(True)
    fig.text(.02, .06, "249 queries; same fixed bank filter. All six frozen hybrids shown; numbers are dense RRF weights.", fontsize=8)
    fig.text(.02, .025, "Focused .50 vs BM25: Holm10 p=.00328; vs Question .50: p=.00045. Exploratory question-level tests.", fontsize=8)
    save(fig, out, "main_results.png", bottom=.18)


def control_chart(out):
    c = table("filter_control.csv")
    c = c[(c.cohort == "remaining249") & (c.k == 10)]
    r = table("representation_control.csv")
    r = r[(r.cohort == "full309") & (r.k == 10)]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    x = np.arange(2)
    for j, (rep, label, color) in enumerate([("question", "Question Hybrid", COLORS["question"]), ("focusedmix", "Focused HyDE Hybrid", COLORS["focused"])]):
        values = [c[(c["filter"] == f) & (c.representation == rep)].iloc[0].ndcg for f in ["no_bank_filter", "bank_filter"]]
        bars = axes[0].bar(x + (j-.5)*.34, values, .34, color=color, label=label)
        for b, v in zip(bars, values):axes[0].text(b.get_x()+b.get_width()/2, v+.012, f"{v:.3f}", ha="center", fontsize=9)
    axes[0].set(xticks=x, xticklabels=["Filter off", "Filter on"], ylabel="nDCG@10", ylim=(0, .76), title="Bank filter × HyDE: 249 queries")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    x = np.arange(3)
    reps = [("page_first256", "First 256 tokens", "#b6c2cc"), ("fullpage_chunkmax", "Full OCR chunk-max", COLORS["focused"])]
    for j, (rep, label, color) in enumerate(reps):
        values = [r[(r.method == m) & (r.representation == rep)].iloc[0].ndcg for m in ["question_dense", "focused160_mix_dense", "short_30_50_mix_dense"]]
        bars = axes[1].bar(x + (j-.5)*.34, values, .34, color=color, label=label)
        for b, v in zip(bars, values):axes[1].text(b.get_x()+b.get_width()/2, v+.012, f"{v:.3f}", ha="center", fontsize=9)
    axes[1].set(xticks=x, xticklabels=["Question", "Focused mix", "Short mix"], ylim=(0, .65), title="Document representation: 309 queries")
    axes[1].legend(frameon=False, fontsize=8, loc="upper left")
    for ax in axes:
        ax.grid(axis="y", alpha=.12)
        ax.set_axisbelow(True)
    fig.text(.5, .025, "Different cohorts. Descriptive controls: OCR coverage and max-chunk aggregation change together; no interaction p-value.", ha="center", fontsize=8)
    save(fig, out, "controls.png", bottom=.14)


def depth_chart(out):
    d = table("hybrid_all_k.csv")
    d = d[d.cohort == "remaining249"]
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    for method, label, color in [("bm25", "BM25", COLORS["bm25"]), ("question_bank_rrf_w050", "Question Hybrid .50", COLORS["question"]), ("focusedmix_bank_rrf_w050", "Focused HyDE Hybrid .50", COLORS["focused"])]:
        s = d[d.method == method].sort_values("k")
        ax.plot(s.k, s.hit, marker="o", label=label, color=color)
    ax.set(xticks=[5, 10, 15, 20, 25], xlabel="Top-k", ylabel="Hit@k", ylim=(.78, 1.01), title="HyDE Hybrid's descriptive advantage extends beyond Top-5")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=.15)
    fig.text(.5, .035, "249 queries; same bank filter. Page-level relevance, not answer accuracy. No extra Top-k significance tests.", ha="center", fontsize=8)
    save(fig, out, "top_k.png", bottom=.17)


def hnsw_chart(out):
    d = table("hnsw.csv")
    d = d[d.representation == "short_mix50"]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.3))
    for ax, y, label in [(axes[0], "neighbor_recall10", "Exact-neighbor Recall@10"), (axes[1], "ndcg10", "Qrel nDCG@10")]:
        for group, color, marker in [("exact", COLORS["bm25"], "s"), ("hnsw_m16", COLORS["short"], "o"), ("hnsw_m32", COLORS["focused"], "^")]:
            g = d[d.configuration == group] if group == "exact" else d[d.configuration.str.startswith(group)]
            ax.plot(g.latency_p50_ms, g[y], marker=marker, color=color, label=group.replace("hnsw_", "HNSW ").upper())
        ax.set(xlabel="Warm search p50 (ms; log scale)", ylabel=label, xscale="log")
        ax.grid(alpha=.15)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("FiQA HNSW: local search speed, neighbor recovery and relevance")
    fig.text(.5, .025, "628 Short-mix queries. Index search only; excludes generation and encoding. All six fixed HNSW settings shown.", ha="center", fontsize=8)
    save(fig, out, "hnsw.png", bottom=.19, top=.88)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=HERE / "reproduced")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    result = verify_statistics()
    hybrid_chart(args.out)
    control_chart(args.out)
    depth_chart(args.out)
    hnsw_chart(args.out)
    (args.out / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
