"""Compare fusion across fixed MultiQuery rewrites on the ViDoRe English split."""

from collections import defaultdict
from statistics import mean

import numpy as np

from compare_fusion import PRIOR, combine_z_scores, comparison_arguments
from benchmark_core import metric_values, save_json, write_csv
from experiment import Experiment, read, rrf


def modality_scores(ocr_ranking, image_ranking, ocr_weight):
    """Expose the existing first-stage weighted RRF scores for a single query."""
    scores = defaultdict(float)
    for ranking, weight in ((ocr_ranking, ocr_weight), (image_ranking, 1 - ocr_weight)):
        for rank, page in enumerate(ranking[:100], 1):
            scores[page] += weight / (60 + rank)
    ranking = sorted(scores, key=lambda page: (-scores[page], page))
    return ranking, np.array([scores[page] for page in ranking], dtype=np.float64)


def scored_candidates(experiment, text, backend, baseline, depth):
    if backend == "combined":
        ocr = experiment.rank(text, "ocr", baseline)
        image = experiment.rank(text, "image", baseline)
        ranking, scores = modality_scores(ocr, image, baseline["ocr_weight"])
        return ranking[:depth], scores[:depth]

    if backend == "bm25":
        scores = experiment.sparse_model(baseline["k1"], baseline["b"]).scores(text)
    else:
        scores = experiment.matrices[backend] @ experiment.vector(text)
    page_ids = np.asarray(experiment.page_ids)
    indices = np.lexsort((page_ids, -scores))[:depth]
    return page_ids[indices].tolist(), scores[indices].astype(np.float64)


def compare(args):
    experiment = Experiment(args)
    previous = read(PRIOR / "results/rankings.json")
    selection = read(PRIOR / "results/selection.json")
    baseline = selection["baseline"]
    configs = {backend: settings["multiquery"] for backend, settings in selection["expansion"].items()}
    per_query = []
    summary = defaultdict(list)
    verified_rankings = 0
    for split in ("validation", "test"):
        for question in experiment.partition(split, english=True):
            qid = question["query_id"]
            for backend, config in configs.items():
                generated = read(args.cache / "generated" / config["strategy"] / f"{qid}.json")["queries"]
                texts = generated[:config["count"]]
                weights = [1.0] * len(texts)
                if config["original_weight"]:
                    texts.insert(0, question["query"])
                    weights.insert(0, config["original_weight"])
                depth = config["depth"]
                candidates = [
                    scored_candidates(experiment, text, backend, baseline, depth) for text in texts
                ]
                lists = [pages for pages, _ in candidates]
                previous_rrf = rrf(lists, weights, config["rrf_constant"], depth)
                if previous_rrf[:20] != previous[str(qid)][f"{backend}/multiquery"]:
                    raise ValueError(f"Baseline MultiQuery ranking changed for {backend}, query {qid}.")
                verified_rankings += 1

                combsum, combmnz = combine_z_scores(candidates, weights)
                _, paper_combmnz = combine_z_scores(candidates)
                rankings = {
                    "original_query": previous[str(qid)][f"{backend}/original"],
                    "multiquery_rrf_previous": previous_rrf,
                    "multiquery_combsum_zmuv_weighted": combsum,
                    "multiquery_combmnz_zmuv_weighted": combmnz,
                    "multiquery_rrf_equal": rrf(lists, [1.0] * len(lists), config["rrf_constant"], depth),
                    "multiquery_combmnz_zmuv_equal": paper_combmnz,
                }
                for method, ranked_pages in rankings.items():
                    for k in (5, 8, 10):
                        scores = metric_values(ranked_pages, experiment.gold[qid], k)
                        row = {
                            "split": split, "query_id": qid, "backend": backend,
                            "method": method, "k": k, **scores,
                        }
                        per_query.append(row)
                        summary[split, backend, method, k].append(row)

    rows = []
    fields = ("hit", "recall", "precision", "mrr", "ndcg", "all_judged_hit", "complete_page_hit")
    for (split, backend, method, k), group in summary.items():
        row = {"split": split, "backend": backend, "method": method, "k": k, "queries": len(group)}
        for field in fields:
            scores = [entry[field] for entry in group if entry[field] is not None]
            row[field] = mean(scores) if scores else None
        rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "multiquery_summary.csv", rows)
    if args.per_query:
        write_csv(args.output_dir / "multiquery_per_query.csv", per_query)
    save_json(args.output_dir / "multiquery_protocol.json", {
        "splits": "Same English validation and optimizer-held-out test as the prior experiment",
        "query_sets": "Prior fixed LangChain MultiQuery rewrites, original question, and depths",
        "selected_baseline": baseline,
        "selected_multiquery_configs": configs,
        "combined_backend_first_stage": "Prior 25% OCR / 75% image RRF, constant 60, depth 100",
        "normalization": "Population z-score within each input list at the selected candidate depth",
        "weighted_combmnz": "Sum signed z-scores weighted by the prior query weights, times list appearances",
        "equal_weight_control": "Standard CombMNZ against an equal-weight RRF control",
        "missing_candidate": "No score contribution and no appearance count",
        "score_ties": "Ascending page ID",
        "prior_multiquery_top_20_rankings_reproduced": verified_rankings,
    })
    print(f"Reproduced {verified_rankings} previous MultiQuery rankings.")
    print(f"Summarized {len(per_query)} per-question metric rows in {len(rows)} rows.")


if __name__ == "__main__":
    compare(comparison_arguments(__doc__))
