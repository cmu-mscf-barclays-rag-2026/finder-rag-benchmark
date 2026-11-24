"""Compare OCR/image fusion rules on fixed ViDoRe English queries and candidates."""

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean
import sys

import numpy as np


ROOT = Path(__file__).resolve().parent
PRIOR = ROOT.parent
sys.path.insert(0, str(PRIOR))

from benchmark_core import metric_values, save_json, write_csv
from experiment import Experiment, read, rrf


def combine_z_scores(scored_lists, weights=None):
    """Return CombSUM and CombMNZ rankings from per-list population z-scores."""
    if weights is None:
        weights = [1.0] * len(scored_lists)
    totals = defaultdict(float)
    appearances = defaultdict(int)
    for (pages, scores), weight in zip(scored_lists, weights, strict=True):
        normalized = (scores - scores.mean()) / max(scores.std(), 1e-9)
        for page, score in zip(pages, normalized, strict=True):
            totals[page] += weight * float(score)
            appearances[page] += 1
    combsum = sorted(totals, key=lambda page: (-totals[page], page))
    combmnz = sorted(totals, key=lambda page: (-totals[page] * appearances[page], page))
    return combsum, combmnz


def compare(args):
    experiment = Experiment(args)
    previous = read(PRIOR / "results/rankings.json")
    ocr_weight = read(PRIOR / "results/baseline_selection.json")["ocr_weight"]
    page_ids = np.array(experiment.page_ids)
    depth = 100
    per_query = []
    summary = defaultdict(list)
    verified_rankings = 0
    for split in ("validation", "test"):
        for question in experiment.partition(split, english=True):
            qid = question["query_id"]
            vector = experiment.vector(question["query"])
            candidates = []
            for modality in ("ocr", "image"):
                scores = experiment.matrices[modality] @ vector
                indices = np.lexsort((page_ids, -scores))[:depth]
                candidates.append((page_ids[indices].tolist(), scores[indices].astype(np.float64)))
            ocr, image = [pages for pages, _ in candidates]
            weighted = rrf([ocr, image], [ocr_weight, 1 - ocr_weight], 60, depth)
            combsum, combmnz = combine_z_scores(candidates)
            for modality, ranking in (("ocr", ocr), ("image", image), ("combined", weighted)):
                if ranking[:20] != previous[str(qid)][f"{modality}/original"]:
                    raise ValueError(f"Baseline ranking changed for {modality}, query {qid}.")
                verified_rankings += 1
            rankings = {
                "ocr_only": ocr,
                "image_only": image,
                "rrf_equal": rrf([ocr, image], [0.5, 0.5], 60, depth),
                "rrf_previous": weighted,
                "combsum_zmuv": combsum,
                "combmnz_zmuv": combmnz,
            }
            for method, ranked_pages in rankings.items():
                for k in (5, 8, 10):
                    scores = metric_values(ranked_pages, experiment.gold[qid], k)
                    row = {"split": split, "query_id": qid, "method": method, "k": k, **scores}
                    per_query.append(row)
                    summary[split, method, k].append(row)
    fields = ("hit", "recall", "precision", "mrr", "ndcg", "all_judged_hit", "complete_page_hit")
    rows = []
    for (split, method, k), group in summary.items():
        row = {"split": split, "method": method, "k": k, "queries": len(group)}
        for field in fields:
            scores = [entry[field] for entry in group if entry[field] is not None]
            row[field] = mean(scores) if scores else None
        rows.append(row)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "modality_summary.csv", rows)
    if args.per_query:
        write_csv(args.output_dir / "modality_per_query.csv", per_query)
    save_json(args.output_dir / "modality_protocol.json", {
        "query": "Original English question",
        "splits": "Same ViDoRe validation and optimizer-held-out test as the prior experiment",
        "candidate_depth_per_modality": depth,
        "rrf_constant": 60,
        "previous_ocr_weight": ocr_weight,
        "normalization": "Population z-score within each retrieved top-100 list",
        "combmnz_convention": "Sum signed z-scores and multiply by number of lists containing the page",
        "missing_candidate": "No score contribution and no appearance count",
        "score_ties": "Ascending page ID",
        "prior_top_20_rankings_reproduced": verified_rankings,
    })


def comparison_arguments(description):
    parser = argparse.ArgumentParser(description=description)
    for option in ("data", "vectors", "cache"):
        parser.add_argument("--" + option, type=Path, required=True, metavar="PATH")
    parser.add_argument("--output-dir", type=Path, default=ROOT, metavar="PATH")
    parser.add_argument("--per-query", action="store_true", help="Also write the per-question metrics.")
    return parser.parse_args()


if __name__ == "__main__":
    compare(comparison_arguments(__doc__))
