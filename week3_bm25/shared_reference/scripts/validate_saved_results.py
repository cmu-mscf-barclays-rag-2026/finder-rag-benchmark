"""Fast checks for the committed dataset and saved result files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    config = json.loads((ROOT / "config" / "benchmark_config.json").read_text(encoding="utf-8"))
    data_path = ROOT / config["dataset_file"]
    checksum = hashlib.sha256(data_path.read_bytes()).hexdigest()
    assert checksum == config["dataset_sha256"], "Dataset checksum mismatch"

    raw = pd.read_csv(ROOT / "results" / "retrieval_metrics.csv")
    assert set(raw["split"]) == {"dev", "test", "all"}
    assert set(raw["k"]) == set(config["evaluation_k"])
    test_counts = set(raw.loc[raw["split"] == "test", "n_queries"])
    assert test_counts == {config["test_queries"]}

    team = pd.read_csv(ROOT / "team_metrics" / "c_hybrid_rrf.csv")
    assert set(team["dataset_id"]) == {config["dataset_id"]}
    assert set(team["corpus_id"]) == {config["corpus_id"]}
    assert set(team["split_id"]) == {config["split_id"]}
    assert set(team["split"]) == {"test"}
    assert set(team["k"]) == set(config["evaluation_k"])

    print("Saved dataset and metric files passed all checks.")


if __name__ == "__main__":
    main()
