"""Run part C, export its standard metrics, and refresh the comparison table."""

from pathlib import Path
import json
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def run(*parts: str) -> None:
    subprocess.run([sys.executable, *parts], cwd=ROOT, check=True)


def main() -> None:
    run("run_retrieval.py")
    summary = json.loads((ROOT / "results" / "run_summary.json").read_text(encoding="utf-8"))
    alpha = float(summary["hybrid"]["best_dense_weight"])
    method = f"Hybrid RRF alpha={alpha:.2f}"
    method_id = f"hybrid_rrf_{int(round((1-alpha)*100))}bm25_{int(round(alpha*100))}dense"
    run(
        "scripts/export_method_metrics.py",
        "--retrieval-csv", "results/retrieval_metrics.csv",
        "--method", method,
        "--method-id", method_id,
        "--owner", "C",
        "--output", "team_metrics/c_hybrid_rrf.csv",
    )
    run("scripts/aggregate_team_metrics.py", "--k", "5")


if __name__ == "__main__":
    main()
