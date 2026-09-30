"""One-command entry point for the reproducible part-C retrieval experiment."""

from pathlib import Path
import hashlib
import json
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def main() -> None:
    config = json.loads((ROOT / "config" / "benchmark_config.json").read_text(encoding="utf-8"))
    data_path = ROOT / config["dataset_file"]
    actual_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()
    if actual_hash != config["dataset_sha256"]:
        raise SystemExit(
            "FinDER Parquet checksum does not match benchmark_config.json. "
            "Do not compare results until the team uses the same data file."
        )
    command = [
        sys.executable,
        str(ROOT / "finder_hybrid_experiment.py"),
        "--parquet", str(data_path),
        "--output-dir", str(ROOT / "results"),
        "--skip-generation",
        "--device", "cpu",
    ]
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
