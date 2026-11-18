"""Spot-check cached vectors against a fresh mixed-length GPU batch."""

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import torch

from benchmark_core import INSTRUCTION, save_json
from experiment import Experiment, ROOT, read


def check(args):
    e = Experiment(args)
    selected = read(e.results / "selection.json")
    originals = [next(q for q in e.queries if q["language"] == language)
                 for language in ("english", "french", "german", "italian", "portuguese", "spanish")]
    strategy = selected["expansion"]["image"]["multiquery"]["strategy"]
    alternatives = [read(e.cache / "generated" / strategy / f"{q['query_id']}.json")["queries"][0]
                    for q in originals]
    texts = [q["query"] for q in originals] + alternatives
    references = np.stack([e.vector(text) for text in texts])
    spec = importlib.util.spec_from_file_location("qwen_checked", args.model / "scripts/qwen3_vl_embedding.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.set_num_threads(4)
    encoder = module.Qwen3VLEmbedder(str(args.model), max_length=4096,
                                   torch_dtype=torch.float16, attn_implementation="sdpa")
    with torch.inference_mode():
        vectors = encoder.process([{"text": text, "instruction": INSTRUCTION} for text in texts]).float().cpu().numpy()
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    agreement = (vectors * references).sum(axis=1)
    assert np.isfinite(vectors).all() and agreement.min() > 0.9999, agreement
    report = {"inputs": len(texts), "fresh_batch_size": len(texts),
              "minimum_cosine_agreement": float(agreement.min()),
              "original_questions_checked": 6, "generated_alternatives_checked": 6,
              "device": torch.cuda.get_device_name(),
              "note": "Fresh mixed-length GPU batch versus saved original batch-1 and generated batch-16 vectors.",
              "status": "passed"}
    save_json(e.results / "encoder_spot_check.json", report)
    print(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for option, default in (("data", "data"), ("model", "model"),
                            ("vectors", "page_vectors"), ("cache", "experiment")):
        parser.add_argument("--" + option, type=Path, default=ROOT / ".cache" / default)
    check(parser.parse_args())
