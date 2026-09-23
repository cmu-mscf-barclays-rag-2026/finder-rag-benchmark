"""Shared data and retrieval evaluation for Legal RAG Bench."""

from .data import load_benchmark, retrieval_text
from .evaluation import evaluate, query_metrics
from .timing import benchmark_retrievers

__all__ = ["load_benchmark", "retrieval_text", "evaluate", "query_metrics", "benchmark_retrievers"]
