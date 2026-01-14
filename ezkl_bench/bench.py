"""
Thin public facade for benchmarking.

NOTE: This file intentionally stays small.
The implementation lives in:
  - ezkl_bench/bench_model.py  (single-model pipeline)
  - ezkl_bench/bench_runner.py (multi-model runner / worker orchestration)
"""

from .bench_model import ModelRunResult, run_single_model
from .bench_runner import run_benchmark

__all__ = [
    "ModelRunResult",
    "run_single_model",
    "run_benchmark",
]
