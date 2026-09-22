"""Measure query latency for each search mode.

    python -m backend.benchmark
Writes benchmarks/latency.md - use these numbers on your resume.
"""
import random
import statistics
import time

import numpy as np

from . import config
from .engine import SearchEngine


def main(n_queries: int = 300):
    engine = SearchEngine()
    random.seed(7)
    sample = random.sample(range(len(engine.texts)), n_queries)
    queries = [" ".join(engine.texts[i].split()[:8]) for i in sample]
    for q in queries[:20]:  # warm-up
        engine.search(q, "hybrid")

    rows = []
    for mode in ("keyword", "semantic", "hybrid"):
        times = []
        for q in queries:
            t = time.perf_counter()
            engine.search(q, mode)
            times.append((time.perf_counter() - t) * 1000)
        rows.append((mode, np.percentile(times, 50), np.percentile(times, 95),
                     np.percentile(times, 99), 1000 / statistics.mean(times)))

    header = f"Documents: {len(engine.texts):,} | Index: {engine.meta['index_type'].upper()} | Queries: {n_queries}\n\n"
    table = "| Mode | p50 (ms) | p95 (ms) | p99 (ms) | Queries/sec (1 thread) |\n|---|---|---|---|---|\n"
    for m, p50, p95, p99, qps in rows:
        table += f"| {m} | {p50:.1f} | {p95:.1f} | {p99:.1f} | {qps:.0f} |\n"
    print("\n" + header + table)
    config.BENCH_DIR.mkdir(exist_ok=True)
    (config.BENCH_DIR / "latency.md").write_text("# Latency benchmark\n\n" + header + table)
    print("Saved to benchmarks/latency.md")


if __name__ == "__main__":
    main()
