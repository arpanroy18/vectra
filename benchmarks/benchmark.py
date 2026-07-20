"""Recall / latency / memory benchmark: flat vs HNSW vs quantized variants.

    python benchmarks/benchmark.py [--n 20000] [--dim 384] [--queries 200] [--k 10]
"""

from __future__ import annotations

import argparse
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vectra import VectorIndex  # noqa: E402
from vectra.flat import exact_topk  # noqa: E402


def bench(idx: VectorIndex, queries: np.ndarray, k: int):
    lat = []
    results = []
    for q in queries:
        t = time.perf_counter()
        results.append(idx.search(q, k=k))
        lat.append((time.perf_counter() - t) * 1e3)
    lat = np.asarray(lat)
    qps = len(queries) / (lat.sum() / 1e3)
    return results, qps, np.percentile(lat, 50), np.percentile(lat, 99)


def recall(results, truths, ids):
    hits = tot = 0
    for res, tr in zip(results, truths):
        hits += len({r[0] for r in res} & {ids[int(i)] for i in tr})
        tot += len(tr)
    return hits / tot


