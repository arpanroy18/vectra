"""Exact (brute-force) search helpers — the recall reference for benchmarks."""

from __future__ import annotations

from typing import Collection, List, Sequence, Tuple

import numpy as np

from . import distance as dist


def exact_topk(vectors: np.ndarray, query: np.ndarray, k: int,
               metric: str = "cosine") -> np.ndarray:
    """Indices of the k nearest rows of ``vectors`` to ``query``."""
    d = dist.distance(metric, query, vectors)
    return np.argsort(d, kind="stable")[:k]


