"""Distance metrics over float32 vectors.

All functions take a 1-D query and a 2-D matrix of candidates and return a
1-D array of *distances* (lower is closer). Inner-product metrics are
returned negated so a single "smallest is best" convention works everywhere.
"""

from __future__ import annotations

import numpy as np

Metric = str  # "cosine" | "l2" | "dot"


def _as_rows(vectors: np.ndarray) -> np.ndarray:
    v = np.asarray(vectors, dtype=np.float32)
    if v.ndim == 1:
        v = v.reshape(1, -1)
    return v


def cosine(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine distance: 1 - cos(q, x)."""
    q = np.asarray(query, dtype=np.float32)
    m = _as_rows(matrix)
    qn = q / max(np.linalg.norm(q), 1e-12)
    mn = m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-12)
    return 1.0 - mn @ qn


def l2(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Euclidean distance."""
    q = np.asarray(query, dtype=np.float32)
    m = _as_rows(matrix)
    # ||q-x||^2 = ||q||^2 + ||x||^2 - 2 q.x ; keep sqrt for a true metric.
    d2 = np.sum(m * m, axis=1) + float(q @ q) - 2.0 * (m @ q)
    return np.sqrt(np.maximum(d2, 0.0))


