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


def dot(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Negative dot product so that smaller is better."""
    q = np.asarray(query, dtype=np.float32)
    m = _as_rows(matrix)
    return -(m @ q)


_METRICS = {"cosine": cosine, "l2": l2, "dot": dot}


def distance(metric: Metric, query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    try:
        fn = _METRICS[metric]
    except KeyError:
        raise ValueError(f"unknown metric {metric!r}; expected one of {sorted(_METRICS)}")
    return fn(query, matrix)


def pairwise(metric: Metric, a: np.ndarray, b: np.ndarray) -> float:
    """Distance between two single vectors."""
    return float(distance(metric, a, b.reshape(1, -1))[0])


def norm_rows(matrix: np.ndarray) -> np.ndarray:
    """L2-normalized copy of ``matrix`` rows (zeros stay zero)."""
    m = _as_rows(matrix)
    return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-12)


def working(matrix: np.ndarray, metric: Metric) -> np.ndarray:
    """Vectors in the space distances are computed in.

    For cosine, rows are pre-normalized once so every subsequent distance is a
    plain dot product; for l2/dot the matrix is returned as-is.
    """
    m = _as_rows(matrix)
    return norm_rows(m) if metric == "cosine" else m


def working_query(query: np.ndarray, metric: Metric) -> np.ndarray:
    q = np.asarray(query, dtype=np.float32)
    if metric == "cosine":
        return q / max(float(np.linalg.norm(q)), 1e-12)
    return q


def work_distance(metric: Metric, q: np.ndarray, wmatrix: np.ndarray) -> np.ndarray:
    """Distance between working-space query/queries and working-space rows.

    ``q`` may be 1-D (returns (m,)) or a 2-D (nq, d) batch (returns (nq, m)).
    """
    if q.ndim == 2:
        if metric == "cosine":
            return 1.0 - q @ wmatrix.T
        if metric == "l2":
            sq_q = np.sum(q * q, axis=1)
            sq_m = np.sum(wmatrix * wmatrix, axis=1)
            d2 = sq_q[:, None] + sq_m[None] - 2.0 * (q @ wmatrix.T)
            return np.sqrt(np.maximum(d2, 0.0))
        return -(q @ wmatrix.T)
    if metric == "cosine":
        return 1.0 - wmatrix @ q
    if metric == "l2":
        d2 = np.sum(wmatrix * wmatrix, axis=1) + float(q @ q) - 2.0 * (wmatrix @ q)
        return np.sqrt(np.maximum(d2, 0.0))
    return -(wmatrix @ q)


def within(metric: Metric, wmatrix: np.ndarray) -> np.ndarray:
    """Pairwise distance matrix among working-space rows."""
    if metric == "cosine":
        return 1.0 - wmatrix @ wmatrix.T
    if metric == "l2":
        sq = np.sum(wmatrix * wmatrix, axis=1)
        d2 = sq[:, None] + sq[None] - 2.0 * (wmatrix @ wmatrix.T)
        return np.sqrt(np.maximum(d2, 0.0))
    return -(wmatrix @ wmatrix.T)
