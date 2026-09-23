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


