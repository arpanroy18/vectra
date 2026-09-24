"""Vector quantization: scalar (uint8) and product quantization.

Scalar quantization maps each dimension to 256 uniform bins fitted on the
training data — 4x smaller vectors with a small recall cost. Product
quantization splits vectors into ``n_sub`` sub-vectors and replaces each with
the index of its nearest of ``n_clusters`` learned centroids (trained with an
in-house Lloyd's-algorithm k-means).
"""

from __future__ import annotations

import json
from typing import Dict, Optional

import numpy as np


class ScalarQuantizer:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.mins: Optional[np.ndarray] = None
        self.ranges: Optional[np.ndarray] = None

    def fit(self, vectors: np.ndarray) -> "ScalarQuantizer":
        v = np.asarray(vectors, dtype=np.float32)
        self.mins = v.min(axis=0)
        ranges = v.max(axis=0) - self.mins
        self.ranges = np.maximum(ranges, 1e-8)
        return self

    def transform(self, vectors: np.ndarray) -> np.ndarray:
        if self.mins is None:
            raise RuntimeError("quantizer is not fitted")
        v = (np.asarray(vectors, np.float32) - self.mins) / self.ranges
        return (np.clip(v, 0, 1) * 255).round().astype(np.uint8)

    def inverse(self, codes: np.ndarray) -> np.ndarray:
        return codes.astype(np.float32) / 255.0 * self.ranges + self.mins

    def state(self) -> Dict:
        return {"mins": self.mins.tolist(), "ranges": self.ranges.tolist()}

    @classmethod
    def from_state(cls, dim: int, state: Dict) -> "ScalarQuantizer":
        q = cls(dim)
        q.mins = np.asarray(state["mins"], np.float32)
        q.ranges = np.asarray(state["ranges"], np.float32)
        return q


