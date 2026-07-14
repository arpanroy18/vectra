"""Public index API: VectorIndex facade over flat and HNSW backends."""

from __future__ import annotations

import asyncio
import math
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from . import distance as dist
from .hnsw import HNSW
from .metadata import MetadataStore
from .quantization import ProductQuantizer, ScalarQuantizer
from . import storage

SearchResult = Tuple[str, float]  # (id, distance — smaller is closer)

_EXECUTOR = ThreadPoolExecutor(max_workers=max(4, (os.cpu_count() or 4)))


class VectorIndex:
    """A vector index supporting brute-force or HNSW search.

    Parameters
    ----------
    dimension : int
        Vector dimensionality; all added vectors must match.
    metric : {"cosine", "l2", "dot"}
        Similarity metric. Results are returned as distances (smaller =
        closer); for ``dot`` the distance is the negated inner product.
    index : {"flat", "hnsw"}
        ``flat`` is exact brute-force search; ``hnsw`` is approximate.
    M, ef_construction : HNSW build parameters.
    quantization : {None, "scalar", "product"}
        When set, vectors are stored as compressed codes and the index is
        built over their dequantized approximations.
    n_sub, n_clusters : product-quantizer parameters.
    """

    def __init__(
        self,
        dimension: int,
        metric: str = "cosine",
        index: str = "flat",
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 100,
        quantization: Optional[str] = None,
        n_sub: int = 8,
        n_clusters: int = 256,
        seed: Optional[int] = None,
    ) -> None:
        if index not in ("flat", "hnsw"):
            raise ValueError("index must be 'flat' or 'hnsw'")
        if quantization not in (None, "scalar", "product"):
            raise ValueError("quantization must be None, 'scalar' or 'product'")
        self.dim = dimension
        self.metric = metric
        self.index_type = index
        self.ef_search = ef_search
        self.seed = seed
        self._hnsw_args = dict(M=M, ef_construction=ef_construction)
        self.quantization = quantization
        self._pq_args = dict(n_sub=n_sub, n_clusters=n_clusters)

        self.ids: List[str] = []
        self._row_of: Dict[str, int] = {}
        self.metadata = MetadataStore()
        self.vectors = np.zeros((0, dimension), np.float32)   # decoded/original
        # Working-space copy for flat search: L2-normalized for cosine.
        self._w: Optional[np.ndarray] = None
        self.codes: Optional[np.ndarray] = None
        self._alive = np.zeros(0, bool)
        self._quantizer: Optional[object] = None
        self._graph: Optional[HNSW] = None
        if index == "hnsw":
            self._graph = HNSW(dimension, metric, M=M,
                               ef_construction=ef_construction, seed=seed)

    # ------------------------------------------------------------------ meta

    def __len__(self) -> int:
        return int(self._alive.sum())

    def __contains__(self, id: str) -> bool:
        return id in self._row_of

    def get(self, id: str) -> np.ndarray:
        return self.vectors[self._row_of[id]].copy()

