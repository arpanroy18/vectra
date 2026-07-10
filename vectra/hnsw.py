"""HNSW (Hierarchical Navigable Small World) index, implemented from scratch.

Follows Malkov & Yashunin (2016): a layered proximity graph where each node
appears on every layer up to a randomly drawn level. Insertion searches the
current graph top-down and links each layer to a diverse set of neighbours;
queries greedy-descend to layer 0 and then run a best-first beam search of
width ``ef_search``.
"""

from __future__ import annotations

import heapq
import math
import random
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import distance as dist

Predicate = Optional[Callable[[int], bool]]


def _level_for(rng: random.Random, level_mult: float) -> int:
    return int(-math.log(rng.random() + 1e-12) * level_mult)


class HNSW:
    def __init__(
        self,
        dim: int,
        metric: str = "cosine",
        M: int = 16,
        ef_construction: int = 200,
        seed: Optional[int] = None,
        select_heuristic: bool = True,
    ) -> None:
        self.dim = dim
        self.metric = metric
        self.M = M
        self.maxM0 = 2 * M  # layer 0 is denser in the reference design
        self.ef_construction = ef_construction
        self.level_mult = 1.0 / math.log(M)
        self.select_heuristic = select_heuristic
        self._rng = random.Random(seed)

        self.vectors = np.zeros((0, dim), dtype=np.float32)
        # Working-space copy: L2-normalized for cosine, alias for l2/dot.
        self._w = self.vectors if metric != "cosine" else np.zeros((0, dim), np.float32)
        self._alive = np.zeros(0, dtype=bool)
        self.levels: List[int] = []
        self.links: List[List[List[int]]] = []  # row -> layer -> neighbour rows
        self.entry: int = -1
        self.max_level: int = -1

    # ------------------------------------------------------------------ util

    def __len__(self) -> int:
        return int(self._alive.sum())

    @property
    def capacity(self) -> int:
        return len(self.vectors)

    def _tq(self, q: np.ndarray) -> np.ndarray:
        return dist.working_query(q, self.metric)

    def _dist(self, qw: np.ndarray, row: int) -> float:
        return float(dist.work_distance(self.metric, qw, self._w[row:row + 1])[0])

    def _dists(self, qw: np.ndarray, rows: Sequence[int]) -> np.ndarray:
        return dist.work_distance(self.metric, qw,
                                  self._w[np.asarray(rows, dtype=np.int64)])

    def _grow(self, n: int) -> None:
        add = n - len(self.vectors)
        if add <= 0:
            return
        cap = max(64, 1 << max(int(math.ceil(math.log2(n))), 6))
        self.vectors = np.vstack([self.vectors, np.zeros((cap - len(self.vectors), self.dim), np.float32)])
        if self.metric == "cosine":
            self._w = np.vstack([self._w, np.zeros((cap - len(self._w), self.dim), np.float32)])
        else:
            self._w = self.vectors
        self._alive = np.concatenate([self._alive, np.zeros(cap - len(self._alive), bool)])

