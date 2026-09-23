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
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

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

    def _dist(self, q: np.ndarray, row: int) -> float:
        return dist.pairwise(self.metric, q, self.vectors[row])

    def _dists(self, q: np.ndarray, rows: Sequence[int]) -> np.ndarray:
        return dist.distance(self.metric, q, self.vectors[np.asarray(rows, dtype=np.int64)])

    def _grow(self, n: int) -> None:
        add = n - len(self.vectors)
        if add <= 0:
            return
        cap = max(64, 1 << max(int(math.ceil(math.log2(n))), 6))
        self.vectors = np.vstack([self.vectors, np.zeros((cap - len(self.vectors), self.dim), np.float32)])
        self._alive = np.concatenate([self._alive, np.zeros(cap - len(self._alive), bool)])

    # ------------------------------------------------------------- insertion

    def add(self, row: int, vector: np.ndarray) -> int:
        """Insert ``vector`` as graph node ``row``. Returns the assigned level."""
        self._grow(row + 1)
        self.vectors[row] = vector
        self._alive[row] = True
        level = _level_for(self._rng, self.level_mult)
        self.levels.append(level)
        self.links.append([[] for _ in range(level + 1)])

        if self.entry == -1:
            self.entry, self.max_level = row, level
            return level

        ep = self.entry
        # Greedy descent on layers above the new node's top.
        for layer in range(self.max_level, level, -1):
            ep = self._greedy_layer(vector, ep, layer)

        for layer in range(min(level, self.max_level), -1, -1):
            cands = self._search_layer(vector, [ep], self.ef_construction, layer)
            maxM = self.maxM0 if layer == 0 else self.M
            neighbours = self._select(vector, cands, maxM, layer)
            self.links[row][layer] = neighbours
            for nb in neighbours:
                nbs = self.links[nb][layer]
                if len(nbs) < maxM:
                    nbs.append(row)
                else:
                    # Re-select including the new node, keeping the best maxM.
                    merged = nbs + [row]
                    self.links[nb][layer] = self._select(
                        self.vectors[nb],
                        [(self._dist(self.vectors[nb], r), r) for r in merged],
                        maxM,
                        layer,
                    )
            if cands:
                ep = min(cands)[1]

        if level > self.max_level:
            self.entry, self.max_level = row, level
        return level

    # --------------------------------------------------------------- search

    def _greedy_layer(self, q: np.ndarray, ep: int, layer: int) -> int:
        best, best_d = ep, self._dist(q, ep)
        improved = True
        while improved:
            improved = False
            nbs = [nb for nb in self.links[best][layer] if self._alive[nb]]
            if not nbs:
                break
            for nb, d in zip(nbs, self._dists(q, nbs)):
                if d < best_d:
                    best, best_d = nb, float(d)
                    improved = True
        return best

