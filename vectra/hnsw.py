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

    # ------------------------------------------------------------- insertion

    def add(self, row: int, vector: np.ndarray) -> int:
        """Insert ``vector`` as graph node ``row``. Returns the assigned level."""
        self._grow(row + 1)
        self.vectors[row] = vector
        self._w[row] = dist.working(vector, self.metric)[0]
        self._alive[row] = True
        level = _level_for(self._rng, self.level_mult)
        self.levels.append(level)
        self.links.append([[] for _ in range(level + 1)])

        if self.entry == -1:
            self.entry, self.max_level = row, level
            return level

        qw = self._tq(vector)
        ep = self.entry
        # Greedy descent on layers above the new node's top.
        for layer in range(self.max_level, level, -1):
            ep = self._greedy_layer(qw, ep, layer)

        for layer in range(min(level, self.max_level), -1, -1):
            cands = self._search_layer(qw, [ep], self.ef_construction, layer)
            maxM = self.maxM0 if layer == 0 else self.M
            neighbours = self._select(cands, maxM)
            self.links[row][layer] = neighbours
            for nb in neighbours:
                nbs = self.links[nb][layer]
                if len(nbs) < maxM:
                    nbs.append(row)
                else:
                    # Re-select including the new node, keeping the best maxM.
                    merged = nbs + [row]
                    nbw = self._w[nb]
                    self.links[nb][layer] = self._select(
                        list(zip(self._dists(nbw, merged).tolist(), merged)),
                        maxM,
                    )
            if cands:
                ep = min(cands)[1]

        if level > self.max_level:
            self.entry, self.max_level = row, level
        return level

    # --------------------------------------------------------------- search

    def _greedy_layer(self, qw: np.ndarray, ep: int, layer: int) -> int:
        best, best_d = ep, self._dist(qw, ep)
        improved = True
        while improved:
            improved = False
            nbs = np.asarray(self.links[best][layer], dtype=np.int64)
            nbs = nbs[self._alive[nbs]]
            if not len(nbs):
                break
            for nb, d in zip(nbs.tolist(), self._dists(qw, nbs)):
                if d < best_d:
                    best, best_d = nb, float(d)
                    improved = True
        return best

    def _search_layer(self, qw: np.ndarray, eps: List[int], ef: int, layer: int) -> List[Tuple[float, int]]:
        """Best-first search; returns up to ``ef`` (distance, row) results."""
        visited = np.zeros(len(self.vectors), bool)
        cand_heap: List[Tuple[float, int]] = []  # min-heap of candidates to expand
        res_heap: List[Tuple[float, int]] = []   # max-heap (via -d) of results
        for ep in eps:
            if ep < 0 or not self._alive[ep]:
                continue
            d = self._dist(qw, ep)
            visited[ep] = True
            heapq.heappush(cand_heap, (d, ep))
            heapq.heappush(res_heap, (-d, ep))
        while cand_heap:
            d, r = heapq.heappop(cand_heap)
            worst = -res_heap[0][0]
            if d > worst and len(res_heap) >= ef:
                break
            new = np.asarray(self.links[r][layer], dtype=np.int64)
            if not len(new):
                continue
            new = new[~visited[new]]
            visited[new] = True
            new = new[self._alive[new]]
            for nb, dnb in zip(new.tolist(), self._dists(qw, new)):
                dnb = float(dnb)
                worst = -res_heap[0][0]
                if len(res_heap) < ef or dnb < worst:
                    heapq.heappush(cand_heap, (dnb, nb))
                    heapq.heappush(res_heap, (-dnb, nb))
                    if len(res_heap) > ef:
                        heapq.heappop(res_heap)
        out = [(-nd, r) for nd, r in res_heap]
        out.sort()
        return out

