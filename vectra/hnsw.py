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

