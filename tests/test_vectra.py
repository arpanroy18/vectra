import numpy as np
import pytest

from vectra import VectorIndex, distance
from vectra.flat import exact_topk, recall_at_k
from vectra.quantization import ScalarQuantizer, ProductQuantizer

rng = np.random.default_rng(42)
DIM = 64


def make(n=500, dim=DIM, seed=42):
    r = np.random.default_rng(seed)
    return r.normal(size=(n, dim)).astype(np.float32)


# ---- distances -------------------------------------------------------------

def test_cosine_and_l2_and_dot_match_naive():
    a, b = rng.normal(size=DIM), rng.normal(size=DIM)
    cos = 1 - (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))
    assert abs(distance.pairwise("cosine", a, b) - cos) < 1e-5
    assert abs(distance.pairwise("l2", a, b) - np.linalg.norm(a - b)) < 1e-5
    assert abs(distance.pairwise("dot", a, b) - (-(a @ b))) < 1e-5


def test_bad_metric_raises():
    with pytest.raises(ValueError):
        distance.distance("manhattan", np.zeros(4), np.zeros((2, 4)))


