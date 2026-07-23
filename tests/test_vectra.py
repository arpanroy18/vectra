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


# ---- flat index ------------------------------------------------------------

def test_flat_exact_topk():
    vecs = make()
    idx = VectorIndex(DIM, metric="l2", index="flat")
    idx.add([f"v{i}" for i in range(len(vecs))], vecs)
    q = vecs[7]
    res = idx.search(q, k=5)
    assert res[0][0] == "v7"
    truth = {f"v{r}" for r in exact_topk(vecs, q, 5, "l2")}
    assert {r[0] for r in res} == truth


def test_flat_sorted_and_duplicate_ids():
    vecs = make(20)
    idx = VectorIndex(DIM)
    idx.add(["a", "b"], vecs[:2])
    with pytest.raises(ValueError):
        idx.add(["a"], vecs[2:3])
    res = idx.search(vecs[10], k=10)
    ds = [d for _, d in res]
    assert ds == sorted(ds)


# ---- deletion / rebuild ----------------------------------------------------

def test_delete_and_rebuild():
    vecs = make(200)
    for index in ("flat", "hnsw"):
        idx = VectorIndex(DIM, index=index, metric="l2")
        idx.add([f"v{i}" for i in range(200)], vecs)
        assert idx.delete(["v0", "v1", "nope"]) == 2
        assert len(idx) == 198
        res = idx.search(vecs[0], k=3)
        assert "v0" not in {r[0] for r in res}
        idx.rebuild()
        assert len(idx) == 198 and len(idx.ids) == 198
        res2 = idx.search(vecs[5], k=3)
        assert res2[0][0] == "v5"


