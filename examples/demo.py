"""vectra demo — exercises the whole library end to end.

    PYTHONPATH=. python3 examples/demo.py
"""

import os
import tempfile
import time

import numpy as np

from vectra import VectorIndex

rng = np.random.default_rng(0)
DIM, N = 384, 5000

# Clustered embeddings — closer to real embedding distributions than noise.
centers = rng.normal(size=(32, DIM)).astype(np.float32)
vecs = centers[rng.integers(0, 32, N)] + 0.2 * rng.normal(size=(N, DIM)).astype(np.float32)
ids = [f"doc{i}" for i in range(N)]
meta = [{"year": 2020 + i % 6, "category": ["finance", "tech", "science"][i % 3]}
        for i in range(N)]

t = time.time()
index = VectorIndex(dimension=DIM, metric="cosine", index="hnsw",
                    M=16, ef_construction=200)
index.add(ids, vecs, metadata=meta)
print(f"hnsw index: {len(index)} x {DIM} vectors, built in {time.time() - t:.1f}s\n")

q = vecs[123]
print("search(q, k=5)")
for id_, d in index.search(q, k=5):
    print(f"  {id_:<8} distance={d:.4f}")

print("\nsearch(q, k=5, filter={'category': 'finance', 'year': 2023})")
for id_, d in index.search(q, k=5, filter={"category": "finance", "year": 2023}):
    print(f"  {id_:<8} distance={d:.4f}  {index.metadata.get(index._row_of[id_])}")

flat = VectorIndex(DIM, metric="cosine")
flat.add(ids, vecs)
truth = {r[0] for r in flat.search(q, k=10)}
print("\nef_search sweep (recall@10 vs exact):")
for ef in (20, 100, 400):
    t = time.perf_counter()
    res = index.search(q, k=10, ef_search=ef)
    ms = (time.perf_counter() - t) * 1e3
    print(f"  ef_search={ef:<4} recall={len({r[0] for r in res} & truth) / 10:.2f}  {ms:.2f}ms")

sq = VectorIndex(DIM, metric="cosine", quantization="scalar")
sq.add(ids, vecs)
print(f"\nscalar quantization: {N * DIM * 4 / 1e6:.1f} MB float32 -> "
      f"{N * DIM / 1e6:.1f} MB uint8")
res = sq.search(q, k=10)
print(f"  recall@10 vs exact: {len({r[0] for r in res} & truth) / 10:.2f}")

path = os.path.join(tempfile.gettempdir(), "demo.vdb")
index.save(path)
loaded = VectorIndex.load(path)
mm = VectorIndex.mmap(path)
print(f"\nsaved {os.path.getsize(path) / 1e6:.1f} MB -> {path}")
print(f"  load()  top-1: {loaded.search(q, k=1)[0]}")
print(f"  mmap()  top-1: {mm.search(q, k=1)[0]}")

import asyncio
batch = index.search_batch(vecs[:3], k=3)
print("\nbatch search:")
for i, r in enumerate(batch):
    print(f"  [{i}] -> {[x[0] for x in r]}")
async def go():
    return await index.search_async(vecs[9], k=3)
print(f"  async -> {[x[0] for x in asyncio.run(go())]}")
