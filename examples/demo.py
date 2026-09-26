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

print("Generating a dataset of 5,000 embeddings (384-d, clustered)...")
centers = rng.normal(size=(32, DIM)).astype(np.float32)
vecs = centers[rng.integers(0, 32, N)] + 0.2 * rng.normal(size=(N, DIM)).astype(np.float32)
ids = [f"doc{i}" for i in range(N)]
meta = [{"year": 2020 + i % 6, "category": ["finance", "tech", "science"][i % 3]}
        for i in range(N)]

print("Building an HNSW index over them (M=16, ef_construction=200)...")
t = time.time()
index = VectorIndex(dimension=DIM, metric="cosine", index="hnsw",
                    M=16, ef_construction=200)
index.add(ids, vecs, metadata=meta)
print(f"Done in {time.time() - t:.1f}s.\n")

q = vecs[123]
print("Searching for the 5 nearest documents to doc123:")
for id_, d in index.search(q, k=5):
    print(f"  {id_:<8} distance={d:.4f}")

print("\nSame search, but only inside category='finance', year=2023:")
for id_, d in index.search(q, k=5, filter={"category": "finance", "year": 2023}):
    print(f"  {id_:<8} distance={d:.4f}  {index.metadata.get(index._row_of[id_])}")

print("\nBuilding an exact brute-force index to measure recall...")
flat = VectorIndex(DIM, metric="cosine")
flat.add(ids, vecs)
truth = {r[0] for r in flat.search(q, k=10)}
print("Sweeping ef_search (higher = slower but more accurate):")
for ef in (20, 100, 400):
    t = time.perf_counter()
    res = index.search(q, k=10, ef_search=ef)
    ms = (time.perf_counter() - t) * 1e3
    print(f"  ef_search={ef:<4} recall={len({r[0] for r in res} & truth) / 10:.2f}  {ms:.2f}ms")

print("\nCompressing all vectors with scalar quantization (float32 -> uint8)...")
sq = VectorIndex(DIM, metric="cosine", quantization="scalar")
sq.add(ids, vecs)
print(f"  storage: {N * DIM * 4 / 1e6:.1f} MB -> {N * DIM / 1e6:.1f} MB (4x smaller)")
res = sq.search(q, k=10)
print(f"  recall@10 vs exact: {len({r[0] for r in res} & truth) / 10:.2f}")

print("\nSaving the index to disk in vectra's .vdb format...")
path = os.path.join(tempfile.gettempdir(), "demo.vdb")
index.save(path)
print(f"  wrote {os.path.getsize(path) / 1e6:.1f} MB -> {path}")
loaded = VectorIndex.load(path)
mm = VectorIndex.mmap(path)
print("Reloading it two ways — full load() and memory-mapped mmap():")
print(f"  load()  top-1: {loaded.search(q, k=1)[0]}")
print(f"  mmap()  top-1: {mm.search(q, k=1)[0]}   (vectors never leave disk)")

import asyncio
print("\nRunning a batch of 3 queries through the thread pool:")
batch = index.search_batch(vecs[:3], k=3)
for i, r in enumerate(batch):
    print(f"  [{i}] -> {[x[0] for x in r]}")
print("And one query through the async API:")
async def go():
    return await index.search_async(vecs[9], k=3)
print(f"  async -> {[x[0] for x in asyncio.run(go())]}")
