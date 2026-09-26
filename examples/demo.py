"""vectra demo — exercises the whole library end to end.

    PYTHONPATH=. python3 examples/demo.py
"""

import os
import tempfile
import time

import numpy as np

from vectra import VectorIndex


def hr(t):
    print("\n" + "=" * 58 + f"\n  {t}\n" + "=" * 58)


rng = np.random.default_rng(0)
DIM, N = 384, 5000

# Clustered embeddings — closer to real embedding distributions than noise.
centers = rng.normal(size=(32, DIM)).astype(np.float32)
vecs = centers[rng.integers(0, 32, N)] + 0.2 * rng.normal(size=(N, DIM)).astype(np.float32)
ids = [f"doc{i}" for i in range(N)]
meta = [{"year": 2020 + i % 6, "category": ["finance", "tech", "science"][i % 3]}
        for i in range(N)]

hr("1. Build an HNSW index over 5,000 x 384 vectors")
t = time.time()
index = VectorIndex(dimension=DIM, metric="cosine", index="hnsw",
                    M=16, ef_construction=200)
index.add(ids, vecs, metadata=meta)
print(f"   built in {time.time() - t:.1f}s   ({len(index)} vectors, M=16, ef_construction=200)")

hr("2. Top-k search")
q = vecs[123]
res = index.search(q, k=5)
for id_, d in res:
    print(f"   {id_:<8} distance={d:.4f}")

hr("3. Metadata-filtered search  (category='finance', year=2023)")
res = index.search(q, k=5, filter={"category": "finance", "year": 2023})
for id_, d in res:
    print(f"   {id_:<8} distance={d:.4f}  meta={index.metadata.get(index._row_of[id_])}")

hr("4. ef_search: recall vs latency tradeoff")
flat = VectorIndex(DIM, metric="cosine")
flat.add(ids, vecs)
truth = {r[0] for r in flat.search(q, k=10)}
for ef in (20, 100, 400):
    t = time.perf_counter()
    res = index.search(q, k=10, ef_search=ef)
    ms = (time.perf_counter() - t) * 1e3
    rec = len({r[0] for r in res} & truth) / 10
    print(f"   ef_search={ef:<4} recall@10={rec:.2f}  latency={ms:.2f}ms")

hr("5. Scalar quantization: float32 -> uint8 (4x smaller)")
sq = VectorIndex(DIM, metric="cosine", quantization="scalar")
sq.add(ids, vecs)
raw = N * DIM * 4
comp = N * DIM
print(f"   storage: {raw / 1e6:.1f} MB float32  ->  {comp / 1e6:.1f} MB uint8")
res = sq.search(q, k=10)
print(f"   recall@10 vs exact: {len({r[0] for r in res} & truth) / 10:.2f}")

hr("6. Persistence: custom .vdb format + memory-mapped load")
path = os.path.join(tempfile.gettempdir(), "demo.vdb")
index.save(path)
loaded = VectorIndex.load(path)
mm = VectorIndex.mmap(path)
print(f"   saved {os.path.getsize(path) / 1e6:.1f} MB -> {path}")
print(f"   load()  top-1: {loaded.search(q, k=1)[0]}")
print(f"   mmap()  top-1: {mm.search(q, k=1)[0]}   (vectors stay on disk)")

hr("7. Batch + async search")
import asyncio
batch = index.search_batch(vecs[:3], k=3)
for i, r in enumerate(batch):
    print(f"   batch[{i}] -> {[x[0] for x in r]}")
async def go():
    return await index.search_async(vecs[9], k=3)
print(f"   async    -> {[x[0] for x in asyncio.run(go())]}")

print("\nDone.")
