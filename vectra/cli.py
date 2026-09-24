"""vectra command line interface.

    vectra build embeddings.npy --index hnsw --metric cosine -o index.vdb
    vectra inspect index.vdb
    vectra benchmark index.vdb --queries queries.npy --k 10
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np


def _cmd_build(args) -> int:
    from .index import VectorIndex
    vecs = np.load(args.embeddings).astype(np.float32)
    idx = VectorIndex(vecs.shape[1], metric=args.metric, index=args.index,
                      M=args.M, ef_construction=args.ef_construction,
                      quantization=args.quantization)
    ids = [f"vec{i}" for i in range(len(vecs))]
    t0 = time.time()
    idx.add(ids, vecs)
    idx.save(args.output)
    print(f"built {args.index} index over {len(vecs)}x{vecs.shape[1]} vectors "
          f"in {time.time() - t0:.2f}s -> {args.output}")
    return 0


def _cmd_inspect(args) -> int:
    from .index import VectorIndex
    idx = VectorIndex.load(args.index)
    print(f"index:      {idx.index_type}")
    print(f"metric:     {idx.metric}")
    print(f"dimension:  {idx.dim}")
    print(f"vectors:    {len(idx)} live / {len(idx.ids)} stored")
    print(f"quantized:  {idx.quantization or 'no'}")
    if idx._graph is not None:
        print(f"hnsw:       M={idx._graph.M} ef_construction={idx._graph.ef_construction} "
              f"max_level={idx._graph.max_level}")
    return 0


