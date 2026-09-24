"""Custom on-disk index format (``.vdb``).

Layout::

    ┌──────────────────────────────────────┐
    │ magic "VTRA" (4B) │ version u32      │
    │ section count u32                    │
    │ per section: name_len u8, name,      │
    │              offset u64, length u64  │
    ├──────────────────────────────────────┤
    │ "config"   JSON                      │
    │ "ids"      u32 count + utf-8 blobs   │
    │ "vectors"  u32 rows, u32 dim, f32[]  │
    │ "hnsw"     JSON graph state          │
    │ "metadata" JSON array                │
    │ "codes"    quantized codes           │
    │ "quantizer" JSON                     │
    └──────────────────────────────────────┘

``load_vectors_mmap`` exposes the vector block as an ``np.memmap`` so a large
index can be queried without reading every float into Python objects.
"""

from __future__ import annotations

import io
import json
import struct
from typing import BinaryIO, Dict, List, Tuple

import numpy as np

MAGIC = b"VTRA"
VERSION = 1
Header = Dict[str, Tuple[int, int]]  # section -> (offset, length)


def write(path: str, sections: Dict[str, bytes]) -> None:
    entries = list(sections.items())
    table_len = sum(1 + len(n.encode()) + 16 for n, _ in entries)
    offset = 4 + 4 + 4 + table_len
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<II", VERSION, len(entries)))
        cursor = offset
        for name, payload in entries:
            nb = name.encode()
            f.write(struct.pack("<B", len(nb)) + nb)
            f.write(struct.pack("<QQ", cursor, len(payload)))
            cursor += len(payload)
        for _, payload in entries:
            f.write(payload)


