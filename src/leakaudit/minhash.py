"""MinHash signatures and banded LSH for near-duplicate candidates.

Candidates from LSH are always verified with exact Jaccard similarity, so
the reported pairs contain no false positives; LSH only controls recall.
"""
from __future__ import annotations

import zlib
from collections import defaultdict
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np

_MERSENNE = np.uint64((1 << 61) - 1)
_MAX32 = np.uint64(0xFFFFFFFF)


def _hash_shingles(sh: Set[str]) -> np.ndarray:
    return np.fromiter((zlib.crc32(s.encode("utf-8")) for s in sh), dtype=np.uint64, count=len(sh))


class MinHasher:
    def __init__(self, num_perm: int = 128, seed: int = 1):
        rng = np.random.RandomState(seed)
        self.num_perm = num_perm
        self.a = rng.randint(1, 2**31 - 1, size=num_perm).astype(np.uint64)
        self.b = rng.randint(0, 2**31 - 1, size=num_perm).astype(np.uint64)

    def signature(self, sh: Set[str]) -> np.ndarray:
        if not sh:
            return np.full(self.num_perm, _MAX32, dtype=np.uint64)
        hv = _hash_shingles(sh)[:, None]
        perm = (hv * self.a[None, :] + self.b[None, :]) % _MERSENNE & _MAX32
        return perm.min(axis=0)


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def near_duplicate_pairs(shingle_sets: Sequence[Set[str]], threshold: float = 0.8,
                         num_perm: int = 128, bands: int = 32, max_bucket: int = 200,
                         seed: int = 1) -> Tuple[List[Tuple[int, int, float]], float]:
    """Return verified pairs (i, j, jaccard) with jaccard >= threshold and the max jaccard seen."""
    mh = MinHasher(num_perm, seed)
    sigs = np.stack([mh.signature(s) for s in shingle_sets]) if shingle_sets else np.zeros((0, num_perm))
    rows = num_perm // bands
    candidates: Set[Tuple[int, int]] = set()
    for b in range(bands):
        buckets: Dict[bytes, List[int]] = defaultdict(list)
        block = sigs[:, b * rows:(b + 1) * rows]
        for d in range(len(shingle_sets)):
            buckets[block[d].tobytes()].append(d)
        for members in buckets.values():
            if 1 < len(members) < max_bucket:
                for x in range(len(members)):
                    for y in range(x + 1, len(members)):
                        candidates.add((members[x], members[y]))
    pairs, best = [], 0.0
    for i, j in candidates:
        jac = jaccard(shingle_sets[i], shingle_sets[j])
        best = max(best, jac)
        if jac >= threshold:
            pairs.append((i, j, jac))
    return pairs, best
