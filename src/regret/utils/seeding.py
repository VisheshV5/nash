"""Deterministic seed derivation.

One master seed fans out into independent, named streams, e.g.
`make_rng(seed, "eval", "deal")` or `derive_seed(seed, "cfr", "worker", "3")` for the C++
PCG streams. Names are hashed with BLAKE2 (not Python's salted `hash`), so a given
(master, names) pair gives the same stream on every run and machine.
"""

from __future__ import annotations

import hashlib

import numpy as np


def _name_key(name: str) -> int:
    return int.from_bytes(hashlib.blake2b(name.encode(), digest_size=4).digest(), "little")


def seed_sequence(master: int, *names: str) -> np.random.SeedSequence:
    if master < 0:
        raise ValueError(f"master seed must be non-negative, got {master}")
    return np.random.SeedSequence(entropy=master, spawn_key=tuple(_name_key(n) for n in names))


def derive_seed(master: int, *names: str) -> int:
    """A 64-bit seed for the stream `names`, e.g. to hand to C++."""
    lo, hi = seed_sequence(master, *names).generate_state(2, dtype=np.uint32)
    return int(lo) | (int(hi) << 32)


def make_rng(master: int, *names: str) -> np.random.Generator:
    """A NumPy generator for the stream `names`."""
    return np.random.Generator(np.random.PCG64(seed_sequence(master, *names)))
