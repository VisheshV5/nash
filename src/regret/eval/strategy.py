"""Read-only blueprint strategies, keyed like the trainer: (abstract action history, bucket)."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from regret import _core


class StrategyTable(Protocol):
    def probs(self, key: int) -> NDArray[np.float64] | None:
        """Average strategy at an infoset (over its abstract actions), or None if never seen."""
        ...


class SolverStrategy:
    """Straight from a live or checkpoint-loaded solver."""

    def __init__(self, solver: _core.CfrSolver) -> None:
        self.solver = solver

    def probs(self, key: int) -> NDArray[np.float64] | None:
        p = self.solver.strategy(key)
        return np.asarray(p) if p else None


class ArrayStrategy:
    """Flat arrays (as exported into a bundle): sorted keys, offsets, probabilities."""

    def __init__(
        self, keys: NDArray[np.uint64], offsets: NDArray[np.uint32], probs: NDArray[np.floating]
    ) -> None:
        self.keys, self.offsets, self.table = keys, offsets, probs

    @classmethod
    def from_solver(cls, solver: _core.CfrSolver) -> ArrayStrategy:
        return cls(*solver.export_strategy())

    @classmethod
    def load(cls, path: Path) -> ArrayStrategy:
        with np.load(path) as z:
            probs = z["probs"]
            if probs.dtype == np.uint8:  # quantized: renormalize per infoset on read
                probs = probs.astype(np.float32)
            return cls(z["keys"], z["offsets"], probs)

    def save(self, path: Path, quantize: bool = True) -> None:
        probs = self.table
        if quantize:
            probs = np.round(np.asarray(probs, np.float32) * 255).astype(np.uint8)
        np.savez(path, keys=self.keys, offsets=self.offsets, probs=probs)

    def __len__(self) -> int:
        return len(self.keys)

    def probs(self, key: int) -> NDArray[np.float64] | None:
        i = int(np.searchsorted(self.keys, np.uint64(key)))
        if i == len(self.keys) or int(self.keys[i]) != key:
            return None
        p = np.asarray(self.table[self.offsets[i] : self.offsets[i + 1]], np.float64)
        total = p.sum()
        return p / total if total > 0 else np.full(len(p), 1 / len(p))
