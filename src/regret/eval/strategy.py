"""Read-only blueprint strategies, looked up by (abstract action history, card bucket).

Three stores:

- SolverStrategy: straight from a live or checkpoint-loaded solver.
- ArrayStrategy: the solver's hash-keyed export (sorted keys, offsets, probabilities).
- DenseStrategy: the bundle format. One block per decision node of the abstract betting tree,
  `buckets x actions` uint8 probabilities, so no per-infoset keys are stored: about a third
  of the size of the keyed export. A block row of zeros means the infoset was never visited.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from regret import _core

U64 = NDArray[np.uint64]


class StrategyTable(Protocol):
    def probs(self, history: int, bucket: int) -> NDArray[np.float64] | None:
        """Average strategy at an infoset (over its abstract actions), or None if never seen."""
        ...


# ---------------------------------------------------------------- the trainer's infoset keys

_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)
_BUCKET_MUL = np.uint64(0xD1B54A32D192ED03)


def splitmix64(x: U64) -> U64:
    with np.errstate(over="ignore"):
        x = x + _GOLDEN
        x = (x ^ (x >> np.uint64(30))) * _M1
        x = (x ^ (x >> np.uint64(27))) * _M2
        out: U64 = x ^ (x >> np.uint64(31))
        return out


def infoset_keys(history: int, buckets: NDArray[np.integer]) -> U64:
    """Vectorized Nlhe::infoset_key (C++): the hash the trainer stores each infoset under."""
    with np.errstate(over="ignore"):
        b = buckets.astype(np.uint64) + np.uint64(1)
        return splitmix64(np.uint64(history) + _BUCKET_MUL * b)


# ---------------------------------------------------------------- stores


class SolverStrategy:
    def __init__(self, solver: _core.CfrSolver) -> None:
        self.solver = solver

    def probs(self, history: int, bucket: int) -> NDArray[np.float64] | None:
        p = self.solver.strategy(_core.nlhe_infoset_key(history, bucket))
        return np.asarray(p) if p else None


class ArrayStrategy:
    """Hash-keyed flat arrays, as the solver exports them."""

    def __init__(self, keys: U64, offsets: NDArray[np.uint32], probs: NDArray[np.floating]) -> None:
        self.keys, self.offsets, self.table = keys, offsets, probs

    @classmethod
    def from_solver(cls, solver: _core.CfrSolver) -> ArrayStrategy:
        return cls(*solver.export_strategy())

    @classmethod
    def load(cls, path: Path) -> ArrayStrategy:
        with np.load(path) as z:
            probs = z["probs"]
            if probs.dtype == np.uint8:
                probs = probs.astype(np.float32)
            return cls(z["keys"], z["offsets"], probs)

    def save(self, path: Path, quantize: bool = True) -> None:
        probs = self.table
        if quantize:
            probs = np.round(np.asarray(probs, np.float32) * 255).astype(np.uint8)
        np.savez(path, keys=self.keys, offsets=self.offsets, probs=probs)

    def __len__(self) -> int:
        return len(self.keys)

    def probs_by_key(self, key: int) -> NDArray[np.float64] | None:
        i = int(np.searchsorted(self.keys, np.uint64(key)))
        if i == len(self.keys) or int(self.keys[i]) != key:
            return None
        return _normalized(self.table[self.offsets[i] : self.offsets[i + 1]])

    def probs(self, history: int, bucket: int) -> NDArray[np.float64] | None:
        return self.probs_by_key(_core.nlhe_infoset_key(history, bucket))


@dataclass(frozen=True)
class Node:
    history: int
    street: int
    actions: int


def decision_nodes(rules: _core.ActionRules, stacks: list[int], sb: int, bb: int) -> list[Node]:
    """Every decision node of the abstract betting tree (no cards)."""
    out: list[Node] = []

    def walk(state: _core.HandState, history: int) -> None:
        if state.is_terminal:
            return
        acts = _core.abstract_actions(state, rules)
        out.append(Node(history, int(state.street), len(acts)))
        for i, a in enumerate(acts):
            child = state.copy()
            child.apply(a)
            walk(child, _core.nlhe_extend_history(history, i))

    walk(_core.HandState(stacks, sb, bb, record_history=False), _core.nlhe_history_root())
    return out


class DenseStrategy:
    """Dense per-node blocks; see the module docstring."""

    def __init__(
        self,
        history: U64,
        offset: NDArray[np.uint64],
        actions: NDArray[np.uint8],
        buckets: NDArray[np.uint16],
        probs: NDArray[np.uint8],
    ) -> None:
        order = np.argsort(history)
        self.history, self.offset = history[order], offset[order]
        self.actions, self.buckets, self.table = actions[order], buckets[order], probs

    @classmethod
    def build(
        cls,
        source: ArrayStrategy,
        nodes: list[Node],
        buckets_per_street: tuple[int, int, int, int],
    ) -> DenseStrategy:
        n = len(nodes)
        history = np.array([nd.history for nd in nodes], np.uint64)
        actions = np.array([nd.actions for nd in nodes], np.uint8)
        buckets = np.array([buckets_per_street[nd.street] for nd in nodes], np.uint16)
        sizes = buckets.astype(np.uint64) * actions.astype(np.uint64)
        offset = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.uint64)
        table = np.zeros(int(sizes.sum()), np.uint8)
        quantized = np.round(np.asarray(source.table, np.float32) * 255).astype(np.uint8)
        for i in range(n):
            k, nb = int(actions[i]), int(buckets[i])
            keys = infoset_keys(int(history[i]), np.arange(nb))
            pos = np.searchsorted(source.keys, keys)
            pos = np.minimum(pos, len(source.keys) - 1)
            found = source.keys[pos] == keys
            starts = source.offsets[pos[found]].astype(np.int64)
            block = table[int(offset[i]) : int(offset[i]) + nb * k].reshape(nb, k)
            if found.any():
                rows = quantized[starts[:, None] + np.arange(k)[None, :]]
                block[np.flatnonzero(found)] = rows
        return cls(history, offset, actions, buckets, table)

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            history=self.history,
            offset=self.offset,
            actions=self.actions,
            buckets=self.buckets,
            probs=self.table,
        )

    @classmethod
    def load(cls, path: Path) -> DenseStrategy:
        with np.load(path) as z:
            return cls(z["history"], z["offset"], z["actions"], z["buckets"], z["probs"])

    def __len__(self) -> int:
        return int((self.buckets.astype(np.int64)).sum())

    def probs(self, history: int, bucket: int) -> NDArray[np.float64] | None:
        i = int(np.searchsorted(self.history, np.uint64(history)))
        if i == len(self.history) or int(self.history[i]) != history:
            return None
        if bucket < 0 or bucket >= int(self.buckets[i]):
            return None
        k = int(self.actions[i])
        start = int(self.offset[i]) + bucket * k
        return _normalized(self.table[start : start + k])


def _normalized(row: NDArray[np.number]) -> NDArray[np.float64] | None:
    p = np.asarray(row, np.float64)
    total = p.sum()
    if total <= 0:
        return None
    out: NDArray[np.float64] = p / total
    return out
