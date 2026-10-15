"""Runtime card bucketing from a built abstraction (see regret.abstraction.build).

Preflop buckets are the 169 hand classes and the flop comes from a lookup table. Turn and river
come from their tables when loaded (training, evaluation on the build machine); a shipped bundle
leaves those tables out and computes features + nearest centroid instead (turn ~0.5 ms, river
~15 µs), which keeps it small.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from regret import _core
from regret.engine.cards import CardLike, indexer, to_ids

_STREET_BY_BOARD = {0: "preflop", 3: "flop", 4: "turn", 5: "river"}


@dataclass
class CardAbstraction:
    manifest: dict[str, Any]
    cluster_of_hole: NDArray[np.uint8]
    river_centroids: NDArray[np.float32]
    flop_table: NDArray[np.uint16]
    turn_table: NDArray[np.uint16] | None
    river_table: NDArray[np.uint16] | None
    turn_centroids: NDArray[np.float32] | None = None

    @classmethod
    def load(cls, directory: Path, river_table: bool = False) -> CardAbstraction:
        def arr(name: str) -> NDArray[Any]:
            a: NDArray[Any] = np.load(
                directory / f"{name}.npy", mmap_mode="r" if name.endswith("table") else None
            )
            return a

        return cls(
            manifest=json.loads((directory / "manifest.json").read_text()),
            cluster_of_hole=arr("cluster_of_hole"),
            river_centroids=arr("river_centroids"),
            flop_table=arr("flop_table"),
            turn_table=arr("turn_table") if (directory / "turn_table.npy").exists() else None,
            river_table=arr("river_table") if river_table else None,
            turn_centroids=arr("turn_centroids")
            if (directory / "turn_centroids.npy").exists()
            else None,
        )

    @property
    def turn_bins(self) -> int:
        return int(self.manifest["config"]["turn_bins"])

    def turn_bucket_live(self, hole: list[int], board: list[int]) -> int:
        """Turn bucket from features: river-equity histogram -> CDF -> nearest centroid."""
        if self.turn_centroids is None:
            raise ValueError("no turn table or turn centroids in this abstraction")
        counts = _core.turn_histogram_one(hole, board, self.turn_bins).astype(np.float32)
        cdf = np.cumsum(counts) / max(1.0, float(counts.sum()))
        return int(((self.turn_centroids - cdf) ** 2).sum(1).argmin())

    @property
    def buckets(self) -> dict[str, int]:
        return dict(self.manifest["buckets"])

    def bucket(self, hole: str | Sequence[CardLike], board: str | Sequence[CardLike] = "") -> int:
        h, b = to_ids(hole), to_ids(board)
        street = _STREET_BY_BOARD.get(len(b))
        if street is None or len(h) != 2:
            raise ValueError("need 2 hole cards and a 0, 3, 4 or 5 card board")
        idx = indexer(street).index(h + b)
        if street == "preflop":
            return idx
        if street == "flop":
            return int(self.flop_table[idx])
        if street == "turn":
            if self.turn_table is not None:
                return int(self.turn_table[idx])
            return self.turn_bucket_live(h, b)
        if self.river_table is not None:
            return int(self.river_table[idx])
        k = len(self.river_centroids[0])
        x = _core.river_ochs_one(h, b, self.cluster_of_hole, k)
        return int(((self.river_centroids - x) ** 2).sum(1).argmin())
