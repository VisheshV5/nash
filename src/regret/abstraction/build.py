"""Build the card abstraction: fit bucket centroids and fill per-street lookup tables.

Pipeline (each street's features depend on the previous step):

1. **Opponent clusters for OCHS.** Preflop classes (169) are grouped into k clusters by their
   all-in equity vs a random hand (1-D k-means).
2. **River.** Sample every hole on random river boards, compute OCHS (equity vs each opponent
   cluster), fit `river_buckets` centroids, then bucket every (hole, board) class.
3. **Turn.** Sample river-equity histograms on random turn boards, fit centroids on their CDFs,
   bucket every class. Turn buckets are ranked by mean equity for the flop step.
4. **Flop.** For every hole on every suit-distinct flop, histogram the turn buckets reached by the
   47 turn cards (in rank order), fit centroids on the CDFs (weighted by how often each flop
   occurs), bucket every class.

Preflop buckets are the 169 classes themselves (lossless).

Output (`artifacts/abstraction/<hash>/`): `manifest.json` plus `.npy` arrays. Tables are indexed
by the suit-isomorphic (hole, board) index for {2,3}, {2,4} and {2,5}. The river table
(~250 MB) is for training; the shipped agent computes river buckets from the centroids.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from regret import _core
from regret.abstraction.kmeans import kmeans
from regret.utils.config import CardAbstractionConfig
from regret.utils.seeding import derive_seed

log = logging.getLogger(__name__)

FORMAT_VERSION = 1
RIVER_ROWS_PER_BOARD = 1081
TURN_ROWS_PER_BOARD = 1128


def artifact_dir(cfg: CardAbstractionConfig, root: Path | None = None) -> Path:
    """Where the abstraction for `cfg` lives: $REGRET_ARTIFACTS (default ./artifacts)/abstraction/
    <hash of the cards config>."""
    if root is None:
        root = Path(os.environ.get("REGRET_ARTIFACTS", "artifacts")) / "abstraction"
    return root / cfg.config_hash()


def _sha(a: NDArray[Any]) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def _cdf(counts: NDArray[np.uint8]) -> NDArray[np.float32]:
    c = counts.astype(np.float32).cumsum(1)
    out: NDArray[np.float32] = (c / np.maximum(c[:, -1:], 1)).astype(np.float32)
    return out


def opponent_clusters(cfg: CardAbstractionConfig, threads: int) -> NDArray[np.uint8]:
    """Cluster id of every hole index (for OCHS), from preflop equity."""
    eq = np.asarray(
        _core.preflop_class_equity(
            cfg.preflop_equity_samples, derive_seed(cfg.seed, "preflop"), threads
        )
    )
    res = kmeans(
        eq[:, None].astype(np.float32),
        cfg.river_opponent_clusters,
        seed=derive_seed(cfg.seed, "ochs"),
    )
    # Relabel clusters in ascending equity so ids are meaningful.
    order = np.argsort(res.centroids[:, 0])
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    cluster_of_class = rank[res.labels].astype(np.uint8)
    return cluster_of_class[_core.preflop_class_of_hole()]


def build_card_abstraction(
    cfg: CardAbstractionConfig, out_dir: Path, threads: int = 6
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    t_all = time.time()

    def step(name: str, t0: float) -> None:
        timings[name] = round(time.time() - t0, 2)
        log.info("%s done in %.1fs", name, timings[name])

    # 1. opponent clusters
    t = time.time()
    cluster_of = opponent_clusters(cfg, threads)
    k_opp = cfg.river_opponent_clusters
    step("opponent_clusters", t)

    # 2. river
    t = time.time()
    boards = -(-cfg.feature_sample_size // RIVER_ROWS_PER_BOARD)
    x = _core.sample_river_ochs(
        boards, derive_seed(cfg.seed, "river-sample"), cluster_of, k_opp, threads
    )
    river = kmeans(x, cfg.river_buckets, seed=derive_seed(cfg.seed, "river-kmeans"))
    del x
    step("river_fit", t)
    t = time.time()
    river_table = _core.build_river_table(cluster_of, k_opp, river.centroids.ravel(), threads)
    step("river_table", t)

    # 3. turn
    t = time.time()
    boards = -(-cfg.feature_sample_size // TURN_ROWS_PER_BOARD)
    counts = _core.sample_turn_histograms(
        boards, derive_seed(cfg.seed, "turn-sample"), cfg.turn_bins, threads
    )
    turn = kmeans(_cdf(counts), cfg.turn_buckets, seed=derive_seed(cfg.seed, "turn-kmeans"))
    del counts
    step("turn_fit", t)
    t = time.time()
    turn_table = _core.build_turn_table(cfg.turn_bins, turn.centroids.ravel(), threads)
    # Mean equity of a CDF over equal-width bins of [0, 1] = sum(1 - F) / bins, up to the
    # half-bin offset, which doesn't change the order.
    mean_eq = (1.0 - turn.centroids).sum(1) / cfg.turn_bins
    order = np.argsort(mean_eq, kind="stable")
    turn_rank = np.empty(cfg.turn_buckets, np.uint16)
    turn_rank[order] = np.arange(cfg.turn_buckets, dtype=np.uint16)
    step("turn_table", t)

    # 4. flop
    t = time.time()
    flop_counts, flop_weights = _core.flop_features(turn_table, turn_rank, threads)
    flop = kmeans(
        _cdf(flop_counts),
        cfg.flop_buckets,
        weights=flop_weights,
        seed=derive_seed(cfg.seed, "flop-kmeans"),
    )
    del flop_counts
    step("flop_fit", t)
    t = time.time()
    flop_table = _core.build_flop_table(turn_table, turn_rank, flop.centroids.ravel(), threads)
    step("flop_table", t)

    arrays: dict[str, NDArray[Any]] = {
        "cluster_of_hole": cluster_of,
        "river_centroids": river.centroids,
        "turn_centroids": turn.centroids,
        "turn_rank": turn_rank,
        "flop_centroids": flop.centroids,
        "flop_table": flop_table,
        "turn_table": turn_table,
        "river_table": river_table,
    }
    for name, a in arrays.items():
        np.save(out_dir / f"{name}.npy", a)

    def usage(table: NDArray[np.uint16], k: int) -> dict[str, int]:
        c = np.bincount(table, minlength=k)
        return {"empty": int((c == 0).sum()), "min": int(c.min()), "max": int(c.max())}

    manifest: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "config": cfg.model_dump(mode="json"),
        "config_hash": cfg.config_hash(),
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "buckets": {
            "preflop": 169,
            "flop": cfg.flop_buckets,
            "turn": cfg.turn_buckets,
            "river": cfg.river_buckets,
        },
        "table_sizes": {
            "flop": len(flop_table),
            "turn": len(turn_table),
            "river": len(river_table),
        },
        "bucket_usage": {
            "flop": usage(flop_table, cfg.flop_buckets),
            "turn": usage(turn_table, cfg.turn_buckets),
            "river": usage(river_table, cfg.river_buckets),
        },
        "kmeans": {
            name: {"inertia": r.inertia, "iterations": r.iterations}
            for name, r in (("river", river), ("turn", turn), ("flop", flop))
        },
        "sha256": {name: _sha(a) for name, a in arrays.items()},
        "bytes": {name: int(a.nbytes) for name, a in arrays.items()},
        "timings_s": timings | {"total": round(time.time() - t_all, 2)},
        "threads": threads,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("card abstraction written to %s in %.1fs", out_dir, time.time() - t_all)
    return manifest
