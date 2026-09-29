"""Agent bundles: everything needed to play a trained blueprint, in one directory.

    bundle/
      manifest.json      version, config hash, training iteration, sizes
      config.yaml        the hold'em config it was trained with
      strategy.npz       average strategy: sorted infoset keys, offsets, uint8 probabilities
      cards/             card abstraction for runtime bucketing (river table left out)

Works on any checkpoint, including one from a run still in progress.
"""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from regret import __version__
from regret.abstraction.build import artifact_dir
from regret.cfr.checkpoint import Checkpoint
from regret.cfr.trainer import RunSpec
from regret.eval.strategy import ArrayStrategy
from regret.utils.config import RegretConfig

_CARD_FILES = (
    "manifest.json",
    "cluster_of_hole.npy",
    "river_centroids.npy",
    "flop_table.npy",
    "turn_table.npy",
)


def export_bundle(cfg: RegretConfig, checkpoint: Checkpoint, out: Path) -> dict[str, Any]:
    spec = RunSpec.from_config(cfg)
    if checkpoint.header.get("fingerprint") != spec.fingerprint:
        raise ValueError(f"{checkpoint.path.name} wasn't trained with this config")
    solver = spec.make_solver()
    solver.load(checkpoint.payload)

    out.mkdir(parents=True, exist_ok=True)
    strategy = ArrayStrategy.from_solver(solver)
    strategy.save(out / "strategy.npz")
    (out / "cards").mkdir(exist_ok=True)
    src = artifact_dir(cfg.cards)
    for name in _CARD_FILES:
        shutil.copy2(src / name, out / "cards" / name)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False))

    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    manifest = {
        "format_version": 1,
        "regret_version": __version__,
        "name": cfg.name,
        "config_hash": cfg.config_hash(),
        "fingerprint": spec.fingerprint,
        "iteration": checkpoint.iteration,
        "trained_seconds": checkpoint.header.get("elapsed_s"),
        "infosets": len(strategy),
        "bytes": size,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def load_bundle(path: Path) -> tuple[RegretConfig, ArrayStrategy, Path]:
    """(config, strategy, card abstraction directory) of a bundle."""
    cfg = RegretConfig.model_validate(yaml.safe_load((path / "config.yaml").read_text()))
    return cfg, ArrayStrategy.load(path / "strategy.npz"), path / "cards"
