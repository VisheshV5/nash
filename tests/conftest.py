"""Shared fixtures: a tiny card abstraction (built once per test session) and a hold'em config
that uses it, so training, export and evaluation can be tested end to end in seconds."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from regret.abstraction.build import artifact_dir, build_card_abstraction
from regret.utils.config import RegretConfig, load_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture(scope="session")
def artifacts_root(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    root = tmp_path_factory.mktemp("artifacts")
    mp = pytest.MonkeyPatch()
    mp.setenv("REGRET_ARTIFACTS", str(root))
    yield root
    mp.undo()


@pytest.fixture(scope="session")
def tiny_holdem(artifacts_root: Path) -> RegretConfig:
    """Heads-up 100bb with the default bet sizes and 6 buckets per postflop street."""
    data = load_config(CONFIGS / "hu_default.yaml").model_dump(mode="json")
    data["name"] = "hu-tiny"
    data["cards"] |= {
        "flop_buckets": 6,
        "turn_buckets": 6,
        "river_buckets": 6,
        "feature_sample_size": 30_000,
        "preflop_equity_samples": 1_000,
    }
    data["training"] |= {"threads": 1, "deterministic": True, "log_every_seconds": 0.5}
    data["cfr"] |= {
        "lcfr_until_iteration": 20_000,
        "discount_interval": 2_000,
        "prune_after_iteration": 30_000,
        "max_infosets": 200_000,
    }
    data["eval"] = {"every_minutes": None, "hands": 40}
    cfg = RegretConfig.model_validate(data)
    build_card_abstraction(cfg.cards, artifact_dir(cfg.cards), threads=6)
    return cfg
