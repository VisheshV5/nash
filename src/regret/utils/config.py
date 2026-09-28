"""YAML → validated config models.

Everything tunable (table, bet sizes, bucket counts, training, search) lives in one
`RegretConfig`. Unknown keys are errors, so a typo in a YAML file fails loudly instead of
silently falling back to a default.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated, Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

PositiveFloat = Annotated[float, Field(gt=0)]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _check_sizes(name: str, sizes: list[float]) -> None:
    if sizes != sorted(set(sizes)):
        raise ValueError(f"{name} must be strictly increasing with no duplicates, got {sizes}")


class TableConfig(_Strict):
    """Table rules. All amounts are in big blinds (the big blind is 1.0)."""

    num_players: int = Field(ge=2, le=6)
    starting_stack_bb: PositiveFloat
    small_blind_bb: float = Field(default=0.5, gt=0, lt=1)


class PreflopSizing(_Strict):
    open_bb: list[PositiveFloat] = Field(min_length=1)
    """Opening raise sizes, as the total amount raised to."""
    reraise_x_ip: list[PositiveFloat] = Field(min_length=1)
    """3-bet sizes in position, as multiples of the bet being faced."""
    reraise_x_oop: list[PositiveFloat] = Field(min_length=1)
    """3-bet sizes out of position, as multiples of the bet being faced."""
    four_bet_x: list[PositiveFloat] = Field(min_length=1)
    """4-bet and later sizes, as multiples of the bet being faced."""
    allow_all_in: bool = True
    max_raises: int = Field(default=4, ge=1, le=8)

    @model_validator(mode="after")
    def _sorted(self) -> Self:
        for name in ("open_bb", "reraise_x_ip", "reraise_x_oop", "four_bet_x"):
            _check_sizes(name, getattr(self, name))
        if any(x <= 1 for x in (*self.reraise_x_ip, *self.reraise_x_oop, *self.four_bet_x)):
            raise ValueError("raise multipliers must be > 1")
        return self


class PostflopSizing(_Strict):
    bet_pot: list[PositiveFloat] = Field(min_length=1)
    """First-bet sizes as fractions of the pot."""
    raise_pot: list[PositiveFloat] = Field(default_factory=list)
    """Raise sizes as fractions of the pot after calling. Empty means all-in only."""
    allow_all_in: bool = True

    @model_validator(mode="after")
    def _sorted(self) -> Self:
        _check_sizes("bet_pot", self.bet_pot)
        _check_sizes("raise_pot", self.raise_pot)
        return self


class ActionAbstractionConfig(_Strict):
    preflop: PreflopSizing
    postflop: dict[int, PostflopSizing]
    """Keyed by the minimum number of active players a rule applies to.

    `{2: A, 3: B}` means A for heads-up pots and B for pots with 3 or more players.
    """

    @model_validator(mode="after")
    def _keys(self) -> Self:
        if 2 not in self.postflop:
            raise ValueError("postflop sizing needs a rule for 2 active players")
        if not all(2 <= k <= 6 for k in self.postflop):
            raise ValueError(f"postflop keys must be in 2..6, got {sorted(self.postflop)}")
        return self

    def postflop_for(self, active_players: int) -> PostflopSizing:
        """Sizing rule for a pot with `active_players` players still in the hand."""
        if active_players < 2:
            raise ValueError(f"need at least 2 active players, got {active_players}")
        key = max(k for k in self.postflop if k <= active_players)
        return self.postflop[key]


class CardAbstractionConfig(_Strict):
    flop_buckets: int = Field(ge=2, le=4096)
    turn_buckets: int = Field(ge=2, le=4096)
    river_buckets: int = Field(ge=2, le=4096)
    river_opponent_clusters: int = Field(default=8, ge=1, le=64)
    """Opponent-hand clusters used for river OCHS features."""
    feature_sample_size: int = Field(default=10_000_000, ge=1)
    """Isomorphic turn/river states sampled to fit centroids (ROADMAP M3)."""


class TrainingConfig(_Strict):
    seed: int = Field(default=0, ge=0)
    deterministic: bool = False
    """Single-threaded, so a run (and any resume of it) is bit-for-bit reproducible."""
    threads: int = Field(default=6, ge=1, le=256)
    memory_budget_gb: PositiveFloat = 10.0
    run_dir: Path | None = None
    """Checkpoints, logs and metrics go here. Defaults to runs/<name>."""
    checkpoint_every_iterations: int | None = Field(default=None, ge=1)
    checkpoint_every_minutes: PositiveFloat = 15.0
    keep_last_checkpoints: int = Field(default=2, ge=1)
    log_every_seconds: PositiveFloat = 10.0

    @model_validator(mode="after")
    def _deterministic_single_thread(self) -> Self:
        if self.deterministic and self.threads != 1:
            raise ValueError("training.deterministic needs training.threads: 1")
        return self


class CfrConfig(_Strict):
    """MCCFR schedule: Linear CFR discounting, negative-regret pruning, table size."""

    lcfr_until_iteration: int = Field(default=0, ge=0)
    """Discount regrets and strategy sums by t/(t+1) every interval until here. 0 = off."""
    discount_interval: int = Field(default=1000, ge=1)
    prune_after_iteration: int | None = Field(default=None, ge=0)
    """Start negative-regret pruning here. None = never."""
    prune_probability: float = Field(default=0.95, ge=0, le=1)
    prune_threshold: float = -1e9
    regret_floor: float | None = None
    max_infosets: int = Field(default=1 << 20, ge=16)
    """Regret table capacity (fixed up front; the memory planner sizes it for hold'em)."""
    target_iterations: int | None = Field(default=None, ge=1)
    """Stop here. None = run until stopped."""


class EvalConfig(_Strict):
    every_iterations: int | None = Field(default=None, ge=1)
    every_minutes: PositiveFloat | None = 10.0


class SearchConfig(_Strict):
    enabled: bool = True
    time_budget_ms: int = Field(default=800, ge=1)
    hard_cap_ms: int = Field(default=1500, ge=1)
    threads: int = Field(default=6, ge=1, le=256)
    continuation_strategies: int = Field(default=4, ge=1, le=8)

    @model_validator(mode="after")
    def _budget(self) -> Self:
        if self.time_budget_ms >= self.hard_cap_ms:
            raise ValueError("search.time_budget_ms must be below search.hard_cap_ms")
        return self


class RegretConfig(_Strict):
    name: str = Field(min_length=1)
    table: TableConfig
    actions: ActionAbstractionConfig
    cards: CardAbstractionConfig
    training: TrainingConfig = TrainingConfig()
    cfr: CfrConfig = CfrConfig()
    eval: EvalConfig = EvalConfig()
    search: SearchConfig = SearchConfig()

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        too_many = [k for k in self.actions.postflop if k > self.table.num_players]
        if too_many:
            raise ValueError(
                f"postflop sizing keys {too_many} exceed table.num_players={self.table.num_players}"
            )
        return self

    def config_hash(self) -> str:
        """Stable short hash of the full config, for bundle manifests and run names."""
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]


ToyGame = Literal["kuhn", "kuhn3", "leduc"]


class ToyGameConfig(_Strict):
    """Training config for the small validation games (exact exploitability is available)."""

    name: str = Field(min_length=1)
    game: ToyGame
    training: TrainingConfig = TrainingConfig(threads=1, deterministic=True)
    cfr: CfrConfig = CfrConfig()
    eval: EvalConfig = EvalConfig()

    def config_hash(self) -> str:
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")
    return data


def load_config(path: str | Path) -> RegretConfig:
    """Load and validate a hold'em YAML config file."""
    return RegretConfig.model_validate(_read_yaml(path))


def load_train_config(path: str | Path) -> RegretConfig | ToyGameConfig:
    """Load any training config: toy games have a top-level `game` key."""
    data = _read_yaml(path)
    if "game" in data:
        return ToyGameConfig.model_validate(data)
    return RegretConfig.model_validate(data)
