"""Wiring for hold'em evaluation: chips from a config, loading players, baseline reports."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from regret import _core
from regret.abstraction.actions import rules_from_config
from regret.abstraction.build import artifact_dir
from regret.abstraction.cards import CardAbstraction
from regret.engine.state import CHIPS_PER_BB
from regret.eval.match import duplicate_match
from regret.eval.players import AlwaysCall, BlueprintPlayer, EquityThreshold, Player, RandomPlayer
from regret.eval.strategy import SolverStrategy, StrategyTable
from regret.utils.config import RegretConfig


def table_chips(cfg: RegretConfig) -> tuple[list[int], int, int]:
    t = cfg.table
    return (
        [round(t.starting_stack_bb * CHIPS_PER_BB)] * t.num_players,
        round(t.small_blind_bb * CHIPS_PER_BB),
        CHIPS_PER_BB,
    )


def blueprint_player(
    cfg: RegretConfig,
    strategy: StrategyTable,
    cards: CardAbstraction | None = None,
    cards_dir: Path | None = None,
    name: str = "blueprint",
) -> BlueprintPlayer:
    if cards is None:
        cards = CardAbstraction.load(cards_dir or artifact_dir(cfg.cards))
    stacks, sb, bb = table_chips(cfg)
    return BlueprintPlayer(strategy, cards, rules_from_config(cfg.actions), stacks, sb, bb, name)


BASELINES: dict[str, Callable[[], Player]] = {
    "random": RandomPlayer,
    "always-call": AlwaysCall,
    "equity-threshold": EquityThreshold,
}


def baseline_report(
    cfg: RegretConfig, solver: _core.CfrSolver, hands: int, seed: int = 0
) -> dict[str, float]:
    """mbb/hand (and 95% CI) of the blueprint against each baseline."""
    bp = blueprint_player(cfg, SolverStrategy(solver))
    stacks, sb, bb = table_chips(cfg)
    out: dict[str, float] = {}
    for name, cls in BASELINES.items():
        r = duplicate_match(bp, cls(), max(1, hands // 2), stacks, sb, bb, seed=seed)
        out[f"vs_{name}_mbb"] = round(r.mbb, 1)
        out[f"vs_{name}_ci95"] = round(r.ci95, 1)
    out["unseen_infosets"] = bp.unseen
    out["fallbacks"] = bp.fallbacks
    return out
