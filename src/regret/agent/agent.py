"""The public agent: load a bundle, hand it a game state, get a strategy and an action.

    agent = RegretAgent.load("bundles/hu-100bb")
    decision = agent.act(state)          # SPEC §4 GameState (dict, JSON string or model)
    decision.strategy                    # {"fold": 0.1, "call": 0.55, "raise_3x": 0.25, ...}
    decision.recommended                 # {"action": "call", "amount": 4.0}

How a decision is made (blueprint mode):

1. The input is validated and replayed through the rules engine (regret.engine.state).
2. The real action history is mapped onto the blueprint's abstract game. A bet the abstraction
   doesn't have is translated to its two neighbouring sizes with pseudo-harmonic weights; the
   agent follows up to `max_branches` of the most likely resulting abstract histories.
3. In each branch, the hero's cards are bucketed for the current street and the blueprint's
   average strategy is looked up. Abstract actions are turned into real ones (same pot
   fraction, or all-in) and the branches are mixed by their probability.
4. The recommended action is sampled from that strategy (or the most likely action with
   `mode="argmax"`). Pass `seed` for reproducible samples.

Amounts are in the input's units (big blinds when big_blind is 1.0). As in the input, a bet,
raise or call amount is the player's total bet on the street after the action.
"""

from __future__ import annotations

import json
import time
import warnings
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from regret import _core
from regret.abstraction.actions import action_label, rules_from_config
from regret.abstraction.cards import CardAbstraction
from regret.engine.state import (
    CHIPS_PER_BB,
    GameStateError,
    GameStateIn,
    ValidatedState,
    parse_game_state,
)
from regret.eval.players import AbstractTracker
from regret.eval.strategy import ArrayStrategy, StrategyTable
from regret.export.bundle import load_bundle
from regret.search.river import Step, bayes_ranges, river_start
from regret.utils.config import RegretConfig

Mode = Literal["sample", "argmax"]
Source = Literal["blueprint", "subgame_solve"]


class UnsupportedTableSize(GameStateError):
    pass


@dataclass
class Decision:
    strategy: dict[str, float]
    recommended: dict[str, Any]
    source: Source
    latency_ms: float
    explain: dict[str, Any] | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "strategy": self.strategy,
            "recommended": self.recommended,
            "source": self.source,
            "latency_ms": self.latency_ms,
        }
        if self.explain is not None:
            out["explain"] = self.explain
        if self.warnings:
            out["warnings"] = self.warnings
        return out


@dataclass
class _Branch:
    prob: float
    tracker: AbstractTracker
    translations: list[str]
    steps: list[Step]


class RegretAgent:
    def __init__(
        self,
        config: RegretConfig,
        strategy: StrategyTable,
        cards: CardAbstraction,
        manifest: dict[str, Any] | None = None,
        max_branches: int = 8,
    ) -> None:
        self.config = config
        self.strategy = strategy
        self.cards = cards
        self.manifest = manifest or {}
        self.rules = rules_from_config(config.actions)
        self.max_branches = max_branches
        t = config.table
        self.trained_stack_bb = t.starting_stack_bb
        self._abstract_stacks = [round(t.starting_stack_bb * CHIPS_PER_BB)] * t.num_players
        self._abstract_blinds = (round(t.small_blind_bb * CHIPS_PER_BB), CHIPS_PER_BB)

    @classmethod
    def load(cls, path: str | Path, max_branches: int = 8) -> RegretAgent:
        """Load a bundle from a directory, or by registry name (downloaded and cached once)."""
        path = Path(path)
        if not path.exists() and len(path.parts) == 1:
            from regret.registry import resolve

            path = resolve(str(path))
        cfg, strategy, cards_dir = load_bundle(path)
        manifest = json.loads((path / "manifest.json").read_text())
        return cls(cfg, strategy, CardAbstraction.load(cards_dir), manifest, max_branches)

    @classmethod
    def from_strategy(
        cls, config: RegretConfig, strategy: StrategyTable, cards: CardAbstraction
    ) -> RegretAgent:
        return cls(config, strategy, cards)

    def info(self) -> dict[str, Any]:
        return {
            "name": self.config.name,
            "players": self.config.table.num_players,
            "starting_stack_bb": self.trained_stack_bb,
            "infosets": len(self.strategy) if isinstance(self.strategy, ArrayStrategy) else None,
            "bundle": self.manifest,
        }

    # ------------------------------------------------------------------ main entry

    def act(
        self,
        state: GameStateIn | dict[str, Any] | str,
        *,
        explain: bool = False,
        mode: Mode = "sample",
        seed: int | None = None,
        search: bool | None = None,
    ) -> Decision:
        """Decide for the player to act in `state` (SPEC §4). `search` overrides the config."""
        t0 = time.perf_counter()
        v = parse_game_state(state)
        if v.engine.num_players != self.config.table.num_players:
            raise UnsupportedTableSize.one(
                "num_players",
                f"this bundle plays {self.config.table.num_players}-handed; got "
                f"{v.engine.num_players} players (6-max arrives in v1.2)",
            )
        e = v.engine
        raw = v.raw
        sb = round(
            (raw.small_blind if raw.small_blind is not None else raw.big_blind / 2)
            * v.chips_per_unit
        )
        decision = self.decide(
            e,
            [e.stacks[i] + e.contributed[i] for i in range(e.num_players)],
            sb,
            round(raw.big_blind * v.chips_per_unit),
            v.hero_cards,
            v.board,
            to_units=v.to_units,
            explain=explain,
            mode=mode,
            seed=seed,
            search=search,
        )
        decision.warnings = self._stack_warnings(v) + decision.warnings
        decision.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        return decision

    def decide(
        self,
        real: _core.HandState,
        starting: list[int],
        sb: int,
        bb: int,
        hero_cards: list[int],
        board: list[int],
        *,
        to_units: Callable[[int], float] = lambda c: c / CHIPS_PER_BB,
        explain: bool = False,
        mode: Mode = "sample",
        seed: int | None = None,
        search: bool | None = None,
    ) -> Decision:
        """Core decision on an engine state that recorded its history (no JSON involved)."""
        t0 = time.perf_counter()
        branches = self._replay(real.history, starting, sb, bb)
        bucket = self.cards.bucket(hero_cards, board)
        use_search = (self.config.search.enabled if search is None else search) and (
            real.street == _core.Street.RIVER and real.num_players == 2
        )

        mixed: dict[tuple[int, int], float] = defaultdict(float)
        actions: dict[tuple[int, int], _core.Action] = {}
        fallback_mass = 0.0
        solve: dict[str, Any] | None = None
        hero = _core.hole_index(*hero_cards)
        if use_search:
            solve = self._solve_river(real, starting, sb, bb, branches[0], board)
            for i, action in enumerate(solve["actions"]):
                p = float(solve["strategy"][i, hero])
                if p > 0:
                    key = (int(action.type), action.amount)
                    mixed[key] += p
                    actions[key] = action
        else:
            for br in branches:
                for action, p in self._branch_policy(br, real, bucket):
                    key = (int(action.type), action.amount)
                    mixed[key] += br.prob * p
                    actions[key] = action
                if not br.tracker.in_sync(real):
                    fallback_mass += br.prob
        total = sum(mixed.values())
        ordered = sorted(mixed, key=lambda k: (k[0], k[1]))

        strategy: dict[str, float] = {}
        labels: dict[tuple[int, int], str] = {}
        for k in ordered:
            label = action_label(real, actions[k])
            labels[k] = label
            strategy[label] = strategy.get(label, 0.0) + mixed[k] / total
        strategy = {k: round(p, 4) for k, p in strategy.items()}

        probs = np.array([mixed[k] / total for k in ordered])
        if mode == "argmax":
            pick = int(probs.argmax())
        else:
            rng = np.random.default_rng(seed)
            pick = int(rng.choice(len(ordered), p=probs / probs.sum()))
        chosen = actions[ordered[pick]]
        recommended = self._recommended(real, chosen, labels[ordered[pick]], to_units)

        explanation = None
        if explain:
            street = ["preflop", "flop", "turn", "river"][int(real.street)]
            explanation = {
                "street": street,
                "bucket": bucket,
                "buckets_on_street": self.cards.buckets[street],
                "equity_vs_random_hand": round(
                    _core.equity_vs_range(
                        hero_cards, board, np.ones(1326, np.float32), 500, seed or 0
                    ),
                    4,
                ),
                "pot": to_units(real.pot),
                "to_call": to_units(real.legal_actions().call_amount),
                "abstract_histories": [
                    {"probability": round(b.prob, 4), "translations": b.translations}
                    for b in branches
                ],
                "fallback_share": round(fallback_mass, 4),
            }
            if solve is not None:
                explanation["search"] = {
                    "iterations": solve["iterations"],
                    "solve_ms": round(solve["seconds"] * 1000, 1),
                    "nodes": solve["nodes"],
                    "exploitability_chips": round(solve["exploitability"], 2),
                    "hand_value": to_units(round(float(solve["values"][hero]))),
                }
        return Decision(
            strategy=strategy,
            recommended=recommended,
            source="subgame_solve" if use_search else "blueprint",
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
            explain=explanation,
        )

    def _solve_river(
        self,
        real: _core.HandState,
        starting: list[int],
        sb: int,
        bb: int,
        branch: _Branch,
        board: list[int],
    ) -> dict[str, Any]:
        start, path = river_start(real.history, starting, sb, bb)
        r0, r1 = bayes_ranges(branch.steps, board, self.cards, self.strategy)
        return _core.solve_river(
            start,
            board,
            path,
            r0,
            r1,
            min_iterations=20,
            max_iterations=100_000,
            max_seconds=self.config.search.time_budget_ms / 1000,
            threads=self.config.search.threads,
        )

    # ------------------------------------------------------------------ pieces

    def _stack_warnings(self, v: ValidatedState) -> list[str]:
        e = v.engine
        starting = [e.stacks[i] + e.contributed[i] for i in range(e.num_players)]
        effective_bb = min(starting) / CHIPS_PER_BB
        if abs(effective_bb - self.trained_stack_bb) > 0.2 * self.trained_stack_bb:
            msg = (
                f"effective stack is {effective_bb:g}bb but the blueprint was trained at "
                f"{self.trained_stack_bb:g}bb; play may be off"
            )
            warnings.warn(msg, stacklevel=3)
            return [msg]
        return []

    def _replay(
        self,
        history: list[tuple[_core.Street, int, _core.Action]],
        starting: list[int],
        sb: int,
        bb: int,
    ) -> list[_Branch]:
        """Abstract histories consistent with the real one, most likely first."""
        real = _core.HandState(starting, sb, bb)
        first = AbstractTracker(self.rules, self._abstract_stacks, *self._abstract_blinds)
        first.reset()
        branches = [_Branch(1.0, first, [], [])]
        for street, seat, action in history:
            grown: list[_Branch] = []
            for br in branches:
                t = br.tracker
                if not t.in_sync(real):
                    t.lost = True
                    grown.append(br)
                    continue
                for index, p in t.translate(real, action):
                    if p <= 0:
                        continue
                    child = _clone(t)
                    step = Step(seat, int(street), t.history, index, len(t.actions()))
                    child.apply_index(index)
                    note = list(br.translations)
                    if action.type == _core.ActionType.BET_RAISE:
                        abstract = t.actions()[index]
                        if abstract.amount != action.amount or p < 1:
                            note.append(
                                f"{action_label(real, action)} -> "
                                f"{action_label(t.state, abstract)} ({p:.0%})"
                            )
                    grown.append(_Branch(br.prob * p, child, note, [*br.steps, step]))
            grown.sort(key=lambda b: -b.prob)
            branches = grown[: self.max_branches]
            total = sum(b.prob for b in branches)
            for b in branches:
                b.prob /= total
            real.apply(action)
        return branches

    def _branch_policy(
        self, br: _Branch, real: _core.HandState, bucket: int
    ) -> list[tuple[_core.Action, float]]:
        t = br.tracker
        if not t.in_sync(real):
            return [(_core.Action.check_call(), 1.0)]
        acts = t.actions()
        p = self.strategy.probs(_core.nlhe_infoset_key(t.history, bucket))
        if p is None or len(p) != len(acts):
            return [(_core.Action.check_call(), 1.0)]
        return [(t.to_real(real, i), float(p[i])) for i in range(len(acts)) if p[i] > 0]

    def _recommended(
        self,
        real: _core.HandState,
        action: _core.Action,
        label: str,
        to_units: Callable[[int], float],
    ) -> dict[str, Any]:
        la = real.legal_actions()
        seat = real.to_act
        if action.type == _core.ActionType.FOLD:
            return {"action": "fold", "amount": None}
        if action.type == _core.ActionType.CHECK_CALL:
            if la.call_amount == 0:
                return {"action": "check", "amount": None}
            return {"action": "call", "amount": to_units(real.bets[seat] + la.call_amount)}
        kind = "all_in" if label == "all_in" else ("bet" if real.current_bet == 0 else "raise")
        return {"action": kind, "amount": to_units(action.amount), "label": label}


def _clone(t: AbstractTracker) -> AbstractTracker:
    c = AbstractTracker(t.rules, t.stacks, t.sb, t.bb)
    c.state = t.state.copy()
    c.history = t.history
    c.lost = t.lost
    return c
