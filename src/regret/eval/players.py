"""Players for matches: baselines (random, always-call, equity threshold) and the blueprint.

Protocol: `new_hand(seat, hole, rng)`, then for every decision in the hand `observe(state,
seat, action, board)` is called on every player *before* the action is applied, and `act(state,
board)` on the player to act. `board` is the part of the board visible on the current street.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from regret import _core
from regret.abstraction.cards import CardAbstraction
from regret.eval.strategy import StrategyTable

Action = _core.Action
_UNIFORM = np.ones(1326, np.float32)


class Player(Protocol):
    name: str

    def new_hand(self, seat: int, hole: list[int], rng: np.random.Generator) -> None: ...
    def observe(
        self, state: _core.HandState, seat: int, action: Action, board: list[int]
    ) -> None: ...
    def act(self, state: _core.HandState, board: list[int]) -> Action: ...


class _Base:
    name = "base"

    def new_hand(self, seat: int, hole: list[int], rng: np.random.Generator) -> None:
        self.seat, self.hole, self.rng = seat, hole, rng

    def observe(self, state: _core.HandState, seat: int, action: Action, board: list[int]) -> None:
        pass


class AlwaysCall(_Base):
    name = "always-call"

    def act(self, state: _core.HandState, board: list[int]) -> Action:
        return Action.check_call()


class RandomPlayer(_Base):
    """Uniform over the legal kinds of action (fold / check-call / bet-raise), then a uniform
    bet size between the minimum and all-in."""

    name = "random"

    def act(self, state: _core.HandState, board: list[int]) -> Action:
        la = state.legal_actions()
        kinds = ["call"] + ["fold"] * la.fold + ["raise"] * la.bet_raise
        kind = kinds[self.rng.integers(len(kinds))]
        if kind == "fold":
            return Action.fold()
        if kind == "raise":
            return Action.bet_raise_to(int(self.rng.integers(la.min_raise_to, la.max_raise_to + 1)))
        return Action.check_call()


class EquityThreshold(_Base):
    """Rule bot: equity vs a random hand decides. Raise pot-size above `raise_above`, call when
    equity beats the pot odds, otherwise check or fold."""

    name = "equity-threshold"

    def __init__(self, raise_above: float = 0.7, samples: int = 300) -> None:
        self.raise_above, self.samples = raise_above, samples
        self._cache: tuple[int, float] | None = None

    def new_hand(self, seat: int, hole: list[int], rng: np.random.Generator) -> None:
        super().new_hand(seat, hole, rng)
        self._cache = None

    def equity(self, board: list[int]) -> float:
        if self._cache is None or self._cache[0] != len(board):
            seed = int(self.rng.integers(2**63))
            eq = _core.equity_vs_range(self.hole, board, _UNIFORM, self.samples, seed)
            self._cache = (len(board), eq)
        return self._cache[1]

    def act(self, state: _core.HandState, board: list[int]) -> Action:
        la = state.legal_actions()
        eq = self.equity(board)
        if eq >= self.raise_above and la.bet_raise:
            pot_size = state.current_bet + state.pot + la.call_amount
            return Action.bet_raise_to(min(max(pot_size, la.min_raise_to), la.max_raise_to))
        if la.call_amount == 0 or eq >= la.call_amount / (state.pot + la.call_amount):
            return Action.check_call()
        return Action.fold()


class AbstractTracker:
    """Follows the real hand inside the abstract game: the abstract state, the abstract action
    history (for infoset keys), and whether the two have drifted apart."""

    def __init__(self, rules: _core.ActionRules, stacks: list[int], sb: int, bb: int) -> None:
        self.rules, self.stacks, self.sb, self.bb = rules, stacks, sb, bb

    def reset(self) -> None:
        self.state = _core.HandState(self.stacks, self.sb, self.bb, record_history=False)
        self.history = _core.nlhe_history_root()
        self.lost = False  # the abstract game can no longer follow the real one

    def in_sync(self, real: _core.HandState) -> bool:
        return (
            not self.lost
            and not self.state.is_terminal
            and self.state.to_act == real.to_act
            and self.state.street == real.street
        )

    def actions(self) -> list[Action]:
        return _core.abstract_actions(self.state, self.rules)

    def apply_index(self, index: int) -> None:
        self.state.apply(self.actions()[index])
        self.history = _core.nlhe_extend_history(self.history, index)

    def translate(self, real: _core.HandState, action: Action) -> list[tuple[int, float]]:
        """Distribution over abstract action indices for a real action taken in `real`."""
        acts = self.actions()
        if action.type != _core.ActionType.BET_RAISE:
            return [(next(i for i, a in enumerate(acts) if a.type == action.type), 1.0)]
        all_in = action.amount == real.legal_actions().max_raise_to
        return _core.translate_bet(
            self.state, self.rules, _core.pot_fraction(real, action.amount), all_in
        )

    def to_real(self, real: _core.HandState, index: int) -> Action:
        """The real action for abstract action `index` (same pot fraction, or all-in)."""
        a = self.actions()[index]
        la = real.legal_actions()
        if a.type == _core.ActionType.FOLD:
            return Action.fold() if la.fold else Action.check_call()
        if a.type == _core.ActionType.CHECK_CALL or not la.bet_raise:
            return Action.check_call()
        if a.amount == self.state.legal_actions().max_raise_to:
            return Action.bet_raise_to(la.max_raise_to)
        frac = _core.pot_fraction(self.state, a.amount)
        to = round(real.current_bet + frac * (real.pot + la.call_amount))
        return Action.bet_raise_to(min(max(to, la.min_raise_to), la.max_raise_to))


class BlueprintPlayer(_Base):
    """Plays the trained blueprint: bucket its cards, look up its abstract infoset, sample."""

    name = "blueprint"

    def __init__(
        self,
        strategy: StrategyTable,
        cards: CardAbstraction,
        rules: _core.ActionRules,
        stacks: list[int],
        sb: int,
        bb: int,
        name: str = "blueprint",
    ) -> None:
        self.strategy, self.cards, self.name = strategy, cards, name
        self.tracker = AbstractTracker(rules, stacks, sb, bb)
        self.fallbacks = 0  # decisions made without the blueprint (abstract game lost track)
        self.unseen = 0  # infosets the blueprint never visited

    def new_hand(self, seat: int, hole: list[int], rng: np.random.Generator) -> None:
        super().new_hand(seat, hole, rng)
        self.tracker.reset()
        self._pending: int | None = None

    def observe(self, state: _core.HandState, seat: int, action: Action, board: list[int]) -> None:
        t = self.tracker
        if not t.in_sync(state):
            t.lost = True
            return
        if seat == self.seat and self._pending is not None:
            index, self._pending = self._pending, None
        else:
            dist = t.translate(state, action)
            index = dist[0][0]
            if len(dist) > 1 and self.rng.random() >= dist[0][1]:
                index = dist[1][0]
        t.apply_index(index)

    def policy(self, board: list[int]) -> np.ndarray | None:
        key = _core.nlhe_infoset_key(self.tracker.history, self.cards.bucket(self.hole, board))
        return self.strategy.probs(key)

    def act(self, state: _core.HandState, board: list[int]) -> Action:
        if not self.tracker.in_sync(state):
            self.fallbacks += 1
            return Action.check_call()
        p = self.policy(board)
        acts = self.tracker.actions()
        if p is None or len(p) != len(acts):
            self.unseen += 1
            index = next(i for i, a in enumerate(acts) if a.type == _core.ActionType.CHECK_CALL)
        else:
            index = int(self.rng.choice(len(p), p=p))
        self._pending = index
        return self.tracker.to_real(state, index)
