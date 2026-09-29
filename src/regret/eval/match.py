"""Heads-up matches with variance reduction, reported in mbb/hand with 95% confidence intervals.

**Duplicate deals.** Each deal (both hole cards and the board) is played twice with the players
swapped between seats, and a player's result for the pair is the sum of the two hands. The card
luck that would decide a single hand largely cancels.

**All-in luck adjustment.** When the chips go in before the river, the hand's result is replaced
by its expectation over every possible runout of the missing board cards (sampled when that's
more than a few thousand boards). This is the chance-node part of AIVAT; the decision-node
corrections (which need the agents' value estimates) are not applied.

Results are from the first player's point of view. mbb/hand = 1000 x big blinds won per hand.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from regret import _core
from regret.eval.players import Player
from regret.utils.seeding import make_rng

BOARD_VISIBLE = {0: 0, 1: 3, 2: 4, 3: 5}  # by street
_MAX_ENUMERATED_RUNOUTS = 2000
_SAMPLED_RUNOUTS = 1000


@dataclass
class HandResult:
    payoffs: list[float]  # big blinds, by seat
    adjusted: list[float]  # after the all-in luck adjustment
    decisions: int


def _allin_ev(
    state: _core.HandState,
    hole: list[list[int]],
    board: list[int],
    seen: int,
    rng: np.random.Generator,
) -> list[float]:
    """Expected payoffs over every runout of the unseen board cards (sampled if too many)."""
    used = {c for h in hole for c in h} | set(board[:seen])
    deck = [c for c in range(52) if c not in used]
    missing = 5 - seen
    if math.comb(len(deck), missing) <= _MAX_ENUMERATED_RUNOUTS:
        runouts = list(itertools.combinations(deck, missing))
    else:
        runouts = [tuple(rng.choice(deck, missing, replace=False)) for _ in range(_SAMPLED_RUNOUTS)]
    total = np.zeros(len(hole))
    for extra in runouts:
        total += state.payoffs(hole, board[:seen] + list(extra))
    return list(total / len(runouts))


def play_hand(
    players: list[Player],
    hole: list[list[int]],
    board: list[int],
    stacks: list[int],
    sb: int,
    bb: int,
    rng: np.random.Generator,
) -> HandResult:
    state = _core.HandState(stacks, sb, bb)
    for seat, p in enumerate(players):
        p.new_hand(seat, hole[seat], np.random.default_rng(rng.integers(2**63)))
    decisions = 0
    while not state.is_terminal:
        seat = state.to_act
        visible = board[: BOARD_VISIBLE[int(state.street)]]
        action = players[seat].act(state, visible)
        if (why := state.why_illegal(action)) is not None:
            raise RuntimeError(f"{players[seat].name} made an illegal action: {why}")
        for p in players:
            p.observe(state, seat, action, visible)
        state.apply(action)
        decisions += 1
    raw = [c / bb for c in state.payoffs(hole, board)]
    seen = BOARD_VISIBLE[int(state.street)]
    if state.is_showdown and seen < 5:
        adjusted = [c / bb for c in _allin_ev(state, hole, board, seen, rng)]
    else:
        adjusted = raw
    return HandResult(raw, adjusted, decisions)


@dataclass
class MatchResult:
    players: tuple[str, str]
    pairs: int
    mbb: float  # first player's win rate
    ci95: float
    mbb_raw: float  # without the all-in adjustment
    ci95_raw: float
    decisions: int

    def summary(self) -> dict[str, Any]:
        return {
            "players": list(self.players),
            "hands": 2 * self.pairs,
            "mbb_per_hand": round(self.mbb, 1),
            "ci95": round(self.ci95, 1),
            "mbb_per_hand_raw": round(self.mbb_raw, 1),
            "ci95_raw": round(self.ci95_raw, 1),
            "significant": abs(self.mbb) > self.ci95,
        }

    def __str__(self) -> str:
        a, b = self.players
        return (
            f"{a} vs {b}: {self.mbb:+.1f} ± {self.ci95:.1f} mbb/hand over {2 * self.pairs:,} "
            f"hands (raw {self.mbb_raw:+.1f} ± {self.ci95_raw:.1f})"
        )


def _mean_ci(per_pair: NDArray[np.float64]) -> tuple[float, float]:
    # Per-hand rate: a pair is two hands.
    x = per_pair / 2 * 1000
    se = x.std(ddof=1) / math.sqrt(len(x)) if len(x) > 1 else float("inf")
    return float(x.mean()), float(1.96 * se)


def duplicate_match(
    a: Player,
    b: Player,
    pairs: int,
    stacks: list[int],
    sb: int,
    bb: int,
    seed: int = 0,
    progress: Callable[[int], None] | None = None,
) -> MatchResult:
    """`pairs` duplicate pairs (2 x pairs hands) between a and b, from a's point of view."""
    if len(stacks) != 2:
        raise ValueError("duplicate_match is heads-up")
    deal_rng = make_rng(seed, "match", "deal")
    play_rng = make_rng(seed, "match", "play")
    raw = np.zeros(pairs)
    adj = np.zeros(pairs)
    decisions = 0
    for i in range(pairs):
        cards = deal_rng.choice(52, 9, replace=False).tolist()
        hole, board = [cards[0:2], cards[2:4]], cards[4:9]
        first = play_hand([a, b], hole, board, stacks, sb, bb, play_rng)  # a in seat 0
        second = play_hand([b, a], hole, board, stacks, sb, bb, play_rng)  # a in seat 1
        raw[i] = first.payoffs[0] + second.payoffs[1]
        adj[i] = first.adjusted[0] + second.adjusted[1]
        decisions += first.decisions + second.decisions
        if progress and (i + 1) % 100 == 0:
            progress(i + 1)
    mbb, ci = _mean_ci(adj)
    mbb_raw, ci_raw = _mean_ci(raw)
    return MatchResult((a.name, b.name), pairs, mbb, ci, mbb_raw, ci_raw, decisions)
