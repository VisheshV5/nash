"""Turn and river re-solving for the agent: beliefs from the blueprint, then the C++ subgame
solver.

Ranges. Both players start with every hole pair; at each of their actions, each pair is
reweighted by the blueprint's probability of that action for the pair's bucket on that street
(Bayes' rule over the abstract history the agent followed). Pairs that touch the board are
removed. A 2% uniform blend keeps an overconfident belief from ruling a hand out entirely.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from regret import _core
from regret.abstraction.cards import CardAbstraction
from regret.eval.lbr import board_buckets
from regret.eval.strategy import StrategyTable

BOARD_VISIBLE = (0, 3, 4, 5)
UNIFORM_BLEND = 0.02
_PAIRS = np.array([(a, b) for b in range(52) for a in range(b)], np.uint8)


@dataclass(frozen=True)
class Step:
    """One action in the abstract history the agent followed."""

    seat: int
    street: int
    history: int  # abstract history hash before the action
    index: int  # abstract action index taken
    num_actions: int


def bayes_ranges(
    steps: list[Step], board: list[int], cards: CardAbstraction, strategy: StrategyTable
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    ranges = [np.ones(len(_PAIRS)), np.ones(len(_PAIRS))]
    buckets: dict[int, NDArray[np.int64]] = {}
    for step in steps:
        visible = board[: BOARD_VISIBLE[step.street]]
        if step.street not in buckets:
            buckets[step.street] = board_buckets(cards, visible)
        b = buckets[step.street]
        like = np.ones(len(_PAIRS))
        for bucket in np.unique(b[b >= 0]):
            p = strategy.probs(step.history, int(bucket))
            if p is not None and len(p) == step.num_actions:
                like[b == bucket] = p[step.index]
        ranges[step.seat] *= like
    blocked = np.isin(_PAIRS, board).any(1)
    out = []
    for r in ranges:
        r[blocked] = 0
        total = r.sum()
        valid = (~blocked).astype(np.float64)
        r = (r / total if total > 0 else valid / valid.sum()) * (1 - UNIFORM_BLEND) + (
            UNIFORM_BLEND * valid / valid.sum()
        )
        out.append(r.astype(np.float32))
    return out[0], out[1]


def street_start(
    history: list[tuple[_core.Street, int, _core.Action]],
    starting: list[int],
    sb: int,
    bb: int,
    street: _core.Street,
) -> tuple[_core.HandState, list[_core.Action]]:
    """The state at the start of `street`'s betting, and the actions taken on it since."""
    h = _core.HandState(starting, sb, bb, record_history=False)
    path: list[_core.Action] = []
    for s, _seat, action in history:
        if s == street:
            path.append(action)
        elif int(s) < int(street):
            h.apply(action)
        else:
            raise ValueError(f"the hand is past the {street.name.lower()}")
    if h.street != street:
        raise ValueError(f"the hand isn't on the {street.name.lower()}")
    return h, path


def river_start(
    history: list[tuple[_core.Street, int, _core.Action]], starting: list[int], sb: int, bb: int
) -> tuple[_core.HandState, list[_core.Action]]:
    """The state at the start of river betting, and the river actions taken since."""
    return street_start(history, starting, sb, bb, _core.Street.RIVER)


@dataclass(frozen=True)
class SearchMenu:
    """Bet sizes (fractions of the pot) for a solve starting on a street."""

    bet_pot: tuple[float, ...]
    raise_pot: tuple[float, ...]
    max_raises: int
    river_bet_pot: tuple[float, ...] = (0.5, 1.0)
    river_raise_pot: tuple[float, ...] = (1.0,)
    river_max_raises: int = 2
    min_iterations: int = 20


# From the river the tree is small, so the menu is rich. From the turn every turn line that
# reaches the river is followed by 48 river subtrees, so both streets get one size plus all-in
# (about 3,200 nodes and 50 ms per iteration on one core; about 80 iterations, 3-4% of the pot
# from equilibrium on uniform ranges, in the 0.8 s budget on six).
RIVER_MENU = SearchMenu(bet_pot=(0.33, 0.5, 0.75, 1.0, 1.5), raise_pot=(0.5, 1.0), max_raises=3)
TURN_MENU = SearchMenu(
    bet_pot=(0.66,),
    raise_pot=(),
    max_raises=2,
    river_bet_pot=(0.75,),
    river_raise_pot=(),
    river_max_raises=2,
    min_iterations=10,
)


def search_menu(street: _core.Street) -> SearchMenu:
    return TURN_MENU if street == _core.Street.TURN else RIVER_MENU
