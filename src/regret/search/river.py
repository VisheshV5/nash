"""River re-solving for the agent: beliefs from the blueprint, then the C++ subgame solver.

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


def river_start(
    history: list[tuple[_core.Street, int, _core.Action]], starting: list[int], sb: int, bb: int
) -> tuple[_core.HandState, list[_core.Action]]:
    """The state at the start of river betting, and the river actions taken since."""
    h = _core.HandState(starting, sb, bb, record_history=False)
    path: list[_core.Action] = []
    for street, _seat, action in history:
        if street == _core.Street.RIVER:
            path.append(action)
        else:
            h.apply(action)
    if h.street != _core.Street.RIVER:
        raise ValueError("the hand isn't on the river")
    return h, path
