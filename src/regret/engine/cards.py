"""Cards, hand evaluation and suit-isomorphism indexing.

Thin, typed wrappers over the C++ core. A card id is `rank * 4 + suit` (0..51), with ranks
2..A as 0..12 and suits c, d, h, s as 0..3, so "2c" is 0 and "As" is 51.
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import IntEnum
from functools import cache
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from regret import _core

Street = Literal["preflop", "flop", "turn", "river"]

# Board cards form one set per street: which card came on the turn doesn't change hand
# strength, so the abstraction indexes {hole, board}, not {hole, flop, turn, river}.
ROUNDS_BY_STREET: dict[Street, list[int]] = {
    "preflop": [2],
    "flop": [2, 3],
    "turn": [2, 4],
    "river": [2, 5],
}
_STREET_BY_BOARD_SIZE: dict[int, Street] = {0: "preflop", 3: "flop", 4: "turn", 5: "river"}


class HandCategory(IntEnum):
    HIGH_CARD = 0
    PAIR = 1
    TWO_PAIR = 2
    THREE_OF_A_KIND = 3
    STRAIGHT = 4
    FLUSH = 5
    FULL_HOUSE = 6
    FOUR_OF_A_KIND = 7
    STRAIGHT_FLUSH = 8


CardLike = int | str


def parse_card(text: str) -> int:
    """'As' -> 51."""
    return _core.parse_card(text)


def parse_cards(text: str) -> list[int]:
    """'AsKd', 'As Kd' or 'As,Kd' -> [51, 45]."""
    compact = "".join(ch for ch in text if not ch.isspace() and ch != ",")
    if len(compact) % 2:
        raise ValueError(f"invalid card string {text!r}")
    return [parse_card(compact[i : i + 2]) for i in range(0, len(compact), 2)]


def card_str(card: int) -> str:
    """51 -> 'As'."""
    return _core.card_to_str(card)


def cards_str(cards: Iterable[int]) -> str:
    return "".join(card_str(c) for c in cards)


def to_ids(cards: str | Iterable[CardLike]) -> list[int]:
    """Accepts 'AsKd', ['As', 'Kd'] or [51, 45]."""
    if isinstance(cards, str):
        return parse_cards(cards)
    return [parse_card(c) if isinstance(c, str) else int(c) for c in cards]


def evaluate(cards: str | Iterable[CardLike]) -> int:
    """Strength of the best 5-card hand in 5 to 7 cards. Higher wins; equal values tie."""
    return _core.evaluate(to_ids(cards))


def evaluate_batch(cards: ArrayLike) -> NDArray[np.uint32]:
    """Evaluate each row of an (n, 5..7) array of card ids."""
    return _core.evaluate_batch(np.ascontiguousarray(cards, dtype=np.uint8))


def hand_category(value: int) -> HandCategory:
    return HandCategory(_core.hand_category(value))


@cache
def indexer(street: Street) -> _core.HandIndexer:
    """Suit-isomorphism indexer for hole cards + board on `street` (built once, then cached)."""
    return _core.HandIndexer(ROUNDS_BY_STREET[street])


def canonical_index(hole: str | Iterable[CardLike], board: str | Iterable[CardLike] = ()) -> int:
    """Index of (hole, board) up to suit isomorphism, in [0, indexer(street).size)."""
    hole_ids, board_ids = to_ids(hole), to_ids(board)
    if len(hole_ids) != 2:
        raise ValueError(f"expected 2 hole cards, got {len(hole_ids)}")
    street = _STREET_BY_BOARD_SIZE.get(len(board_ids))
    if street is None:
        raise ValueError(f"board must have 0, 3, 4 or 5 cards, got {len(board_ids)}")
    return indexer(street).index(hole_ids + board_ids)
