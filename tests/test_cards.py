from collections import Counter
from itertools import combinations, pairwise

import numpy as np
import pytest

from regret.engine.cards import (
    HandCategory,
    canonical_index,
    card_str,
    cards_str,
    evaluate,
    evaluate_batch,
    hand_category,
    indexer,
    parse_card,
    parse_cards,
)

# ---------------------------------------------------------------- cards


def test_every_card_round_trips() -> None:
    for c in range(52):
        assert parse_card(card_str(c)) == c
    assert parse_card("2c") == 0
    assert parse_card("As") == 51
    assert parse_card("tD") == parse_card("Td")


@pytest.mark.parametrize("bad", ["", "A", "Asd", "1s", "Ax", "10s", "  "])
def test_invalid_cards_are_rejected(bad: str) -> None:
    with pytest.raises(ValueError, match="invalid card"):
        parse_card(bad)


def test_parse_cards_formats() -> None:
    assert parse_cards("AsKd") == parse_cards("As Kd") == parse_cards("As,Kd") == [51, 45]
    assert cards_str([51, 45]) == "AsKd"
    with pytest.raises(ValueError, match="invalid card string"):
        parse_cards("AsK")


# ---------------------------------------------------------------- evaluator


def _reference_5(cards: tuple[int, ...]) -> tuple[int, tuple[int, ...]]:
    """Slow, obviously-correct 5-card evaluator: (category, tiebreak ranks)."""
    ranks = [c // 4 for c in cards]
    flush = len({c % 4 for c in cards}) == 1
    counts = Counter(ranks)
    by_group = sorted(counts, key=lambda r: (counts[r], r), reverse=True)
    shape = sorted(counts.values(), reverse=True)
    uniq = sorted(set(ranks), reverse=True)
    straight_top = None
    if len(uniq) == 5 and uniq[0] - uniq[4] == 4:
        straight_top = uniq[0]
    elif uniq == [12, 3, 2, 1, 0]:
        straight_top = 3
    if straight_top is not None and flush:
        return 8, (straight_top,)
    if shape == [4, 1]:
        return 7, tuple(by_group)
    if shape == [3, 2]:
        return 6, tuple(by_group)
    if flush:
        return 5, tuple(uniq)
    if straight_top is not None:
        return 4, (straight_top,)
    if shape == [3, 1, 1]:
        return 3, tuple(by_group)
    if shape == [2, 2, 1]:
        return 2, tuple(by_group)
    if shape == [2, 1, 1, 1]:
        return 1, tuple(by_group)
    return 0, tuple(uniq)


def _reference(cards: list[int]) -> tuple[int, tuple[int, ...]]:
    return max(_reference_5(combo) for combo in combinations(cards, 5))


@pytest.mark.parametrize("n_cards", [5, 6, 7])
def test_matches_reference_evaluator(n_cards: int) -> None:
    rng = np.random.default_rng(n_cards)
    hands = [rng.choice(52, size=n_cards, replace=False).tolist() for _ in range(3000)]
    ours = [evaluate(h) for h in hands]
    ref = [_reference(h) for h in hands]
    for value, (cat, _) in zip(ours, ref, strict=True):
        assert hand_category(value) == cat
    # Same ordering, same ties: ref -> ours must be strictly monotone.
    pairs = sorted(zip(ref, ours, strict=True))
    for (r1, o1), (r2, o2) in pairwise(pairs):
        assert (o1 == o2) if r1 == r2 else (o1 < o2)


@pytest.mark.parametrize(
    ("hand", "category"),
    [
        ("AsKsQsJsTs", HandCategory.STRAIGHT_FLUSH),
        ("5h4h3h2hAh", HandCategory.STRAIGHT_FLUSH),
        ("9c9d9h9s2c", HandCategory.FOUR_OF_A_KIND),
        ("KcKdKh2c2d", HandCategory.FULL_HOUSE),
        ("Ac9c7c4c2c", HandCategory.FLUSH),
        ("5c4d3h2sAc", HandCategory.STRAIGHT),
        ("7c7d7h2cKd", HandCategory.THREE_OF_A_KIND),
        ("7c7d2h2cKd", HandCategory.TWO_PAIR),
        ("7c7d3h2cKd", HandCategory.PAIR),
        ("Ac9d7h4c2d", HandCategory.HIGH_CARD),
    ],
)
def test_known_hands(hand: str, category: HandCategory) -> None:
    assert hand_category(evaluate(hand)) is category


def test_seven_card_specifics() -> None:
    # Wheel straight flush beats quads; six-high straight beats the wheel.
    assert evaluate("5h4h3h2hAhAcAd") > evaluate("AcAdAhAs2c3d4h")
    assert evaluate("6c5d4h3s2cKdQh") > evaluate("5c4d3h2sAcKdQh")
    # Two trips make a full house with the lower trips as the pair.
    assert hand_category(evaluate("KcKdKh2c2d2hQs")) is HandCategory.FULL_HOUSE
    assert evaluate("KcKdKh2c2d2hQs") == evaluate("KcKdKh2c2dQsJh")
    # Board plays: identical best five tie regardless of suits or unused cards.
    assert evaluate("AsKsQdJcTh2c3d") == evaluate("AhKdQcJsTd4c5h")


def test_evaluate_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="5 to 7"):
        evaluate("AsKsQs")
    with pytest.raises(ValueError, match="duplicate"):
        evaluate("AsAsKdQh2c")
    with pytest.raises(ValueError, match=r"0\.\.51"):
        evaluate([0, 1, 2, 3, 52])


def test_evaluate_batch_matches_single() -> None:
    rng = np.random.default_rng(0)
    batch = np.stack([rng.choice(52, size=7, replace=False) for _ in range(500)]).astype(np.uint8)
    out = evaluate_batch(batch)
    assert out.dtype == np.uint32
    assert out.tolist() == [evaluate(row.tolist()) for row in batch]
    with pytest.raises(ValueError, match="cards per row"):
        evaluate_batch(batch[:, :4])
    batch[0, 1] = batch[0, 0]
    with pytest.raises(ValueError, match="duplicate card in row 0"):
        evaluate_batch(batch)


# ---------------------------------------------------------------- isomorphism


@pytest.mark.parametrize(
    ("street", "size"),
    [("preflop", 169), ("flop", 1_286_792), ("turn", 13_960_050), ("river", 123_156_254)],
)
def test_indexer_sizes(street: str, size: int) -> None:
    assert indexer(street).size == size


def test_preflop_classes() -> None:
    assert canonical_index("AsKs") == canonical_index("AhKh")
    assert canonical_index("AsKd") == canonical_index("KcAh")
    assert canonical_index("AsKs") != canonical_index("AsKd")
    reps = [cards_str(indexer("preflop").unindex(i)) for i in range(169)]
    pairs = sum(r[0] == r[2] for r in reps)
    suited = sum(r[0] != r[2] and r[1] == r[3] for r in reps)
    assert (pairs, suited, 169 - pairs - suited) == (13, 78, 78)


def test_suit_relabeling_and_board_order_do_not_matter() -> None:
    assert canonical_index("AsKs", "Qs7h2d") == canonical_index("AhKh", "2c7dQh")
    assert canonical_index("AsKs", "Qs7h2d9c") == canonical_index("AdKd", "9h2sQd7c")
    # ...but which cards are the hole cards does.
    assert canonical_index("AsKs", "Qs7h2d") != canonical_index("Qs7h", "AsKs2d")


@pytest.mark.parametrize("street", ["preflop", "flop", "turn", "river"])
def test_round_trip_and_batch(street: str) -> None:
    idx = indexer(street)
    rng = np.random.default_rng(1)
    indices = rng.integers(0, idx.size, size=2000, dtype=np.uint64)
    reps = idx.unindex_batch(indices)
    assert reps.shape == (2000, idx.total_cards)
    np.testing.assert_array_equal(idx.index_batch(reps), indices)
    assert idx.unindex(int(indices[0])) == reps[0].tolist()
    with pytest.raises(IndexError):
        idx.unindex(idx.size)


def test_canonical_index_validates() -> None:
    with pytest.raises(ValueError, match="2 hole cards"):
        canonical_index("As", "KsQsJs")
    with pytest.raises(ValueError, match="0, 3, 4 or 5"):
        canonical_index("AsKs", "QsJs")
    with pytest.raises(ValueError, match="duplicate"):
        canonical_index("AsKs", "AsQsJs")
