"""Hand-written rule fixtures for the betting engine (the PokerKit fuzz covers the rest)."""

import pytest

from regret._core import Action, HandState, PlayerStatus, Street
from regret.engine.cards import parse_cards

call, fold = Action.check_call, Action.fold
raise_to = Action.bet_raise_to


def cards(*hands: str) -> list[list[int]]:
    return [parse_cards(h) for h in hands]


def test_blinds_and_first_actor() -> None:
    h = HandState([10_000] * 6, 50, 100)
    assert h.bets == [50, 100, 0, 0, 0, 0]
    assert h.to_act == 2  # UTG
    assert h.pot == 150
    la = h.legal_actions()
    assert (la.fold, la.call_amount, la.min_raise_to, la.max_raise_to) == (True, 100, 200, 10_000)


def test_heads_up_order() -> None:
    h = HandState([10_000, 10_000], 50, 100)
    assert h.button == 0
    assert h.to_act == 0  # SB/button first preflop
    h.apply(call())
    assert h.to_act == 1  # BB option
    h.apply(call())
    assert h.street == Street.FLOP
    assert h.to_act == 1  # BB first postflop


def test_min_raise_tracks_largest_increment() -> None:
    h = HandState([10_000] * 3, 50, 100)
    h.apply(raise_to(300))  # BTN: increment 200
    assert h.legal_actions().min_raise_to == 500
    h.apply(raise_to(1000))  # SB: increment 700
    assert h.legal_actions().min_raise_to == 1700


def test_short_all_in_does_not_reopen_raising() -> None:
    h = HandState([400, 10_000, 10_000], 50, 100)
    h.apply(raise_to(300))  # BTN raises by 200
    h.apply(raise_to(400))  # SB all-in: only 100 more, not a full raise
    h.apply(fold())  # BB
    la = h.legal_actions()
    assert h.to_act == 2
    assert (la.fold, la.call_amount, la.bet_raise) == (True, 100, False)


def test_short_all_ins_that_add_up_to_a_full_raise_reopen() -> None:
    h = HandState([500, 10_000, 10_000, 400, 10_000], 50, 100)  # SB, BB, HJ, CO, BTN
    h.apply(raise_to(300))  # HJ raises by 200
    h.apply(raise_to(400))  # CO all-in, +100
    h.apply(fold())  # BTN
    h.apply(raise_to(500))  # SB all-in, +100: 200 in total, a full raise
    h.apply(call())  # BB calls and still has chips behind
    assert h.to_act == 2
    assert h.legal_actions().bet_raise


def test_cannot_raise_when_nobody_can_call_more() -> None:
    h = HandState([1000, 5000], 50, 100)
    h.apply(raise_to(1000))  # SB/button all-in
    la = h.legal_actions()
    assert (la.fold, la.call_amount, la.bet_raise) == (True, 900, False)


def test_three_way_all_in_side_pots_and_uncalled_chips() -> None:
    h = HandState([1000, 3000, 5000], 50, 100)  # SB, BB, BTN
    h.apply(raise_to(5000))  # BTN all-in
    h.apply(call())  # SB calls all-in for 1000
    h.apply(call())  # BB calls all-in for 3000
    assert h.is_terminal
    assert h.is_showdown
    assert h.statuses == [PlayerStatus.ALL_IN] * 3
    assert [(p.amount, p.eligible) for p in h.pots()] == [
        (3000, [0, 1, 2]),
        (4000, [1, 2]),
        (2000, [2]),
    ]
    # SB best, BB second, BTN worst.
    hole = cards("AsAd", "KsKd", "7c2h")
    board = parse_cards("AhKh9c5d3s")
    assert h.payoffs(hole, board) == [2000, 1000, -3000]


def test_everyone_folds_to_big_blind() -> None:
    h = HandState([10_000] * 4, 50, 100)
    for _ in range(3):
        h.apply(fold())
    assert h.is_terminal
    assert not h.is_showdown
    assert h.payoffs([[0, 1]] * 4, []) == [-50, 50, 0, 0]


def test_split_pot_odd_chip_goes_clockwise_from_button() -> None:
    h = HandState([100] * 3, 1, 2)  # SB, BB, BTN
    h.apply(call())  # BTN limps
    h.apply(fold())  # SB folds, 1 dead chip
    h.apply(call())  # BB checks
    for _ in range(3):
        h.apply(call())
        h.apply(call())
    assert h.is_showdown
    # Board is a royal flush: BB and BTN split 5 chips; BB is first after the button.
    hole = cards("2c3d", "4c5d", "6c7d")
    assert h.payoffs(hole, parse_cards("AsKsQsJsTs")) == [-1, 1, 0]


def test_player_nobody_can_bet_against_is_skipped() -> None:
    # SB has 504 behind but nobody else can match more than 2: no action for SB.
    h = HandState([506, 1, 2], 2, 3)
    h.apply(call())  # BTN calls all-in for 2
    assert h.is_terminal


def test_illegal_actions_are_rejected() -> None:
    h = HandState([10_000] * 3, 50, 100)
    with pytest.raises(ValueError, match="below the minimum"):
        h.apply(raise_to(150))
    with pytest.raises(ValueError, match="more than the player's stack"):
        h.apply(raise_to(20_000))
    h.apply(call())
    h.apply(call())
    assert h.why_illegal(fold()) == "cannot fold when not facing a bet (check instead)"
    with pytest.raises(ValueError, match="not facing a bet"):
        h.apply(fold())
    h.apply(call())
    for _ in range(3):
        for _ in range(3):
            h.apply(call())
    assert h.is_terminal
    with pytest.raises(ValueError, match="hand is over"):
        h.apply(call())


@pytest.mark.parametrize(
    ("stacks", "sb", "bb", "msg"),
    [
        ([100], 1, 2, "2-6 players"),
        ([100] * 7, 1, 2, "2-6 players"),
        ([100, 100], 2, 2, "small blind < big blind"),
        ([100, 0], 1, 2, "positive"),
    ],
)
def test_bad_tables_are_rejected(stacks: list[int], sb: int, bb: int, msg: str) -> None:
    with pytest.raises(ValueError, match=msg):
        HandState(stacks, sb, bb)


def test_payoffs_need_a_finished_hand_and_valid_cards() -> None:
    h = HandState([1000, 1000], 50, 100)
    with pytest.raises(RuntimeError, match="isn't over"):
        h.payoffs(cards("AsAd", "KsKd"), parse_cards("2c3c4c5c7d"))
    h.apply(raise_to(1000))
    h.apply(call())
    with pytest.raises(ValueError, match="duplicate"):
        h.payoffs(cards("AsAd", "AsKd"), parse_cards("2c3c4c5c7d"))
    with pytest.raises(ValueError, match="5-card board"):
        h.payoffs(cards("AsAd", "KsKd"), parse_cards("2c3c4c"))
