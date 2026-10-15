"""Real-time river search: convergence, a textbook equilibrium, and forced real actions."""

from __future__ import annotations

import numpy as np
import pytest

from regret import _core
from regret.engine.cards import parse_cards

A = _core.Action
PAIRS = [(a, b) for b in range(52) for a in range(b)]
BOARD = parse_cards("Kd9s5h2c7d")


def river_start(open_to: int = 250) -> _core.HandState:
    """Heads-up 100bb: button opens, big blind calls, checked down to the river (pot 500)."""
    h = _core.HandState([10_000, 10_000], 50, 100, record_history=False)
    for a in [A.bet_raise_to(open_to), A.check_call(), *[A.check_call()] * 4]:
        h.apply(a)
    assert h.street == _core.Street.RIVER
    return h


def hands(spec: str) -> list[int]:
    """Hole indices for a rank pair like '86' or 'QJ' (all suit combos not on the board)."""
    ranks = "23456789TJQKA"
    r1, r2 = ranks.index(spec[0]), ranks.index(spec[1])
    out = []
    for i, (a, b) in enumerate(PAIRS):
        if {a // 4, b // 4} == {r1, r2} and a // 4 != b // 4 and not {a, b} & set(BOARD):
            out.append(i)
    return out


def range_of(*specs: str) -> np.ndarray:
    r = np.zeros(1326, np.float32)
    for spec in specs:
        r[hands(spec)] = 1
    return r


def test_exploitability_falls_with_iterations() -> None:
    u = np.ones(1326, np.float32)
    prev = float("inf")
    for its in (20, 80, 320):
        r = _core.solve_river(
            river_start(), BOARD, [], u, u, min_iterations=its, max_iterations=its, max_seconds=60
        )
        assert r["exploitability"] < prev
        prev = r["exploitability"]
    assert prev < 0.01 * 500  # under 1% of the pot


def test_polarized_bettor_vs_bluff_catcher() -> None:
    # Seat 1 (big blind, first to act) has the nuts (86: a 5-9 straight) or air (QJ).
    # Seat 0 has a bluff catcher (A9: second pair). Only a pot-size bet is allowed.
    nuts, air, catcher = hands("86"), hands("QJ"), hands("A9")
    r0, r1 = range_of("A9"), range_of("86", "QJ")
    menu = {"bet_pot": [1.0], "raise_pot": [], "max_raises": 1, "all_in": False}
    root = _core.solve_river(
        river_start(),
        BOARD,
        [],
        r0,
        r1,
        min_iterations=3000,
        max_iterations=3000,
        max_seconds=60,
        **menu,
    )
    assert root["player"] == 1
    labels = [a.type for a in root["actions"]]
    bet = labels.index(_core.ActionType.BET_RAISE)
    s = root["strategy"]
    # Equilibrium: value-bet every nut hand, bluff so bluffs:value = B/(P+B) = 1:2.
    assert s[bet, nuts].mean() > 0.95
    bluffs = s[bet, air].mean() * len(air)
    assert bluffs / len(nuts) == pytest.approx(0.5, abs=0.1)
    assert root["exploitability"] < 0.005 * 500
    # Facing a bet B into pot P, the bluff catcher calls P/(P+B) of the time: 1/2 against a
    # pot-size bet (500), and 1/3 against an off-menu 2x-pot bet (1,000) forced into the tree.
    for to, want in ((500, 1 / 2), (1000, 1 / 3)):
        faced = _core.solve_river(
            river_start(),
            BOARD,
            [A.bet_raise_to(to)],
            r0,
            r1,
            min_iterations=3000,
            max_iterations=3000,
            max_seconds=60,
            **menu,
        )
        assert faced["player"] == 0
        call = [a.type for a in faced["actions"]].index(_core.ActionType.CHECK_CALL)
        assert faced["strategy"][call, catcher].mean() == pytest.approx(want, abs=0.05)


def test_real_off_menu_bet_is_in_the_tree() -> None:
    # The big blind bets 300 into 500 (60% pot), a size the search menu doesn't have.
    u = np.ones(1326, np.float32)
    r = _core.solve_river(
        river_start(), BOARD, [A.bet_raise_to(300)], u, u, min_iterations=30, max_iterations=30
    )
    assert r["player"] == 0
    assert [a.type for a in r["actions"]][:2] == [
        _core.ActionType.FOLD,
        _core.ActionType.CHECK_CALL,
    ]
    np.testing.assert_allclose(r["strategy"].sum(0)[hands("A9")], 1.0, atol=1e-4)
    assert np.isfinite(r["values"]).all()


def test_time_budget_is_respected() -> None:
    u = np.ones(1326, np.float32)
    r = _core.solve_river(
        river_start(), BOARD, [], u, u, min_iterations=5, max_iterations=100_000, max_seconds=0.3
    )
    assert r["seconds"] < 0.6
    assert r["iterations"] >= 5


def test_search_rejects_bad_problems() -> None:
    u = np.ones(1326, np.float32)
    flop = _core.HandState([10_000, 10_000], 50, 100, record_history=False)
    flop.apply(A.check_call())
    flop.apply(A.check_call())
    with pytest.raises(ValueError, match="start of the river"):
        _core.solve_river(flop, BOARD, [], u, u)
    with pytest.raises(ValueError, match="end the hand"):
        _core.solve_river(river_start(), BOARD, [A.check_call(), A.check_call()], u, u)
