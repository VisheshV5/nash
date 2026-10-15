"""Real-time turn and river search: convergence, a textbook equilibrium, forced real actions,
and the river card as a chance node."""

from __future__ import annotations

from typing import Any

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
    menu: dict[str, Any] = {"bet_pot": [1.0], "raise_pot": [], "max_raises": 1, "all_in": False}
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
    with pytest.raises(ValueError, match="start of the turn or river"):
        _core.solve_river(flop, BOARD, [], u, u)
    with pytest.raises(ValueError, match="end the hand"):
        _core.solve_river(river_start(), BOARD, [A.check_call(), A.check_call()], u, u)


def test_parallel_solve_is_reproducible_and_converges() -> None:
    u = np.ones(1326, np.float32)
    runs = [
        _core.solve_river(
            river_start(),
            BOARD,
            [],
            u,
            u,
            min_iterations=100,
            max_iterations=100,
            max_seconds=60,
            threads=4,
        )
        for _ in range(2)
    ]
    np.testing.assert_array_equal(runs[0]["strategy"], runs[1]["strategy"])
    serial = _core.solve_river(
        river_start(), BOARD, [], u, u, min_iterations=100, max_iterations=100, max_seconds=60
    )
    assert runs[0]["exploitability"] == pytest.approx(serial["exploitability"], rel=0.05)


# ---------------------------------------------------------------- from the turn

TURN_MENU: dict[str, Any] = {
    "bet_pot": [0.5, 1.0],
    "raise_pot": [1.0],
    "max_raises": 2,
    "river_bet_pot": [1.0],
    "river_raise_pot": [],
    "river_max_raises": 1,
}


def turn_start() -> _core.HandState:
    """Heads-up 100bb: button opens to 2.5bb, big blind calls, flop checks through (pot 500)."""
    h = _core.HandState([10_000, 10_000], 50, 100, record_history=False)
    for a in [A.bet_raise_to(250), A.check_call(), A.check_call(), A.check_call()]:
        h.apply(a)
    assert h.street == _core.Street.TURN
    return h


def solve_turn(board: list[int], r0: np.ndarray, r1: np.ndarray, its: int, **kw: Any) -> Any:
    return _core.solve_subgame(
        turn_start(),
        board,
        kw.pop("path", []),
        r0,
        r1,
        min_iterations=its,
        max_iterations=its,
        max_seconds=120,
        **(TURN_MENU | kw),
    )


def test_turn_exploitability_falls_with_iterations() -> None:
    u = np.ones(1326, np.float32)
    prev = float("inf")
    for its in (10, 40, 160):
        r = solve_turn(BOARD[:4], u, u, its, threads=4)
        assert r["exploitability"] < prev
        prev = r["exploitability"]
    assert prev < 0.02 * 500  # under 2% of the pot


def test_turn_values_average_over_river_cards() -> None:
    # Quads on the turn: the big blind (first to act) wins whatever river comes, so the button
    # never puts in another chip and the big blind's value is exactly the button's 250.
    board = parse_cards("KdKs5h2c")
    quads, junk = _core.hole_index(*parse_cards("KhKc")), _core.hole_index(*parse_cards("7c3d"))
    r0, r1 = np.zeros(1326, np.float32), np.zeros(1326, np.float32)
    r0[junk], r1[quads] = 1, 1
    r = solve_turn(board, r0, r1, 300)
    assert r["player"] == 1
    assert r["values"][quads] == pytest.approx(250, abs=2)
    # And the button, facing a bet with no outs, folds.
    faced = solve_turn(board, r0, r1, 300, path=[A.bet_raise_to(250)])
    fold = [a.type for a in faced["actions"]].index(_core.ActionType.FOLD)
    assert faced["strategy"][fold, junk] > 0.99


def test_turn_solve_is_the_same_on_any_number_of_threads() -> None:
    u = np.ones(1326, np.float32)
    serial = solve_turn(BOARD[:4], u, u, 20, exploitability=False)
    threaded = solve_turn(BOARD[:4], u, u, 20, threads=4, exploitability=False)
    np.testing.assert_array_equal(serial["strategy"], threaded["strategy"])
    assert serial["exploitability"] == -1


def test_turn_search_rejects_a_river_board() -> None:
    u = np.ones(1326, np.float32)
    with pytest.raises(ValueError, match="4-card board"):
        _core.solve_subgame(turn_start(), BOARD, [], u, u)
