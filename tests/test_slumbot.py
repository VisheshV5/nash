"""Slumbot client, tested offline against a fake server that speaks the same protocol."""

from __future__ import annotations

import random
from typing import Any

import pytest

from regret import _core
from regret.engine.cards import card_str
from regret.eval import slumbot

BOARD_VISIBLE = (0, 3, 4, 5)


class FakeSlumbot:
    """Deals heads-up 200bb hands; its bot calls or checks, and sometimes pots it."""

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)
        self.hands = 0
        self.total = 0  # sum of client winnings it reported

    def __call__(self, endpoint: str, body: dict[str, Any]) -> dict[str, Any]:
        if endpoint == "new_hand":
            self.client_pos = self.hands % 2  # client starts in the big blind (pos 0)
            self.hands += 1
            deck = list(range(52))
            self.rng.shuffle(deck)
            self.hole = {0: deck[0:2], 1: deck[2:4]}  # by our-engine seat
            self.board = deck[4:9]
            self.state = _core.HandState([slumbot.STACK, slumbot.STACK], *slumbot.BLINDS)
            self.tokens: list[str] = []
            self.streets: list[int] = []
        else:
            assert body["token"] == "tok"
            self._apply(body["incr"])
        self._bot_moves()
        return self._response()

    @property
    def client_seat(self) -> int:
        return 1 - self.client_pos

    def _apply(self, incr: str) -> None:
        s = self.state
        if incr == "f":
            a = _core.Action.fold()
        elif incr in "kc":
            a = _core.Action.check_call()
        else:
            a = _core.Action.bet_raise_to(int(incr[1:]))
        self.streets.append(int(s.street))
        self.tokens.append(incr)
        s.apply(a)

    def _bot_moves(self) -> None:
        s = self.state
        while not s.is_terminal and s.to_act != self.client_seat:
            la = s.legal_actions()
            if la.bet_raise and self.rng.random() < 0.2:
                to = min(
                    max(s.current_bet + s.pot + la.call_amount, la.min_raise_to), la.max_raise_to
                )
                self._apply(f"b{to}")
            else:
                self._apply("c" if la.call_amount else "k")

    def _response(self) -> dict[str, Any]:
        s = self.state
        out, street = "", 0
        for st, tok in zip(self.streets, self.tokens, strict=True):
            while st > street:
                out += "/"
                street += 1
            out += tok
        final = int(s.street) if not s.is_terminal else 3 if s.is_showdown else int(s.street)
        while street < final:
            out += "/"
            street += 1
        resp: dict[str, Any] = {
            "token": "tok",
            "action": out,
            "client_pos": self.client_pos,
            "hole_cards": [card_str(c) for c in self.hole[self.client_seat]],
            "board": [card_str(c) for c in self.board[: BOARD_VISIBLE[min(int(s.street), 3)]]],
        }
        if s.is_terminal:
            pay = s.payoffs([self.hole[0], self.hole[1]], self.board)
            resp["winnings"] = pay[self.client_seat]
            self.total += pay[self.client_seat]
        return resp


def test_replay_parses_slumbot_strings() -> None:
    h = slumbot.replay("b200c/kk/kk/kb200")
    assert h.street == _core.Street.RIVER
    assert h.to_act == 1  # the big blind checked, the button bet: the big blind faces it
    assert h.legal_actions().call_amount == 200
    allin = slumbot.replay("b20000c///")
    assert allin.is_terminal
    assert allin.is_showdown
    with pytest.raises(ValueError, match="unknown action"):
        slumbot.replay("x")


def test_encode() -> None:
    h = slumbot.replay("")
    assert slumbot.encode(h, _core.Action.check_call()) == "c"  # the button calls the blind
    assert slumbot.encode(h, _core.Action.fold()) == "f"
    assert slumbot.encode(h, _core.Action.bet_raise_to(250)) == "b250"
    h = slumbot.replay("cb300c/")
    assert slumbot.encode(h, _core.Action.check_call()) == "k"


def test_full_session_against_a_fake_server() -> None:
    server = FakeSlumbot(seed=3)
    rng = random.Random(9)

    def decide(
        state: _core.HandState, seat: int, hole: list[int], board: list[int]
    ) -> _core.Action:
        assert state.to_act == seat
        assert len(board) in (0, 3, 4, 5)
        la = state.legal_actions()
        r = rng.random()
        if la.fold and r < 0.1:
            return _core.Action.fold()
        if la.bet_raise and r > 0.85:
            return _core.Action.bet_raise_to(la.min_raise_to)
        return _core.Action.check_call()

    result = slumbot.play(decide, 60, server, pause=0)
    assert result.hands == 60
    assert sum(result.winnings) == server.total
    assert result.summary()["hands"] == 60
