"""Demo table sessions: play heads-up against the agent, or watch it play a baseline bot.

A session deals hands one at a time with the real engine. In "play" mode the human is one seat
and the agent the other; in "watch" mode the agent plays a baseline bot and the client steps
through decisions. Every agent decision keeps its `explain` data (strategy, equity, how bets
were translated, search stats) for the decision panel.
"""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from regret import _core
from regret.agent import RegretAgent
from regret.engine.cards import card_str
from regret.engine.state import CHIPS_PER_BB
from regret.eval.players import BASELINE_FACTORIES, Player

Mode = Literal["play", "watch"]
STREETS = ("preflop", "flop", "turn", "river")
BOARD_VISIBLE = (0, 3, 4, 5)
MAX_SESSIONS = 200
SESSION_TTL_S = 3600


def _bb(chips: int) -> float:
    return round(chips / CHIPS_PER_BB, 2)


@dataclass
class Session:
    agent: RegretAgent
    mode: Mode
    opponent: str
    rng: np.random.Generator
    agent_seat: int = 1
    hands: int = 0
    net_bb: float = 0.0  # agent's running result
    log: list[dict[str, Any]] = field(default_factory=list)
    decisions: list[dict[str, Any]] = field(default_factory=list)  # the agent's, this hand
    result: dict[str, Any] | None = None
    touched: float = field(default_factory=time.time)
    bot: Player | None = None

    def __post_init__(self) -> None:
        t = self.agent.config.table
        self.stacks = [round(t.starting_stack_bb * CHIPS_PER_BB)] * 2
        self.sb = round(t.small_blind_bb * CHIPS_PER_BB)
        self.bb = CHIPS_PER_BB
        if self.mode == "watch":
            self.bot = BASELINE_FACTORIES[self.opponent]()
        self.new_hand()

    # ------------------------------------------------------------------ hand flow

    def new_hand(self) -> None:
        self.agent_seat = 1 - self.agent_seat  # alternate the button
        cards: list[int] = [int(c) for c in self.rng.choice(52, 9, replace=False)]
        self.hole = [cards[0:2], cards[2:4]]
        self.board = cards[4:9]
        self.state = _core.HandState(self.stacks, self.sb, self.bb)
        self.log = []
        self.result = None
        self.decisions = []
        if self.bot is not None:
            self.bot.new_hand(
                1 - self.agent_seat,
                self.hole[1 - self.agent_seat],
                np.random.default_rng(self.rng.integers(2**63)),
            )
        self.hands += 1
        if self.mode == "play":
            self._advance_agent()

    def visible_board(self) -> list[int]:
        return self.board[: BOARD_VISIBLE[min(int(self.state.street), 3)]]

    def _apply(self, seat: int, action: _core.Action, who: str) -> None:
        if (why := self.state.why_illegal(action)) is not None:
            raise ValueError(why)
        if self.bot is not None:
            self.bot.observe(self.state, seat, action, self.visible_board())
        before = self.state.legal_actions()
        street = STREETS[int(self.state.street)]
        self.state.apply(action)
        self.log.append(
            {"street": street, "seat": seat, "who": who, "text": _describe(action, before)}
        )
        if self.state.is_terminal:
            self._finish()

    def _finish(self) -> None:
        pay = self.state.payoffs(self.hole, self.board)
        self.net_bb += _bb(pay[self.agent_seat])
        self.result = {
            "payoffs_bb": [_bb(p) for p in pay],
            "showdown": self.state.is_showdown,
            "board": [card_str(c) for c in self.board]
            if self.state.is_showdown
            else [card_str(c) for c in self.visible_board()],
        }

    def agent_move(self) -> None:
        seat = self.agent_seat
        d = self.agent.decide(
            self.state,
            self.stacks,
            self.sb,
            self.bb,
            self.hole[seat],
            self.visible_board(),
            explain=True,
            seed=int(self.rng.integers(2**63)),
        )
        rec = d.recommended
        la = self.state.legal_actions()
        if rec["action"] == "fold":
            action = _core.Action.fold() if la.fold else _core.Action.check_call()
        elif rec["action"] in ("check", "call"):
            action = _core.Action.check_call()
        else:
            action = _core.Action.bet_raise_to(round(rec["amount"] * CHIPS_PER_BB))
        self.decisions.append(
            {
                "street": STREETS[int(self.state.street)],
                "hole": [card_str(c) for c in self.hole[seat]],
                **d.to_dict(),
            }
        )
        self._apply(seat, action, "Regret")

    def _advance_agent(self) -> None:
        while not self.state.is_terminal and self.state.to_act == self.agent_seat:
            self.agent_move()

    def human_act(self, kind: str, amount_bb: float | None) -> None:
        if self.mode != "play":
            raise ValueError("this session is for watching")
        if self.state.is_terminal:
            raise ValueError("the hand is over; deal the next one")
        seat = 1 - self.agent_seat
        if self.state.to_act != seat:
            raise ValueError("it's not your turn")
        if kind == "fold":
            action = _core.Action.fold()
        elif kind in ("check", "call"):
            action = _core.Action.check_call()
        elif kind in ("bet", "raise", "all_in"):
            la = self.state.legal_actions()
            to = (
                la.max_raise_to
                if kind == "all_in" or amount_bb is None
                else round(amount_bb * CHIPS_PER_BB)
            )
            action = _core.Action.bet_raise_to(to)
        else:
            raise ValueError(f"unknown action '{kind}'")
        self._apply(seat, action, "You")
        self._advance_agent()

    def step(self) -> None:
        """Watch mode: play the next decision (agent or bot)."""
        if self.mode != "watch" or self.bot is None:
            raise ValueError("this session is for playing")
        if self.state.is_terminal:
            raise ValueError("the hand is over; deal the next one")
        if self.state.to_act == self.agent_seat:
            self.agent_move()
        else:
            seat = 1 - self.agent_seat
            self._apply(seat, self.bot.act(self.state, self.visible_board()), self.opponent)

    # ------------------------------------------------------------------ view

    def view(self) -> dict[str, Any]:
        s = self.state
        other = 1 - self.agent_seat
        show_all = self.mode == "watch" or (self.result is not None and self.result["showdown"])
        seats = []
        for seat in (0, 1):
            is_agent = seat == self.agent_seat
            seats.append(
                {
                    "seat": seat,
                    "name": "Regret"
                    if is_agent
                    else ("You" if self.mode == "play" else self.opponent),
                    "position": "BTN" if seat == s.button else "BB",
                    "stack_bb": _bb(s.stacks[seat]),
                    "bet_bb": _bb(s.bets[seat]),
                    "status": str(s.statuses[seat]).split(".")[-1].lower(),
                    "cards": [card_str(c) for c in self.hole[seat]]
                    if (not is_agent or show_all)
                    else None,
                }
            )
        legal = None
        if self.mode == "play" and not s.is_terminal and s.to_act == other:
            la = s.legal_actions()
            legal = {
                "fold": la.fold,
                "call_bb": _bb(la.call_amount),
                "can_raise": la.bet_raise,
                "min_raise_to_bb": _bb(la.min_raise_to),
                "max_raise_to_bb": _bb(la.max_raise_to),
                "pot_bb": _bb(s.pot),
                "current_bet_bb": _bb(s.current_bet),
            }
        return {
            "mode": self.mode,
            "hand": self.hands,
            "street": STREETS[int(s.street)],
            "pot_bb": _bb(s.pot),
            "board": [card_str(c) for c in self.visible_board()],
            "seats": seats,
            "to_act": None if s.is_terminal else s.to_act,
            "agent_seat": self.agent_seat,
            "legal": legal,
            "log": self.log,
            # In play mode the agent's reasoning would reveal its cards, so it's shown once the
            # hand is over; in watch mode it's shown as it happens.
            "decisions": self.decisions if (self.mode == "watch" or self.result) else [],
            "decisions_hidden": self.mode == "play" and not self.result and bool(self.decisions),
            "result": self.result,
            "agent_net_bb": round(self.net_bb, 2),
        }


def _describe(action: _core.Action, la: _core.LegalActions) -> str:
    if action.type == _core.ActionType.FOLD:
        return "folds"
    if action.type == _core.ActionType.CHECK_CALL:
        return f"calls {_bb(la.call_amount):g}" if la.call_amount else "checks"
    if action.amount == la.max_raise_to:
        return f"all-in ({_bb(action.amount):g})"
    return f"raises to {_bb(action.amount):g}" if la.call_amount else f"bets {_bb(action.amount):g}"


class Sessions:
    """In-memory sessions with a size cap and idle expiry (a demo, not a game server)."""

    def __init__(self, agent: RegretAgent) -> None:
        self.agent = agent
        self.items: dict[str, Session] = {}

    def create(
        self, mode: Mode, opponent: str = "equity-threshold", seed: int | None = None
    ) -> tuple[str, Session]:
        if mode == "watch" and opponent not in BASELINE_FACTORIES:
            raise ValueError(f"unknown opponent '{opponent}' ({', '.join(BASELINE_FACTORIES)})")
        self._expire()
        sid = secrets.token_urlsafe(9)
        self.items[sid] = Session(self.agent, mode, opponent, np.random.default_rng(seed))
        return sid, self.items[sid]

    def get(self, sid: str) -> Session:
        s = self.items.get(sid)
        if s is None:
            raise KeyError(sid)
        s.touched = time.time()
        return s

    def _expire(self) -> None:
        now = time.time()
        for sid in [k for k, v in self.items.items() if now - v.touched > SESSION_TTL_S]:
            del self.items[sid]
        while len(self.items) >= MAX_SESSIONS:
            del self.items[min(self.items, key=lambda k: self.items[k].touched)]
