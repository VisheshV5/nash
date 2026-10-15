"""Benchmark against Slumbot (slumbot.com) through its public API.

Slumbot's rules: heads-up no-limit, blinds 50/100, 200bb stacks reset every hand. Its action
strings use "k" check, "c" call, "f" fold and "b<N>" bet or raise to N chips on the current
street, with "/" between streets, e.g. "b200c/kk/kk/kb200". Chips are 1/100 of a big blind,
the same scale as our engine.

`client_pos` 0 is the big blind (our seat 1), 1 is the button (our seat 0). Play an even number
of hands so each seat comes up equally often.

The HTTP transport is injectable so the whole protocol can be tested offline against a fake
server (tests/test_slumbot.py); `play` only talks to slumbot.com when given the real transport.
"""

from __future__ import annotations

import json
import math
import ssl
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from regret import _core
from regret.engine.cards import parse_card

URL = "https://slumbot.com/slumbot/api/"
BLINDS = (50, 100)
STACK = 20_000
Transport = Callable[[str, dict[str, Any]], dict[str, Any]]


def http_transport(timeout: float = 30) -> Transport:
    try:
        import certifi

        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = ssl.create_default_context()

    def post(endpoint: str, body: dict[str, Any]) -> dict[str, Any]:
        req = urllib.request.Request(
            URL + endpoint,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, context=ctx, timeout=timeout) as r:
            out: dict[str, Any] = json.loads(r.read())
        if "error_msg" in out:
            raise RuntimeError(f"slumbot: {out['error_msg']}")
        return out

    return post


def replay(action: str) -> _core.HandState:
    """Engine state after a Slumbot action string (heads-up, 200bb)."""
    h = _core.HandState([STACK, STACK], *BLINDS)
    streets = action.split("/")
    for i, street in enumerate(streets):
        j = 0
        while j < len(street):
            ch = street[j]
            if ch in "kc":
                h.apply(_core.Action.check_call())
                j += 1
            elif ch == "f":
                h.apply(_core.Action.fold())
                j += 1
            elif ch == "b":
                k = j + 1
                while k < len(street) and street[k].isdigit():
                    k += 1
                h.apply(_core.Action.bet_raise_to(int(street[j + 1 : k])))
                j = k
            else:
                raise ValueError(f"unknown action '{ch}' in {action!r}")
        if i < len(streets) - 1 and int(h.street) == i and not h.is_terminal:
            raise ValueError(f"street {i} didn't close in {action!r}")
    return h


def encode(state: _core.HandState, action: _core.Action) -> str:
    """Our action as Slumbot's increment ("f", "k", "c" or "b<N>")."""
    if action.type == _core.ActionType.FOLD:
        return "f"
    if action.type == _core.ActionType.CHECK_CALL:
        return "c" if state.legal_actions().call_amount else "k"
    return f"b{action.amount}"


@dataclass
class SlumbotResult:
    hands: int = 0
    winnings: list[int] = field(default_factory=list)  # chips per hand, our side

    @property
    def mbb(self) -> float:
        return 1000 * sum(self.winnings) / max(1, len(self.winnings)) / BLINDS[1]

    @property
    def ci95(self) -> float:
        n = len(self.winnings)
        if n < 2:
            return float("inf")
        mean = sum(self.winnings) / n
        var = sum((w - mean) ** 2 for w in self.winnings) / (n - 1)
        return 1.96 * math.sqrt(var / n) * 1000 / BLINDS[1]

    def summary(self) -> dict[str, Any]:
        return {
            "hands": self.hands,
            "mbb_per_hand": round(self.mbb, 1),
            "ci95": round(self.ci95, 1),
            "significant": abs(self.mbb) > self.ci95,
        }


def play(
    decide: Callable[[_core.HandState, int, list[int], list[int]], _core.Action],
    hands: int,
    transport: Transport,
    pause: float = 0.1,
    progress: Callable[[SlumbotResult], None] | None = None,
) -> SlumbotResult:
    """Play `hands` hands. `decide(state, our_seat, hole, board)` returns our engine action."""
    result = SlumbotResult()
    token: str | None = None
    for _ in range(hands):
        resp = transport("new_hand", {} if token is None else {"token": token})
        while True:
            token = resp.get("token", token)
            if "winnings" in resp:
                result.winnings.append(int(resp["winnings"]))
                result.hands += 1
                break
            seat = 1 - int(resp["client_pos"])
            state = replay(resp["action"])
            if state.to_act != seat:
                raise RuntimeError(f"not our turn in {resp['action']!r}")
            hole = [parse_card(c) for c in resp["hole_cards"]]
            board = [parse_card(c) for c in resp["board"]]
            action = decide(state, seat, hole, board)
            time.sleep(pause)
            resp = transport("act", {"token": token, "incr": encode(state, action)})
        if progress:
            progress(result)
    return result
