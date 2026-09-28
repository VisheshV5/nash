"""GameState input: schema, replay validation, and error messages."""

import copy
import json
from collections.abc import Callable
from typing import Any

import pytest

from regret._core import PlayerStatus, Street
from regret.engine.state import GameStateError, parse_game_state

# SPEC §4's example with consistent numbers (100bb stacks): HJ opens 2.5, BTN and BB call,
# BB checks the flop, HJ bets 6, BTN to act.
SPEC_EXAMPLE: dict[str, Any] = {
    "num_players": 6,
    "hero_seat": "BTN",
    "hero_cards": ["As", "Kd"],
    "board": ["Qh", "Jc", "2s"],
    "street": "flop",
    "pot": 14.0,
    "players": [
        {"seat": "UTG", "stack": 100.0, "status": "folded"},
        {"seat": "HJ", "stack": 91.5, "status": "active"},
        {"seat": "CO", "stack": 100.0, "status": "folded"},
        {"seat": "BTN", "stack": 97.5, "status": "active"},
        {"seat": "SB", "stack": 99.5, "status": "folded"},
        {"seat": "BB", "stack": 97.5, "status": "active"},
    ],
    "action_history": {
        "preflop": [
            ["UTG", "fold"],
            ["HJ", "raise", 2.5],
            ["CO", "fold"],
            ["BTN", "call", 2.5],
            ["SB", "fold"],
            ["BB", "call", 2.5],
        ],
        "flop": [["BB", "check"], ["HJ", "bet", 6.0]],
    },
    "to_act": "BTN",
    "big_blind": 1.0,
}

HEADS_UP_PREFLOP: dict[str, Any] = {
    "num_players": 2,
    "hero_seat": "BB",
    "hero_cards": ["7h", "7d"],
    "street": "preflop",
    "pot": 4.0,
    "players": [
        {"seat": "SB", "stack": 97.0, "status": "active"},
        {"seat": "BB", "stack": 99.0, "status": "active"},
    ],
    "action_history": {"preflop": [["SB", "raise", 3.0]]},
    "to_act": "BB",
}


def test_spec_example_replays() -> None:
    v = parse_game_state(SPEC_EXAMPLE)
    e = v.engine
    assert v.labels[v.hero] == "BTN"
    assert e.to_act == v.hero
    assert e.street == Street.FLOP
    assert v.to_units(e.pot) == 14.0
    assert v.to_units(e.current_bet) == 6.0
    la = e.legal_actions()
    assert v.to_units(la.call_amount) == 6.0
    assert v.to_units(la.min_raise_to) == 12.0
    assert v.to_units(la.max_raise_to) == 97.5
    assert v.hero_cards == [51, 45]


def test_json_string_and_lowercase_seats() -> None:
    data = copy.deepcopy(SPEC_EXAMPLE)
    data["hero_seat"] = data["to_act"] = "btn"
    v = parse_game_state(json.dumps(data))
    assert v.labels[v.hero] == "BTN"


def test_heads_up_accepts_sb_alias() -> None:
    v = parse_game_state(HEADS_UP_PREFLOP)
    assert v.hero == 1
    assert v.to_units(v.engine.legal_actions().call_amount) == 2.0


def test_big_blind_in_chips() -> None:
    data = copy.deepcopy(HEADS_UP_PREFLOP)
    data |= {"big_blind": 100, "small_blind": 50, "pot": 400}
    data["players"] = [
        {"seat": "BTN", "stack": 9700, "status": "active"},
        {"seat": "BB", "stack": 9900, "status": "active"},
    ]
    data["action_history"] = {"preflop": [["BTN", "raise", 300]]}
    v = parse_game_state(data)
    assert v.to_units(v.engine.legal_actions().call_amount) == 200


def test_all_in_and_short_stack() -> None:
    data = {
        "num_players": 3,
        "hero_seat": "BB",
        "hero_cards": ["Ac", "Qc"],
        "street": "preflop",
        "pot": 26.5,
        "players": [
            {"seat": "BTN", "stack": 0.0, "status": "all_in"},
            {"seat": "SB", "stack": 60.0, "status": "active"},
            {"seat": "BB", "stack": 99.0, "status": "active"},
        ],
        "action_history": {"preflop": [["BTN", "all_in", 20.0], ["SB", "raise", 40.0]]},
        "to_act": "BB",
    }
    with pytest.raises(GameStateError, match="pot"):
        parse_game_state(data)  # 20 + 40 + 1 = 61, not 26.5
    data["pot"] = 61.0
    v = parse_game_state(data)
    assert v.engine.statuses[2] == PlayerStatus.ALL_IN
    assert v.to_units(v.engine.legal_actions().min_raise_to) == 60.0


def test_errors_are_structured() -> None:
    data = copy.deepcopy(SPEC_EXAMPLE)
    data["pot"] = 20.5
    data["to_act"] = "HJ"
    with pytest.raises(GameStateError) as info:
        parse_game_state(data)
    fields = {i.field for i in info.value.issues}
    assert fields == {"pot", "to_act"}
    assert info.value.to_dict()["errors"][0]["field"] in fields


Mutation = Callable[[dict[str, Any]], None]


def _set(path: str, value: Any) -> Mutation:
    def apply(d: dict[str, Any]) -> None:
        *head, last = path.split(".")
        node: Any = d
        for k in head:
            node = node[int(k)] if k.isdigit() else node[k]
        node[int(last) if last.isdigit() else last] = value

    return apply


def _flop(*actions: list[Any]) -> Mutation:
    return _set("action_history.flop", list(actions))


def _pre(i: int, action: list[Any]) -> Mutation:
    def apply(d: dict[str, Any]) -> None:
        d["action_history"]["preflop"][i] = action

    return apply


def _spec_literal(d: dict[str, Any]) -> None:
    d["pot"] = 20.5


def _all_fold(d: dict[str, Any]) -> None:
    d["action_history"]["preflop"] = [
        ["UTG", "fold"],
        ["HJ", "fold"],
        ["CO", "fold"],
        ["BTN", "fold"],
        ["SB", "fold"],
    ]


def _drop_bb_call(d: dict[str, Any]) -> None:
    d["action_history"]["preflop"].pop()


MALFORMED: list[tuple[str, Mutation, str, str]] = [
    ("spec literal pot", _spec_literal, "pot", "doesn't match 14.0"),
    ("unknown seat", _set("players.0.seat", "MP"), "players", "unknown seat 'MP'"),
    ("duplicate seat", _set("players.2.seat", "UTG"), "players", "listed twice"),
    ("player count", _set("num_players", 5), "players", "num_players is 5 but 6 players"),
    ("duplicate card", _set("board.0", "As"), "board", "appears twice"),
    ("invalid card", _set("hero_cards.1", "Ax"), "hero_cards[1]", "invalid card"),
    ("board size", _set("board", ["Qh", "Jc"]), "board", "needs 3 board cards"),
    ("street name", _set("street", "fourth"), "street", "Input should be"),
    ("out of turn", _pre(1, ["CO", "raise", 2.5]), "action_history.preflop[1]", "out of turn"),
    ("check vs bet", _pre(0, ["UTG", "check"]), "action_history.preflop[0]", "check facing a bet"),
    ("call nothing", _flop(["BB", "call"]), "action_history.flop[0]", "nothing to call"),
    ("fold no bet", _flop(["BB", "fold"]), "action_history.flop[0]", "no bet to face"),
    ("preflop bet", _pre(1, ["HJ", "bet", 2.5]), "action_history.preflop[1]", "use 'raise'"),
    ("raise no bet", _flop(["BB", "raise", 3.0]), "action_history.flop[0]", "use 'bet'"),
    ("min raise", _pre(1, ["HJ", "raise", 1.5]), "action_history.preflop[1]", "below the minimum"),
    ("no amount", _pre(1, ["HJ", "raise"]), "action_history.preflop.1", "needs an amount"),
    (
        "check amount",
        _flop(["BB", "check", 1.0], ["HJ", "bet", 6.0]),
        "action_history.flop.0",
        "takes no amount",
    ),
    (
        "call amount",
        _pre(3, ["BTN", "call", 3.0]),
        "action_history.preflop[3]",
        "doesn't match the replay",
    ),
    ("bad action shape", _pre(0, ["UTG"]), "action_history.preflop.0", "[seat, kind]"),
    ("unknown kind", _pre(0, ["UTG", "limp"]), "action_history.preflop.0.kind", "Input should be"),
    ("to_act", _set("to_act", "BB"), "to_act", "BTN is to act"),
    ("hero not to act", _set("hero_seat", "HJ"), "hero_seat", "not the hero's"),
    ("status", _set("players.4.status", "active"), "players[SB].status", "replay says 'folded'"),
    ("street ahead", _set("street", "turn"), "board", "needs 4 board cards"),
    (
        "future actions",
        _set("action_history.turn", [["BB", "check"]]),
        "action_history.turn",
        "hand is on the flop",
    ),
    ("sub-cent", _set("pot", 14.004), "pot", "finer than 0.01"),
    ("extra field", _set("rake", 0.5), "rake", "Extra inputs"),
    ("negative stack", _set("players.1.stack", -3), "players.1.stack", "greater than or equal"),
    ("hand over", _all_fold, "action_history.flop[0]", "already over"),
    ("blinds", _set("small_blind", 1.0), "small_blind", "below the big blind"),
    ("street unfinished", _drop_bb_call, "action_history.preflop", "isn't finished"),
]


@pytest.mark.parametrize(
    ("name", "mutate", "field", "message"), MALFORMED, ids=[m[0] for m in MALFORMED]
)
def test_malformed_states_are_rejected(
    name: str, mutate: Mutation, field: str, message: str
) -> None:
    data = copy.deepcopy(SPEC_EXAMPLE)
    mutate(data)
    with pytest.raises(GameStateError) as info:
        parse_game_state(data)
    issues = info.value.issues
    assert any(i.field == field and message in i.message for i in issues), issues
