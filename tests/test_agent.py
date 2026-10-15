"""The public agent (RegretAgent) and the REST API, on a tiny trained bundle."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from regret import RegretAgent
from regret.agent import UnsupportedTableSize
from regret.api.app import create_app
from regret.cfr.checkpoint import latest_checkpoint
from regret.cfr.trainer import RunSpec, Trainer
from regret.engine.state import GameStateError
from regret.export.bundle import export_bundle
from regret.utils.config import RegretConfig
from test_state import SPEC_EXAMPLE


@pytest.fixture(scope="module")
def bundle(tiny_holdem: RegretConfig, tmp_path_factory: pytest.TempPathFactory) -> Path:
    run_dir = tmp_path_factory.mktemp("agent-run")
    Trainer(RunSpec.from_config(tiny_holdem), run_dir).run(max_iterations=40_000)
    ckpt = latest_checkpoint(run_dir / "checkpoints")
    assert ckpt is not None
    out = tmp_path_factory.mktemp("bundle") / "hu-tiny"
    export_bundle(tiny_holdem, ckpt, out)
    return out


@pytest.fixture(scope="module")
def agent(bundle: Path) -> RegretAgent:
    return RegretAgent.load(bundle)


# Heads-up, 100bb: button opens to 2.5, big blind to act.
PREFLOP: dict[str, Any] = {
    "num_players": 2,
    "hero_seat": "BB",
    "hero_cards": ["Ah", "Kh"],
    "street": "preflop",
    "pot": 3.5,
    "players": [
        {"seat": "BTN", "stack": 97.5, "status": "active"},
        {"seat": "BB", "stack": 99.0, "status": "active"},
    ],
    "action_history": {"preflop": [["BTN", "raise", 2.5]]},
    "to_act": "BB",
}

# Flop after a 2.5bb open and call: BB checks, button bets 2.2 (44% pot, between abstract sizes).
FLOP: dict[str, Any] = {
    "num_players": 2,
    "hero_seat": "BB",
    "hero_cards": ["Qs", "Qd"],
    "board": ["7h", "4c", "2s"],
    "street": "flop",
    "pot": 7.2,
    "players": [
        {"seat": "BTN", "stack": 95.3, "status": "active"},
        {"seat": "BB", "stack": 97.5, "status": "active"},
    ],
    "action_history": {
        "preflop": [["BTN", "raise", 2.5], ["BB", "call", 2.5]],
        "flop": [["BB", "check"], ["BTN", "bet", 2.2]],
    },
    "to_act": "BB",
}


def test_act_returns_a_valid_strategy_and_legal_action(agent: RegretAgent) -> None:
    d = agent.act(PREFLOP, seed=1)
    assert d.source == "blueprint"
    assert sum(d.strategy.values()) == pytest.approx(1.0, abs=1e-3)
    assert all(k in {"fold", "call", "all_in"} or k.startswith("raise_") for k in d.strategy)
    assert d.recommended["action"] in {"fold", "call", "raise", "all_in"}
    assert d.latency_ms < 1000


def test_seed_makes_samples_reproducible(agent: RegretAgent) -> None:
    a = [agent.act(FLOP, seed=s).recommended for s in range(10)]
    b = [agent.act(FLOP, seed=s).recommended for s in range(10)]
    assert a == b


def test_argmax_picks_the_most_likely_action(agent: RegretAgent) -> None:
    d = agent.act(FLOP, mode="argmax")
    best = max(d.strategy, key=lambda k: d.strategy[k])
    rec = d.recommended
    assert rec.get("label", rec["action"]) == best or rec["action"] == best


def test_off_tree_bet_is_translated_and_explained(agent: RegretAgent) -> None:
    d = agent.act(FLOP, explain=True, seed=0)
    assert d.explain is not None
    hist = d.explain["abstract_histories"]
    assert len(hist) == 2  # 2.2 into 4.8 (44% pot) splits between 33% and 75% pot
    assert sum(h["probability"] for h in hist) == pytest.approx(1.0, abs=1e-3)
    assert any("->" in t for h in hist for t in h["translations"])
    assert d.explain["street"] == "flop"
    assert 0 <= d.explain["equity_vs_random_hand"] <= 1
    assert d.explain["to_call"] == pytest.approx(2.2)


def test_amounts_are_in_input_units(agent: RegretAgent) -> None:
    chips = copy.deepcopy(PREFLOP)
    chips |= {"big_blind": 100, "small_blind": 50, "pot": 350}
    chips["players"] = [
        {"seat": "BTN", "stack": 9750, "status": "active"},
        {"seat": "BB", "stack": 9900, "status": "active"},
    ]
    chips["action_history"] = {"preflop": [["BTN", "raise", 250]]}
    a = agent.act(PREFLOP, mode="argmax")
    b = agent.act(chips, mode="argmax")
    assert a.strategy == b.strategy
    if a.recommended["amount"] is not None:
        assert b.recommended["amount"] == pytest.approx(a.recommended["amount"] * 100)


def test_wrong_table_size_and_bad_input(agent: RegretAgent) -> None:
    with pytest.raises(UnsupportedTableSize, match="2-handed"):
        agent.act(SPEC_EXAMPLE)
    bad = copy.deepcopy(PREFLOP)
    bad["pot"] = 9.0
    with pytest.raises(GameStateError, match="pot"):
        agent.act(bad)


def test_deep_stacks_warn(agent: RegretAgent) -> None:
    deep = copy.deepcopy(PREFLOP)
    deep["players"] = [
        {"seat": "BTN", "stack": 197.5, "status": "active"},
        {"seat": "BB", "stack": 199.0, "status": "active"},
    ]
    with pytest.warns(UserWarning, match="trained at 100bb"):
        d = agent.act(deep)
    assert d.warnings


# ---------------------------------------------------------------- REST API


@pytest.fixture(scope="module")
def client(agent: RegretAgent) -> TestClient:
    return TestClient(create_app(agent))


def test_api_health_and_info(client: TestClient) -> None:
    assert client.get("/health").json()["status"] == "ok"
    info = client.get("/info").json()
    assert info["players"] == 2
    assert info["name"] == "hu-tiny"


def test_api_act(client: TestClient) -> None:
    r = client.post("/act", json=FLOP, params={"seed": 3, "explain": True})
    assert r.status_code == 200
    body = r.json()
    assert set(body) >= {"strategy", "recommended", "source", "latency_ms", "explain"}
    again = client.post("/act", json=FLOP, params={"seed": 3, "explain": True}).json()
    assert again["recommended"] == body["recommended"]


def test_api_rejects_invalid_states_with_field_errors(client: TestClient) -> None:
    bad = copy.deepcopy(FLOP)
    bad["to_act"] = "BTN"
    r = client.post("/act", json=bad)
    assert r.status_code == 422
    fields = [e["field"] for e in r.json()["errors"]]
    assert "to_act" in fields
    schema_bad = copy.deepcopy(FLOP)
    schema_bad["hero_cards"] = ["As"]
    assert client.post("/act", json=schema_bad).status_code == 422


# Checked down to the river, where the big blind bets 3 into 5 (60% pot, off the blueprint's menu).
RIVER: dict[str, Any] = {
    "num_players": 2,
    "hero_seat": "BTN",
    "hero_cards": ["Ah", "9h"],
    "board": ["Kd", "9s", "5h", "2c", "7d"],
    "street": "river",
    "pot": 8.0,
    "players": [
        {"seat": "BTN", "stack": 97.5, "status": "active"},
        {"seat": "BB", "stack": 94.5, "status": "active"},
    ],
    "action_history": {
        "preflop": [["BTN", "raise", 2.5], ["BB", "call", 2.5]],
        "flop": [["BB", "check"], ["BTN", "check"]],
        "turn": [["BB", "check"], ["BTN", "check"]],
        "river": [["BB", "bet", 3.0]],
    },
    "to_act": "BTN",
}


def test_river_decisions_use_search(agent: RegretAgent) -> None:
    d = agent.act(RIVER, explain=True, seed=0)
    assert d.source == "subgame_solve"
    assert set(d.strategy) <= {"fold", "call"} | {
        k for k in d.strategy if k.startswith("raise_")
    } | {"all_in"}
    assert sum(d.strategy.values()) == pytest.approx(1.0, abs=1e-3)
    assert d.explain is not None
    s = d.explain["search"]
    assert s["iterations"] >= 20
    assert s["solve_ms"] < 1500
    assert d.latency_ms < 1500  # the 1-second budget, with headroom for a loaded machine
    off = agent.act(RIVER, search=False, seed=0)
    assert off.source == "blueprint"


def test_search_ranges_follow_the_blueprint(agent: RegretAgent) -> None:
    from regret.search.river import bayes_ranges

    # No actions: uniform over hands not on the board.
    board = [0, 17, 33, 42, 51]
    r0, r1 = bayes_ranges([], board, agent.cards, agent.strategy)
    assert r0.sum() == pytest.approx(1.0, abs=1e-5)
    assert (r0 > 0).sum() == 1081
    np.testing.assert_allclose(r0, r1)
