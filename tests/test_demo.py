"""Demo table API: play hands against the agent, watch it play a baseline, static page."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from regret import RegretAgent
from regret.api.app import create_app
from regret.cfr.checkpoint import latest_checkpoint
from regret.cfr.trainer import RunSpec, Trainer
from regret.export.bundle import export_bundle
from regret.utils.config import RegretConfig


@pytest.fixture(scope="module")
def client(tiny_holdem: RegretConfig, tmp_path_factory: pytest.TempPathFactory) -> TestClient:
    run = tmp_path_factory.mktemp("demo-run")
    Trainer(RunSpec.from_config(tiny_holdem), run).run(max_iterations=20_000)
    ckpt = latest_checkpoint(run / "checkpoints")
    assert ckpt is not None
    out = tmp_path_factory.mktemp("demo-bundle") / "hu-tiny"
    export_bundle(tiny_holdem, ckpt, out)
    fast = RegretAgent.load(out)
    fast.config = fast.config.model_copy(
        update={
            "search": fast.config.search.model_copy(
                update={"time_budget_ms": 30, "hard_cap_ms": 60}
            )
        }
    )
    return TestClient(create_app(fast))


def _play_out(client: TestClient, s: dict[str, Any]) -> dict[str, Any]:
    for _ in range(40):
        if s["result"]:
            return s
        legal = s["legal"]
        assert legal is not None
        kind = "call" if legal["call_bb"] else "check"
        s = client.post(f"/demo/api/sessions/{s['id']}/act", json={"kind": kind}).json()
    raise AssertionError("hand didn't finish")


def test_play_a_few_hands(client: TestClient) -> None:
    s = client.post("/demo/api/sessions", json={"mode": "play", "seed": 1}).json()
    assert s["mode"] == "play"
    for hand in range(4):
        s = _play_out(client, s)
        assert sum(s["result"]["payoffs_bb"]) == pytest.approx(0)
        assert s["seats"][s["agent_seat"]]["cards"] is not None or not s["result"]["showdown"]
        s = client.post(f"/demo/api/sessions/{s['id']}/next").json()
        assert s["hand"] == hand + 2
    assert all("strategy" in d for d in s["decisions"])


def test_agent_cards_and_reasoning_stay_hidden_during_play(client: TestClient) -> None:
    for seed in range(2, 8):
        s = client.post("/demo/api/sessions", json={"mode": "play", "seed": seed}).json()
        if s["result"]:
            continue
        assert s["seats"][s["agent_seat"]]["cards"] is None
        assert s["seats"][1 - s["agent_seat"]]["cards"] is not None
        assert s["decisions"] == []  # would reveal the agent's hand
        done = _play_out(client, s)
        if done["log"] and any(entry["who"] == "Regret" for entry in done["log"]):
            assert done["decisions"]


def test_watch_mode_steps_through_hands(client: TestClient) -> None:
    s = client.post(
        "/demo/api/sessions", json={"mode": "watch", "opponent": "equity-threshold", "seed": 3}
    ).json()
    decided = False
    for _ in range(60):
        if s["result"]:
            s = client.post(f"/demo/api/sessions/{s['id']}/next").json()
            continue
        s = client.post(f"/demo/api/sessions/{s['id']}/step").json()
        if s["decisions"]:
            decided = True
            d = s["decisions"][-1]
            assert sum(d["strategy"].values()) == pytest.approx(1, abs=1e-3)
            assert "equity_vs_random_hand" in d["explain"]
    assert decided


def test_errors(client: TestClient) -> None:
    assert client.post("/demo/api/sessions/nope/step").status_code == 404
    r = client.post("/demo/api/sessions", json={"mode": "watch", "opponent": "shark"})
    assert r.status_code == 422
    s = client.post("/demo/api/sessions", json={"mode": "play", "seed": 4}).json()
    assert client.post(f"/demo/api/sessions/{s['id']}/step").status_code == 422
    if s["legal"] and s["legal"]["can_raise"]:
        bad = client.post(
            f"/demo/api/sessions/{s['id']}/act", json={"kind": "raise", "amount_bb": 0.01}
        )
        assert bad.status_code == 422


def test_static_page_is_served(client: TestClient) -> None:
    r = client.get("/demo/")
    assert r.status_code == 200
    assert "Regret's thinking" in r.text
    assert client.get("/demo/app.js").status_code == 200
    assert Path(__file__).parents[1].joinpath("src/regret/demo/static/style.css").exists()
