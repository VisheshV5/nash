"""REST API for a Regret bundle.

    REGRET_BUNDLE=bundles/hu-100bb uv run uvicorn regret.api.app:app --port 8000

    POST /act      body: SPEC §4 GameState   query: explain, mode, seed
    GET  /health   liveness
    GET  /info     which bundle is loaded
    GET  /demo/    the demo table (play against the agent, or watch it play a baseline bot)

Invalid game states get a 422 whose body lists each problem with the field it's about.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from regret import __version__
from regret.agent import RegretAgent
from regret.engine.state import GameStateError, GameStateIn

log = logging.getLogger("regret.api")


class NewSession(BaseModel):
    mode: Literal["play", "watch"] = "play"
    opponent: str = "equity-threshold"
    seed: int | None = None


class HumanAction(BaseModel):
    kind: Literal["fold", "check", "call", "bet", "raise", "all_in"]
    amount_bb: float | None = None


def create_app(agent: RegretAgent | None = None) -> FastAPI:
    app = FastAPI(
        title="Regret",
        version=__version__,
        description="Heads-up no-limit hold'em agent: game state in, strategy out.",
    )
    app.state.agent = agent
    app.state.started = time.time()

    def get_agent() -> RegretAgent:
        if app.state.agent is None:
            bundle = os.environ.get("REGRET_BUNDLE")
            if not bundle:
                raise RuntimeError("no bundle loaded; set REGRET_BUNDLE")
            app.state.agent = RegretAgent.load(bundle)
        agent_: RegretAgent = app.state.agent
        return agent_

    @app.exception_handler(GameStateError)
    async def invalid_state(request: Request, exc: GameStateError) -> JSONResponse:
        return JSONResponse(status_code=422, content=exc.to_dict())

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "uptime_s": round(time.time() - app.state.started, 1)}

    @app.get("/info")
    def info() -> dict[str, Any]:
        return {"version": __version__, **get_agent().info()}

    # ---- demo table
    from regret.demo.session import Sessions

    sessions: Sessions | None = None

    def demo() -> Sessions:
        nonlocal sessions
        if sessions is None:
            sessions = Sessions(get_agent())
        return sessions

    def session_or_404(sid: str) -> Any:
        try:
            return demo().get(sid)
        except KeyError:
            raise HTTPException(404, "session expired; start a new one") from None

    @app.post("/demo/api/sessions")
    def new_session(body: NewSession) -> dict[str, Any]:
        try:
            sid, s = demo().create(body.mode, body.opponent, body.seed)
        except ValueError as e:
            raise HTTPException(422, str(e)) from None
        return {"id": sid, **s.view()}

    @app.post("/demo/api/sessions/{sid}/act")
    def human_act(sid: str, body: HumanAction) -> dict[str, Any]:
        s = session_or_404(sid)
        try:
            s.human_act(body.kind, body.amount_bb)
        except ValueError as e:
            raise HTTPException(422, str(e)) from None
        return {"id": sid, **s.view()}

    @app.post("/demo/api/sessions/{sid}/step")
    def step(sid: str) -> dict[str, Any]:
        s = session_or_404(sid)
        try:
            s.step()
        except ValueError as e:
            raise HTTPException(422, str(e)) from None
        return {"id": sid, **s.view()}

    @app.post("/demo/api/sessions/{sid}/next")
    def next_hand(sid: str) -> dict[str, Any]:
        s = session_or_404(sid)
        s.new_hand()
        return {"id": sid, **s.view()}

    @app.get("/demo", include_in_schema=False)
    def demo_root() -> RedirectResponse:
        return RedirectResponse("/demo/")

    app.mount(
        "/demo",
        _RevalidatedStatic(directory=Path(__file__).parent.parent / "demo" / "static", html=True),
        name="demo",
    )

    @app.post("/act")
    def act(
        state: GameStateIn,
        explain: bool = False,
        mode: Literal["sample", "argmax"] = "sample",
        seed: int | None = None,
    ) -> dict[str, Any]:
        decision = get_agent().act(state, explain=explain, mode=mode, seed=seed)
        log.info(
            "act %s -> %s (%.1f ms)",
            state.street,
            decision.recommended["action"],
            decision.latency_ms,
        )
        return decision.to_dict()

    return app


class _RevalidatedStatic(StaticFiles):
    """Static files that browsers re-check on every load, so an updated demo page shows up."""

    def file_response(self, *args: Any, **kwargs: Any) -> Response:
        response = super().file_response(*args, **kwargs)
        response.headers["cache-control"] = "no-cache"
        return response


app = create_app()
