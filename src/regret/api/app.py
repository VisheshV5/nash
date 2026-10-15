"""REST API for a Regret bundle.

    REGRET_BUNDLE=bundles/hu-100bb uv run uvicorn regret.api.app:app --port 8000

    POST /act      body: SPEC §4 GameState   query: explain, mode, seed
    GET  /health   liveness
    GET  /info     which bundle is loaded

Invalid game states get a 422 whose body lists each problem with the field it's about.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Literal

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from regret import __version__
from regret.agent import RegretAgent
from regret.engine.state import GameStateError, GameStateIn

log = logging.getLogger("regret.api")


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


app = create_app()
