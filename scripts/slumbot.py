"""Play a bundle against Slumbot over its public API and report mbb/hand with a 95% CI.

    uv run python scripts/slumbot.py bundles/hu-200bb --hands 2000

Slumbot plays 200bb stacks, so the bundle must be trained at 200bb. Requests are paced (0.1 s
apart by default); play an even number of hands so both seats come up equally often.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from regret import _core
from regret.agent import RegretAgent
from regret.eval import slumbot


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("bundle", type=Path)
    ap.add_argument("--hands", type=int, default=1000)
    ap.add_argument("--no-search", action="store_true", help="blueprint only")
    ap.add_argument("--pause", type=float, default=0.1, help="seconds between requests")
    args = ap.parse_args()

    agent = RegretAgent.load(args.bundle)
    if agent.trained_stack_bb != slumbot.STACK / slumbot.BLINDS[1]:
        print(
            f"this bundle is trained at {agent.trained_stack_bb:g}bb; Slumbot plays "
            f"{slumbot.STACK // slumbot.BLINDS[1]}bb",
            file=sys.stderr,
        )
        return 2

    def decide(
        state: _core.HandState, seat: int, hole: list[int], board: list[int]
    ) -> _core.Action:
        d = agent.decide(
            state, [slumbot.STACK] * 2, *slumbot.BLINDS, hole, board, search=not args.no_search
        )
        rec = d.recommended
        if rec["action"] == "fold":
            return _core.Action.fold() if state.legal_actions().fold else _core.Action.check_call()
        if rec["action"] in ("check", "call"):
            return _core.Action.check_call()
        return _core.Action.bet_raise_to(round(rec["amount"] * slumbot.BLINDS[1]))

    def progress(r: slumbot.SlumbotResult) -> None:
        if r.hands % 50 == 0:
            print(f"{r.hands:6d} hands  {r.mbb:+8.1f} ± {r.ci95:.1f} mbb/hand", file=sys.stderr)

    result = slumbot.play(decide, args.hands, slumbot.http_transport(), args.pause, progress)
    print(json.dumps(result.summary(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
