"""Blueprint + real-time search vs blueprint alone, in resumable chunks, only on AC power.

uv run python scripts/search_benchmark.py bundles/hu-100bb runs/search-bench
uv run python scripts/search_benchmark.py bundles/hu-100bb runs/search-bench --streets turn river

Plays duplicate chunks of --chunk hands (seed = --seed + chunk number) until stopped (Ctrl+C or
SIGTERM) or --chunks are done, appending each chunk's result to <out>/chunks.jsonl and the
combined result so far to <out>/summary.json. Re-running continues with the next chunk. Every
100 pairs it checks the power source: on battery it pauses until the machine is plugged in
again. A chunk interrupted midway is discarded and replayed.

Chunks are independent, so the combined win rate is their hand-weighted mean and its 95% CI
comes from the chunks' standard errors.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from regret.abstraction.cards import CardAbstraction
from regret.agent import RegretAgent
from regret.eval.match import duplicate_match
from regret.eval.players import AgentPlayer
from regret.eval.setup import table_chips
from regret.export.bundle import load_bundle


class Stop(Exception):
    pass


def on_ac() -> bool:
    forced = os.environ.get("REGRET_FORCE_POWER_FILE")
    if forced:
        return Path(forced).read_text().strip() == "ac"
    if shutil.which("pmset") is None:
        return True
    out = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout
    return "AC Power" in out


def combine(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    hands = sum(c["hands"] for c in chunks)
    mbb = sum(c["mbb_per_hand"] * c["hands"] for c in chunks) / hands
    se = math.sqrt(sum((c["ci95"] / 1.96 * c["hands"]) ** 2 for c in chunks)) / hands
    searches = sum(c["searches"] for c in chunks)
    seconds = sum(c["search_seconds"] for c in chunks)
    return {
        "chunks": len(chunks),
        "hands": hands,
        "mbb_per_hand": round(mbb, 1),
        "ci95": round(1.96 * se, 1),
        "significant": abs(mbb) > 1.96 * se,
        "searches": searches,
        "ms_per_search": round(seconds / max(1, searches) * 1000, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("bundle", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--chunk", type=int, default=2000, help="hands per chunk")
    ap.add_argument("--chunks", type=int, default=1_000_000, help="stop after this many")
    ap.add_argument("--streets", nargs="+", default=["river"], choices=["turn", "river"])
    ap.add_argument("--search-ms", type=int, help="search time budget per decision")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--poll", type=float, default=60, help="seconds between power checks")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    log_path = args.out / "chunks.jsonl"
    chunks = (
        [json.loads(line) for line in log_path.read_text().splitlines()]
        if log_path.exists()
        else []
    )

    def say(msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        with (args.out / "bench.log").open("a") as f:
            f.write(line + "\n")

    stopping = False

    def request_stop(*_: Any) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    def wait_for_power(_: int = 0) -> None:
        if stopping:
            raise Stop
        if on_ac():
            return
        say("on battery: paused")
        while not on_ac():
            if stopping:
                raise Stop
            time.sleep(args.poll)
        say("plugged in: resuming")

    cfg, strategy, cards_dir = load_bundle(args.bundle)
    search = cfg.search.model_copy(update={"streets": args.streets})
    if args.search_ms:
        search = search.model_copy(
            update={
                "time_budget_ms": args.search_ms,
                "hard_cap_ms": max(search.hard_cap_ms, 2 * args.search_ms),
            }
        )
    agent = RegretAgent(
        cfg.model_copy(update={"search": search}), strategy, CardAbstraction.load(cards_dir)
    )
    stacks, sb, bb = table_chips(cfg)
    say(f"benchmark: {args.bundle} search on {args.streets}, {len(chunks)} chunks done")

    try:
        while len(chunks) < args.chunks:
            wait_for_power()
            index = len(chunks)
            with_search = AgentPlayer(agent, stacks, sb, bb, search=True)
            without = AgentPlayer(agent, stacks, sb, bb, search=False)
            t0 = time.time()
            r = duplicate_match(
                with_search,
                without,
                max(1, args.chunk // 2),
                stacks,
                sb,
                bb,
                seed=args.seed + index,
                progress=wait_for_power,
            )
            chunk = r.summary() | {
                "chunk": index,
                "seed": args.seed + index,
                "streets": args.streets,
                "searches": with_search.searches,
                "search_seconds": round(with_search.search_seconds, 2),
                "seconds": round(time.time() - t0, 1),
            }
            chunks.append(chunk)
            with log_path.open("a") as f:
                f.write(json.dumps(chunk) + "\n")
            summary = combine(chunks)
            (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
            say(
                f"chunk {index}: {chunk['mbb_per_hand']:+.1f} ± {chunk['ci95']:.1f}; "
                f"combined {summary['mbb_per_hand']:+.1f} ± {summary['ci95']:.1f} mbb/hand "
                f"over {summary['hands']:,} hands ({summary['ms_per_search']:.0f} ms/search)"
            )
    except Stop:
        say("stopped (an unfinished chunk is replayed next time)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
