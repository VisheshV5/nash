"""Print the progress of a training run (safe to run over SSH while training continues).

uv run python scripts/status.py runs/leduc
uv run python scripts/status.py            # most recently active run under runs/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from regret.cfr.checkpoint import list_checkpoints


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # a line being written right now
    return rows


def _alive(run_dir: Path) -> int | None:
    pid_file = run_dir / "train.pid"
    try:
        pid = int(pid_file.read_text())
        os.kill(pid, 0)
        return pid
    except (OSError, ValueError):
        return None


def _duration(seconds: float | None) -> str:
    if seconds is None:
        return "?"
    s = int(seconds)
    d, s = divmod(s, 86400)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return (f"{d}d " if d else "") + f"{h:02d}:{m:02d}:{s:02d}"


def status(run_dir: Path) -> str:
    rows = _rows(run_dir / "metrics.jsonl")
    if not rows:
        return f"{run_dir}: no training events yet"
    last = {r["event"]: r for r in rows}
    start = last.get("start", {})
    progress = last.get("progress", {})
    pid = _alive(run_dir)
    end = last.get("end")
    if pid:
        state = f"RUNNING (pid {pid})"
    elif end and rows.index(end) > rows.index(start):
        state = f"STOPPED ({end['reason']})"
    else:
        state = "NOT RUNNING (no clean exit recorded; resume with --resume)"

    iteration = max(r.get("iteration", 0) for r in rows)
    target = start.get("target")
    lines = [
        f"run        {run_dir}",
        f"state      {state}",
        f"iteration  {iteration:,}"
        + (f" / {target:,} ({100 * iteration / target:.1f}%)" if target else ""),
    ]
    if progress:
        eta = f"   ETA {_duration(progress['eta_s'])}" if pid and progress.get("eta_s") else ""
        lines += [
            f"speed      {progress['it_per_s']:,.0f} it/s"
            f"   elapsed {_duration(progress['elapsed_s'])}{eta}",
            f"memory     RSS {progress['rss_mb']:,.0f} MB"
            f"   regret table {progress['store_mb']:,.0f} MB"
            f"   {progress['infosets']:,} infosets",
        ]
    if "eval" in last:
        e = last["eval"]
        lines.append(f"eval       exploitability {e['exploitability']:.6f} @ {e['iteration']:,}")
    ckpts = list_checkpoints(run_dir / "checkpoints")
    if ckpts:
        age = time.time() - ckpts[-1].stat().st_mtime
        lines.append(f"checkpoint {ckpts[-1].name} ({_duration(age)} ago, {len(ckpts)} kept)")
    lines.append(
        f"updated    {rows[-1]['time']} (now {datetime.now(UTC).isoformat(timespec='seconds')})"
    )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("run_dir", type=Path, nargs="?")
    args = ap.parse_args()
    run_dir = args.run_dir
    if run_dir is None:
        runs = [p.parent for p in Path("runs").glob("*/metrics.jsonl")]
        if not runs:
            print("no runs under runs/", file=sys.stderr)
            return 1
        run_dir = max(runs, key=lambda p: (p / "metrics.jsonl").stat().st_mtime)
    print(status(run_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
