"""MCCFR convergence on toy games, and the long-running training guarantees (SPEC §6.1)."""

from __future__ import annotations

import json
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest
import yaml

from regret import _core
from regret.cfr.checkpoint import (
    CheckpointError,
    checkpoint_path,
    latest_checkpoint,
    list_checkpoints,
    load_checkpoint,
    save_checkpoint,
)
from regret.cfr.trainer import RunSpec, Trainer, TrainingError
from regret.utils.config import CfrConfig, EvalConfig, ToyGameConfig, TrainingConfig

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------- convergence


def test_kuhn_converges_to_the_known_value() -> None:
    s = _core.CfrSolver("kuhn", seed=3, lcfr_until=2_000_000, discount_interval=20_000)
    s.run(20_000_000)
    r = s.nash_conv()
    assert r.nash_conv / 2 < 1e-3  # exploitability
    assert r.on_policy[0] == pytest.approx(-1 / 18, abs=1e-3)
    assert sum(r.on_policy) == pytest.approx(0, abs=1e-9)


def test_leduc_converges() -> None:
    s = _core.CfrSolver("leduc", seed=3, lcfr_until=1_000_000, discount_interval=10_000)
    history = []
    for _ in range(3):
        s.run(4_000_000)
        history.append(s.nash_conv().nash_conv / 2)
    assert history[-1] < 0.01
    assert history[-1] < history[0]


def test_three_player_kuhn_nash_conv_shrinks() -> None:
    s = _core.CfrSolver("kuhn3", seed=3, lcfr_until=2_000_000, discount_interval=20_000)
    s.run(1_000_000)
    early = s.nash_conv().nash_conv
    s.run(19_000_000)
    late = s.nash_conv().nash_conv
    assert late < early
    assert late < 0.005


def test_multithreaded_solver_converges_too() -> None:
    s = _core.CfrSolver("leduc", seed=3, threads=4, lcfr_until=1_000_000, discount_interval=10_000)
    s.run(8_000_000)
    assert s.nash_conv().nash_conv / 2 < 0.01


def test_unknown_game_and_bad_params() -> None:
    with pytest.raises(ValueError, match="unknown game"):
        _core.CfrSolver("chess")
    with pytest.raises(ValueError, match="threads"):
        _core.CfrSolver("kuhn", threads=0)


def test_table_capacity_is_enforced() -> None:
    s = _core.CfrSolver("leduc", max_infosets=100)
    with pytest.raises(RuntimeError, match="table is full"):
        s.run(100_000)


# ---------------------------------------------------------------- determinism


def test_resumed_solver_is_bit_identical() -> None:
    kwargs: dict[str, Any] = {
        "seed": 9,
        "lcfr_until": 30_000,
        "discount_interval": 1_000,
        "prune_after": 20_000,
        "prune_threshold": -5.0,
    }
    straight = _core.CfrSolver("leduc", **kwargs)
    straight.run(50_000)

    first = _core.CfrSolver("leduc", **kwargs)
    first.run(21_337)  # stop between discount boundaries
    second = _core.CfrSolver("leduc", **kwargs)
    second.load(first.save())
    second.run(50_000 - 21_337)

    assert second.iteration == straight.iteration == 50_000
    assert second.save() == straight.save()


def test_different_seeds_differ() -> None:
    a, b = _core.CfrSolver("kuhn", seed=1), _core.CfrSolver("kuhn", seed=2)
    a.run(10_000)
    b.run(10_000)
    assert a.save() != b.save()


def test_load_rejects_garbage() -> None:
    with pytest.raises(ValueError, match="not a solver checkpoint"):
        _core.CfrSolver("kuhn").load(b"nope")


# ---------------------------------------------------------------- checkpoint files


def test_checkpoint_round_trip_and_keep_last(tmp_path: Path) -> None:
    for it in (10, 20, 30):
        save_checkpoint(tmp_path, it, f"payload {it}".encode(), {"elapsed_s": it}, keep_last=2)
    assert [p.name for p in list_checkpoints(tmp_path)] == [
        "ckpt-000000000020.rgc",
        "ckpt-000000000030.rgc",
    ]
    ckpt = latest_checkpoint(tmp_path)
    assert ckpt is not None
    assert (ckpt.iteration, ckpt.payload, ckpt.header["elapsed_s"]) == (30, b"payload 30", 30)


def test_damaged_checkpoints_are_skipped(tmp_path: Path) -> None:
    save_checkpoint(tmp_path, 1, b"good" * 100, {}, keep_last=5)
    p2 = save_checkpoint(tmp_path, 2, b"newer" * 100, {}, keep_last=5)
    p2.write_bytes(p2.read_bytes()[:-10])  # truncated, e.g. disk full
    with pytest.raises(CheckpointError):
        load_checkpoint(p2)
    ckpt = latest_checkpoint(tmp_path)
    assert ckpt is not None
    assert ckpt.iteration == 1


def test_failed_write_leaves_previous_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    save_checkpoint(tmp_path, 1, b"good", {}, keep_last=2)

    def boom(*args: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError, match="disk full"):
        save_checkpoint(tmp_path, 2, b"never lands", {}, keep_last=2)
    assert not checkpoint_path(tmp_path, 2).exists()
    ckpt = latest_checkpoint(tmp_path)
    assert ckpt is not None
    assert ckpt.payload == b"good"


# ---------------------------------------------------------------- trainer


def _spec(target: int, **training: Any) -> RunSpec:
    cfg = ToyGameConfig(
        name="leduc-test",
        game="leduc",
        training=TrainingConfig(
            **{
                "threads": 1,
                "deterministic": True,
                "checkpoint_every_iterations": 20_000,
                "log_every_seconds": 0.2,
                "keep_last_checkpoints": 3,
            }
            | training
        ),
        cfr=CfrConfig(
            lcfr_until_iteration=40_000,
            discount_interval=1_000,
            prune_after_iteration=30_000,
            prune_threshold=-5.0,
            max_infosets=4096,
            target_iterations=target,
        ),
        eval=EvalConfig(every_iterations=40_000, every_minutes=None),
    )
    return RunSpec.from_config(cfg)


def _final_payload(run_dir: Path) -> bytes:
    ckpt = latest_checkpoint(run_dir / "checkpoints")
    assert ckpt is not None
    return ckpt.payload


def test_trainer_resume_matches_uninterrupted_run(tmp_path: Path) -> None:
    assert Trainer(_spec(80_000), tmp_path / "a").run() == "target"

    assert Trainer(_spec(80_000), tmp_path / "b").run(max_iterations=33_333) == "max_iterations"
    assert Trainer(_spec(80_000), tmp_path / "b", resume=True).run() == "target"

    assert _final_payload(tmp_path / "a") == _final_payload(tmp_path / "b")
    events = [json.loads(line)["event"] for line in (tmp_path / "b" / "metrics.jsonl").open()]
    assert events.count("start") == 2
    assert "eval" in events


def test_trainer_refuses_to_clobber_or_mix_runs(tmp_path: Path) -> None:
    Trainer(_spec(20_000), tmp_path).run()
    with pytest.raises(TrainingError, match="--resume"):
        Trainer(_spec(20_000), tmp_path)
    other_seed = _spec(20_000, seed=5, deterministic=True)
    with pytest.raises(TrainingError, match="different game/CFR/seed"):
        Trainer(other_seed, tmp_path, resume=True)
    # Extending the target is fine: it isn't part of the fingerprint.
    assert Trainer(_spec(40_000), tmp_path, resume=True).run() == "target"


# ---------------------------------------------------------------- processes and signals


def _write_config(tmp_path: Path, checkpoint_every: int) -> Path:
    cfg = {
        "name": "leduc-proc",
        "game": "leduc",
        "training": {
            "threads": 1,
            "deterministic": True,
            "log_every_seconds": 0.2,
            "checkpoint_every_iterations": checkpoint_every,
            "keep_last_checkpoints": 2,
        },
        "cfr": {
            "lcfr_until_iteration": 1_000_000,
            "discount_interval": 10_000,
            "max_infosets": 4096,
        },
        "eval": {"every_minutes": None},
    }
    path = tmp_path / "leduc.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def _start(config: Path, run_dir: Path, *extra: str) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [
            sys.executable,
            str(ROOT / "scripts" / "train.py"),
            str(config),
            "--run-dir",
            str(run_dir),
            *extra,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


def _wait_for(pred: Any, timeout: float = 30) -> None:
    deadline = time.time() + timeout
    while not pred():
        assert time.time() < deadline, "timed out"
        time.sleep(0.05)


def test_sigterm_checkpoints_and_exits_cleanly(tmp_path: Path) -> None:
    config, run_dir = _write_config(tmp_path, 10**9), tmp_path / "run"
    proc = _start(config, run_dir)
    metrics = run_dir / "metrics.jsonl"
    _wait_for(lambda: metrics.exists() and '"progress"' in metrics.read_text())
    proc.send_signal(signal.SIGTERM)
    assert proc.wait(timeout=20) == 0, proc.stderr.read() if proc.stderr else ""

    end = json.loads(metrics.read_text().splitlines()[-1])
    assert (end["event"], end["reason"]) == ("end", "stopped")
    ckpt = latest_checkpoint(run_dir / "checkpoints")
    assert ckpt is not None
    assert ckpt.iteration == end["iteration"] > 0
    assert not (run_dir / "train.pid").exists()

    out = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "status.py"), str(run_dir)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "STOPPED (stopped)" in out
    assert f"{ckpt.iteration:,}" in out


def test_kill_9_never_corrupts_the_latest_checkpoint(tmp_path: Path) -> None:
    # Checkpoint every 100k iterations: several writes per second, so kills land mid-write.
    config, run_dir = _write_config(tmp_path, 100_000), tmp_path / "run"
    rng = random.Random(0)
    ckpts = run_dir / "checkpoints"
    for attempt in range(4):
        proc = _start(config, run_dir, *(["--resume"] if attempt else []))
        _wait_for(lambda: len(list_checkpoints(ckpts)) >= 1)
        time.sleep(rng.uniform(0.05, 0.6))
        proc.kill()
        proc.wait()
        ckpt = latest_checkpoint(ckpts)
        assert ckpt is not None
        for path in list_checkpoints(ckpts):
            load_checkpoint(path)  # every surviving checkpoint verifies

    # And the run resumes from it.
    before = latest_checkpoint(run_dir / "checkpoints")
    assert before is not None
    done = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "train.py"),
            str(config),
            "--run-dir",
            str(run_dir),
            "--resume",
            "--iterations",
            "50000",
        ],
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr
    after = latest_checkpoint(run_dir / "checkpoints")
    assert after is not None
    assert after.iteration == before.iteration + 50_000
