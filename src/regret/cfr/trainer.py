"""Long-running MCCFR training: checkpoints, resume, graceful stop, progress logging (SPEC §6.1).

A run directory holds everything about one training run:

    config.yaml      resolved config the run was started with
    checkpoints/     ckpt-<iteration>.rgc (newest `keep_last_checkpoints` kept)
    metrics.jsonl    one JSON object per progress/eval/checkpoint event (read by status.py)
    train.log        human-readable log
    tb/              TensorBoard scalars
    train.pid        present while a trainer is running

The C++ solver runs in short time-boxed chunks (≤ 1 s), so a stop request (Ctrl+C / SIGTERM)
is noticed within a second; the trainer then writes a checkpoint and exits cleanly. A second
signal aborts immediately (the last checkpoint is still intact).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import signal
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import FrameType
from typing import Any

import psutil
import yaml

from regret import _core
from regret.cfr.checkpoint import (
    latest_checkpoint,
    list_checkpoints,
    remove_stale_temp_files,
    save_checkpoint,
)
from regret.utils.config import (
    CfrConfig,
    EvalConfig,
    RegretConfig,
    ToyGameConfig,
    TrainingConfig,
)
from regret.utils.seeding import derive_seed

log = logging.getLogger("regret.train")


class TrainingError(Exception):
    pass


@dataclass(frozen=True)
class RunSpec:
    """What the trainer needs from a config, whatever the game."""

    name: str
    game: str
    training: TrainingConfig
    cfr: CfrConfig
    eval: EvalConfig
    config: ToyGameConfig | RegretConfig

    @classmethod
    def from_config(cls, cfg: ToyGameConfig | RegretConfig) -> RunSpec:
        if isinstance(cfg, RegretConfig):
            raise TrainingError("hold'em training arrives in M4; train a toy game for now")
        return cls(cfg.name, cfg.game, cfg.training, cfg.cfr, cfg.eval, cfg)

    @property
    def fingerprint(self) -> str:
        """Hash of everything that changes the results. Resuming requires a match.

        Excluded on purpose: checkpoint/log/eval cadence, target iterations (so a run can be
        extended), table capacity, and thread count for non-deterministic runs.
        """
        cfr = self.cfr.model_dump(mode="json", exclude={"target_iterations", "max_infosets"})
        algo = {
            "game": self.game,
            "cfr": cfr,
            "seed": self.training.seed,
            "deterministic": self.training.deterministic,
        }
        return hashlib.sha256(json.dumps(algo, sort_keys=True).encode()).hexdigest()[:16]

    def make_solver(self) -> _core.CfrSolver:
        c = self.cfr
        return _core.CfrSolver(
            self.game,
            seed=derive_seed(self.training.seed, "cfr"),
            threads=self.training.threads,
            lcfr_until=c.lcfr_until_iteration,
            discount_interval=c.discount_interval,
            prune_after=-1 if c.prune_after_iteration is None else c.prune_after_iteration,
            prune_probability=c.prune_probability,
            prune_threshold=c.prune_threshold,
            regret_floor=-1e30 if c.regret_floor is None else c.regret_floor,
            max_infosets=c.max_infosets,
        )


class _StopFlag:
    def __init__(self) -> None:
        self.requested = False

    def handler(self, signum: int, frame: FrameType | None) -> None:
        if self.requested:
            raise KeyboardInterrupt  # second signal: abort now
        self.requested = True
        log.info(
            "%s received: finishing the current chunk, then checkpointing and exiting",
            signal.Signals(signum).name,
        )


@contextmanager
def _signals(flag: _StopFlag) -> Iterator[None]:
    old = {s: signal.signal(s, flag.handler) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        yield
    finally:
        for s, h in old.items():
            signal.signal(s, h)


class Trainer:
    CHUNK_SECONDS = 1.0

    def __init__(self, spec: RunSpec, run_dir: Path | None = None, resume: bool = False) -> None:
        self.spec = spec
        self.run_dir = run_dir or spec.training.run_dir or Path("runs") / spec.name
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.solver = spec.make_solver()
        self.elapsed_before = 0.0
        self.stop = _StopFlag()
        self._tb: Any = None
        self._setup(resume)

    # ------------------------------------------------------------------ setup

    def _setup(self, resume: bool) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(self.run_dir / "train.log")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        log.addHandler(handler)
        log.setLevel(logging.INFO)
        self._log_handler = handler

        remove_stale_temp_files(self.ckpt_dir)
        existing = list_checkpoints(self.ckpt_dir)
        if existing and not resume:
            raise TrainingError(
                f"{self.run_dir} already has checkpoints; pass --resume to continue it or "
                "choose another run directory"
            )
        if resume:
            ckpt = latest_checkpoint(self.ckpt_dir)
            if ckpt is None:
                log.info("nothing to resume in %s; starting a new run", self.ckpt_dir)
            else:
                if ckpt.header.get("fingerprint") != self.spec.fingerprint:
                    raise TrainingError(
                        f"{ckpt.path.name} was made with a different game/CFR/seed config "
                        f"(fingerprint {ckpt.header.get('fingerprint')} vs "
                        f"{self.spec.fingerprint}); refusing to resume"
                    )
                self.solver.load(ckpt.payload)
                self.elapsed_before = float(ckpt.header["elapsed_s"])
                log.info("resumed from %s at iteration %d", ckpt.path.name, ckpt.iteration)

        (self.run_dir / "config.yaml").write_text(
            yaml.safe_dump(self.spec.config.model_dump(mode="json"), sort_keys=False)
        )
        try:
            from tensorboardX import SummaryWriter  # type: ignore[import-untyped]

            self._tb = SummaryWriter(str(self.run_dir / "tb"))
        except ImportError:  # pragma: no cover - optional
            log.warning("tensorboardX not installed; TensorBoard logging is off")

    # ------------------------------------------------------------------ main loop

    def run(self, max_iterations: int | None = None) -> str:
        """Train until the target, `max_iterations` more, or a stop signal. Returns why."""
        s = self.spec
        start_iter = self.solver.iteration
        ends = [
            x
            for x in (
                s.cfr.target_iterations,
                None if max_iterations is None else start_iter + max_iterations,
            )
            if x is not None
        ]
        end = min(ends) if ends else None
        ckpt_every = s.training.checkpoint_every_iterations
        eval_every = s.eval.every_iterations

        t0 = time.monotonic()
        last_ckpt_t = last_eval_t = last_log_t = t0
        last_ckpt_iter = start_iter
        last_log_iter = start_iter
        self._pid_file(True)
        self._event(
            "start",
            iteration=start_iter,
            fingerprint=s.fingerprint,
            threads=s.training.threads,
            target=end,
        )
        log.info(
            "training %s (%s) from iteration %d to %s in %s",
            s.name,
            s.game,
            start_iter,
            end if end is not None else "∞",
            self.run_dir,
        )

        reason = "stopped"
        aborted = False
        try:
            with _signals(self.stop):
                while True:
                    it = self.solver.iteration
                    if end is not None and it >= end:
                        reason = "target" if end == s.cfr.target_iterations else "max_iterations"
                        break
                    if self.stop.requested:
                        reason = "stopped"
                        break
                    n = (end - it) if end is not None else 10**15
                    if ckpt_every:
                        n = min(n, ckpt_every - it % ckpt_every)
                    if eval_every:
                        n = min(n, eval_every - it % eval_every)
                    self.solver.run(n, min(self.CHUNK_SECONDS, s.training.log_every_seconds))

                    now = time.monotonic()
                    it = self.solver.iteration
                    elapsed = self.elapsed_before + now - t0
                    if now - last_log_t >= s.training.log_every_seconds:
                        rate = (it - last_log_iter) / (now - last_log_t)
                        self._progress(it, elapsed, rate, end)
                        last_log_t, last_log_iter = now, it
                    if (eval_every and it % eval_every == 0) or (
                        s.eval.every_minutes and now - last_eval_t >= 60 * s.eval.every_minutes
                    ):
                        self.evaluate()
                        last_eval_t = time.monotonic()
                    if (ckpt_every and it % ckpt_every == 0 and it != last_ckpt_iter) or (
                        now - last_ckpt_t >= 60 * s.training.checkpoint_every_minutes
                    ):
                        self.checkpoint(elapsed)
                        last_ckpt_t, last_ckpt_iter = time.monotonic(), it
        except KeyboardInterrupt:
            aborted, reason = True, "aborted"
            raise
        finally:
            it = self.solver.iteration
            elapsed = self.elapsed_before + time.monotonic() - t0
            if it != last_ckpt_iter and not aborted:
                self.checkpoint(elapsed)
            self._event("end", iteration=it, reason=reason, elapsed_s=round(elapsed, 3))
            log.info("%s at iteration %d after %.1fs of training", reason, it, elapsed)
            self._pid_file(False)
            self.close()
        return reason

    # ------------------------------------------------------------------ pieces

    def checkpoint(self, elapsed: float) -> Path:
        it = self.solver.iteration
        path = save_checkpoint(
            self.ckpt_dir,
            it,
            self.solver.save(),
            {
                "fingerprint": self.spec.fingerprint,
                "config_hash": self.spec.config.config_hash(),
                "game": self.spec.game,
                "elapsed_s": elapsed,
                "created": datetime.now(UTC).isoformat(timespec="seconds"),
            },
            keep_last=self.spec.training.keep_last_checkpoints,
        )
        self._event("checkpoint", iteration=it, path=path.name, elapsed_s=round(elapsed, 3))
        log.info("checkpoint %s", path.name)
        return path

    def evaluate(self) -> dict[str, float]:
        """Exact exploitability (toy games). Two-player: NashConv / 2."""
        r = self.solver.nash_conv()
        players = self.solver.num_players
        out = {
            "nash_conv": r.nash_conv,
            "exploitability": r.nash_conv / 2 if players == 2 else r.nash_conv,
        }
        for p, v in enumerate(r.on_policy):
            out[f"value_p{p}"] = v
        self._event("eval", iteration=self.solver.iteration, **out)
        log.info(
            "eval @ %d: exploitability %.6f (NashConv %.6f)",
            self.solver.iteration,
            out["exploitability"],
            r.nash_conv,
        )
        return out

    def _progress(self, it: int, elapsed: float, rate: float, end: int | None) -> None:
        rss = psutil.Process().memory_info().rss
        eta = (end - it) / rate if end is not None and rate > 0 else None
        self._event(
            "progress",
            iteration=it,
            elapsed_s=round(elapsed, 3),
            it_per_s=round(rate, 1),
            eta_s=None if eta is None else round(eta, 1),
            infosets=self.solver.num_infosets,
            store_mb=round(self.solver.memory_bytes / 2**20, 1),
            rss_mb=round(rss / 2**20, 1),
        )
        log.info(
            "iter %d  %.0f it/s  %d infosets  RSS %.0f MB%s",
            it,
            rate,
            self.solver.num_infosets,
            rss / 2**20,
            f"  ETA {eta / 60:.1f} min" if eta is not None else "",
        )

    def _event(self, event: str, **fields: Any) -> None:
        row = {"event": event, "time": datetime.now(UTC).isoformat(timespec="seconds"), **fields}
        with open(self.run_dir / "metrics.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
        if self._tb is not None and "iteration" in fields:
            for k, v in fields.items():
                if k != "iteration" and isinstance(v, int | float) and not isinstance(v, bool):
                    self._tb.add_scalar(f"{event}/{k}", v, fields["iteration"])

    def _pid_file(self, create: bool) -> None:
        pid = self.run_dir / "train.pid"
        if create:
            pid.write_text(str(os.getpid()))
        else:
            pid.unlink(missing_ok=True)

    def close(self) -> None:
        if self._tb is not None:
            self._tb.close()
            self._tb = None
        log.removeHandler(self._log_handler)
        self._log_handler.close()
