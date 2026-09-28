"""Train a blueprint with MCCFR.

    uv run python scripts/train.py configs/leduc.yaml                  # new run in runs/leduc
    uv run python scripts/train.py configs/leduc.yaml --resume         # continue it
    uv run python scripts/train.py configs/leduc.yaml --run-dir runs/x --iterations 1000000

Ctrl+C or SIGTERM: finish the current chunk (< 1 s), checkpoint, exit 0.
Check progress from anywhere with `uv run python scripts/status.py runs/leduc`.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from regret.cfr.trainer import RunSpec, Trainer, TrainingError
from regret.utils.config import load_train_config


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("config", type=Path)
    ap.add_argument("--run-dir", type=Path, help="default: training.run_dir or runs/<name>")
    ap.add_argument("--resume", action="store_true", help="continue from the latest checkpoint")
    ap.add_argument("--iterations", type=int, help="stop after this many more iterations")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stderr)
    try:
        spec = RunSpec.from_config(load_train_config(args.config))
        reason = Trainer(spec, args.run_dir, resume=args.resume).run(args.iterations)
    except TrainingError as e:
        logging.error("%s", e)
        return 2
    except KeyboardInterrupt:
        logging.error("aborted; the last checkpoint is intact")
        return 130
    logging.info("done (%s)", reason)
    return 0


if __name__ == "__main__":
    sys.exit(main())
