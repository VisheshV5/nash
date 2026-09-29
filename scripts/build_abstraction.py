"""Build the card abstraction (bucket centroids + lookup tables) for a hold'em config.

    uv run python scripts/build_abstraction.py configs/hu_default.yaml
    uv run python scripts/build_abstraction.py configs/hu_default.yaml --out artifacts/abstraction/x

Output goes to artifacts/abstraction/<hash of the `cards` config>/ unless --out is given; an
existing complete build is reused unless --force.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from regret.abstraction.build import artifact_dir, build_card_abstraction
from regret.utils.config import load_config


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("config", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--threads", type=int, help="default: training.threads")
    ap.add_argument("--force", action="store_true", help="rebuild even if the output exists")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stderr)
    cfg = load_config(args.config)
    out = args.out or artifact_dir(cfg.cards)
    if (out / "manifest.json").exists() and not args.force:
        logging.info("%s already built (use --force to rebuild)", out)
        return 0
    manifest = build_card_abstraction(cfg.cards, out, threads=args.threads or cfg.training.threads)
    logging.info("timings: %s", manifest["timings_s"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
