"""Worst-case blueprint size for a hold'em config, checked against training.memory_budget_gb.

    uv run python scripts/plan_memory.py configs/hu_default.yaml

Exit code 0 if it fits, 1 if it doesn't.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from regret.abstraction.planner import plan
from regret.utils.config import load_config


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("config", type=Path)
    args = ap.parse_args()
    p = plan(load_config(args.config))
    print(json.dumps(p.summary(), indent=2))
    return 0 if p.fits else 1


if __name__ == "__main__":
    sys.exit(main())
