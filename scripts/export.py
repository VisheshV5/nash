"""Export a checkpoint (default: the run's latest) into an agent bundle.

uv run python scripts/export.py configs/hu_default.yaml runs/hu-100bb bundles/hu-100bb
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from regret.cfr.checkpoint import latest_checkpoint, load_checkpoint
from regret.export.bundle import export_bundle
from regret.utils.config import load_config


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("config", type=Path)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--checkpoint", type=Path, help="a specific checkpoint file")
    args = ap.parse_args()
    ckpt = (
        load_checkpoint(args.checkpoint)
        if args.checkpoint
        else latest_checkpoint(args.run_dir / "checkpoints")
    )
    if ckpt is None:
        print(f"no checkpoint in {args.run_dir}", file=sys.stderr)
        return 1
    print(json.dumps(export_bundle(load_config(args.config), ckpt, args.out), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
