"""Pack a bundle for a GitHub release and print its registry entry.

    uv run python scripts/package_bundle.py bundles/hu-100bb --version v1.0.0

Upload the archive to the release, then add the printed entry to src/regret/registry.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from regret.registry import pack

REPO = "https://github.com/VisheshV5/nash"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("bundle", type=Path)
    ap.add_argument("--version", required=True, help="release tag, e.g. v1.0.0")
    ap.add_argument("--out", type=Path, default=Path("dist"))
    args = ap.parse_args()
    archive = args.out / f"{args.bundle.name}.tar.gz"
    url = f"{REPO}/releases/download/{args.version}/{archive.name}"
    entry = pack(args.bundle, archive, url)
    print(f"wrote {archive} ({entry['bytes'] / 2**20:.1f} MB)", file=sys.stderr)
    print(json.dumps({args.bundle.name: entry}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
