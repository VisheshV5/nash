"""Evaluate a bundle: duplicate matches vs the baselines (and optionally another bundle), and LBR.

uv run python scripts/evaluate.py bundles/hu-100bb --hands 20000
uv run python scripts/evaluate.py bundles/hu-100bb --vs bundles/older --hands 20000
uv run python scripts/evaluate.py bundles/hu-100bb --lbr 2000 --no-baselines
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from regret.abstraction.actions import rules_from_config
from regret.abstraction.cards import CardAbstraction
from regret.eval.lbr import LBRPlayer
from regret.eval.match import duplicate_match
from regret.eval.setup import BASELINES, blueprint_player, table_chips
from regret.export.bundle import load_bundle


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("bundle", type=Path)
    ap.add_argument("--hands", type=int, default=10_000, help="per matchup (duplicate pairs x 2)")
    ap.add_argument("--vs", type=Path, action="append", default=[], help="another bundle")
    ap.add_argument("--lbr", type=int, default=0, help="LBR hands (0 = skip)")
    ap.add_argument("--no-baselines", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg, strategy, cards_dir = load_bundle(args.bundle)
    cards = CardAbstraction.load(cards_dir)
    stacks, sb, bb = table_chips(cfg)
    me = blueprint_player(cfg, strategy, cards, name=args.bundle.name)
    results: dict[str, Any] = {}
    pairs = max(1, args.hands // 2)

    opponents: list[Any] = [] if args.no_baselines else [cls() for cls in BASELINES.values()]
    for other in args.vs:
        ocfg, ostrat, ocards = load_bundle(other)
        opponents.append(
            blueprint_player(ocfg, ostrat, CardAbstraction.load(ocards), name=other.name)
        )
    for opp in opponents:
        t = time.time()
        r = duplicate_match(me, opp, pairs, stacks, sb, bb, seed=args.seed)
        print(f"{r}  [{time.time() - t:.0f}s]", file=sys.stderr)
        results[f"vs {opp.name}"] = r.summary()

    if args.lbr:
        t = time.time()
        lbr = LBRPlayer(strategy, cards, rules_from_config(cfg.actions), stacks, sb, bb)
        r = duplicate_match(lbr, me, max(1, args.lbr // 2), stacks, sb, bb, seed=args.seed)
        print(f"LBR: {r}  [{time.time() - t:.0f}s]", file=sys.stderr)
        results["lbr"] = r.summary() | {"note": "LBR's win rate: a lower bound on exploitability"}
    results["blueprint"] = {"unseen_infosets": me.unseen, "fallbacks": me.fallbacks}
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
