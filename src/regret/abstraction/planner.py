"""Memory planner: how big is the blueprint for a config, before training it?

Walks the abstract betting tree (action abstraction only, no cards) and counts decision nodes
and action slots per street; street-start states with identical chips and statuses share one
subtree, so the walk stays fast even when the tree has billions of paths. Each node is an
infoset per card bucket of that street, so:

    infosets(street) = nodes(street) * buckets(street)
    regret bytes     = Σ action slots * buckets * 2 floats (regret + strategy sum) * 4 bytes
    table bytes      = infosets / 0.9 load * (8 key + 8 pointer + 1 action count)

This is a worst case: pruning and unreachable lines mean training touches fewer infosets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from regret import _core
from regret.abstraction.actions import rules_from_config
from regret.engine.state import CHIPS_PER_BB
from regret.utils.config import RegretConfig

STREETS = ("preflop", "flop", "turn", "river")
_TABLE_BYTES_PER_SLOT = 17
_LOAD = 0.9


@dataclass
class Plan:
    nodes: list[int]
    action_slots: list[int]
    buckets: list[int]
    infosets: list[int] = field(default_factory=list)
    regret_bytes: int = 0
    table_bytes: int = 0
    abstraction_bytes: int = 0
    budget_bytes: int = 0

    @property
    def total_infosets(self) -> int:
        return sum(self.infosets)

    @property
    def total_bytes(self) -> int:
        return self.regret_bytes + self.table_bytes + self.abstraction_bytes

    @property
    def fits(self) -> bool:
        return self.total_bytes <= self.budget_bytes

    def summary(self) -> dict[str, Any]:
        return {
            "streets": {
                s: {"nodes": n, "buckets": b, "infosets": i}
                for s, n, b, i in zip(STREETS, self.nodes, self.buckets, self.infosets, strict=True)
            },
            "total_infosets": self.total_infosets,
            "gb": {
                "regrets": round(self.regret_bytes / 1e9, 2),
                "hash_table": round(self.table_bytes / 1e9, 2),
                "card_tables": round(self.abstraction_bytes / 1e9, 2),
                "total": round(self.total_bytes / 1e9, 2),
                "budget": round(self.budget_bytes / 1e9, 2),
            },
            "fits": self.fits,
            "suggested_max_infosets": self.total_infosets,
        }


def count_tree(cfg: RegretConfig) -> tuple[list[int], list[int]]:
    rules = rules_from_config(cfg.actions)
    t = cfg.table
    stacks = [round(t.starting_stack_bb * CHIPS_PER_BB)] * t.num_players
    root = _core.HandState(stacks, round(t.small_blind_bb * CHIPS_PER_BB), CHIPS_PER_BB)
    memo: dict[tuple[Any, ...], tuple[list[int], list[int]]] = {}

    def walk(state: _core.HandState, street_start: bool) -> tuple[list[int], list[int]]:
        if state.is_terminal:
            return [0] * 4, [0] * 4
        key: tuple[Any, ...] | None = None
        if street_start:
            key = (
                int(state.street),
                tuple(state.contributed),
                tuple(int(s) for s in state.statuses),
            )
            if key in memo:
                return memo[key]
        nodes, slots = [0] * 4, [0] * 4
        street = int(state.street)
        actions = _core.abstract_actions(state, rules)
        nodes[street] += 1
        slots[street] += len(actions)
        for a in actions:
            child = state.copy()
            child.apply(a)
            cn, cs = walk(child, int(child.street) != street)
            for i in range(4):
                nodes[i] += cn[i]
                slots[i] += cs[i]
        if key is not None:
            memo[key] = (nodes, slots)
        return nodes, slots

    return walk(root, True)


def plan(cfg: RegretConfig) -> Plan:
    nodes, slots = count_tree(cfg)
    c = cfg.cards
    buckets = [169, c.flop_buckets, c.turn_buckets, c.river_buckets]
    p = Plan(nodes, slots, buckets)
    p.infosets = [n * b for n, b in zip(nodes, buckets, strict=True)]
    p.regret_bytes = sum(s * b for s, b in zip(slots, buckets, strict=True)) * 2 * 4
    p.table_bytes = int(p.total_infosets / _LOAD * _TABLE_BYTES_PER_SLOT)
    # uint16 bucket tables for flop, turn and river held in memory while training.
    p.abstraction_bytes = (1_286_792 + 13_960_050 + 123_156_254) * 2
    p.budget_bytes = int(cfg.training.memory_budget_gb * 1e9)
    return p
