"""Action abstraction: config -> C++ rules, plus a readable view of the abstract actions."""

from __future__ import annotations

from regret import _core
from regret.engine.state import CHIPS_PER_BB
from regret.utils.config import ActionAbstractionConfig


def rules_from_config(
    cfg: ActionAbstractionConfig, chips_per_bb: int = CHIPS_PER_BB
) -> _core.ActionRules:
    rules = _core.ActionRules()
    rules.chips_per_bb = chips_per_bb
    pre = _core.PreflopSizing()
    pre.open_bb = list(cfg.preflop.open_bb)
    pre.reraise_x_ip = list(cfg.preflop.reraise_x_ip)
    pre.reraise_x_oop = list(cfg.preflop.reraise_x_oop)
    pre.four_bet_x = list(cfg.preflop.four_bet_x)
    pre.all_in = cfg.preflop.allow_all_in
    pre.max_raises = cfg.preflop.max_raises
    rules.preflop = pre
    post = []
    for players, sizing in sorted(cfg.postflop.items()):
        p = _core.PostflopSizing()
        p.bet_pot = list(sizing.bet_pot)
        p.raise_pot = list(sizing.raise_pot)
        p.all_in = sizing.allow_all_in
        p.max_raises = sizing.max_raises
        post.append((players, p))
    rules.postflop = post
    return rules


def action_label(
    state: _core.HandState, action: _core.Action, chips_per_bb: int = CHIPS_PER_BB
) -> str:
    """Stable, human-readable label, e.g. fold, check, call, bet_0.75pot, raise_3x, all_in."""
    if action.type == _core.ActionType.FOLD:
        return "fold"
    if action.type == _core.ActionType.CHECK_CALL:
        return "call" if state.legal_actions().call_amount else "check"
    la = state.legal_actions()
    if action.amount == la.max_raise_to:
        return "all_in"
    cur = state.current_bet
    if cur == 0:
        return f"bet_{_core.pot_fraction(state, action.amount):.2f}pot"
    if state.street == _core.Street.PREFLOP and all(
        e[2].type != _core.ActionType.BET_RAISE
        for e in state.history
        if e[0] == _core.Street.PREFLOP
    ):
        return f"raise_{action.amount / chips_per_bb:g}bb"
    return f"raise_{action.amount / cur:.2g}x"
