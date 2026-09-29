"""Local best response (Lisý & Bowling 2017): a lower bound on how exploitable a blueprint is.

LBR plays against the blueprint knowing its strategy exactly. It tracks the blueprint's range
(1,326 hole-card weights, updated by Bayes' rule with the blueprint's action probabilities) and
at each of its own decisions picks the action with the best *myopic* expected value, assuming
the hand is then checked down (or, after a bet, that the blueprint folds or calls):

    EV(fold) = 0
    EV(call) = eq * pot - (1 - eq) * to_call
    EV(bet)  = p_fold * pot
             + (1 - p_fold) * (eq_call * (pot + their_call) - (1 - eq_call) * our_add)

where eq is showdown equity against the range, p_fold the range-weighted chance the blueprint
folds to the bet, and eq_call equity against the part of the range that doesn't fold.

Two standard restrictions keep it tractable and exact:
- preflop, LBR only checks/calls;
- LBR bets only the blueprint's own abstract sizes (and all-in), so the blueprint maps them
  back exactly and LBR always knows which abstract infoset the blueprint is in.

Whatever LBR wins (mbb/hand) is a lower bound on the blueprint's exploitability.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from regret import _core
from regret.abstraction.cards import CardAbstraction
from regret.engine.cards import indexer
from regret.eval.players import AbstractTracker, _Base
from regret.eval.strategy import StrategyTable

Action = _core.Action
_PAIRS = np.array([(a, b) for b in range(52) for a in range(b)], np.uint8)


def board_buckets(cards: CardAbstraction, board: list[int]) -> NDArray[np.int64]:
    """Bucket of every hole index on `board` (-1 where the hole overlaps the board)."""
    out = np.full(len(_PAIRS), -1, np.int64)
    ok = ~np.isin(_PAIRS, board).any(1)
    rows = np.concatenate([_PAIRS[ok], np.tile(np.array(board, np.uint8), (ok.sum(), 1))], 1)
    street = {0: "preflop", 3: "flop", 4: "turn", 5: "river"}[len(board)]
    idx = indexer(street).index_batch(rows)
    if street == "preflop":
        out[ok] = idx
    else:
        table = {"flop": cards.flop_table, "turn": cards.turn_table}.get(street)
        if table is None:  # river: per hand from centroids (tables may not be loaded)
            if cards.river_table is not None:
                out[ok] = cards.river_table[idx]
            else:
                out[ok] = [cards.bucket(list(r[:2]), board) for r in rows]
        else:
            out[ok] = table[idx]
    return out


class LBRPlayer(_Base):
    name = "lbr"

    def __init__(
        self,
        strategy: StrategyTable,
        cards: CardAbstraction,
        rules: _core.ActionRules,
        stacks: list[int],
        sb: int,
        bb: int,
        flop_samples: int = 0,
    ) -> None:
        self.strategy, self.cards = strategy, cards
        self.tracker = AbstractTracker(rules, stacks, sb, bb)
        self.flop_samples = flop_samples
        self._buckets: tuple[int, NDArray[np.int64]] | None = None

    def new_hand(self, seat: int, hole: list[int], rng: np.random.Generator) -> None:
        super().new_hand(seat, hole, rng)
        self.tracker.reset()
        self.range = np.ones(len(_PAIRS), np.float64)
        self.range[np.isin(_PAIRS, hole).any(1)] = 0
        self._buckets = None

    # -- the blueprint's policy for every hand in its range

    def _buckets_on(self, board: list[int]) -> NDArray[np.int64]:
        if self._buckets is None or self._buckets[0] != len(board):
            self._buckets = (len(board), board_buckets(self.cards, board))
        return self._buckets[1]

    def _policy_matrix(self, history: int, n_actions: int, board: list[int]) -> NDArray[np.float64]:
        """Rows: hole index; columns: the blueprint's probability of each abstract action."""
        buckets = self._buckets_on(board)
        out = np.full((len(_PAIRS), n_actions), 1.0 / n_actions)
        for b in np.unique(buckets[buckets >= 0]):
            p = self.strategy.probs(_core.nlhe_infoset_key(history, int(b)))
            if p is not None and len(p) == n_actions:
                out[buckets == b] = p
        return out

    def observe(self, state: _core.HandState, seat: int, action: Action, board: list[int]) -> None:
        t = self.tracker
        if not t.in_sync(state):
            t.lost = True
            return
        dist = t.translate(state, action)
        index = max(dist, key=lambda d: d[1])[0]
        if seat != self.seat:  # Bayes update of the blueprint's range
            probs = self._policy_matrix(t.history, len(t.actions()), board)[:, index]
            self.range *= probs
            blocked = np.isin(_PAIRS, board).any(1)
            self.range[blocked] = 0
        t.apply_index(index)

    # -- LBR's own decision

    def _equity(self, weights: NDArray[np.float64], board: list[int]) -> float:
        w = weights.astype(np.float32)
        if w.sum() <= 0:
            return 0.5
        samples = self.flop_samples if len(board) == 3 and self.flop_samples else 300
        seed = int(self.rng.integers(2**63))
        return _core.equity_vs_range(self.hole, board, w, samples, seed)

    def act(self, state: _core.HandState, board: list[int]) -> Action:
        t = self.tracker
        la = state.legal_actions()
        if state.street == _core.Street.PREFLOP or not t.in_sync(state):
            return Action.check_call()
        weights = self.range.copy()
        weights[np.isin(_PAIRS, board).any(1)] = 0
        pot = state.pot
        eq = self._equity(weights, board)
        best_ev = eq * pot - (1 - eq) * la.call_amount
        best = Action.check_call()
        if la.fold and best_ev < 0:
            best_ev, best = 0.0, Action.fold()

        acts = t.actions()
        for i, a in enumerate(acts):
            if a.type != _core.ActionType.BET_RAISE:
                continue
            real = t.to_real(state, i)
            if real.type != _core.ActionType.BET_RAISE:
                continue
            # The blueprint's response to this (exactly translated) bet.
            child = t.state.copy()
            child.apply(a)
            if child.is_terminal:
                continue
            child_acts = _core.abstract_actions(child, t.rules)
            policy = self._policy_matrix(
                _core.nlhe_extend_history(t.history, i), len(child_acts), board
            )
            fold_col = [j for j, c in enumerate(child_acts) if c.type == _core.ActionType.FOLD]
            p_fold_h = policy[:, fold_col[0]] if fold_col else np.zeros(len(_PAIRS))
            total = weights.sum()
            p_fold = float((weights * p_fold_h).sum() / total) if total > 0 else 0.0
            calling = weights * (1 - p_fold_h)
            eq_call = self._equity(calling, board) if calling.sum() > 0 else 0.5
            our_add = real.amount - state.bets[self.seat]
            their_call = real.amount - state.current_bet
            ev = p_fold * pot + (1 - p_fold) * (
                eq_call * (pot + their_call) - (1 - eq_call) * our_add
            )
            if ev > best_ev:
                best_ev, best = ev, real
        return best
