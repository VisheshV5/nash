"""Differential fuzzing of regret's betting engine against PokerKit.

Random 2-6 player hands (stacks from sub-blind to 300bb, so plenty of all-ins and side pots)
are played through both engines with random legal actions. At every decision the actor,
street, bets, stacks and full legal action set (fold / call amount / min and max raise-to)
must match, and the final stacks must match after showdown.

pytest runs a small batch; for the full run:

    uv run python tests/fuzz_pokerkit.py --hands 1000000 --workers 6
"""

from __future__ import annotations

import argparse
import random
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

from pokerkit import Automation, NoLimitTexasHoldem

from regret import _core
from regret.engine.cards import card_str

AUTOMATIONS = (
    Automation.ANTE_POSTING,
    Automation.BET_COLLECTION,
    Automation.BLIND_OR_STRADDLE_POSTING,
    Automation.CARD_BURNING,
    Automation.HOLE_CARDS_SHOWING_OR_MUCKING,
    Automation.HAND_KILLING,
    Automation.CHIPS_PUSHING,
    Automation.CHIPS_PULLING,
)


class Mismatch(AssertionError):
    pass


@dataclass
class Deal:
    stacks: list[int]  # by our seat
    small_blind: int
    big_blind: int
    hole: list[list[int]]  # by our seat
    board: list[int]


def random_deal(rng: random.Random) -> Deal:
    n = rng.randint(2, 6)
    bb = rng.choice([2, 3, 10, 100, rng.randint(2, 200)])
    sb = rng.randint(1, bb - 1)
    stacks = []
    for _ in range(n):
        r = rng.random()
        if r < 0.2:
            stacks.append(rng.randint(1, 2 * bb))
        elif r < 0.5:
            stacks.append(rng.randint(bb, 20 * bb))
        else:
            stacks.append(rng.randint(20 * bb, 300 * bb))
    deck = list(range(52))
    rng.shuffle(deck)
    hole = [deck[2 * i : 2 * i + 2] for i in range(n)]
    return Deal(stacks, sb, bb, hole, deck[2 * n : 2 * n + 5])


def pk_index(seat: int, n: int) -> int:
    """PokerKit puts the big blind first heads-up; otherwise seats line up."""
    return 1 - seat if n == 2 else seat


def play_hand(deal: Deal, rng: random.Random) -> int:
    """Play one hand through both engines. Returns the number of decisions compared."""
    n = len(deal.stacks)
    to_pk = [pk_index(s, n) for s in range(n)]
    to_ours = {pk: seat for seat, pk in enumerate(to_pk)}

    ours = _core.HandState(deal.stacks, deal.small_blind, deal.big_blind)
    pk_stacks = [0] * n
    for seat in range(n):
        pk_stacks[to_pk[seat]] = deal.stacks[seat]
    pk = NoLimitTexasHoldem.create_state(
        AUTOMATIONS,
        True,
        0,
        (deal.small_blind, deal.big_blind),
        deal.big_blind,
        tuple(pk_stacks),
        n,
    )

    board_dealt = 0
    decisions = 0

    def check(cond: bool, what: str) -> None:
        if not cond:
            raise Mismatch(f"{what}\n  deal={deal}\n  history={ours.history}")

    while pk.status:
        if pk.can_deal_hole():
            dealee = pk.hole_dealee_index
            assert dealee is not None
            seat = to_ours[dealee]
            pk.deal_hole("".join(card_str(c) for c in deal.hole[seat]))
            continue
        if pk.can_deal_board():
            k = 3 if board_dealt == 0 else 1
            pk.deal_board("".join(card_str(c) for c in deal.board[board_dealt : board_dealt + k]))
            board_dealt += k
            continue
        actor = pk.actor_index
        check(actor is not None, "PokerKit is stuck with no actor")
        assert actor is not None
        seat = to_ours[actor]
        decisions += 1

        check(not ours.is_terminal, "ours ended the hand early")
        check(ours.to_act == seat, f"actor: ours {ours.to_act}, PokerKit seat {seat}")
        check(
            int(ours.street) == pk.street_index, f"street: ours {ours.street}, pk {pk.street_index}"
        )
        for s in range(n):
            check(ours.stacks[s] == pk.stacks[to_pk[s]], f"stack of seat {s}")
            check(ours.bets[s] == pk.bets[to_pk[s]], f"bet of seat {s}")

        la = ours.legal_actions()
        facing = pk.bets[actor] < max(pk.bets)
        check(la.fold == facing, "fold legality")
        check(la.call_amount == pk.checking_or_calling_amount, "call amount")
        pk_can_raise = pk.can_complete_bet_or_raise_to()
        check(la.bet_raise == pk_can_raise, f"raise legality: ours {la.bet_raise}")
        if pk_can_raise:
            check(la.min_raise_to == pk.min_completion_betting_or_raising_to_amount, "min raise")
            check(la.max_raise_to == pk.max_completion_betting_or_raising_to_amount, "max raise")

        r = rng.random()
        if la.fold and r < 0.15:
            ours.apply(_core.Action.fold())
            pk.fold()
        elif la.bet_raise and r > 0.6:
            pick = rng.random()
            if pick < 0.3:
                to = la.min_raise_to
            elif pick < 0.6:
                to = la.max_raise_to
            else:
                to = rng.randint(la.min_raise_to, la.max_raise_to)
            ours.apply(_core.Action.bet_raise_to(to))
            pk.complete_bet_or_raise_to(to)
        else:
            ours.apply(_core.Action.check_call())
            pk.check_or_call()

    check(ours.is_terminal, "PokerKit finished but ours didn't")
    check(ours.to_act == -1, "terminal state still has an actor")
    payoffs = ours.payoffs(deal.hole, deal.board)
    for s in range(n):
        final = deal.stacks[s] + payoffs[s]
        check(
            final == pk.stacks[to_pk[s]],
            f"final stack of seat {s}: ours {final}, pk {pk.stacks[to_pk[s]]}",
        )
    check(sum(payoffs) == 0, "payoffs don't sum to zero")
    return decisions


def run(seed: int, hands: int) -> tuple[int, int]:
    rng = random.Random(seed)
    decisions = 0
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # e.g. PokerKit burning chips would be a bug on our side
        # PokerKit burns a random card before each street; it may pick one we then deal.
        warnings.filterwarnings("ignore", message="A card being dealt")
        for _ in range(hands):
            decisions += play_hand(random_deal(rng), rng)
    return hands, decisions


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--hands", type=int, default=100_000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    chunk = 10_000
    jobs = [
        (args.seed * 1_000_003 + i, min(chunk, args.hands - i * chunk))
        for i in range((args.hands + chunk - 1) // chunk)
    ]
    t0 = time.time()
    total_hands = total_decisions = 0
    with ProcessPoolExecutor(args.workers) as pool:
        for h, d in pool.map(run, *zip(*jobs, strict=True)):
            total_hands += h
            total_decisions += d
            print(
                f"{total_hands:>9,} hands  {total_decisions:>11,} decisions  "
                f"{time.time() - t0:7.1f}s",
                flush=True,
            )
    print(f"OK: {total_hands:,} hands, {total_decisions:,} decisions matched PokerKit")


if __name__ == "__main__":
    main()
