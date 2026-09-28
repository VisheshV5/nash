"""The public GameState input (SPEC §4): schema, semantic validation, and replay.

Validation happens in two layers:

1. **Schema** (pydantic): types, ranges, allowed seat labels and action kinds.
2. **Replay**: the action history is replayed through the C++ engine from the blinds. Every
   action must be legal and in turn, and the reported pot, stacks, statuses, street, board and
   player to act must all agree with the replay. Starting stacks are recovered as
   `reported stack + chips put in` (from a first replay with unlimited stacks), so stacks
   don't need to match any configured depth.

Any problem raises `GameStateError` with one or more `Issue`s that point at the offending field
(e.g. `action_history.flop[1]`).

Amounts are in the same unit as `big_blind` (big blinds when `big_blind` is 1.0). Bet, raise,
call and all-in amounts are the player's **total bet on the street after the action** ("raise
to"); `call` may omit its amount, `bet`, `raise` and `all_in` may not. Internally amounts
become integer chips at 1/100 of a big blind.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, Self

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)

from regret import _core
from regret.engine.cards import cards_str, parse_card

CHIPS_PER_BB = 100
_UNLIMITED = 10**15  # chips; stands in for unknown stacks during the first replay

StreetName = Literal["preflop", "flop", "turn", "river"]
STREETS: tuple[StreetName, ...] = ("preflop", "flop", "turn", "river")
BOARD_SIZE: dict[StreetName, int] = {"preflop": 0, "flop": 3, "turn": 4, "river": 5}

# Seat labels by table size, in engine seat order (SB, BB, ..., BTN). Heads-up, the button
# posts the small blind; "SB" is accepted as an alias for "BTN".
SEAT_LABELS: dict[int, tuple[str, ...]] = {
    2: ("BTN", "BB"),
    3: ("SB", "BB", "BTN"),
    4: ("SB", "BB", "CO", "BTN"),
    5: ("SB", "BB", "HJ", "CO", "BTN"),
    6: ("SB", "BB", "UTG", "HJ", "CO", "BTN"),
}
_HEADS_UP_ALIASES = {"SB": "BTN"}

ActionKind = Literal["fold", "check", "call", "bet", "raise", "all_in"]
PlayerStatusName = Literal["active", "folded", "all_in"]

_STATUS_NAMES: dict[_core.PlayerStatus, PlayerStatusName] = {
    _core.PlayerStatus.ACTIVE: "active",
    _core.PlayerStatus.FOLDED: "folded",
    _core.PlayerStatus.ALL_IN: "all_in",
}


# ---------------------------------------------------------------- errors


@dataclass(frozen=True)
class Issue:
    field: str
    message: str

    def __str__(self) -> str:
        return f"{self.field}: {self.message}"


class GameStateError(ValueError):
    """The input is malformed or describes an impossible hand."""

    def __init__(self, issues: list[Issue]) -> None:
        self.issues = issues
        super().__init__("; ".join(str(i) for i in issues))

    @classmethod
    def one(cls, field: str, message: str) -> GameStateError:
        return cls([Issue(field, message)])

    def to_dict(self) -> dict[str, Any]:
        return {"errors": [{"field": i.field, "message": i.message} for i in self.issues]}


# ---------------------------------------------------------------- schema


def _upper(v: object) -> object:
    return v.upper() if isinstance(v, str) else v


SeatLabel = Annotated[str, BeforeValidator(_upper)]
Amount = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PlayerIn(_Strict):
    seat: SeatLabel
    stack: Amount
    status: PlayerStatusName


class ActionIn(_Strict):
    seat: SeatLabel
    kind: ActionKind
    amount: Amount | None = None

    @model_validator(mode="before")
    @classmethod
    def _from_list(cls, v: object) -> object:
        # Accept the spec's compact form: ["HJ", "raise", 2.5] or ["UTG", "fold"].
        if isinstance(v, list | tuple):
            if len(v) not in (2, 3):
                raise ValueError("an action is [seat, kind] or [seat, kind, amount]")
            return dict(zip(("seat", "kind", "amount"), v, strict=False))
        return v

    @model_validator(mode="after")
    def _amount_needed(self) -> Self:
        if self.kind in ("bet", "raise", "all_in") and self.amount is None:
            raise ValueError(f"'{self.kind}' needs an amount (the total bet after the action)")
        if self.kind in ("fold", "check") and self.amount is not None:
            raise ValueError(f"'{self.kind}' takes no amount")
        return self


class GameStateIn(_Strict):
    num_players: int = Field(ge=2, le=6)
    hero_seat: SeatLabel
    hero_cards: list[str] = Field(min_length=2, max_length=2)
    board: list[str] = Field(default_factory=list, max_length=5)
    street: StreetName
    pot: Amount
    players: list[PlayerIn]
    action_history: dict[StreetName, list[ActionIn]] = Field(default_factory=dict)
    to_act: SeatLabel
    big_blind: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    small_blind: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    """Defaults to half the big blind."""


# ---------------------------------------------------------------- validated result


@dataclass
class ValidatedState:
    """A GameStateIn that replayed cleanly, with the engine state at the decision point."""

    raw: GameStateIn
    engine: _core.HandState
    labels: tuple[str, ...]  # engine seat -> label
    hero: int  # engine seat
    hero_cards: list[int]
    board: list[int]
    chips_per_unit: float  # chips per input unit (CHIPS_PER_BB / big_blind)
    warnings: list[str] = field(default_factory=list)

    def to_units(self, chips: int) -> float:
        return round(chips / self.chips_per_unit, 6)


# ---------------------------------------------------------------- validation


def parse_game_state(data: GameStateIn | dict[str, Any] | str) -> ValidatedState:
    """Validate a GameState (model, dict, or JSON string) and replay it."""
    if isinstance(data, GameStateIn):
        state = data
    else:
        try:
            state = (
                GameStateIn.model_validate_json(data)
                if isinstance(data, str)
                else GameStateIn.model_validate(data)
            )
        except ValidationError as e:
            raise GameStateError(
                [
                    Issue(".".join(str(p) for p in err["loc"]) or "(root)", err["msg"])
                    for err in e.errors()
                ]
            ) from None
    return _Replay(state).run()


class _Replay:
    def __init__(self, state: GameStateIn) -> None:
        self.s = state
        self.n = state.num_players
        self.labels = SEAT_LABELS[self.n]
        self.chips_per_unit = CHIPS_PER_BB / state.big_blind

    # -- helpers

    def seat(self, label: str, where: str) -> int:
        if self.n == 2:
            label = _HEADS_UP_ALIASES.get(label, label)
        if label not in self.labels:
            raise GameStateError.one(
                where, f"unknown seat '{label}' for {self.n} players (use {', '.join(self.labels)})"
            )
        return self.labels.index(label)

    def chips(self, amount: float, where: str) -> int:
        c = amount * self.chips_per_unit
        r = round(c)
        if abs(c - r) > 1e-6 * max(1.0, abs(c)):
            raise GameStateError.one(
                where, f"{amount} is finer than 0.01 big blinds, which isn't supported"
            )
        return int(r)

    def units(self, chips: int) -> float:
        return round(chips / self.chips_per_unit, 6)

    # -- main

    def run(self) -> ValidatedState:
        s = self.s
        hero = self.seat(s.hero_seat, "hero_seat")
        hero_cards, board = self.cards()
        stacks = self.stacks()

        sb = self.chips(
            s.small_blind if s.small_blind is not None else s.big_blind / 2, "small_blind"
        )
        bb = self.chips(s.big_blind, "big_blind")
        if not 0 < sb < bb:
            raise GameStateError.one("small_blind", "must be positive and below the big blind")

        # Pass 1, unlimited stacks: how much did each seat put in? Then starting stack =
        # reported stack + that. Pass 2 replays for real and checks everything.
        probe = _core.HandState([_UNLIMITED] * self.n, sb, bb)
        self.replay(probe, probe=True)
        starting = [stacks[i] + probe.contributed[i] for i in range(self.n)]
        for i, st in enumerate(starting):
            if st <= 0:
                raise GameStateError.one(
                    f"players[{self.labels[i]}].stack", "player has no chips and put none in"
                )

        engine = _core.HandState(starting, sb, bb)
        self.replay(engine)
        self.check_final(engine, stacks, hero)

        return ValidatedState(
            raw=s,
            engine=engine,
            labels=self.labels,
            hero=hero,
            hero_cards=hero_cards,
            board=board,
            chips_per_unit=self.chips_per_unit,
        )

    def cards(self) -> tuple[list[int], list[int]]:
        s = self.s
        ids: list[int] = []
        for name, cards in (("hero_cards", s.hero_cards), ("board", s.board)):
            for i, text in enumerate(cards):
                try:
                    ids.append(parse_card(text))
                except ValueError:
                    raise GameStateError.one(
                        f"{name}[{i}]", f"invalid card '{text}' (expected e.g. 'As', 'Td')"
                    ) from None
        if len(set(ids)) != len(ids):
            dup = next(c for c in ids if ids.count(c) > 1)
            raise GameStateError.one("board", f"card {cards_str([dup])} appears twice")
        want = BOARD_SIZE[s.street]
        if len(s.board) != want:
            raise GameStateError.one(
                "board", f"the {s.street} needs {want} board cards, got {len(s.board)}"
            )
        return ids[:2], ids[2:]

    def stacks(self) -> list[int]:
        s = self.s
        if len(s.players) != self.n:
            raise GameStateError.one(
                "players", f"num_players is {self.n} but {len(s.players)} players are listed"
            )
        stacks: list[int | None] = [None] * self.n
        for p in s.players:
            i = self.seat(p.seat, "players")
            if stacks[i] is not None:
                raise GameStateError.one("players", f"seat '{p.seat}' is listed twice")
            stacks[i] = self.chips(p.stack, f"players[{p.seat}].stack")
        return [x for x in stacks if x is not None]

    def replay(self, engine: _core.HandState, probe: bool = False) -> None:
        s = self.s
        unknown = [k for k in s.action_history if k not in STREETS]
        if unknown:
            raise GameStateError.one("action_history", f"unknown street {unknown[0]}")
        current = STREETS.index(s.street)
        for k in s.action_history:
            if STREETS.index(k) > current and s.action_history[k]:
                raise GameStateError.one(
                    f"action_history.{k}", f"has actions but the hand is on the {s.street}"
                )

        for si, street in enumerate(STREETS[: current + 1]):
            for ai, a in enumerate(s.action_history.get(street, [])):
                where = f"action_history.{street}[{ai}]"
                if engine.is_terminal:
                    raise GameStateError.one(where, "the hand was already over")
                if int(engine.street) != si:
                    raise GameStateError.one(
                        where,
                        f"betting on the {STREETS[int(engine.street)]} isn't finished yet",
                    )
                seat = self.seat(a.seat, where)
                if seat != engine.to_act:
                    raise GameStateError.one(
                        where,
                        f"{a.seat} acted out of turn; {self.labels[engine.to_act]} was to act",
                    )
                engine.apply(self.to_engine(engine, a, where, probe))
            if si < current and not engine.is_terminal and int(engine.street) == si:
                raise GameStateError.one(
                    f"action_history.{street}",
                    f"betting on the {street} isn't finished but the hand is on the {s.street}",
                )

    def to_engine(
        self, engine: _core.HandState, a: ActionIn, where: str, probe: bool
    ) -> _core.Action:
        la = engine.legal_actions()
        seat = engine.to_act
        cur = engine.current_bet
        bet = engine.bets[seat]
        all_in_to = bet + engine.stacks[seat]

        def expect(amount_chips: int) -> None:
            if a.amount is not None and self.chips(a.amount, where) != amount_chips:
                raise GameStateError.one(
                    where,
                    f"{a.kind} amount {a.amount} doesn't match the replay "
                    f"({self.units(amount_chips)}; amounts are the total street bet)",
                )

        match a.kind:
            case "fold":
                if not la.fold:
                    raise GameStateError.one(where, "fold with no bet to face (that's a check)")
                return _core.Action.fold()
            case "check":
                if la.call_amount:
                    raise GameStateError.one(
                        where, f"check facing a bet of {self.units(cur)} (call or fold)"
                    )
                return _core.Action.check_call()
            case "call":
                if not la.call_amount:
                    raise GameStateError.one(where, "call with nothing to call (that's a check)")
                expect(bet + la.call_amount)
                return _core.Action.check_call()
            case "all_in":
                assert a.amount is not None
                if probe:  # stacks are unknown yet: trust the amount
                    to = self.chips(a.amount, where)
                    if to <= cur:
                        return _core.Action.check_call()
                    return self.raise_to(engine, to, where, probe)
                if all_in_to <= cur:  # calling all-in
                    expect(all_in_to)
                    return _core.Action.check_call()
                expect(all_in_to)
                return self.raise_to(engine, all_in_to, where, probe)
            case "bet" | "raise":
                assert a.amount is not None
                if a.kind == "bet" and cur > 0:
                    raise GameStateError.one(
                        where,
                        "'bet' when there's already a bet (use 'raise'; preflop the big blind "
                        "counts as a bet)",
                    )
                if a.kind == "raise" and cur == 0:
                    raise GameStateError.one(where, "'raise' with no bet to raise (use 'bet')")
                return self.raise_to(engine, self.chips(a.amount, where), where, probe)
        raise AssertionError(a.kind)

    def raise_to(self, engine: _core.HandState, to: int, where: str, probe: bool) -> _core.Action:
        action = _core.Action.bet_raise_to(to)
        why = engine.why_illegal(action)
        if why is not None:
            la = engine.legal_actions()
            if not la.bet_raise:
                detail = ""
            elif probe:
                detail = f" (minimum {self.units(la.min_raise_to)})"
            else:
                detail = f" (legal: {self.units(la.min_raise_to)} to {self.units(la.max_raise_to)})"
            raise GameStateError.one(where, f"{why}{detail}")
        return action

    def check_final(self, engine: _core.HandState, stacks: list[int], hero: int) -> None:
        s = self.s
        if engine.is_terminal:
            raise GameStateError.one("action_history", "the hand is already over")
        if STREETS[int(engine.street)] != s.street:
            raise GameStateError.one(
                "street",
                f"the history puts the hand on the {STREETS[int(engine.street)]}, "
                f"not the {s.street}",
            )

        issues: list[Issue] = []
        pot = self.chips(s.pot, "pot")
        if pot != engine.pot:
            issues.append(
                Issue(
                    "pot",
                    f"{s.pot} doesn't match {self.units(engine.pot)} from the blinds and "
                    "action history (pot counts every chip put in, including this street's bets)",
                )
            )
        for i in range(self.n):
            label = self.labels[i]
            want = _STATUS_NAMES[engine.statuses[i]]
            got = next(p.status for p in s.players if self.seat(p.seat, "players") == i)
            if got != want:
                issues.append(
                    Issue(f"players[{label}].status", f"is '{got}' but replay says '{want}'")
                )
            if engine.stacks[i] != stacks[i]:  # only possible via all-in/blind corner cases
                issues.append(
                    Issue(
                        f"players[{label}].stack",
                        f"is {self.units(stacks[i])} but replay says "
                        f"{self.units(engine.stacks[i])}",
                    )
                )
        to_act = self.seat(s.to_act, "to_act")
        if to_act != engine.to_act:
            issues.append(
                Issue("to_act", f"is {s.to_act} but {self.labels[engine.to_act]} is to act")
            )
        elif to_act != hero:
            issues.append(
                Issue("hero_seat", f"it's {s.to_act}'s turn, not the hero's ({s.hero_seat})")
            )
        if issues:
            raise GameStateError(issues)
