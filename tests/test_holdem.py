"""Hold'em training, export, players, matches and LBR, on a tiny abstraction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from regret import _core
from regret.abstraction.actions import rules_from_config
from regret.abstraction.build import artifact_dir
from regret.abstraction.cards import CardAbstraction
from regret.cfr.checkpoint import latest_checkpoint
from regret.cfr.trainer import RunSpec, Trainer, TrainingError
from regret.eval.lbr import LBRPlayer, board_buckets
from regret.eval.match import duplicate_match, play_hand
from regret.eval.players import AlwaysCall, EquityThreshold, RandomPlayer, _Base
from regret.eval.setup import blueprint_player, table_chips
from regret.eval.strategy import ArrayStrategy, SolverStrategy
from regret.export.bundle import export_bundle, load_bundle
from regret.utils.config import RegretConfig

STACKS, SB, BB = [10_000, 10_000], 50, 100


@pytest.fixture(scope="module")
def trained(
    tiny_holdem: RegretConfig, tmp_path_factory: pytest.TempPathFactory
) -> tuple[RegretConfig, Path]:
    run_dir = tmp_path_factory.mktemp("run")
    Trainer(RunSpec.from_config(tiny_holdem), run_dir).run(max_iterations=60_000)
    return tiny_holdem, run_dir


@pytest.fixture(scope="module")
def solver(trained: tuple[RegretConfig, Path]) -> _core.CfrSolver:
    cfg, run_dir = trained
    s = RunSpec.from_config(cfg).make_solver()
    ckpt = latest_checkpoint(run_dir / "checkpoints")
    assert ckpt is not None
    s.load(ckpt.payload)
    return s


# ---------------------------------------------------------------- training


def test_holdem_training_and_eval_events(
    trained: tuple[RegretConfig, Path], solver: _core.CfrSolver
) -> None:
    assert solver.iteration == 60_000
    assert solver.num_infosets > 1_000
    keys, offsets, probs = solver.export_strategy()
    assert len(keys) == solver.num_infosets == len(offsets) - 1
    assert np.all(np.diff(keys.astype(np.float64)) > 0)
    sums = np.add.reduceat(probs, offsets[:-1])
    np.testing.assert_allclose(sums, 1.0, atol=1e-4)
    with pytest.raises(RuntimeError, match="small games"):
        solver.nash_conv()


def test_holdem_resume_is_bit_identical(tiny_holdem: RegretConfig) -> None:
    spec = RunSpec.from_config(tiny_holdem)
    a = spec.make_solver()
    a.run(8_000)
    b = spec.make_solver()
    b.run(3_333)
    c = spec.make_solver()
    c.load(b.save())
    c.run(8_000 - 3_333)
    assert c.save() == a.save()


def test_holdem_needs_a_built_abstraction(tiny_holdem: RegretConfig) -> None:
    other = tiny_holdem.model_copy(
        update={"cards": tiny_holdem.cards.model_copy(update={"seed": 99})}
    )
    with pytest.raises(TrainingError, match="build_abstraction"):
        RunSpec.from_config(other).make_solver()


def test_six_max_is_v1_2(tiny_holdem: RegretConfig) -> None:
    six = tiny_holdem.model_copy(
        update={"table": tiny_holdem.table.model_copy(update={"num_players": 6})}
    )
    with pytest.raises(TrainingError, match="heads-up"):
        RunSpec.from_config(six)


# ---------------------------------------------------------------- bundle


def test_export_and_load_bundle(
    trained: tuple[RegretConfig, Path], solver: _core.CfrSolver, tmp_path: Path
) -> None:
    cfg, run_dir = trained
    ckpt = latest_checkpoint(run_dir / "checkpoints")
    assert ckpt is not None
    manifest = export_bundle(cfg, ckpt, tmp_path / "bundle")
    assert manifest["iteration"] == 60_000
    assert manifest["infosets"] == solver.num_infosets
    assert not (tmp_path / "bundle" / "cards" / "river_table.npy").exists()
    cfg2, strategy, cards_dir = load_bundle(tmp_path / "bundle")
    assert cfg2 == cfg
    live = SolverStrategy(solver)
    keys, _, _ = solver.export_strategy()
    for key in keys[:: max(1, len(keys) // 50)]:
        a, b = strategy.probs(int(key)), live.probs(int(key))
        assert a is not None
        assert b is not None
        np.testing.assert_allclose(a, b, atol=0.01)  # uint8 quantization
    assert strategy.probs(12345) is None
    assert CardAbstraction.load(cards_dir).bucket("AsKs", "Qs7h2d") >= 0


# ---------------------------------------------------------------- players and matches


class Shover(_Base):
    name = "shover"

    def act(self, state: _core.HandState, board: list[int]) -> _core.Action:
        la = state.legal_actions()
        return (
            _core.Action.bet_raise_to(la.max_raise_to)
            if la.bet_raise
            else _core.Action.check_call()
        )


def test_identical_players_break_even_exactly() -> None:
    r = duplicate_match(AlwaysCall(), AlwaysCall(), 200, STACKS, SB, BB, seed=1)
    assert (r.mbb, r.ci95) == (0.0, 0.0)


def test_allin_adjustment_reduces_variance() -> None:
    r = duplicate_match(Shover(), EquityThreshold(), 150, STACKS, SB, BB, seed=2)
    assert r.ci95 < r.ci95_raw


def test_blueprint_plays_legal_poker(tiny_holdem: RegretConfig, solver: _core.CfrSolver) -> None:
    bp = blueprint_player(tiny_holdem, SolverStrategy(solver))
    rng = np.random.default_rng(0)
    for opp in (RandomPlayer(), AlwaysCall(), EquityThreshold()):
        for _ in range(40):  # play_hand raises on any illegal action
            cards = rng.choice(52, 9, replace=False).tolist()
            play_hand([bp, opp], [cards[:2], cards[2:4]], cards[4:], STACKS, SB, BB, rng)
            play_hand([opp, bp], [cards[:2], cards[2:4]], cards[4:], STACKS, SB, BB, rng)
    assert bp.unseen == 0


def test_blueprint_stays_on_tree_against_a_caller(
    tiny_holdem: RegretConfig, solver: _core.CfrSolver
) -> None:
    bp = blueprint_player(tiny_holdem, SolverStrategy(solver))
    duplicate_match(bp, AlwaysCall(), 50, STACKS, SB, BB, seed=3)
    assert bp.fallbacks == 0  # calls always translate exactly


def test_equity_bot_folds_trash_to_a_shove() -> None:
    bot = EquityThreshold()
    bot.new_hand(1, [5, 20], np.random.default_rng(0))  # 3d 7c
    h = _core.HandState(STACKS, SB, BB)
    h.apply(_core.Action.bet_raise_to(10_000))
    assert bot.act(h, []).type == _core.ActionType.FOLD


# ---------------------------------------------------------------- LBR


def test_board_buckets_match_single_lookups(tiny_holdem: RegretConfig) -> None:
    cards = CardAbstraction.load(artifact_dir(tiny_holdem.cards))
    board = [0, 17, 33, 42]
    b = board_buckets(cards, board)
    pairs = [(x, y) for y in range(52) for x in range(y)]
    for h in range(0, 1326, 97):
        if set(pairs[h]) & set(board):
            assert b[h] == -1
        else:
            assert b[h] == cards.bucket(list(pairs[h]), board)


def test_lbr_crushes_an_untrained_blueprint(tiny_holdem: RegretConfig) -> None:
    fresh = RunSpec.from_config(tiny_holdem).make_solver()  # uniform random strategy everywhere
    cards = CardAbstraction.load(artifact_dir(tiny_holdem.cards))
    stacks, sb, bb = table_chips(tiny_holdem)
    rules = rules_from_config(tiny_holdem.actions)
    lbr = LBRPlayer(SolverStrategy(fresh), cards, rules, stacks, sb, bb, flop_samples=200)
    victim = blueprint_player(tiny_holdem, SolverStrategy(fresh), cards)
    r = duplicate_match(lbr, victim, 60, stacks, sb, bb, seed=4)
    assert r.mbb > r.ci95 > 0


def test_array_strategy_round_trip(solver: _core.CfrSolver, tmp_path: Path) -> None:
    s = ArrayStrategy.from_solver(solver)
    s.save(tmp_path / "s.npz")
    t = ArrayStrategy.load(tmp_path / "s.npz")
    key = int(s.keys[len(s) // 2])
    a, b = s.probs(key), t.probs(key)
    assert a is not None
    assert b is not None
    np.testing.assert_allclose(a, b, atol=0.01)


def test_periodic_eval_report(tiny_holdem: RegretConfig, solver: _core.CfrSolver) -> None:
    from regret.eval.setup import baseline_report

    out = baseline_report(tiny_holdem, solver, hands=40, seed=1)
    assert set(out) >= {
        "vs_random_mbb",
        "vs_always-call_mbb",
        "vs_equity-threshold_mbb",
        "vs_random_ci95",
        "unseen_infosets",
        "fallbacks",
    }
