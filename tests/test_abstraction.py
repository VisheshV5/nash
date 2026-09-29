"""Card abstraction (features, k-means, tables, runtime bucketing) and action abstraction."""

from __future__ import annotations

import itertools
import time
from pathlib import Path

import numpy as np
import pytest

from regret import _core
from regret.abstraction.actions import action_label, rules_from_config
from regret.abstraction.build import build_card_abstraction
from regret.abstraction.cards import CardAbstraction
from regret.abstraction.kmeans import kmeans
from regret.abstraction.planner import plan
from regret.engine.cards import evaluate, indexer, parse_cards
from regret.utils.config import ActionAbstractionConfig, CardAbstractionConfig, load_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
PAIRS = [(a, b) for b in range(52) for a in range(b)]

# ---------------------------------------------------------------- features


def test_hole_index_matches_pair_order() -> None:
    assert [_core.hole_index(a, b) for a, b in PAIRS] == list(range(1326))
    assert _core.hole_index(51, 0) == _core.hole_index(0, 51)


def test_river_equities_match_brute_force() -> None:
    board = parse_cards("AsKd7h7c2s")
    eq = _core.river_equities(board)
    rng = np.random.default_rng(0)
    for h in rng.choice(1326, 15, replace=False):
        a, b = PAIRS[h]
        if {a, b} & set(board):
            assert eq[h] == -1
            continue
        me = evaluate([a, b, *board])
        rest = [c for c in range(52) if c not in board and c not in (a, b)]
        vals = [evaluate([x, y, *board]) for x, y in itertools.combinations(rest, 2)]
        want = (sum(v < me for v in vals) + 0.5 * sum(v == me for v in vals)) / len(vals)
        assert eq[h] == pytest.approx(want, abs=1e-6)


def test_batch_and_single_hand_features_agree() -> None:
    board = parse_cards("9h8h2cQsTd")
    clusters = np.random.default_rng(1).integers(0, 8, 1326).astype(np.uint8)
    ochs = _core.river_ochs(board, clusters, 8)
    hist = _core.turn_histograms(board[:4], 50)
    for h in np.random.default_rng(2).choice(1326, 40, replace=False):
        a, b = PAIRS[h]
        if {a, b} & set(board):
            continue
        np.testing.assert_allclose(
            ochs[h], _core.river_ochs_one([a, b], board, clusters, 8), atol=1e-6
        )
        if not {a, b} & set(board[:4]):
            np.testing.assert_array_equal(hist[h], _core.turn_histogram_one([a, b], board[:4], 50))
    assert set(hist.sum(1).tolist()) == {0, 46}


def test_runtime_features_are_fast() -> None:
    board = parse_cards("9h8h2cQsTd")
    clusters = np.zeros(1326, np.uint8)
    t = time.perf_counter()
    for _ in range(20):
        _core.turn_histogram_one([0, 1], board[:4], 50)
    assert (time.perf_counter() - t) / 20 < 5e-3  # roadmap: turn <= 5 ms
    t = time.perf_counter()
    for _ in range(200):
        _core.river_ochs_one([0, 1], board, clusters, 8)
    assert (time.perf_counter() - t) / 200 < 1e-3  # roadmap: river <= 1 ms


# ---------------------------------------------------------------- k-means


def test_kmeans_finds_separated_clusters_and_iterates() -> None:
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(m, 0.05, (500, 2)) for m in (0.0, 1.0, 2.0)]).astype(np.float32)
    r = kmeans(x, 3, seed=1)
    np.testing.assert_allclose(np.sort(r.centroids[:, 0]), [0, 1, 2], atol=0.02)
    assert r.iterations >= 2
    assert len(set(r.labels[:500])) == 1


def test_kmeans_is_deterministic_and_weighted() -> None:
    rng = np.random.default_rng(3)
    x = rng.random((3000, 5)).astype(np.float32)
    a, b = kmeans(x, 7, seed=5), kmeans(x, 7, seed=5)
    np.testing.assert_array_equal(a.centroids, b.centroids)
    # One cluster, two points: the centroid moves toward the heavier point.
    pts = np.array([[0.0], [1.0]], np.float32)
    r = kmeans(pts, 1, weights=np.array([3.0, 1.0]), seed=0)
    assert r.centroids[0, 0] == pytest.approx(0.25)


# ---------------------------------------------------------------- tables (tiny build)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory: pytest.TempPathFactory) -> CardAbstraction:
    cfg = CardAbstractionConfig(
        flop_buckets=6,
        turn_buckets=6,
        river_buckets=6,
        feature_sample_size=30_000,
        preflop_equity_samples=1_000,
    )
    out = tmp_path_factory.mktemp("abs")
    build_card_abstraction(cfg, out, threads=6)
    return CardAbstraction.load(out, river_table=True)


def test_tables_are_complete_and_use_every_bucket(tiny: CardAbstraction) -> None:
    m = tiny.manifest
    assert m["table_sizes"] == {"flop": 1_286_792, "turn": 13_960_050, "river": 123_156_254}
    for street in ("flop", "turn", "river"):
        assert m["bucket_usage"][street]["empty"] == 0
    assert tiny.flop_table.max() < 6


def test_table_lookups_match_direct_computation(tiny: CardAbstraction) -> None:
    rng = np.random.default_rng(7)
    k = tiny.manifest["config"]["river_opponent_clusters"]
    for _ in range(30):
        cards = rng.choice(52, 7, replace=False).tolist()
        hole, board = cards[:2], cards[2:]
        # River: table vs OCHS features + nearest centroid.
        x = _core.river_ochs_one(hole, board, tiny.cluster_of_hole, k)
        direct = int(((tiny.river_centroids - x) ** 2).sum(1).argmin())
        assert tiny.bucket(hole, board) == direct
        # And the same answer without the river table.
        no_table = CardAbstraction(
            tiny.manifest,
            tiny.cluster_of_hole,
            tiny.river_centroids,
            tiny.flop_table,
            tiny.turn_table,
            None,
        )
        assert no_table.bucket(hole, board) == direct


def test_isomorphic_hands_share_buckets(tiny: CardAbstraction) -> None:
    assert tiny.bucket("AsKs", "Qs7h2d") == tiny.bucket("AhKh", "2c7dQh")
    assert tiny.bucket("AsKs", "Qs7h2d9c") == tiny.bucket("AdKd", "9h2sQd7c")
    assert tiny.bucket("AsKs") == indexer("preflop").index(parse_cards("AsKs"))


def test_strong_hands_land_in_strong_river_buckets(tiny: CardAbstraction) -> None:
    order = np.argsort(tiny.river_centroids.mean(1))  # weakest .. strongest bucket
    rank = {int(b): i for i, b in enumerate(order)}
    nuts = rank[tiny.bucket("AsKs", "QsJsTs2d3c")]  # royal flush
    air = rank[tiny.bucket("7c2d", "QsJsTs9h4h")]
    assert nuts > air


# ---------------------------------------------------------------- action abstraction


@pytest.fixture
def rules() -> _core.ActionRules:
    """Pinned sizes (independent of the shipped config, which gets tuned)."""
    return rules_from_config(
        ActionAbstractionConfig.model_validate(
            {
                "preflop": {"open_bb": [2.5], "reraise_x_ip": [3.0], "reraise_x_oop": [4.0],
                            "four_bet_x": [2.3], "max_raises": 4},
                "postflop": {2: {"bet_pot": [0.33, 0.75, 1.25], "raise_pot": [0.75],
                                 "max_raises": 3}},
            }
        )
    )


def labels(h: _core.HandState, rules: _core.ActionRules) -> list[str]:
    return [action_label(h, a) for a in _core.abstract_actions(h, rules)]


def test_heads_up_preflop_sizes(rules: _core.ActionRules) -> None:
    h = _core.HandState([10_000, 10_000], 50, 100)
    assert labels(h, rules) == ["fold", "call", "raise_2.5bb", "all_in"]
    h.apply(_core.Action.bet_raise_to(250))
    acts = _core.abstract_actions(h, rules)
    assert [a.amount for a in acts[2:]] == [1000, 10_000]  # BB is out of position: 4x
    h.apply(acts[2])
    assert [a.amount for a in _core.abstract_actions(h, rules)[2:]] == [2300, 10_000]  # 2.3x


def _raised_flop() -> _core.HandState:
    h = _core.HandState([10_000, 10_000], 50, 100)
    h.apply(_core.Action.bet_raise_to(250))
    h.apply(_core.Action.check_call())  # flop, pot 500
    return h


def test_postflop_sizes_and_raise_cap(rules: _core.ActionRules) -> None:
    h = _raised_flop()
    assert labels(h, rules) == ["check", "bet_0.33pot", "bet_0.75pot", "bet_1.25pot", "all_in"]
    assert [a.amount for a in _core.abstract_actions(h, rules)[1:4]] == [165, 375, 625]
    for _ in range(3):  # bet, raise, raise: the cap (3) is reached
        h.apply(_core.abstract_actions(h, rules)[-2])
    assert labels(h, rules) == ["fold", "call", "all_in"]


def test_sizes_are_clamped_to_the_legal_range(rules: _core.ActionRules) -> None:
    # Limped pot of 200: 33% pot (66) is below the 1bb minimum bet, so it becomes 100.
    h = _core.HandState([10_000, 10_000], 50, 100)
    h.apply(_core.Action.check_call())
    h.apply(_core.Action.check_call())
    assert [a.amount for a in _core.abstract_actions(h, rules)[1:4]] == [100, 150, 250]
    # A short stack whose open size would be all-in anyway just gets all-in, once.
    h = _core.HandState([240, 10_000], 50, 100)
    assert labels(h, rules) == ["fold", "call", "all_in"]
    for a in _core.abstract_actions(h, rules):
        assert h.why_illegal(a) is None


def test_pseudo_harmonic_mapping() -> None:
    # Closed form: f(x) = (B - x)(1 + A) / ((B - A)(1 + x)).
    for a, b, x, want in [
        (0.5, 1.0, 0.75, 0.375 / 0.875),
        (0.33, 0.75, 0.5, 0.25 * 1.33 / (0.42 * 1.5)),
        (0.0, 1.0, 0.5, 1 / 3),
        (1.0, 2.0, 1.0, 1.0),
        (1.0, 2.0, 2.0, 0.0),
    ]:
        assert _core.pseudo_harmonic(a, b, x) == pytest.approx(want)


def test_translation(rules: _core.ActionRules) -> None:
    h = _raised_flop()  # pot 500; sizes 165 (0.33), 375 (0.75), 625 (1.25), all-in
    assert _core.translate(h, rules, _core.Action.bet_raise_to(375)) == [(2, 1.0)]
    half = _core.translate(h, rules, _core.Action.bet_raise_to(250))  # 0.5 pot
    assert [i for i, _ in half] == [1, 2]
    assert half[0][1] == pytest.approx(_core.pseudo_harmonic(0.33, 0.75, 0.5))
    assert sum(p for _, p in half) == pytest.approx(1.0)
    # Below the smallest size: between check (size 0) and 33% pot.
    small = _core.translate(h, rules, _core.Action.bet_raise_to(120))
    assert [i for i, _ in small] == [0, 1]
    # Bigger than every sized bet but short of all-in: between 125% and all-in.
    big = _core.translate(h, rules, _core.Action.bet_raise_to(3000))
    assert [i for i, _ in big] == [3, 4]
    assert _core.translate(h, rules, _core.Action.check_call()) == [(0, 1.0)]
    with pytest.raises(ValueError, match="below the minimum"):
        _core.translate(h, rules, _core.Action.bet_raise_to(10))


# ---------------------------------------------------------------- memory planner


def test_memory_planner_counts_the_heads_up_tree() -> None:
    p = plan(load_config(CONFIGS / "hu_default.yaml"))
    assert p.nodes[0] > 0
    assert all(n > 0 for n in p.nodes)
    assert p.infosets[0] == p.nodes[0] * 169
    assert p.total_bytes > p.regret_bytes > 0


def test_same_seed_gives_identical_tables(tiny: CardAbstraction, tmp_path: Path) -> None:
    cfg = CardAbstractionConfig(**tiny.manifest["config"])
    again = build_card_abstraction(cfg, tmp_path, threads=3)  # thread count mustn't matter
    assert again["sha256"] == tiny.manifest["sha256"]
