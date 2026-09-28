"""A quick differential fuzz vs PokerKit on every test run (full run: tests/fuzz_pokerkit.py)."""

import pytest

from fuzz_pokerkit import run


@pytest.mark.parametrize("seed", [0, 1])
def test_engine_matches_pokerkit(seed: int) -> None:
    hands, decisions = run(seed, 750)
    assert hands == 750
    assert decisions > 3000
