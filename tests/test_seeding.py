import numpy as np
import pytest

from regret.utils.seeding import derive_seed, make_rng


def test_same_inputs_same_stream() -> None:
    a = make_rng(42, "eval", "deal").integers(0, 2**32, size=16)
    b = make_rng(42, "eval", "deal").integers(0, 2**32, size=16)
    np.testing.assert_array_equal(a, b)


def test_names_and_master_seed_change_the_stream() -> None:
    base = make_rng(42, "eval", "deal").integers(0, 2**32, size=16)
    for other in (make_rng(42, "eval", "shuffle"), make_rng(43, "eval", "deal"), make_rng(42)):
        assert not np.array_equal(base, other.integers(0, 2**32, size=16))


def test_derived_seed_is_fixed_64_bit_value() -> None:
    seed = derive_seed(7, "cfr", "worker", "0")
    assert seed == derive_seed(7, "cfr", "worker", "0")
    assert 0 <= seed < 2**64
    assert seed != derive_seed(7, "cfr", "worker", "1")


def test_derived_seed_is_stable_across_releases() -> None:
    # Pinned so checkpoints and published eval runs stay reproducible. Changing the
    # derivation scheme must be a deliberate, versioned decision.
    assert derive_seed(0, "regret") == 382386978998307302


def test_negative_master_seed_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        derive_seed(-1)
