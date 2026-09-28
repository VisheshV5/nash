from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from regret.utils.config import RegretConfig, load_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _hu() -> dict[str, Any]:
    data = yaml.safe_load((CONFIGS / "hu_default.yaml").read_text())
    assert isinstance(data, dict)
    return data


@pytest.mark.parametrize("path", sorted(CONFIGS.glob("*.yaml")), ids=lambda p: p.name)
def test_shipped_configs_load(path: Path) -> None:
    load_config(path)


def test_hu_default_values() -> None:
    cfg = load_config(CONFIGS / "hu_default.yaml")
    assert cfg.table.num_players == 2
    assert cfg.table.starting_stack_bb == 100
    assert cfg.actions.postflop_for(2).bet_pot == [0.33, 0.75, 1.25]
    assert cfg.search.time_budget_ms < cfg.search.hard_cap_ms


def test_unknown_key_is_rejected() -> None:
    data = _hu()
    data["table"]["stack"] = 100
    with pytest.raises(ValidationError, match="stack"):
        RegretConfig.model_validate(data)


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("table", "num_players", 7),
        ("table", "starting_stack_bb", 0),
        ("cards", "flop_buckets", 1),
        ("search", "time_budget_ms", 2000),
        ("training", "threads", 0),
    ],
)
def test_out_of_range_values_are_rejected(section: str, key: str, value: object) -> None:
    data = _hu()
    data[section][key] = value
    with pytest.raises(ValidationError):
        RegretConfig.model_validate(data)


def test_bet_sizes_must_be_increasing() -> None:
    data = _hu()
    data["actions"]["postflop"][2]["bet_pot"] = [0.75, 0.33]
    with pytest.raises(ValidationError, match="strictly increasing"):
        RegretConfig.model_validate(data)


def test_raise_multiplier_must_exceed_one() -> None:
    data = _hu()
    data["actions"]["preflop"]["four_bet_x"] = [1.0]
    with pytest.raises(ValidationError, match="> 1"):
        RegretConfig.model_validate(data)


def test_postflop_rules_cannot_exceed_table_size() -> None:
    data = _hu()
    data["actions"]["postflop"][3] = {"bet_pot": [1.0]}
    with pytest.raises(ValidationError, match="exceed"):
        RegretConfig.model_validate(data)


def test_postflop_rule_selection_by_active_players() -> None:
    data = _hu()
    data["table"]["num_players"] = 6
    data["actions"]["postflop"][3] = {"bet_pot": [0.5, 1.0]}
    data["actions"]["postflop"][4] = {"bet_pot": [1.0]}
    cfg = RegretConfig.model_validate(data)
    assert cfg.actions.postflop_for(2).bet_pot == [0.33, 0.75, 1.25]
    assert cfg.actions.postflop_for(3).bet_pot == [0.5, 1.0]
    assert cfg.actions.postflop_for(6).bet_pot == [1.0]
    with pytest.raises(ValueError, match="at least 2"):
        cfg.actions.postflop_for(1)


def test_config_hash_is_stable_and_sensitive() -> None:
    a = RegretConfig.model_validate(_hu())
    b = RegretConfig.model_validate(_hu())
    assert a.config_hash() == b.config_hash()
    data = _hu()
    data["cards"]["river_buckets"] = 201
    assert RegretConfig.model_validate(data).config_hash() != a.config_hash()


def test_top_level_must_be_a_mapping(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("- just\n- a list\n")
    with pytest.raises(ValueError, match="mapping"):
        load_config(bad)
