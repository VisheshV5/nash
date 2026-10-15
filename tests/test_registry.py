"""Named bundles: pack, download (file:// in tests), verify, cache, load."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from regret.registry import RegistryError, pack, resolve
from regret.utils.config import RegretConfig


@pytest.fixture
def registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    bundle = tmp_path / "hu-demo"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(json.dumps({"iteration": 7, "format_version": 1}))
    (bundle / "strategy.npz").write_bytes(b"\0" * 1000)
    archive = tmp_path / "dist" / "hu-demo.tar.gz"
    entry = pack(bundle, archive, f"file://{archive}")
    reg = tmp_path / "registry.json"
    reg.write_text(json.dumps({"bundles": {"hu-demo": entry}}))
    monkeypatch.setenv("REGRET_CACHE", str(tmp_path / "cache"))
    return reg, archive


def test_resolve_downloads_once_and_caches(registry: tuple[Path, Path]) -> None:
    reg, archive = registry
    path = resolve("hu-demo", reg)
    assert json.loads((path / "manifest.json").read_text())["iteration"] == 7
    archive.unlink()  # a second resolve must not need the network
    assert resolve("hu-demo", reg) == path


def test_tampered_download_is_rejected(registry: tuple[Path, Path]) -> None:
    reg, archive = registry
    archive.write_bytes(archive.read_bytes()[:-5] + b"evil!")
    with pytest.raises(RegistryError, match="checksum mismatch"):
        resolve("hu-demo", reg)
    assert not any((Path(reg.parent) / "cache").glob("hu-demo-*/manifest.json"))


def test_unknown_name(registry: tuple[Path, Path]) -> None:
    reg, _ = registry
    with pytest.raises(RegistryError, match="unknown bundle 'nope'"):
        resolve("nope", reg)


def test_agent_loads_a_registered_name(
    tiny_holdem: RegretConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import regret.registry as registry_mod
    from regret import RegretAgent
    from regret.cfr.checkpoint import latest_checkpoint
    from regret.cfr.trainer import RunSpec, Trainer
    from regret.export.bundle import export_bundle

    Trainer(RunSpec.from_config(tiny_holdem), tmp_path / "run").run(max_iterations=5_000)
    ckpt = latest_checkpoint(tmp_path / "run" / "checkpoints")
    assert ckpt is not None
    export_bundle(tiny_holdem, ckpt, tmp_path / "hu-tiny")
    archive = tmp_path / "dist" / "hu-tiny.tar.gz"
    reg = tmp_path / "registry.json"
    reg.write_text(
        json.dumps(
            {"bundles": {"hu-tiny": pack(tmp_path / "hu-tiny", archive, f"file://{archive}")}}
        )
    )
    monkeypatch.setattr(registry_mod, "REGISTRY", reg)
    monkeypatch.setenv("REGRET_CACHE", str(tmp_path / "cache"))
    monkeypatch.chdir(tmp_path / "dist")  # so "hu-tiny" isn't also a local directory
    agent = RegretAgent.load("hu-tiny")
    assert agent.info()["name"] == "hu-tiny"
