"""Named bundles that download on first use.

    RegretAgent.load("hu-100bb")   # downloads, verifies and caches it once, then loads locally

`registry.json` (shipped with the package) maps names to a release archive URL, its SHA-256 and
size. Archives are cached under $REGRET_CACHE (default ~/.cache/regret) and verified before
they're unpacked; a damaged or tampered download is deleted and an error raised.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import ssl
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

REGISTRY = Path(__file__).with_name("registry.json")


class RegistryError(Exception):
    pass


def cache_dir() -> Path:
    return Path(os.environ.get("REGRET_CACHE", Path.home() / ".cache" / "regret"))


def entries(registry: Path | None = None) -> dict[str, dict[str, Any]]:
    data: dict[str, dict[str, Any]] = json.loads((registry or REGISTRY).read_text())["bundles"]
    return data


def _ssl_context() -> ssl.SSLContext:
    try:  # python.org builds on macOS ship without root certificates
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _download(url: str, dest: Path) -> None:
    if url.startswith("file://"):
        shutil.copyfile(url[len("file://") :], dest)
        return
    with (
        urllib.request.urlopen(url, context=_ssl_context(), timeout=60) as r,
        open(dest, "wb") as f,
    ):
        shutil.copyfileobj(r, f)


def resolve(name: str, registry: Path | None = None) -> Path:
    """Local directory of bundle `name`, downloading it first if needed."""
    known = entries(registry)
    if name not in known:
        raise RegistryError(
            f"unknown bundle '{name}' (known: {', '.join(sorted(known)) or 'none'})"
        )
    entry = known[name]
    target = cache_dir() / f"{name}-{entry['sha256'][:12]}"
    if (target / "manifest.json").exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent) as tmp:
        archive = Path(tmp) / "bundle.tar.gz"
        _download(entry["url"], archive)
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        if digest != entry["sha256"]:
            raise RegistryError(
                f"{name}: checksum mismatch (got {digest[:12]}, want "
                f"{entry['sha256'][:12]}); the download was deleted"
            )
        unpacked = Path(tmp) / "unpacked"
        with tarfile.open(archive) as tar:
            tar.extractall(unpacked, filter="data")
        roots = [p for p in unpacked.iterdir() if p.is_dir()]
        if len(roots) != 1 or not (roots[0] / "manifest.json").exists():
            raise RegistryError(f"{name}: archive doesn't contain one bundle directory")
        roots[0].rename(target)  # atomic within the cache directory
    return target


def pack(bundle: Path, out: Path, url: str) -> dict[str, Any]:
    """Archive a bundle directory and return its registry entry."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out, "w:gz") as tar:
        tar.add(bundle, arcname=bundle.name)
    data = out.read_bytes()
    manifest = json.loads((bundle / "manifest.json").read_text())
    return {
        "url": url,
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "iteration": manifest.get("iteration"),
        "format_version": manifest.get("format_version"),
    }
