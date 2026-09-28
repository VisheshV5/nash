"""Crash-safe training checkpoints.

A checkpoint is one file, `ckpt-<iteration>.rgc`:

    b"RGCKPT01" | u32 header length | JSON header | zstd(payload)

The header records the iteration, elapsed training time, config fingerprint, and a SHA-256 of
the payload, so a truncated or corrupted file is detected on load rather than trusted.

Writes are atomic: write to a temporary file in the same directory, fsync it, rename it into
place, then fsync the directory. A crash at any point leaves either the old set of checkpoints
or the old set plus the complete new one, never a half-written file under a real name.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import zstandard

log = logging.getLogger(__name__)

MAGIC = b"RGCKPT01"
_NAME = re.compile(r"^ckpt-(\d{12})\.rgc$")
_TMP_PREFIX = ".tmp-ckpt-"


class CheckpointError(Exception):
    pass


@dataclass(frozen=True)
class Checkpoint:
    path: Path
    iteration: int
    header: dict[str, Any]
    payload: bytes


def checkpoint_path(directory: Path, iteration: int) -> Path:
    return directory / f"ckpt-{iteration:012d}.rgc"


def save_checkpoint(
    directory: Path, iteration: int, payload: bytes, header: dict[str, Any], keep_last: int
) -> Path:
    """Atomically write a checkpoint, then delete all but the newest `keep_last`."""
    directory.mkdir(parents=True, exist_ok=True)
    full_header = {
        **header,
        "iteration": iteration,
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "payload_bytes": len(payload),
    }
    head = json.dumps(full_header, sort_keys=True).encode()
    body = zstandard.ZstdCompressor(level=3).compress(payload)

    final = checkpoint_path(directory, iteration)
    tmp = directory / f"{_TMP_PREFIX}{iteration:012d}-{os.getpid()}"
    with open(tmp, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", len(head)))
        f.write(head)
        f.write(body)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, final)
    _fsync_dir(directory)

    for old in list_checkpoints(directory)[:-keep_last]:
        old.unlink(missing_ok=True)
    return final


def load_checkpoint(path: Path) -> Checkpoint:
    """Read and verify a checkpoint. Raises CheckpointError if it's damaged."""
    try:
        data = path.read_bytes()
    except OSError as e:
        raise CheckpointError(f"{path}: {e}") from e
    if not data.startswith(MAGIC) or len(data) < len(MAGIC) + 4:
        raise CheckpointError(f"{path}: not a checkpoint")
    (n,) = struct.unpack_from("<I", data, len(MAGIC))
    start = len(MAGIC) + 4
    try:
        header = json.loads(data[start : start + n])
        payload = zstandard.ZstdDecompressor().decompress(
            data[start + n :], max_output_size=int(header["payload_bytes"])
        )
    except (ValueError, KeyError, zstandard.ZstdError) as e:
        raise CheckpointError(f"{path}: damaged ({e})") from e
    if hashlib.sha256(payload).hexdigest() != header["payload_sha256"]:
        raise CheckpointError(f"{path}: payload checksum mismatch")
    return Checkpoint(path, int(header["iteration"]), header, payload)


def list_checkpoints(directory: Path) -> list[Path]:
    """Checkpoint files, oldest first (by iteration)."""
    if not directory.is_dir():
        return []
    found = [(int(m.group(1)), p) for p in directory.iterdir() if (m := _NAME.match(p.name))]
    return [p for _, p in sorted(found)]


def latest_checkpoint(directory: Path) -> Checkpoint | None:
    """Newest checkpoint that verifies; damaged ones are skipped with a warning."""
    for path in reversed(list_checkpoints(directory)):
        try:
            return load_checkpoint(path)
        except CheckpointError as e:
            log.warning("skipping %s", e)
    return None


def remove_stale_temp_files(directory: Path) -> None:
    """Leftovers from a crash mid-write are never valid; delete them."""
    if directory.is_dir():
        for p in directory.glob(f"{_TMP_PREFIX}*"):
            p.unlink(missing_ok=True)


def _fsync_dir(directory: Path) -> None:
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
