"""Fetch a registered artifact from private Supabase ``run_artifacts`` into a local cache.

Trust chain: the run row's ``hyperparameters.artifact_sha256`` (written by the
registration command) is the expected SHA-256. The downloaded bytes are hashed
BEFORE anything is written to the cache or handed to ``torch.load``; a mismatch
raises and the bytes are discarded. A cached file is re-hashed on every use, so
a stale or tampered cache file is deleted and fetched again. Loading itself is
still the existing ``weights_only`` loader. Errors carry reason codes only: no
paths, URLs or database messages.
"""
from __future__ import annotations

import base64
import binascii
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict

from shared.forecast_registration import file_sha256

MAX_B64_CHARS = 30_000_000   # artifacts are ~60 KB of base64 today; this only bounds a bad row
_SHA = re.compile(r"[0-9a-f]{64}")


class ArtifactError(Exception):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def expected_sha256(run_row: Dict[str, Any]) -> str:
    hp = run_row.get("hyperparameters")
    sha = hp.get("artifact_sha256") if isinstance(hp, dict) else None
    if not isinstance(sha, str) or not _SHA.fullmatch(sha):
        raise ArtifactError("artifact_hash_missing_in_run")
    return sha


def cache_path(cache_dir: Path, run_id: int, sha: str) -> Path:
    return Path(cache_dir) / f"run{run_id}-{sha[:16]}.pt"


def ensure_artifact(repo, run_id: int, run_row: Dict[str, Any], cache_dir: Path) -> Path:
    """Return a local file whose SHA-256 equals the run's recorded artifact hash."""
    sha = expected_sha256(run_row)
    path = cache_path(cache_dir, run_id, sha)
    if path.is_file():
        try:
            if file_sha256(path) == sha:
                return path
            path.unlink()                      # stale or tampered cache entry
        except OSError:
            raise ArtifactError("artifact_cache_unusable") from None
    try:
        b64 = repo.get_run_artifact(run_id)
    except Exception:  # noqa: BLE001 - repository errors may hold URLs
        raise ArtifactError("database_unavailable") from None
    if not b64 or not isinstance(b64, str):
        raise ArtifactError("artifact_not_in_database")
    if len(b64) > MAX_B64_CHARS:
        raise ArtifactError("artifact_too_large")
    try:
        raw = base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError):
        raise ArtifactError("artifact_invalid") from None
    import hashlib
    if hashlib.sha256(raw).hexdigest() != sha:
        raise ArtifactError("artifact_hash_mismatch")
    try:
        Path(cache_dir).mkdir(parents=True, exist_ok=True, mode=0o700)
        fd, tmp = tempfile.mkstemp(dir=cache_dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(raw)
            os.chmod(tmp, 0o600)
            os.replace(tmp, path)             # atomic: no half-written cache file is ever visible
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
    except OSError:
        raise ArtifactError("artifact_cache_unusable") from None
    return path
