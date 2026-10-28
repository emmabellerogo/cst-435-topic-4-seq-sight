"""Environment-driven configuration for the forecasting service.

Artifact paths default to the seed-0 runs under ``artifacts/epoch50/final``. That
is a DEVELOPMENT default only: it is not a final deployment selection (the
final choice waits for the pending test evaluation).

Run ids have NO default. Each served artifact must be explicitly mapped to the
database run that records it; an unmapped artifact is never served.

    FORECAST_LSTM_ARTIFACT        path to the lstm artifact (.pt)
    FORECAST_ATTENTION_ARTIFACT   path to the attention artifact (.pt)
    FORECAST_LSTM_RUN_ID          runs.id registered for that lstm artifact
    FORECAST_ATTENTION_RUN_ID     runs.id registered for that attention artifact
    FORECAST_ARTIFACT_SOURCE      "local" (default) or "supabase"
    FORECAST_ARTIFACT_CACHE_DIR   cache for downloaded artifacts (supabase source only)
    FORECAST_RETRY_COOLDOWN_S     minimum seconds between registration/download retries

With ``FORECAST_ARTIFACT_SOURCE=supabase`` the artifact paths are ignored: each
artifact is downloaded from ``run_artifacts`` for its configured run id and checked
against the SHA-256 in that run's metadata (see api/artifact_store.py).
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Mapping, Optional

from shared.forecast_models import MODEL_TYPES

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_DIR = REPO / "artifacts" / "epoch50" / "final"


@dataclass(frozen=True)
class ModelSetting:
    artifact_path: Path
    run_id: Optional[int]               # None -> not configured
    run_id_error: Optional[str] = None  # set when the env value was present but invalid
    path_is_default: bool = True


DEFAULT_CACHE_DIR = Path(tempfile.gettempdir()) / "seqsight-artifacts"
SOURCES = ("local", "supabase")


@dataclass(frozen=True)
class ForecastSettings:
    models: Dict[str, ModelSetting] = field(default_factory=dict)
    artifact_source: str = "local"
    cache_dir: Path = DEFAULT_CACHE_DIR
    retry_cooldown_s: float = 30.0
    config_error: Optional[str] = None      # set when an environment value is invalid

    @property
    def artifacts_are_development_defaults(self) -> bool:
        return self.artifact_source == "local" and any(m.path_is_default for m in self.models.values())

    @classmethod
    def from_env(cls, environ: Optional[Mapping[str, str]] = None) -> "ForecastSettings":
        env = os.environ if environ is None else environ
        models: Dict[str, ModelSetting] = {}
        for kind in MODEL_TYPES:
            prefix = f"FORECAST_{kind.upper()}"
            raw_path = env.get(f"{prefix}_ARTIFACT")
            path = Path(raw_path) if raw_path else DEFAULT_ARTIFACT_DIR / f"{kind}_seed0.pt"
            if not path.is_absolute():
                path = REPO / path
            raw_id = (env.get(f"{prefix}_RUN_ID") or "").strip()
            run_id, err = None, None
            if raw_id:
                if raw_id.isascii() and raw_id.isdigit() and int(raw_id) >= 1:
                    run_id = int(raw_id)
                else:
                    err = "run_id_invalid"
            models[kind] = ModelSetting(path, run_id, err, path_is_default=not raw_path)
        source = (env.get("FORECAST_ARTIFACT_SOURCE") or "local").strip().lower()
        err = None if source in SOURCES else "artifact_source_invalid"
        cache = Path(env.get("FORECAST_ARTIFACT_CACHE_DIR") or DEFAULT_CACHE_DIR)
        try:
            cooldown = float(env.get("FORECAST_RETRY_COOLDOWN_S") or 30.0)
            if not 0 <= cooldown <= 3600:
                raise ValueError
        except ValueError:
            cooldown, err = 30.0, err or "retry_cooldown_invalid"
        return cls(models, source if source in SOURCES else "local", cache, cooldown, err)
