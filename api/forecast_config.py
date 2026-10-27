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
"""
from __future__ import annotations

import os
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


@dataclass(frozen=True)
class ForecastSettings:
    models: Dict[str, ModelSetting] = field(default_factory=dict)

    @property
    def artifacts_are_development_defaults(self) -> bool:
        return any(m.path_is_default for m in self.models.values())

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
        return cls(models)
