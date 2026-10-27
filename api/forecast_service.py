"""Forecast inference and model/run registration checks. No web framework, no SQL.

``ForecastService`` owns the loaded artifacts. It touches the database only
through the injected ``ForecastRepository`` and only to (a) verify that each
configured run id really describes the artifact being served and (b) write
audit rows.

Serving rules
-------------
* A model type is served only if its artifact loaded AND its configured run id
  exists in ``runs`` AND that run's persisted metadata matches the artifact
  (model type, data version, seed, scaler, validation MAE, artifact SHA-256).
  Otherwise the type is "not ready" and requests get an explicit 503.
  Run ids are never invented, guessed or defaulted.
* Inference runs in eval mode under ``torch.inference_mode`` using the scaler
  stored in the artifact, and only after the input is standardized with it.
* Artifacts are trusted local files (``weights_only`` loading, see
  ``shared.forecast_training.load_artifact``).

Canonical input hash (``seqsight-input-v1``)
--------------------------------------------
SHA-256 (lowercase hex) of the UTF-8 text::

    seqsight-input-v1|degC|72|<v0>,<v1>,...,<v71>

where every value is converted to a Python float (float64) and written with
``repr`` (the shortest string that round-trips), with ``-0.0`` normalized to
``0.0``. So ``5`` and ``5.0`` hash identically, key order and JSON whitespace
play no part, and the model type is not part of the hash (it is the INPUT hash).
"""
from __future__ import annotations

import hashlib
import math
import subprocess
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from api.forecast_config import ForecastSettings
from shared.forecast_models import INPUT_HOURS, MODEL_TYPES, TARGET_HOURS
from shared.forecast_training import load_artifact
from shared.forecast_registration import file_sha256
from shared.forecasting import Standardizer

HASH_FORMAT = "seqsight-input-v1"
API_VERSION = "2.0.0"


class ServeError(Exception):
    """A request cannot be served. ``code`` is machine-readable; the message is user-safe."""

    def __init__(self, code: str, message: str, status: int = 503):
        super().__init__(message)
        self.code, self.message, self.status = code, message, status


def canonical_input_hash(values_c: Sequence[float]) -> str:
    if len(values_c) != INPUT_HOURS:
        raise ValueError(f"expected {INPUT_HOURS} values, got {len(values_c)}")
    parts = []
    for v in values_c:
        f = float(v)
        if not math.isfinite(f):
            raise ValueError("cannot hash non-finite values")
        parts.append(repr(0.0 if f == 0.0 else f))
    text = f"{HASH_FORMAT}|degC|{INPUT_HOURS}|" + ",".join(parts)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class LoadedModel:
    model_type: str
    model: torch.nn.Module
    scaler: Standardizer
    meta: Dict[str, Any]
    artifact_sha256: str


@dataclass
class Forecast:
    input_hash: str
    forecast_c: List[float]
    attention: Optional[List[List[float]]]


def load_model(model_type: str, path) -> LoadedModel:
    try:
        sha = file_sha256(path)
    except OSError:
        raise ServeError("artifact_not_found", f"{model_type} artifact file is missing or unreadable") from None
    try:
        model, scaler, meta = load_artifact(path)
    except Exception:  # noqa: BLE001 - torch/pickle errors may quote the path
        raise ServeError("artifact_invalid", f"{model_type} artifact could not be loaded") from None
    if meta["model_type"] != model_type:
        raise ServeError("artifact_type_mismatch",
                         f"artifact configured for {model_type} holds a {meta['model_type']} model")
    return LoadedModel(model_type, model, scaler, meta, sha)


def run_forecasts(loaded: LoadedModel, histories_c: Sequence[Sequence[float]]) -> List[Forecast]:
    """Forecast every history in one forward pass. Eval mode, no gradients."""
    x_c = np.asarray(histories_c, dtype=np.float64)
    if x_c.ndim != 2 or x_c.shape[1] != INPUT_HOURS or not np.isfinite(x_c).all():
        raise ServeError("bad_input", f"expected finite histories of {INPUT_HOURS} values", status=422)
    x = torch.from_numpy(loaded.scaler.transform(x_c).astype(np.float32)).unsqueeze(-1)  # [B, 72, 1]
    model = loaded.model
    model.eval()
    with torch.inference_mode():
        z, attn = model(x)
    pred_c = loaded.scaler.inverse_transform(z.double().numpy())                      # [B, 24] degC
    if pred_c.shape[1] != TARGET_HOURS or not np.isfinite(pred_c).all():
        raise ServeError("non_finite_forecast", "model produced a non-finite forecast", status=500)
    attn_np = None
    if attn is not None:
        attn_np = attn.double().numpy()
        if not np.isfinite(attn_np).all():
            raise ServeError("non_finite_forecast", "model produced non-finite attention", status=500)
    return [Forecast(canonical_input_hash(x_c[i]), pred_c[i].tolist(),
                     None if attn_np is None else attn_np[i].tolist())
            for i in range(x_c.shape[0])]


def _close(a, b, rel=1e-9) -> bool:
    return (isinstance(a, (int, float)) and not isinstance(a, bool)
            and isinstance(b, (int, float)) and math.isclose(a, b, rel_tol=rel, abs_tol=1e-12))


def check_run_matches(loaded: LoadedModel, row: Dict[str, Any]) -> Optional[str]:
    """Return None if run row ``row`` describes ``loaded``, else a reason code."""
    meta = loaded.meta
    if row.get("model_type") != loaded.model_type:
        return "run_mismatch_model_type"
    if row.get("data_version") != meta["data_version"]:
        return "run_mismatch_data_version"
    if row.get("seed") != meta["seed"]:
        return "run_mismatch_seed"
    pre = row.get("preprocessing") or {}
    if not (_close(pre.get("mean"), loaded.scaler.mean) and _close(pre.get("std"), loaded.scaler.std)):
        return "run_mismatch_scaler"
    if not _close(row.get("val_mae_c"), meta["best_val_mae_c"]):
        return "run_mismatch_val_mae"
    if (row.get("hyperparameters") or {}).get("artifact_sha256") != loaded.artifact_sha256:
        return "run_mismatch_artifact"
    return None


@dataclass
class _Slot:
    model_type: str
    run_id: Optional[int]
    loaded: Optional[LoadedModel] = None
    reason: Optional[str] = None     # why this type is not ready; None when ready
    ready: bool = False


class ForecastService:
    def __init__(self, settings: ForecastSettings, repo):
        self.settings, self.repo = settings, repo
        self._lock = threading.Lock()
        self._slots: Dict[str, _Slot] = {
            k: _Slot(k, settings.models[k].run_id, reason="not_loaded") for k in MODEL_TYPES}

    # ------------------------------------------------------------ startup
    def startup(self) -> None:
        """Load every artifact once, then try to verify run registration."""
        for kind, slot in self._slots.items():
            cfg = self.settings.models[kind]
            try:
                slot.loaded = load_model(kind, cfg.artifact_path)
                slot.reason = None
            except ServeError as e:
                slot.reason = e.code
        self.verify_registration()

    def verify_registration(self) -> None:
        """(Re)check unverified models against persisted runs. Cheap when all are ready."""
        with self._lock:
            for kind, slot in self._slots.items():
                if slot.ready or slot.loaded is None:
                    continue
                cfg = self.settings.models[kind]
                if cfg.run_id_error:
                    slot.reason = cfg.run_id_error
                    continue
                if slot.run_id is None:
                    slot.reason = "run_id_not_configured"
                    continue
                try:
                    row = self.repo.get_run(slot.run_id)
                except Exception:  # noqa: BLE001 - any repository failure
                    slot.reason = "database_unavailable"
                    continue
                if row is None:
                    slot.reason = "run_not_found"
                    continue
                mismatch = check_run_matches(slot.loaded, row)
                slot.reason, slot.ready = mismatch, mismatch is None

    # ---------------------------------------------------------- accessors
    def require_ready(self, model_type: str) -> LoadedModel:
        slot = self._slots[model_type]
        if not slot.ready:
            self.verify_registration()   # DB may have been down at startup or the run registered since
        if not slot.ready:
            raise ServeError(slot.reason or "not_ready", f"{model_type} model is not ready ({slot.reason})")
        return slot.loaded

    def run_id(self, model_type: str) -> int:
        return self._slots[model_type].run_id

    def served_run_ids(self) -> set:
        return {s.run_id for s in self._slots.values() if s.ready}

    def readiness(self) -> Dict[str, Any]:
        self.verify_registration()
        models = {k: {"artifact_loaded": s.loaded is not None,
                      "run_registered": s.ready,
                      "ready": s.ready, "reason": s.reason}
                  for k, s in self._slots.items()}
        return {"ready": all(m["ready"] for m in models.values()),
                "database": self.repo.ping(), "models": models}

    def version_info(self) -> Dict[str, Any]:
        models = {}
        for k, s in self._slots.items():
            m = s.loaded
            models[k] = {
                "ready": s.ready,
                "serving_run_id": s.run_id if s.ready else None,
                "seed": m.meta["seed"] if m else None,
                "data_version": m.meta["data_version"] if m else None,
                "best_epoch": m.meta["best_epoch"] if m else None,
                "best_val_mae_c": m.meta["best_val_mae_c"] if m else None,
                "parameter_count": m.meta.get("parameter_count") if m else None,
                "artifact_sha256_prefix": m.artifact_sha256[:12] if m else None,
                "trained_with_torch": m.meta.get("torch_version") if m else None,
            }
        return models


def git_sha() -> str:
    import os
    if os.environ.get("RENDER_GIT_COMMIT"):
        return os.environ["RENDER_GIT_COMMIT"][:7]
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:  # noqa: BLE001
        return "unknown"
