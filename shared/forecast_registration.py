"""Build and (explicitly) write the `runs` + `run_artifacts` rows for trained artifacts.

Pure planning is separated from writing. ``plan_registration`` only reads local
artifact files. ``execute_registration`` takes an injected client and is called
only by ``scripts/register_forecast_runs.py --execute``.

``runs.test_mae_c`` is NOT NULL in the applied 002 schema, and the real test
evaluation has not been run. A plan therefore carries ``test_mae_c = None`` until
a caller supplies measured values (``attach_test_mae``), and ``execute_registration``
refuses any plan item whose value is missing. Nothing here ever substitutes zero,
the validation MAE, or any other placeholder.

Provenance: ``hyperparameters.artifact_sha256`` records the SHA-256 of the exact
artifact file, so the serving API can confirm a configured run id describes the
file it loaded (runs are anon-readable; the hash is not sensitive).
"""
from __future__ import annotations

import base64
import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from shared.forecast_models import MODEL_TYPES
from shared.forecast_training import load_artifact
from shared.forecasting import ForecastDataError


RESULT_FORMAT = "seqsight-test-eval-v1"  # written by scripts/evaluate_forecast_artifacts.py


class RegistrationError(RuntimeError):
    pass


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class PlannedRun:
    key: str                               # artifact stem, e.g. "lstm_seed0"
    path: Path
    run: Dict[str, Any]                    # runs row WITHOUT test_mae_c
    artifact_sha256: str
    model_b64_chars: int
    test_mae_c: Optional[float] = None     # None == not yet measured
    _b64: str = field(default="", repr=False)

    def run_row(self) -> Dict[str, Any]:
        if self.test_mae_c is None:
            raise RegistrationError(f"{self.key}: test_mae_c is not available (final test evaluation pending)")
        return {**self.run, "test_mae_c": self.test_mae_c}


def default_keys(seeds=(0, 1, 2)) -> List[str]:
    return [f"{kind}_seed{s}" for kind in MODEL_TYPES for s in seeds]


def _plan_one(key: str, path: Path) -> PlannedRun:
    raw = path.read_bytes()
    _model, scaler, meta = load_artifact(path)
    sha = file_sha256(path)
    history = meta["history"]
    extra = meta.get("extra", {})
    run = {
        "model_type": meta["model_type"],
        "data_version": meta["data_version"],
        "seed": meta["seed"],
        "hyperparameters": {"model_config": meta["model_config"], "train_config": meta["train_config"],
                            "best_epoch": meta["best_epoch"], "artifact_sha256": sha},
        "preprocessing": scaler.to_dict(),
        "compute_budget": {
            "definition": "matched epochs and optimizer steps on identical windows in identical order",
            "epoch_cap": meta["train_config"]["max_epochs"], "epochs_run": len(history),
            "optimizer_steps": meta.get("optimizer_steps"), "batch_size": meta["train_config"]["batch_size"],
            "parameter_count": meta.get("parameter_count"), "train_windows": extra.get("train_windows_used"),
            "train_seconds": meta.get("train_seconds"),
        },
        "val_mae_c": meta["best_val_mae_c"],
    }
    b64 = base64.b64encode(raw).decode("ascii")
    return PlannedRun(key, path, run, sha, len(b64), _b64=b64)


def plan_registration(artifact_dir: Path, keys: Optional[List[str]] = None) -> List[PlannedRun]:
    keys = keys or default_keys()
    plans = []
    for key in keys:
        path = Path(artifact_dir) / f"{key}.pt"
        if not path.is_file():
            raise RegistrationError(f"artifact not found: {key}.pt")
        plans.append(_plan_one(key, path))
    return plans


def attach_test_mae(plans: List[PlannedRun], values: Mapping[str, Any]) -> None:
    """Attach MEASURED test MAE (degC) per artifact key. Every plan must be covered.

    ``values`` is either a plain ``{key: mae}`` mapping or an evaluation result file
    (``format == RESULT_FORMAT``). For the latter, each artifact's SHA-256 and data
    version must match the plan, so MAE measured for a different file is rejected.
    """
    if isinstance(values, Mapping) and values.get("format") == RESULT_FORMAT:
        models = values.get("models") or {}
        for p in plans:
            m = models.get(p.key)
            if m is None:
                raise RegistrationError(f"no test MAE supplied for: {p.key}")
            if m.get("artifact_sha256") != p.artifact_sha256:
                raise RegistrationError(f"{p.key}: result was measured on a different artifact file")
            if values.get("data_version") != p.run["data_version"]:
                raise RegistrationError(f"{p.key}: result data_version differs from the artifact's")
        values = values.get("test_mae_c") or {}
    missing = [p.key for p in plans if p.key not in values]
    if missing:
        raise RegistrationError(f"no test MAE supplied for: {', '.join(missing)}")
    for p in plans:
        v = values[p.key]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
            raise RegistrationError(f"{p.key}: test MAE must be a finite number >= 0")
        p.test_mae_c = float(v)


def execute_registration(client, plans: List[PlannedRun]) -> List[Dict[str, Any]]:
    """Insert runs + run_artifacts. Skips artifacts already registered (same sha). Not transactional:
    if the artifact insert fails, the just-created run is deleted again (best effort)."""
    for p in plans:                      # refuse BEFORE writing anything
        p.run_row()
    out = []
    for p in plans:
        existing = (client.table("runs").select("id,hyperparameters").eq("model_type", p.run["model_type"])
                    .eq("data_version", p.run["data_version"]).eq("seed", p.run["seed"]).execute().data)
        dup = [r["id"] for r in existing
               if (r.get("hyperparameters") or {}).get("artifact_sha256") == p.artifact_sha256]
        if dup:
            out.append({"key": p.key, "run_id": dup[0], "action": "already_registered"})
            continue
        run_id = client.table("runs").insert(p.run_row()).execute().data[0]["id"]
        try:
            client.table("run_artifacts").insert({"run_id": run_id, "model_b64": p._b64}).execute()
        except Exception as e:  # noqa: BLE001
            try:
                client.table("runs").delete().eq("id", run_id).execute()
                cleanup = "run row removed"
            except Exception:  # noqa: BLE001
                cleanup = f"ORPHAN run row {run_id} remains; delete it manually"
            raise RegistrationError(f"{p.key}: run_artifacts insert failed ({type(e).__name__}); {cleanup}") from None
        out.append({"key": p.key, "run_id": run_id, "action": "registered"})
    return out
