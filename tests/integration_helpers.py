"""Helpers for the opt-in Supabase integration test (tests/test_supabase_integration.py).

Everything here is pure or takes an injected client, so it is tested offline
(tests/test_integration_helpers.py). Nothing in this module reads a ``.env`` file, and
nothing reads the production variable names (``SUPABASE_URL`` / ``SUPABASE_SERVICE_KEY``)
except to REFUSE a configuration that points at the same project.

Configuration comes only from ``SEQSIGHT_IT_*`` variables that a person set deliberately.
No credential, URL or run id has a default, and run ids are never discovered or guessed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from api.forecast_repo import SupabaseForecastRepository
from api.forecast_service import canonical_input_hash
from shared.forecast_models import INPUT_HOURS, MODEL_TYPES, TARGET_HOURS

ENABLE = "SEQSIGHT_IT_ENABLE"
URL = "SEQSIGHT_IT_SUPABASE_URL"
KEY = "SEQSIGHT_IT_SUPABASE_SERVICE_KEY"
RUN_IDS = {"lstm": "SEQSIGHT_IT_LSTM_RUN_ID", "attention": "SEQSIGHT_IT_ATTENTION_RUN_ID"}
SOURCE = "SEQSIGHT_IT_ARTIFACT_SOURCE"                      # "supabase" (default) or "local"
LOCAL_ARTIFACTS = {"lstm": "SEQSIGHT_IT_LSTM_ARTIFACT", "attention": "SEQSIGHT_IT_ATTENTION_ARTIFACT"}
ALLOW_SAME_PROJECT = "SEQSIGHT_IT_ALLOW_PRIMARY_PROJECT"    # explicit override of the safety guard
PRIMARY_URL = "SUPABASE_URL"                                # only compared, never used

FLOAT_REL = 1e-9
ATTENTION_ROW_SUM_TOL = 1e-6


class IntegrationConfigError(RuntimeError):
    """Configuration is present but unsafe or inconsistent (a failure, not a skip)."""


@dataclass(frozen=True)
class IntegrationConfig:
    url: str
    service_key: str
    run_ids: Dict[str, int]
    artifact_source: str
    local_artifacts: Dict[str, Path] = field(default_factory=dict)

    def __repr__(self) -> str:                  # never leak the key or URL into a failure report
        return f"IntegrationConfig(run_ids={self.run_ids}, artifact_source={self.artifact_source!r})"


def missing_config(environ: Mapping[str, str]) -> List[str]:
    """Names of absent variables; empty means the integration test is configured."""
    if environ.get(ENABLE) != "1":
        return [f"{ENABLE}=1"]
    need = [URL, KEY, *RUN_IDS.values()]
    if (environ.get(SOURCE) or "supabase").lower() == "local":
        need += list(LOCAL_ARTIFACTS.values())
    return [n for n in need if not (environ.get(n) or "").strip()]


def load_config(environ: Mapping[str, str]) -> IntegrationConfig:
    """Validate a fully present configuration. Call only when ``missing_config`` is empty."""
    if missing_config(environ):
        raise IntegrationConfigError("integration configuration is incomplete")
    run_ids = {}
    for kind, name in RUN_IDS.items():
        raw = environ[name].strip()
        if not (raw.isascii() and raw.isdigit() and int(raw) >= 1):
            raise IntegrationConfigError(f"{name} must be a positive integer")
        run_ids[kind] = int(raw)
    if run_ids["lstm"] == run_ids["attention"]:
        raise IntegrationConfigError("the lstm and attention run ids must differ")
    source = (environ.get(SOURCE) or "supabase").strip().lower()
    if source not in ("supabase", "local"):
        raise IntegrationConfigError(f"{SOURCE} must be 'supabase' or 'local'")
    url = environ[URL].strip()
    primary = (environ.get(PRIMARY_URL) or "").strip()
    if primary and url.rstrip("/") == primary.rstrip("/") and environ.get(ALLOW_SAME_PROJECT) != "1":
        raise IntegrationConfigError(
            f"{URL} equals {PRIMARY_URL}: use a separate test project, or set {ALLOW_SAME_PROJECT}=1 "
            "to accept writing (and deleting) test prediction rows in that project")
    local = {k: Path(environ[v]) for k, v in LOCAL_ARTIFACTS.items()} if source == "local" else {}
    return IntegrationConfig(url, environ[KEY].strip(), run_ids, source, local)


def forecast_env(cfg: IntegrationConfig, cache_dir: Path) -> Dict[str, str]:
    """The ``FORECAST_*`` mapping for ``ForecastSettings.from_env`` (explicit, not os.environ)."""
    env = {"FORECAST_ARTIFACT_SOURCE": cfg.artifact_source, "FORECAST_ARTIFACT_CACHE_DIR": str(cache_dir),
           "FORECAST_RETRY_COOLDOWN_S": "0"}
    for kind in MODEL_TYPES:
        env[f"FORECAST_{kind.upper()}_RUN_ID"] = str(cfg.run_ids[kind])
        if cfg.artifact_source == "local":
            env[f"FORECAST_{kind.upper()}_ARTIFACT"] = str(cfg.local_artifacts[kind])
    return env


def make_client(cfg: IntegrationConfig):
    from supabase import create_client
    return create_client(cfg.url, cfg.service_key)


# ------------------------------------------------------------------ inputs

def sample_history(model_type: str) -> List[float]:
    """A plausible, deterministic 72 h history in degC. Differs per model type so the two
    tests log different input hashes. Synthetic: it is a connectivity/logging fixture."""
    phase = {"lstm": 0.0, "attention": 1.3}[model_type]
    return [round(9.0 + 6.0 * math.sin(2 * math.pi * (h / 24.0) + phase) + 0.01 * h, 4)
            for h in range(INPUT_HOURS)]


# --------------------------------------------------- exact-row bookkeeping

TrackedRow = Tuple[int, str, int]       # (prediction id, input_hash, served_by_run_id)


class TrackingRepository:
    """Wraps a real repository and records exactly which prediction rows it inserted."""

    def __init__(self, inner):
        self._inner = inner
        self.created: List[TrackedRow] = []

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def insert_prediction(self, *, input_hash, forecast, attention, model_type, served_by_run_id):
        row = self._inner.insert_prediction(input_hash=input_hash, forecast=forecast, attention=attention,
                                            model_type=model_type, served_by_run_id=served_by_run_id)
        self.created.append((int(row["id"]), input_hash, int(served_by_run_id)))
        return row


def cleanup_predictions(client, tracked: List[TrackedRow]) -> List[str]:
    """Delete exactly the tracked rows, one by one, each matched on id AND hash AND run.

    Never truncates, never filters on anything broader than a single primary key. Returns
    problems (an empty list means every tracked row was removed or already absent); it
    does not raise, so a cleanup problem cannot mask the test's own failure.
    """
    problems = []
    for pid, input_hash, run_id in tracked:
        if isinstance(pid, bool) or not isinstance(pid, int) or pid < 1:
            problems.append(f"refused to delete: invalid id {pid!r}")
            continue
        try:
            (client.table("predictions").delete().eq("id", pid).eq("input_hash", input_hash)
             .eq("served_by_run_id", run_id).execute())
        except Exception as e:  # noqa: BLE001
            problems.append(f"prediction {pid} could not be deleted ({type(e).__name__}); remove it manually")
    return problems


# ------------------------------------------------------------ verification

def _close(a, b) -> bool:
    return bool(np.allclose(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64),
                            rtol=FLOAT_REL, atol=1e-12))


def verify_logged_prediction(response: Mapping[str, Any], stored: Mapping[str, Any],
                             run: Mapping[str, Any], *, model_type: str, history_c: List[float],
                             expected_run_id: int, expected_data_version: Optional[str] = None) -> List[str]:
    """Compare an API response, the stored ``predictions`` row and its ``runs`` row.

    Returns a list of mismatches (empty == all checks hold). Values only; no secrets.
    """
    bad = []

    def check(ok: bool, what: str) -> None:
        if not ok:
            bad.append(what)

    check(response.get("model_type") == model_type, "response model_type")
    check(response.get("served_by_run_id") == expected_run_id, "response served_by_run_id != configured run id")
    check(stored.get("id") == response.get("prediction_id"), "stored id != response prediction_id")
    check(stored.get("input_hash") == response.get("input_hash"), "stored input_hash != response input_hash")
    check(response.get("input_hash") == canonical_input_hash(history_c), "response input_hash != hash of the sent history")
    check(stored.get("model_type") == model_type, "stored model_type")
    check(stored.get("served_by_run_id") == expected_run_id, "stored served_by_run_id != configured run id")
    f_resp, f_db = response.get("forecast_c"), stored.get("forecast")
    check(isinstance(f_db, list) and len(f_db) == TARGET_HOURS and isinstance(f_resp, list)
          and len(f_resp) == TARGET_HOURS and _close(f_resp, f_db), "stored forecast != response forecast_c")

    a_resp, a_db = response.get("attention"), stored.get("attention")
    if model_type == "lstm":
        check(a_resp is None, "lstm response attention must be null")
        check(a_db is None, "lstm stored attention must be NULL")
    else:
        for label, a in (("response", a_resp), ("stored", a_db)):
            arr = None
            try:
                arr = np.asarray(a, dtype=np.float64)
            except (TypeError, ValueError):
                pass
            if arr is None or arr.shape != (INPUT_HOURS, INPUT_HOURS) or not np.isfinite(arr).all():
                bad.append(f"{label} attention is not a finite 72x72 matrix")
            elif not np.allclose(arr.sum(axis=1), 1.0, atol=ATTENTION_ROW_SUM_TOL):
                bad.append(f"{label} attention rows do not sum to 1")
        if not bad or all("attention" not in b for b in bad):
            check(_close(a_resp, a_db), "stored attention != response attention")

    check(run.get("id") == expected_run_id, "joined run id")
    check(run.get("id") == stored.get("served_by_run_id"), "run row is not the one the prediction references")
    check(run.get("model_type") == model_type == stored.get("model_type"), "run model_type != prediction model_type")
    if expected_data_version is not None:
        check(run.get("data_version") == expected_data_version, "run data_version")
    return bad


def fetch_logged(client, prediction_id: int) -> Tuple[Optional[dict], Optional[dict]]:
    """Read the stored prediction row and the runs row it references (two exact-key reads)."""
    rows = client.table("predictions").select("*").eq("id", prediction_id).limit(1).execute().data
    if not rows:
        return None, None
    runs = client.table("runs").select("id,model_type,data_version,seed,hyperparameters") \
        .eq("id", rows[0]["served_by_run_id"]).limit(1).execute().data
    return rows[0], (runs[0] if runs else None)
