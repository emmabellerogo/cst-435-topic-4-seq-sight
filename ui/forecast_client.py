"""Pure helpers for the Seq Sight UI: API client, parsing, validation, Supabase reads.

No Streamlit and no torch here, so everything is unit-testable with mocked
responses (tests/test_ui_helpers.py). The request/response shapes mirror
shared/forecast_schemas.py and docs/forecasting-api.md; they are copied, not
imported, because the UI is deployed on its own with only ui/requirements.txt.
"""
from __future__ import annotations

import base64
import json
import math
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import requests

INPUT_HOURS = 72
FORECAST_HOURS = 24
MAX_ABS_TEMPERATURE_C = 1000.0   # same numerical guard as the API
MODEL_TYPES = ("lstm", "attention")
MODEL_LABELS = {"lstm": "LSTM (baseline)", "attention": "Self-attention"}
CONNECT_TIMEOUT_S = 5
READ_TIMEOUT_S = 60              # covers a Render cold start that answers late
STATUS_READ_TIMEOUT_S = 15

# 503 reason codes from the API that mean "no model is registered/ready yet".
PENDING_CODES = {"run_id_not_configured", "run_id_invalid", "run_not_found", "database_unavailable",
                 "artifact_not_found", "artifact_invalid", "artifact_type_mismatch", "not_ready"}


# ------------------------------------------------------------------ input parsing
def parse_history(text: str) -> List[float]:
    """Parse pasted temperatures: separated by commas, semicolons, spaces or newlines."""
    tokens = [t for t in re.split(r"[,;\s]+", (text or "").strip()) if t]
    if len(tokens) != INPUT_HOURS:
        raise ValueError(f"Exactly {INPUT_HOURS} hourly temperatures are required; found {len(tokens)}.")
    out = []
    for i, t in enumerate(tokens, 1):
        if t.lower() in {"true", "false"}:
            raise ValueError(f"Value {i} ({t!r}) is not a number.")
        try:
            v = float(t)
        except ValueError:
            raise ValueError(f"Value {i} ({t!r}) is not a number.") from None
        if not math.isfinite(v):
            raise ValueError(f"Value {i} ({t!r}) is not finite.")
        if abs(v) > MAX_ABS_TEMPERATURE_C:
            raise ValueError(f"Value {i} ({t!r}) is outside ±{MAX_ABS_TEMPERATURE_C:g} °C.")
        out.append(v)
    return out


def format_history(values: Sequence[float]) -> str:
    return ", ".join(f"{v:.1f}" for v in values)


def demo_history(kind: str) -> List[float]:
    """Synthetic DEMONSTRATION inputs, not measurements."""
    t = np.arange(INPUT_HOURS)
    if kind == "daily_cycle":
        v = 12.0 + 6.0 * np.sin(2 * np.pi * (t - 9) / 24)
    elif kind == "warming_trend":
        v = 8.0 + 0.08 * t + 3.0 * np.sin(2 * np.pi * (t - 9) / 24)
    elif kind == "constant":
        v = np.full(INPUT_HOURS, 10.0)
    else:
        raise ValueError(f"unknown demo kind {kind!r}")
    return [round(float(x), 2) for x in v]


DEMO_LABELS = {"daily_cycle": "Synthetic daily cycle (sine wave)",
               "warming_trend": "Synthetic cycle + warming trend",
               "constant": "Constant 10 °C"}


# --------------------------------------------------------------- response parsing
class ApiError(Exception):
    """kind: validation | pending | unavailable | cold_start | connection | server | bad_response"""

    def __init__(self, kind: str, message: str, status: Optional[int] = None, code: Optional[str] = None):
        super().__init__(message)
        self.kind, self.message, self.status, self.code = kind, message, status, code


@dataclass
class Prediction:
    prediction_id: int
    model_type: str
    served_by_run_id: int
    input_hash: str
    forecast_c: List[float]
    attention: Optional[np.ndarray]    # [72, 72] or None


def _finite_list(values: Any, n: int, what: str) -> List[float]:
    if (not isinstance(values, list) or len(values) != n
            or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in values)):
        raise ApiError("bad_response", f"The API returned a malformed {what}.")
    return [float(v) for v in values]


def parse_attention(raw: Any) -> Optional[np.ndarray]:
    if raw is None:
        return None
    try:
        a = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError):
        raise ApiError("bad_response", "The API returned malformed attention weights.") from None
    if a.shape != (INPUT_HOURS, INPUT_HOURS) or not np.isfinite(a).all():
        raise ApiError("bad_response", "The API returned malformed attention weights.")
    return a


def parse_predict_response(body: Any, requested_model: str) -> Prediction:
    try:
        pid, run_id = body["prediction_id"], body["served_by_run_id"]
        model, h = body["model_type"], body["input_hash"]
    except (TypeError, KeyError):
        raise ApiError("bad_response", "The API response is missing expected fields.") from None
    if model != requested_model or isinstance(pid, bool) or not isinstance(pid, int) \
            or isinstance(run_id, bool) or not isinstance(run_id, int) \
            or not isinstance(h, str) or not re.fullmatch(r"[0-9a-f]{64}", h):
        raise ApiError("bad_response", "The API response does not match the request.")
    att = parse_attention(body.get("attention"))
    if (model == "attention") != (att is not None):
        raise ApiError("bad_response", "Attention weights are present only for the attention model.")
    return Prediction(pid, model, run_id, h, _finite_list(body.get("forecast_c"), FORECAST_HOURS, "forecast"), att)


def _error_from_response(status: int, body: Any) -> ApiError:
    detail = body.get("detail") if isinstance(body, dict) else None
    if status == 422:
        msgs = []
        if isinstance(detail, list):
            for e in detail[:5]:
                if isinstance(e, dict):
                    loc = ".".join(str(x) for x in e.get("loc", []) if x != "body")
                    msgs.append(f"{loc}: {e.get('msg', 'invalid')}")
        return ApiError("validation", "The API rejected the input. " + "; ".join(msgs), status)
    if isinstance(detail, dict) and "code" in detail:
        code = str(detail["code"])
        if status == 503 and code in PENDING_CODES:
            return ApiError("pending", "No registered model is ready to serve this request yet.", status, code)
        if status == 503:
            return ApiError("unavailable", str(detail.get("message", "The service is unavailable.")), status, code)
    if status in (502, 503, 504):     # proxy error pages while a free-tier instance spins up
        return ApiError("cold_start", "The API is not answering yet; it may be waking up.", status)
    return ApiError("server", f"The API returned an unexpected error (HTTP {status}).", status)


class ForecastApi:
    """Every UI->API call goes through ``_request`` (finite timeouts, uniform errors)."""

    def __init__(self, base_url: str, http=requests, sleep: Callable[[float], None] = time.sleep):
        self.base_url, self._http, self._sleep = base_url.rstrip("/"), http, sleep

    def _request(self, method: str, path: str, read_timeout: float = READ_TIMEOUT_S, **kw):
        try:
            r = self._http.request(method, self.base_url + path, timeout=(CONNECT_TIMEOUT_S, read_timeout), **kw)
        except requests.exceptions.Timeout:
            raise ApiError("cold_start", "The API did not answer in time; a free-tier service may be waking "
                                         "up (this can take about a minute). Try again shortly.") from None
        except requests.exceptions.RequestException:
            raise ApiError("connection", "Could not reach the API. Check API_URL and your network.") from None
        try:
            body = r.json()
        except ValueError:
            body = None
        return r.status_code, body

    def predict(self, model_type: str, history_c: Sequence[float]) -> Prediction:
        status, body = self._request("POST", "/predict", json={"model_type": model_type,
                                                              "history_c": list(history_c)})
        if status != 200:
            raise _error_from_response(status, body)
        return parse_predict_response(body, model_type)

    def health(self) -> Dict[str, Any]:
        """Returns the /healthz body for 200 and 503 (503 means 'not ready', still informative)."""
        status, body = self._request("GET", "/healthz", read_timeout=STATUS_READ_TIMEOUT_S)
        if status in (200, 503) and isinstance(body, dict) and "models" in body:
            return {"http_status": status, **body}
        raise _error_from_response(status, body)

    def wait_until_up(self, max_wait_s: float = 90, interval_s: float = 5) -> Dict[str, Any]:
        """Poll /healthz until the service answers (ready or not). Bounded."""
        waited, last = 0.0, None
        while True:
            try:
                return self.health()
            except ApiError as e:
                last = e
                if e.kind not in ("cold_start", "connection") or waited >= max_wait_s:
                    raise last
            self._sleep(interval_s)
            waited += interval_s


def describe_error(e: ApiError) -> str:
    if e.kind == "pending":
        return ("Predictions are not available yet: the trained models have not been registered in the "
                f"database (reason: `{e.code}`). The API refuses to serve a model that is not registered "
                "and audited. See the setup notes in docs/forecasting-api.md.")
    return e.message


def health_summary(h: Dict[str, Any]) -> List[str]:
    lines = [f"API reports **{h.get('status')}**; database reachable: {h.get('database')}."]
    for name, m in (h.get("models") or {}).items():
        lines.append(f"- {name}: {'ready' if m.get('ready') else 'not ready'}"
                     + ("" if m.get("ready") else f" (`{m.get('reason')}`)"))
    return lines


# ----------------------------------------------------------- Supabase (anon key)
PREDICTION_COLUMNS = "id,created_at,model_type,served_by_run_id,input_hash,forecast"   # no attention
SORTABLE = {"created_at": "Time", "id": "Prediction ID", "model_type": "Model", "served_by_run_id": "Run"}
MAX_PAGE = 100
RUN_COLUMNS = ("id,model_type,data_version,seed,hyperparameters,preprocessing,compute_budget,"
               "val_mae_c,test_mae_c,created_at")


def check_anon_key(key: str) -> None:
    """Refuse a key that is detectably a service-role / secret key."""
    if not isinstance(key, str) or not key.strip():
        raise ValueError("SUPABASE_ANON_KEY is not set.")
    if key.startswith("sb_secret_"):
        raise ValueError("SUPABASE_ANON_KEY holds a secret key; use the public anon key only.")
    parts = key.split(".")
    if len(parts) == 3:
        try:
            payload = json.loads(base64.urlsafe_b64decode(parts[1] + "=" * (-len(parts[1]) % 4)))
        except Exception:  # noqa: BLE001 - not a decodable JWT; cannot tell, let Supabase judge
            return
        if isinstance(payload, dict) and payload.get("role") == "service_role":
            raise ValueError("SUPABASE_ANON_KEY holds a service-role key. Remove it; the UI must use the anon key.")


def fetch_predictions(client, sort_by: str = "created_at", descending: bool = True,
                      limit: int = 25, offset: int = 0) -> List[Dict[str, Any]]:
    """One page of audited predictions WITHOUT attention (72x72 per row is too heavy for a list)."""
    if sort_by not in SORTABLE:
        raise ValueError(f"cannot sort by {sort_by!r}")
    if not 1 <= limit <= MAX_PAGE or offset < 0:
        raise ValueError("limit must be 1..100 and offset >= 0")
    q = client.table("predictions").select(PREDICTION_COLUMNS).order(sort_by, desc=descending)
    if sort_by != "id":
        q = q.order("id", desc=True)          # deterministic paging for ties
    return q.range(offset, offset + limit - 1).execute().data


def fetch_attention(client, prediction_id: int) -> Optional[np.ndarray]:
    """Attention for ONE selected prediction (None for the LSTM)."""
    rows = client.table("predictions").select("id,attention").eq("id", int(prediction_id)).limit(1).execute().data
    if not rows:
        raise ValueError("prediction not found")
    return parse_attention(rows[0].get("attention"))


def fetch_runs(client, limit: int = 100) -> List[Dict[str, Any]]:
    return (client.table("runs").select(RUN_COLUMNS).order("created_at", desc=True).order("id", desc=True)
            .limit(min(limit, MAX_PAGE)).execute().data)


def summarize_runs(runs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Per-model mean / sample SD / n of validation and test MAE over the runs present. No ranking."""
    out = []
    for kind in MODEL_TYPES:
        rs = [r for r in runs if r.get("model_type") == kind]
        if not rs:
            continue
        row: Dict[str, Any] = {"model": MODEL_LABELS[kind], "runs": len(rs)}
        for col, label in (("val_mae_c", "validation MAE (°C)"), ("test_mae_c", "test MAE (°C)")):
            vals = np.array([r[col] for r in rs if isinstance(r.get(col), (int, float))], dtype=float)
            row[f"mean {label}"] = float(vals.mean()) if vals.size else None
            row[f"SD {label}"] = float(vals.std(ddof=1)) if vals.size > 1 else None
        out.append(row)
    return out


# ------------------------------------------------------------ curated examples
def _num_list(v: Any, n: int) -> bool:
    return (isinstance(v, list) and len(v) == n
            and all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in v))


def validate_failure_case(c: Any) -> Optional[str]:
    """None if ``c`` is a complete, measured failure record; else why not."""
    if not isinstance(c, dict):
        return "not an object"
    if not _num_list(c.get("history_c"), INPUT_HOURS):
        return "history_c must be 72 numbers"
    for k in ("actual_c", "predicted_c"):
        if not _num_list(c.get(k), FORECAST_HOURS):
            return f"{k} must be 24 numbers"
    if c.get("model_type") not in MODEL_TYPES:
        return "model_type must be lstm or attention"
    if isinstance(c.get("run_id"), bool) or not isinstance(c.get("run_id"), int):
        return "run_id (the persisted run that produced the forecast) is required"
    for k in ("title", "origin", "discussion"):
        if not isinstance(c.get(k), str) or not c[k].strip():
            return f"{k} is required"
    if c.get("attention") is not None:
        try:
            parse_attention(c["attention"])
        except ApiError:
            return "attention must be 72x72"
    return None


def validate_comparison_example(c: Any) -> Optional[str]:
    if not isinstance(c, dict):
        return "not an object"
    if not _num_list(c.get("history_c"), INPUT_HOURS):
        return "history_c must be 72 numbers"
    for k in ("actual_c", "lstm_predicted_c", "attention_predicted_c"):
        if not _num_list(c.get(k), FORECAST_HOURS):
            return f"{k} must be 24 numbers"
    for k in ("lstm_run_id", "attention_run_id"):
        if isinstance(c.get(k), bool) or not isinstance(c.get(k), int):
            return f"{k} is required"
    for k in ("title", "origin", "discussion"):
        if not isinstance(c.get(k), str) or not c[k].strip():
            return f"{k} is required"
    if c.get("attention") is not None:
        try:
            parse_attention(c["attention"])
        except ApiError:
            return "attention must be 72x72"
    return None


def load_examples(text: Optional[str]) -> Dict[str, Any]:
    """Parse ui/content/examples.json. Returns valid cases plus the reasons others were dropped."""
    result = {"comparison_examples": [], "failure_cases": [], "problems": []}
    if not text:
        return result
    try:
        data = json.loads(text)
    except ValueError:
        result["problems"].append("examples.json is not valid JSON")
        return result
    for key, validator in (("comparison_examples", validate_comparison_example),
                           ("failure_cases", validate_failure_case)):
        items = data.get(key, []) if isinstance(data, dict) else []
        for i, item in enumerate(items if isinstance(items, list) else []):
            why = validator(item)
            if why:
                result["problems"].append(f"{key}[{i}] ignored: {why}")
            else:
                result[key].append(item)
    return result


# -------------------------------------------------------------- concepts example
def softmax(x: Sequence[float]) -> np.ndarray:
    z = np.asarray(x, dtype=np.float64)
    e = np.exp(z - z.max())
    return e / e.sum()


def scaling_example(raw_scores: Sequence[float] = (16.0, 8.0, 0.0), d_k: int = 64,
                    values: Sequence[float] = (10.0, 20.0, 30.0)) -> Dict[str, Any]:
    """Worked example: softmax of raw q·k scores with and without the 1/sqrt(d_k) factor."""
    s = np.asarray(raw_scores, dtype=np.float64)
    out = {"d_k": d_k, "raw": s, "scaled": s / math.sqrt(d_k)}
    for name in ("raw", "scaled"):
        p = softmax(out[name])
        out[f"weights_{name}"] = p
        out[f"output_{name}"] = float(p @ np.asarray(values, dtype=np.float64))
        out[f"max_slope_{name}"] = float((p * (1 - p)).max())   # largest diagonal softmax-Jacobian entry
    return out
