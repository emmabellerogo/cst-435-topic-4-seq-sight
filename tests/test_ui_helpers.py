"""Tests for ui/forecast_client.py and ui/forecast_plots.py with mocked HTTP and Supabase.

No network, no credentials, no Streamlit server. A headless AppTest smoke test checks that
the app builds all six tabs and degrades gracefully when no secrets are configured.
"""
from __future__ import annotations

import base64
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import requests

UI = Path(__file__).resolve().parents[1] / "ui"
sys.path.insert(0, str(UI))

import forecast_client as fc  # noqa: E402
import forecast_plots as fp  # noqa: E402

HIST = [10.0 + 0.1 * i for i in range(72)]
HASH = "a" * 64


def ok_body(model="attention", with_att=True, **over):
    b = {"prediction_id": 7, "model_type": model, "served_by_run_id": 3, "input_hash": HASH,
         "forecast_c": [11.0 + i * 0.1 for i in range(24)],
         "attention": np.full((72, 72), 1 / 72).tolist() if with_att else None}
    b.update(over)
    return b


class Resp:
    def __init__(self, status, body):
        self.status_code, self._b = status, body

    def json(self):
        if self._b is ValueError:
            raise ValueError("not json")
        return self._b


class FakeHttp:
    def __init__(self, *responses):
        self.responses, self.calls = list(responses), []

    def request(self, method, url, **kw):
        self.calls.append((method, url, kw))
        r = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(r, Exception):
            raise r
        return r


def api(*responses):
    http = FakeHttp(*responses)
    return fc.ForecastApi("https://x.example/", http=http, sleep=lambda s: None), http


# ----------------------------------------------------------------------- parsing input
def test_parse_history_accepts_separators():
    assert fc.parse_history(", ".join(str(v) for v in HIST)) == HIST
    assert len(fc.parse_history("\n".join(["1.5"] * 72))) == 72
    assert len(fc.parse_history(" ".join(["-3"] * 72))) == 72


@pytest.mark.parametrize("text", ["", " ".join(["1"] * 71), " ".join(["1"] * 73), " ".join(["1"] * 71 + ["abc"]),
                                  " ".join(["1"] * 71 + ["nan"]), " ".join(["1"] * 71 + ["inf"]),
                                  " ".join(["1"] * 71 + ["true"]), " ".join(["1"] * 71 + ["5000"])])
def test_parse_history_rejects(text):
    with pytest.raises(ValueError):
        fc.parse_history(text)


def test_demo_histories_are_valid_and_72_long():
    for kind in fc.DEMO_LABELS:
        assert fc.parse_history(fc.format_history(fc.demo_history(kind)))
    with pytest.raises(ValueError):
        fc.demo_history("nope")


# --------------------------------------------------------------------- API client
def test_predict_sends_exact_schema_with_finite_timeouts():
    a, http = api(Resp(200, ok_body()))
    p = a.predict("attention", HIST)
    method, url, kw = http.calls[0]
    assert (method, url) == ("POST", "https://x.example/predict")
    assert kw["json"] == {"model_type": "attention", "history_c": HIST}
    assert all(0 < t < 120 for t in kw["timeout"])
    assert p.served_by_run_id == 3 and p.attention.shape == (72, 72) and len(p.forecast_c) == 24


def test_lstm_response_has_no_attention():
    p, = [api(Resp(200, ok_body("lstm", with_att=False)))[0].predict("lstm", HIST)]
    assert p.attention is None


@pytest.mark.parametrize("body", [
    ok_body(forecast_c=[1.0] * 23), ok_body(forecast_c=[float("nan")] * 24), ok_body(forecast_c=[True] * 24),
    ok_body(model="lstm"), ok_body(with_att=False), ok_body("lstm", with_att=True),
    ok_body(input_hash="xyz"), ok_body(served_by_run_id=True), {"detail": "x"}, None, [],
    ok_body(attention=[[1.0] * 72] * 71)])
def test_malformed_success_responses_are_rejected(body):
    a, _ = api(Resp(200, body))
    with pytest.raises(fc.ApiError) as e:
        a.predict("attention", HIST)
    assert e.value.kind == "bad_response"


def test_422_is_reported_as_validation():
    body = {"detail": [{"loc": ["body", "history_c"], "msg": "List should have at least 72 items", "type": "x"}]}
    a, _ = api(Resp(422, body))
    with pytest.raises(fc.ApiError) as e:
        a.predict("lstm", HIST)
    assert e.value.kind == "validation" and "history_c" in e.value.message and e.value.status == 422


def test_503_registration_pending_is_explained():
    a, _ = api(Resp(503, {"detail": {"code": "run_id_not_configured", "message": "..."}}))
    with pytest.raises(fc.ApiError) as e:
        a.predict("lstm", HIST)
    assert e.value.kind == "pending" and e.value.code == "run_id_not_configured"
    assert "not been registered" in fc.describe_error(e.value)


def test_503_not_audited_is_unavailable_not_pending():
    a, _ = api(Resp(503, {"detail": {"code": "prediction_not_audited", "message": "withheld"}}))
    with pytest.raises(fc.ApiError) as e:
        a.predict("lstm", HIST)
    assert e.value.kind == "unavailable" and fc.describe_error(e.value) == "withheld"


@pytest.mark.parametrize("status", [502, 503, 504])
def test_proxy_errors_without_api_body_are_cold_start(status):
    a, _ = api(Resp(status, ValueError))
    with pytest.raises(fc.ApiError) as e:
        a.predict("lstm", HIST)
    assert e.value.kind == "cold_start"


def test_timeouts_and_connection_errors():
    for exc, kind in ((requests.exceptions.ReadTimeout(), "cold_start"),
                      (requests.exceptions.ConnectTimeout(), "cold_start"),
                      (requests.exceptions.ConnectionError("https://secret"), "connection")):
        a, _ = api(exc)
        with pytest.raises(fc.ApiError) as e:
            a.predict("lstm", HIST)
        assert e.value.kind == kind and "secret" not in e.value.message


def test_unexpected_500():
    a, _ = api(Resp(500, {"detail": "boom"}))
    with pytest.raises(fc.ApiError) as e:
        a.predict("lstm", HIST)
    assert e.value.kind == "server"


HEALTH_503 = {"status": "degraded", "ready": False, "database": True,
              "models": {"lstm": {"ready": False, "reason": "run_id_not_configured"},
                         "attention": {"ready": True, "reason": None}}}


def test_health_accepts_503_body_and_summarizes():
    a, _ = api(Resp(503, HEALTH_503))
    h = a.health()
    text = "\n".join(fc.health_summary(h))
    assert "degraded" in text and "run_id_not_configured" in text and "attention: ready" in text


def test_wait_until_up_retries_then_succeeds_and_is_bounded():
    a, http = api(requests.exceptions.ReadTimeout(), Resp(502, ValueError), Resp(200, {**HEALTH_503, "status": "ok"}))
    assert a.wait_until_up(max_wait_s=30)["http_status"] == 200
    assert len(http.calls) == 3
    a2, http2 = api(requests.exceptions.ReadTimeout())
    with pytest.raises(fc.ApiError):
        a2.wait_until_up(max_wait_s=10, interval_s=5)
    assert len(http2.calls) == 3                       # initial + 2 retries, then gives up
    a3, http3 = api(requests.exceptions.ConnectionError())
    a3_ = fc.ForecastApi("https://x", http=http3, sleep=lambda s: None)
    with pytest.raises(fc.ApiError):
        a3_.wait_until_up(max_wait_s=0)


# ------------------------------------------------------------------- Supabase reads
class Q:
    def __init__(self, log, data):
        self.log, self.data = log, data

    def __getattr__(self, name):
        def f(*a, **k):
            self.log.append((name, a, k))
            return self
        return f

    def execute(self):
        return type("R", (), {"data": self.data})()


class Client:
    def __init__(self, data):
        self.log, self.data = [], data

    def table(self, name):
        self.log.append(("table", (name,), {}))
        return Q(self.log, self.data)


def test_fetch_predictions_never_selects_attention_and_pages():
    c = Client([{"id": 1}])
    fc.fetch_predictions(c, "model_type", False, 25, 50)
    sel = [x for x in c.log if x[0] == "select"][0][1][0]
    assert "attention" not in sel and "forecast" in sel
    assert [x for x in c.log if x[0] == "order"][0] == ("order", ("model_type",), {"desc": False})
    assert ("range", (50, 74), {}) in c.log


@pytest.mark.parametrize("kw", [{"sort_by": "forecast"}, {"limit": 0}, {"limit": 101}, {"offset": -1}])
def test_fetch_predictions_validates(kw):
    with pytest.raises(ValueError):
        fc.fetch_predictions(Client([]), **kw)


def test_fetch_attention_single_row_and_shape_check():
    c = Client([{"id": 4, "attention": np.eye(72).tolist()}])
    a = fc.fetch_attention(c, 4)
    assert a.shape == (72, 72) and ("eq", ("id", 4), {}) in c.log
    assert fc.fetch_attention(Client([{"id": 1, "attention": None}]), 1) is None
    with pytest.raises(ValueError):
        fc.fetch_attention(Client([]), 9)
    with pytest.raises(fc.ApiError):
        fc.fetch_attention(Client([{"id": 1, "attention": [[1.0]]}]), 1)


def jwt(role):
    enc = lambda d: base64.urlsafe_b64encode(json.dumps(d).encode()).rstrip(b"=").decode()  # noqa: E731
    return f"{enc({'alg': 'HS256'})}.{enc({'role': role})}.sig"


def test_service_role_keys_are_refused():
    fc.check_anon_key(jwt("anon"))
    fc.check_anon_key("sb_publishable_abc")
    fc.check_anon_key("not-a-jwt-but-undecidable.a.b")
    for bad in (jwt("service_role"), "sb_secret_abc", "", "  "):
        with pytest.raises(ValueError):
            fc.check_anon_key(bad)


def test_summarize_runs_reports_val_and_test_separately_without_ranking():
    runs = [{"model_type": "lstm", "val_mae_c": 1.8, "test_mae_c": 2.0},
            {"model_type": "lstm", "val_mae_c": 1.9, "test_mae_c": 2.2},
            {"model_type": "attention", "val_mae_c": 1.95, "test_mae_c": 2.1}]
    s = {r["model"]: r for r in fc.summarize_runs(runs)}
    l = s["LSTM (baseline)"]
    assert l["runs"] == 2 and l["mean validation MAE (°C)"] == pytest.approx(1.85)
    assert l["mean test MAE (°C)"] == pytest.approx(2.1) and l["SD test MAE (°C)"] == pytest.approx(0.1414, abs=1e-3)
    assert s["Self-attention"]["SD test MAE (°C)"] is None
    assert not any("best" in k or "rank" in k for r in s.values() for k in r)
    assert fc.summarize_runs([]) == []


# -------------------------------------------------------------------------- examples
def failure(**over):
    c = {"title": "t", "origin": "2022-01-01 00:00", "model_type": "attention", "run_id": 3,
         "history_c": HIST, "actual_c": [1.0] * 24, "predicted_c": [2.0] * 24, "discussion": "why"}
    c.update(over)
    return c


def test_shipped_examples_file_is_valid_complete_and_measured():
    text = (UI / "content" / "examples.json").read_text()
    ex = fc.load_examples(text)
    assert ex["problems"] == [] and len(ex["comparison_examples"]) == 3 and len(ex["failure_cases"]) == 3
    meta = json.loads(text)["_meta"]
    assert meta["split"].startswith("validation") and "DELIBERATELY" in meta["purpose"]
    origins = sorted(datetime.fromisoformat(c["origin"]) for c in ex["comparison_examples"] + ex["failure_cases"])
    # six windows of 96 h each never overlap (the curation rule requires >= 168 h between origins)
    assert all((b - a).total_seconds() >= 168 * 3600 for a, b in zip(origins, origins[1:]))
    for c in ex["comparison_examples"]:
        assert (c["lstm_run_id"], c["attention_run_id"]) == (1, 4)
        for key, pred in (("lstm_mae_c", "lstm_predicted_c"), ("attention_mae_c", "attention_predicted_c")):
            assert c[key] == pytest.approx(np.mean(np.abs(np.array(c[pred]) - c["actual_c"])), abs=1e-4)
        assert len(c["lstm_artifact_sha256"]) == len(c["attention_artifact_sha256"]) == 64
        assert c["input_provenance"]["split"] == "validation"
        att = np.array(c["attention"])
        assert att.shape == (72, 72) and np.allclose(att.sum(axis=1), 1.0, atol=1e-4)
    for c in ex["failure_cases"]:
        assert c["model_type"] == "attention" and c["run_id"] == 4 and c["mae_c"] >= 5.0
        assert c["mae_c"] == pytest.approx(np.mean(np.abs(np.array(c["predicted_c"]) - c["actual_c"])), abs=1e-4)
        att = np.array(c["attention"])
        assert att.shape == (72, 72) and np.allclose(att.sum(axis=1), 1.0, atol=1e-4)
        assert "no calibrated confidence" in c["discussion"] and "does not represent average performance" in c["discussion"]


def test_examples_validation_drops_incomplete_entries():
    comp = {"title": "t", "origin": "o", "history_c": HIST, "actual_c": [1.0] * 24, "lstm_predicted_c": [1.0] * 24,
            "attention_predicted_c": [1.0] * 24, "lstm_run_id": 1, "attention_run_id": 2, "discussion": "d"}
    data = {"comparison_examples": [comp, {**comp, "lstm_run_id": None}],
            "failure_cases": [failure(), failure(run_id=None), failure(predicted_c=[1.0] * 23),
                              failure(discussion=" "), failure(model_type="gru"), failure(attention=[[1.0]])]}
    ex = fc.load_examples(json.dumps(data))
    assert len(ex["comparison_examples"]) == 1 and len(ex["failure_cases"]) == 1
    assert len(ex["problems"]) == 6
    bad_att = fc.load_examples(json.dumps({"comparison_examples": [{**comp, "attention": [[1.0]]}]}))
    assert bad_att["comparison_examples"] == [] and bad_att["problems"]
    assert fc.load_examples("{bad")["problems"] and fc.load_examples(None)["failure_cases"] == []


# ------------------------------------------------------------------- concepts example
def test_scaling_example_numbers():
    e = fc.scaling_example()
    assert e["weights_raw"][0] > 0.999 and e["output_raw"] == pytest.approx(10.0, abs=0.01)
    assert e["weights_scaled"] == pytest.approx([0.6652, 0.2447, 0.0900], abs=1e-3)
    assert e["output_scaled"] == pytest.approx(14.248, abs=1e-2)
    assert e["max_slope_scaled"] > 100 * e["max_slope_raw"]
    assert e["weights_scaled"].sum() == pytest.approx(1.0)


# ----------------------------------------------------------------------------- plots
def test_plots_build_and_validate_shape():
    assert fp.forecast_figure(HIST, [1.0] * 24, [2.0] * 24)
    assert fp.forecast_figure(None, [1.0] * 24)
    assert fp.attention_figure(np.full((72, 72), 1 / 72))
    assert fp.mean_attention_profile_figure(np.full((72, 72), 1 / 72))
    assert fp.comparison_figure(HIST, [1.0] * 24, [1.0] * 24, [2.0] * 24)
    with pytest.raises(ValueError):
        fp.attention_figure(np.zeros((3, 3)))


def test_attention_figure_states_input_to_input_in_axes():
    ax = fp.attention_figure(np.full((72, 72), 1 / 72)).axes[0]
    assert "attended-to input hour" in ax.get_xlabel() and "attending input hour" in ax.get_ylabel()


# ----------------------------------------------------------------------- app smoke test
def test_app_renders_six_tabs_without_secrets():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(UI / "app.py"), default_timeout=30).run()
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Concepts", "Try It", "Baseline vs. Attention", "Recent Predictions",
                                          "Failure Gallery", "Model Card"]
    text = " ".join(e.value for els in (at.markdown, at.info, at.warning, at.error) for e in els)
    assert "Predictions work only after" in text
    assert "deliberately shows the attention model's largest validation errors" in text
    assert "0 of 3 documented failures" not in text and "0 of 3 measured comparison examples" not in text


def test_app_renders_attention_heatmap_for_each_comparison_example():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(UI / "app.py"), default_timeout=60).run()
    assert not at.exception
    captions = [c.value for c in at.caption]
    assert sum("not an attribution of any forecast hour" in c for c in captions) == 3
    assert sum("Input-to-input attention; not an explanation of the error." in c for c in captions) == 3
