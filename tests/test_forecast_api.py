"""Forecast API tests: synthetic artifacts + an in-memory fake repository.

No network, no credentials, no real artifacts. Models are untrained (random
weights from a fixed seed): these tests check serving behavior, shapes, logging
and failure handling, not forecast quality.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from api.forecast_config import ForecastSettings, ModelSetting
from api.forecast_repo import RepositoryError, SupabaseForecastRepository
from api.forecast_service import ForecastService, canonical_input_hash, file_sha256
from api.main import create_app
from shared import forecast_models as fm
from shared import forecast_training as ft
from shared.forecast_registration import (RegistrationError, attach_test_mae, execute_registration,
                                          plan_registration)
from shared.forecasting import Standardizer

SCALER = Standardizer(mean=9.5, std=8.0)
DATA_VERSION = "synthetic-v1"
RUN_IDS = {"lstm": 11, "attention": 12}


def make_artifact(path: Path, model_type: str, seed: int = 0, scaler=SCALER) -> None:
    torch.manual_seed(seed)
    model = fm.build_model(model_type)
    cfg = {"seed": seed, "max_epochs": 1, "batch_size": 128, "lr": 1e-3}
    res = ft.TrainResult(model, model_type, model.config(), cfg, 1, 1.5, [{"epoch": 1}], 10,
                         fm.count_parameters(model), 1.0, copy.deepcopy(model.state_dict()))
    ft.save_artifact(path, res, scaler, DATA_VERSION, extra={"train_windows_used": 100})


class FakeRepo:
    def __init__(self):
        self.runs, self.predictions = {}, []
        self.db_up = True
        self.fail_insert_on = set()        # 0-based insert attempt numbers that fail
        self.insert_attempts = 0
        self.b64 = {}

    def add_run(self, run_id, kind, artifact: Path, **over):
        _, scaler, meta = ft.load_artifact(artifact)
        row = {"id": run_id, "model_type": kind, "data_version": meta["data_version"],
               "seed": meta["seed"], "hyperparameters": {"artifact_sha256": file_sha256(artifact)},
               "preprocessing": scaler.to_dict(), "compute_budget": {"epochs_run": 1},
               "val_mae_c": meta["best_val_mae_c"], "test_mae_c": 2.0,
               "created_at": "2026-10-27T00:00:00+00:00"}
        row.update(over)
        self.runs[run_id] = row

    def ping(self):
        return self.db_up

    def get_run(self, run_id):
        if not self.db_up:
            raise RepositoryError("down")
        return self.runs.get(run_id)

    def get_run_artifact(self, run_id):
        if not self.db_up:
            raise RepositoryError("down")
        return self.b64.get(run_id)

    def list_runs(self, limit, offset):
        if not self.db_up:
            raise RepositoryError("down")
        rows = sorted(self.runs.values(), key=lambda r: -r["id"])
        return rows[offset:offset + limit]

    def insert_prediction(self, *, input_hash, forecast, attention, model_type, served_by_run_id):
        n = self.insert_attempts
        self.insert_attempts += 1
        if not self.db_up or n in self.fail_insert_on:
            raise RepositoryError("insert failed")
        if served_by_run_id not in self.runs or self.runs[served_by_run_id]["model_type"] != model_type:
            raise RepositoryError("fk violation")
        row = {"id": len(self.predictions) + 1, "created_at": f"2026-10-27T00:00:{len(self.predictions):02d}+00:00",
               "input_hash": input_hash, "forecast": forecast, "attention": attention,
               "model_type": model_type, "served_by_run_id": served_by_run_id}
        self.predictions.append(row)
        return {"id": row["id"]}

    def list_predictions(self, limit, offset, include_attention):
        if not self.db_up:
            raise RepositoryError("down")
        rows = sorted(self.predictions, key=lambda r: -r["id"])[offset:offset + limit]
        return [r if include_attention else {k: v for k, v in r.items() if k != "attention"} for r in rows]


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    d = tmp_path_factory.mktemp("arts")
    for kind in fm.MODEL_TYPES:
        make_artifact(d / f"{kind}.pt", kind)
    return d


def settings_for(artifacts, run_ids=RUN_IDS, **paths):
    return ForecastSettings({k: ModelSetting(Path(paths.get(k, artifacts / f"{k}.pt")), run_ids.get(k),
                                             path_is_default=False) for k in fm.MODEL_TYPES},
                            retry_cooldown_s=0)


@pytest.fixture
def repo(artifacts):
    r = FakeRepo()
    for kind, rid in RUN_IDS.items():
        r.add_run(rid, kind, artifacts / f"{kind}.pt")
    return r


@pytest.fixture
def client(artifacts, repo):
    with TestClient(create_app(settings_for(artifacts), repo)) as c:
        c.repo = repo
        yield c


HIST = [10.0 + 0.1 * i for i in range(72)]


def post(client, model_type="attention", history=HIST):
    return client.post("/predict", json={"model_type": model_type, "history_c": history})


# ------------------------------------------------------------------ validation
@pytest.mark.parametrize("history", [
    HIST[:71], HIST + [1.0], [], [True] + HIST[1:], ["10.0"] + HIST[1:], [None] + HIST[1:],
    [float("nan")] + HIST[1:], [float("inf")] + HIST[1:], [2000.0] + HIST[1:]])
def test_predict_rejects_bad_history(client, history):
    # json.dumps writes NaN/Infinity as bare tokens, which Starlette's parser accepts, so the
    # finite check (not the JSON parser) is what must reject them.
    r = client.post("/predict", content=json.dumps({"model_type": "lstm", "history_c": history}),
                    headers={"content-type": "application/json"})
    assert r.status_code == 422
    assert client.repo.predictions == []


def test_predict_rejects_bad_model_type_and_extra_fields(client):
    assert post(client, "gru").status_code == 422
    assert client.post("/predict", json={"history_c": HIST}).status_code == 422
    assert client.post("/predict", json={"model_type": "lstm", "history_c": HIST, "run_id": 3}).status_code == 422


def test_predict_accepts_integers(client):
    assert post(client, "lstm", [10] * 72).status_code == 200


# ---------------------------------------------------------------- response shape
def test_attention_response_shape_and_logging(client):
    r = post(client, "attention")
    assert r.status_code == 200
    b = r.json()
    assert b["model_type"] == "attention" and b["served_by_run_id"] == RUN_IDS["attention"]
    assert len(b["forecast_c"]) == 24 and all(math.isfinite(v) for v in b["forecast_c"])
    a = np.array(b["attention"])
    assert a.shape == (72, 72) and np.allclose(a.sum(axis=1), 1.0, atol=1e-5)
    (row,) = client.repo.predictions
    assert row["id"] == b["prediction_id"]
    assert row["input_hash"] == b["input_hash"] == canonical_input_hash(HIST)
    assert row["forecast"] == b["forecast_c"] and row["attention"] == b["attention"]
    assert row["model_type"] == "attention" and row["served_by_run_id"] == RUN_IDS["attention"]


def test_lstm_response_has_null_attention(client):
    b = post(client, "lstm").json()
    assert b["attention"] is None and b["served_by_run_id"] == RUN_IDS["lstm"] and len(b["forecast_c"]) == 24
    assert client.repo.predictions[0]["attention"] is None


def test_prediction_uses_artifact_scaler_in_degrees(artifacts, client):
    model, scaler, _ = ft.load_artifact(artifacts / "lstm.pt")
    x = torch.from_numpy(scaler.transform(np.array(HIST)).astype(np.float32)).reshape(1, 72, 1)
    with torch.no_grad():
        z, _ = model(x)
    expected = scaler.inverse_transform(z.double().numpy())[0]
    assert np.allclose(post(client, "lstm").json()["forecast_c"], expected, atol=1e-9)


def test_inference_is_eval_mode_and_leaves_no_grad(artifacts, repo):
    svc = ForecastService(settings_for(artifacts), repo)
    svc.startup()
    loaded = svc.require_ready("attention")
    loaded.model.train()                      # simulate a stray mode flip
    from api.forecast_service import run_forecasts
    run_forecasts(loaded, [HIST])
    assert not loaded.model.training
    assert all(p.grad is None for p in loaded.model.parameters())


def test_deterministic_and_input_dependent(client):
    a, b = post(client).json(), post(client).json()
    assert a["forecast_c"] == b["forecast_c"] and a["input_hash"] == b["input_hash"]
    c = post(client, history=[v + 1 for v in HIST]).json()
    assert c["forecast_c"] != a["forecast_c"] and c["input_hash"] != a["input_hash"]


# ------------------------------------------------------------------------- hash
def test_canonical_hash_format():
    assert canonical_input_hash([5] * 72) == canonical_input_hash([5.0] * 72)
    assert canonical_input_hash([0.0] * 72) == canonical_input_hash([-0.0] * 72)
    import hashlib
    expected = hashlib.sha256(("seqsight-input-v1|degC|72|" + ",".join(["1.5"] * 72)).encode()).hexdigest()
    assert canonical_input_hash([1.5] * 72) == expected
    assert len(expected) == 64 and expected == expected.lower()
    with pytest.raises(ValueError):
        canonical_input_hash([1.0] * 71)
    with pytest.raises(ValueError):
        canonical_input_hash([float("nan")] * 72)


# ------------------------------------------------------------------------ batch
def test_batch_success_matches_single(client):
    h2 = [v - 3 for v in HIST]
    r = client.post("/predict_batch", json={"model_type": "attention", "histories_c": [HIST, h2]})
    assert r.status_code == 200
    b = r.json()
    assert (b["n_requested"], b["n_audited"], b["n_not_audited"]) == (2, 2, 0)
    assert [i["index"] for i in b["results"]] == [0, 1]
    single = post(client, "attention", h2).json()
    assert np.allclose(b["results"][1]["prediction"]["forecast_c"], single["forecast_c"], atol=1e-4)
    assert len(client.repo.predictions) == 3
    assert {p["served_by_run_id"] for p in client.repo.predictions} == {RUN_IDS["attention"]}


def test_batch_validation_is_all_or_nothing(client):
    bad = HIST[:-1]
    r = client.post("/predict_batch", json={"model_type": "lstm", "histories_c": [HIST, bad]})
    assert r.status_code == 422 and client.repo.predictions == []
    assert client.post("/predict_batch", json={"model_type": "lstm", "histories_c": []}).status_code == 422
    too_many = [HIST] * 33
    assert client.post("/predict_batch", json={"model_type": "lstm", "histories_c": too_many}).status_code == 422
    assert client.post("/predict_batch", json={"model_type": "lstm", "histories_c": [HIST] * 32}).status_code == 200


def test_batch_partial_logging_failure_is_207_and_withholds_forecast(client):
    client.repo.fail_insert_on = {1}
    r = client.post("/predict_batch", json={"model_type": "lstm", "histories_c": [HIST] * 3})
    assert r.status_code == 207
    b = r.json()
    assert (b["n_audited"], b["n_not_audited"]) == (2, 1)
    bad = b["results"][1]
    assert bad["status"] == "not_audited" and bad["prediction"] is None and bad["error"]
    assert b["results"][0]["status"] == b["results"][2]["status"] == "ok"
    assert len(client.repo.predictions) == 2


def test_batch_total_logging_failure_is_503(client):
    client.repo.db_up = False
    r = client.post("/predict_batch", json={"model_type": "lstm", "histories_c": [HIST] * 2})
    assert r.status_code == 503
    assert r.json()["n_audited"] == 0 and all(i["prediction"] is None for i in r.json()["results"])


# ------------------------------------------------------------- logging failures
def test_single_logging_failure_withholds_forecast(client):
    client.repo.fail_insert_on = {0}
    r = post(client)
    assert r.status_code == 503
    assert r.json()["detail"]["code"] == "prediction_not_audited"
    assert "forecast_c" not in r.text and client.repo.predictions == []


# ---------------------------------------------------------------------- history
def test_predictions_pagination(client):
    for i in range(5):
        post(client, "lstm", [float(i)] * 72)
    p1 = client.get("/predictions", params={"limit": 2}).json()
    assert [x["id"] for x in p1["items"]] == [5, 4] and p1["has_more"] is True
    p2 = client.get("/predictions", params={"limit": 2, "offset": 2}).json()
    assert [x["id"] for x in p2["items"]] == [3, 2] and p2["has_more"] is True
    p3 = client.get("/predictions", params={"limit": 2, "offset": 4}).json()
    assert [x["id"] for x in p3["items"]] == [1] and p3["has_more"] is False
    assert p1["items"][0]["attention"] is None
    assert len(p1["items"][0]["forecast_c"]) == 24


def test_predictions_include_attention_flag(client):
    post(client, "attention")
    item = client.get("/predictions", params={"include_attention": True}).json()["items"][0]
    assert np.array(item["attention"]).shape == (72, 72)


@pytest.mark.parametrize("params", [{"limit": 0}, {"limit": 101}, {"offset": -1}, {"offset": 10_001}])
def test_predictions_bounds_enforced(client, params):
    assert client.get("/predictions", params=params).status_code == 422


def test_history_db_down_is_503(client):
    client.repo.db_up = False
    assert client.get("/predictions").status_code == 503
    assert client.get("/runs").status_code == 503


# ------------------------------------------------------------------------- runs
def test_runs_lists_and_marks_served(client, artifacts):
    client.repo.add_run(99, "lstm", artifacts / "lstm.pt")     # registered but not the served one
    body = client.get("/runs", params={"limit": 2}).json()
    assert [r["id"] for r in body["items"]] == [99, 12] and body["has_more"] is True
    served = {r["id"]: r["served_by_this_api"] for r in client.get("/runs").json()["items"]}
    assert served == {99: False, 12: True, 11: True}
    assert "artifact_sha256" in body["items"][1]["hyperparameters"]


# -------------------------------------------------------------- readiness/health
def test_healthz_ready_and_hides_secrets(client, artifacts):
    r = client.get("/healthz")
    assert r.status_code == 200
    b = r.json()
    assert b["status"] == "ok" and b["ready"] and b["database"]
    assert all(m["ready"] and m["reason"] is None for m in b["models"].values())
    assert str(artifacts) not in r.text


def test_missing_artifact_not_ready(artifacts, repo):
    s = settings_for(artifacts, lstm=artifacts / "nope.pt")
    with TestClient(create_app(s, repo)) as c:
        h = c.get("/healthz")
        assert h.status_code == 503
        assert h.json()["models"]["lstm"] == {"artifact_loaded": False, "run_registered": False,
                                               "ready": False, "reason": "artifact_not_found"}
        assert h.json()["models"]["attention"]["ready"] is True
        r = post(c, "lstm")
        assert r.status_code == 503 and r.json()["detail"]["code"] == "artifact_not_found"
        assert str(artifacts) not in r.text
        assert post(c, "attention").status_code == 200       # the healthy type still serves


def test_corrupt_artifact_not_ready(artifacts, repo, tmp_path):
    bad = tmp_path / "bad.pt"
    bad.write_bytes(b"not a checkpoint")
    with TestClient(create_app(settings_for(artifacts, lstm=bad), repo)) as c:
        assert post(c, "lstm").json()["detail"]["code"] == "artifact_invalid"


def test_artifact_of_wrong_type_not_ready(artifacts, repo):
    with TestClient(create_app(settings_for(artifacts, lstm=artifacts / "attention.pt"), repo)) as c:
        assert post(c, "lstm").json()["detail"]["code"] == "artifact_type_mismatch"


def test_unconfigured_run_id_never_served(artifacts, repo):
    s = settings_for(artifacts, run_ids={"attention": 12})
    with TestClient(create_app(s, repo)) as c:
        r = post(c, "lstm")
        assert r.status_code == 503 and r.json()["detail"]["code"] == "run_id_not_configured"
        assert c.get("/healthz").status_code == 503
    assert repo.predictions == []


def test_run_not_in_database(artifacts, repo):
    s = settings_for(artifacts, run_ids={"lstm": 404, "attention": 12})
    with TestClient(create_app(s, repo)) as c:
        assert post(c, "lstm").json()["detail"]["code"] == "run_not_found"


@pytest.mark.parametrize("over,code", [
    ({"model_type": "attention"}, "run_mismatch_model_type"),
    ({"data_version": "other"}, "run_mismatch_data_version"),
    ({"seed": 5}, "run_mismatch_seed"),
    ({"preprocessing": {"mean": 1.0, "std": 8.0}}, "run_mismatch_scaler"),
    ({"val_mae_c": 9.9}, "run_mismatch_val_mae"),
    ({"hyperparameters": {"artifact_sha256": "0" * 64}}, "run_mismatch_artifact"),
    ({"hyperparameters": {}}, "run_mismatch_artifact"),
])
def test_registration_mismatch_blocks_serving(artifacts, over, code):
    repo = FakeRepo()
    repo.add_run(11, "lstm", artifacts / "lstm.pt", **over)
    repo.add_run(12, "attention", artifacts / "attention.pt")
    with TestClient(create_app(settings_for(artifacts), repo)) as c:
        r = post(c, "lstm")
        assert r.status_code == 503 and r.json()["detail"]["code"] == code
    assert repo.predictions == []


def test_invalid_run_id_env_value():
    s = ForecastSettings.from_env({"FORECAST_LSTM_RUN_ID": "abc", "FORECAST_ATTENTION_RUN_ID": "7"})
    assert s.models["lstm"].run_id is None and s.models["lstm"].run_id_error == "run_id_invalid"
    assert s.models["attention"].run_id == 7
    assert ForecastSettings.from_env({"FORECAST_LSTM_RUN_ID": "0"}).models["lstm"].run_id_error
    assert ForecastSettings.from_env({}).models["lstm"].run_id is None


def test_default_paths_are_dev_seed0_and_flagged():
    s = ForecastSettings.from_env({})
    assert s.artifacts_are_development_defaults
    assert s.models["lstm"].artifact_path.parts[-4:] == ("epoch50", "final", "lstm_seed0.pt") or \
        s.models["lstm"].artifact_path.name == "lstm_seed0.pt"
    assert s.models["attention"].artifact_path.name == "attention_seed0.pt"
    s2 = ForecastSettings.from_env({"FORECAST_LSTM_ARTIFACT": "/x/l.pt", "FORECAST_ATTENTION_ARTIFACT": "/x/a.pt"})
    assert s2.models["lstm"].artifact_path == Path("/x/l.pt")


def test_db_down_at_startup_then_recovers(artifacts, repo):
    repo.db_up = False
    with TestClient(create_app(settings_for(artifacts), repo)) as c:
        h = c.get("/healthz")
        assert h.status_code == 503 and h.json()["database"] is False
        assert h.json()["models"]["lstm"]["reason"] == "database_unavailable"
        assert post(c).status_code == 503
        repo.db_up = True
        assert c.get("/healthz").status_code == 200
        assert post(c).status_code == 200


def test_db_down_with_ready_models_is_degraded(client):
    client.repo.db_up = False
    h = client.get("/healthz")
    assert h.status_code == 503 and h.json()["status"] == "degraded" and h.json()["ready"] is True


# ---------------------------------------------------------------------- version
def test_version_has_metadata_without_paths(client, artifacts):
    r = client.get("/version")
    b = r.json()
    assert r.status_code == 200
    assert b["input_hours"] == 72 and b["forecast_hours"] == 24 and b["max_batch_size"] == 32
    assert b["input_hash_format"] == "seqsight-input-v1"
    assert b["artifacts_are_development_defaults"] is False
    m = b["models"]["attention"]
    assert m["serving_run_id"] == 12 and m["data_version"] == DATA_VERSION and len(m["artifact_sha256_prefix"]) == 12
    assert str(artifacts) not in r.text and "supabase" not in r.text.lower()


# ------------------------------------------------------------------ real repo
def test_supabase_repo_is_lazy_and_sanitizes_errors():
    calls = []

    def factory():
        calls.append(1)
        raise RuntimeError("https://secret-ref.supabase.co key=SECRET")

    repo = SupabaseForecastRepository(factory)
    assert calls == []                              # nothing connects at construction
    assert repo.ping() is False
    with pytest.raises(RepositoryError) as e:
        repo.get_run(1)
    assert "SECRET" not in str(e.value) and "secret-ref" not in str(e.value)


def test_supabase_repo_query_shape():
    class Q:
        def __init__(self, log): self.log = log
        def __getattr__(self, name):
            def f(*a, **k):
                self.log.append((name, a, k))
                return self
            return f
        data = [{"id": 1}]
    log = []
    repo = SupabaseForecastRepository(lambda: type("C", (), {"table": lambda s, n: (log.append(("table", n)), Q(log))[1]})())
    repo.list_predictions(11, 20, False)
    names = [x[0] for x in log]
    assert ("table", "predictions") in log and ("range", (20, 30), {}) in log
    assert [x for x in log if x[0] == "order"][0][2] == {"desc": True}


# ------------------------------------------------------------------ registration
class FakeSb:
    """Records inserts; mimics the few supabase-py calls the registration code uses."""
    def __init__(self, fail_artifact=False):
        self.tables = {"runs": [], "run_artifacts": []}
        self.fail_artifact, self._t, self._op, self._eq = fail_artifact, None, None, {}

    def table(self, name): self._t, self._op, self._eq = name, "select", {}; return self
    def select(self, *_): self._op = "select"; return self
    def eq(self, k, v): self._eq[k] = v; return self
    def insert(self, row): self._op, self._row = "insert", row; return self
    def delete(self): self._op = "delete"; return self

    def execute(self):
        t = self.tables[self._t]
        if self._op == "insert":
            if self._t == "run_artifacts" and self.fail_artifact:
                raise RuntimeError("boom")
            row = dict(self._row)
            if self._t == "runs":
                row["id"] = len(t) + 100
            t.append(row)
            return type("R", (), {"data": [row]})()
        if self._op == "delete":
            self.tables[self._t] = [r for r in t if any(r.get(k) != v for k, v in self._eq.items())]
            return type("R", (), {"data": []})()
        return type("R", (), {"data": [r for r in t if all(r.get(k) == v for k, v in self._eq.items())]})()


def regdir(tmp_path, keys=("lstm_seed0", "attention_seed0")):
    for k in keys:
        make_artifact(tmp_path / f"{k}.pt", k.split("_")[0], seed=0)
    return list(keys)


def test_registration_plan_matches_schema_and_lacks_test_mae(tmp_path):
    keys = regdir(tmp_path)
    plans = plan_registration(tmp_path, keys)
    for p in plans:
        assert p.test_mae_c is None
        assert "test_mae_c" not in p.run
        assert p.run["preprocessing"]["mean"] == SCALER.mean and p.run["preprocessing"]["std"] == SCALER.std
        assert p.run["hyperparameters"]["artifact_sha256"] == file_sha256(p.path)
        assert set(p.run) == {"model_type", "data_version", "seed", "hyperparameters", "preprocessing",
                              "compute_budget", "val_mae_c"}
        with pytest.raises(RegistrationError):
            p.run_row()


def test_execute_refuses_without_test_mae_and_writes_nothing(tmp_path):
    plans = plan_registration(tmp_path if regdir(tmp_path) else None, ["lstm_seed0", "attention_seed0"])
    sb = FakeSb()
    with pytest.raises(RegistrationError):
        execute_registration(sb, plans)
    assert sb.tables == {"runs": [], "run_artifacts": []}


@pytest.mark.parametrize("bad", [True, "1.0", -1.0, float("nan"), float("inf"), None])
def test_attach_test_mae_rejects_placeholders(tmp_path, bad):
    keys = regdir(tmp_path)
    plans = plan_registration(tmp_path, keys)
    with pytest.raises(RegistrationError):
        attach_test_mae(plans, {"lstm_seed0": 1.0, "attention_seed0": bad})
    with pytest.raises(RegistrationError):
        attach_test_mae(plans, {"lstm_seed0": 1.0})          # incomplete coverage


def test_execute_with_measured_values_is_idempotent_and_servable(tmp_path):
    keys = regdir(tmp_path)
    plans = plan_registration(tmp_path, keys)
    attach_test_mae(plans, {"lstm_seed0": 2.5, "attention_seed0": 2.6})
    sb = FakeSb()
    out = execute_registration(sb, plans)
    assert [o["action"] for o in out] == ["registered", "registered"]
    assert [r["test_mae_c"] for r in sb.tables["runs"]] == [2.5, 2.6]
    assert len(sb.tables["run_artifacts"]) == 2
    assert [o["action"] for o in execute_registration(sb, plans)] == ["already_registered"] * 2
    assert len(sb.tables["runs"]) == 2
    # the registered rows satisfy the API's mapping verification
    repo = FakeRepo()
    for r in sb.tables["runs"]:
        repo.runs[r["id"]] = r
    ids = {r["model_type"]: r["id"] for r in sb.tables["runs"]}
    s = ForecastSettings({k: ModelSetting(tmp_path / f"{k}_seed0.pt", ids[k], path_is_default=False)
                          for k in fm.MODEL_TYPES})
    with TestClient(create_app(s, repo)) as c:
        assert c.get("/healthz").status_code == 200


def test_execute_cleans_up_run_when_artifact_insert_fails(tmp_path):
    keys = regdir(tmp_path, ("lstm_seed0",))
    plans = plan_registration(tmp_path, keys)
    attach_test_mae(plans, {"lstm_seed0": 2.5})
    sb = FakeSb(fail_artifact=True)
    with pytest.raises(RegistrationError, match="run row removed"):
        execute_registration(sb, plans)
    assert sb.tables["runs"] == []


def test_cli_dry_run_default_reports_blocker(tmp_path, capsys):
    import scripts.register_forecast_runs as cli
    regdir(tmp_path)
    called = []
    rc = cli.main(["--artifact-dir", str(tmp_path), "--keys", "lstm_seed0", "attention_seed0"],
                  client_factory=lambda: called.append(1))
    out = capsys.readouterr().out
    assert rc == 0 and not called
    assert "DRY-RUN" in out and "BLOCKED" in out and "test_mae_c=MISSING" in out
    assert "nothing was written" in out


def test_cli_execute_without_test_mae_refused(tmp_path, capsys):
    import scripts.register_forecast_runs as cli
    regdir(tmp_path)
    called = []
    rc = cli.main(["--artifact-dir", str(tmp_path), "--keys", "lstm_seed0", "--execute"],
                  client_factory=lambda: called.append(1))
    assert rc == 2 and not called
    assert "refusing" in capsys.readouterr().err


def test_cli_execute_with_file_writes_via_injected_client(tmp_path, capsys):
    import scripts.register_forecast_runs as cli
    regdir(tmp_path)
    f = tmp_path / "t.json"
    f.write_text(json.dumps({"lstm_seed0": 2.5}))
    sb = FakeSb()
    rc = cli.main(["--artifact-dir", str(tmp_path), "--keys", "lstm_seed0", "--execute",
                   "--test-mae-file", str(f)], client_factory=lambda: sb)
    assert rc == 0 and sb.tables["runs"][0]["test_mae_c"] == 2.5
