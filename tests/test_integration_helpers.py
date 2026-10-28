"""Offline tests for tests/integration_helpers.py and the integration test's skip behavior.

No network, no credentials. These do NOT verify anything against Supabase; they check the
helper logic (configuration gating, exact-row cleanup, response/row/run comparison) using
the in-memory repository and synthetic artifacts from test_forecast_api.py.
"""
from __future__ import annotations

import copy
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from shared.forecast_models import MODEL_TYPES
from tests import integration_helpers as ih
from tests.test_forecast_api import FakeRepo, RUN_IDS, make_artifact, settings_for

REPO = Path(__file__).resolve().parents[1]

FULL = {ih.ENABLE: "1", ih.URL: "https://test.example", ih.KEY: "k-secret", ih.RUN_IDS["lstm"]: "5",
        ih.RUN_IDS["attention"]: "6"}


# ------------------------------------------------------------- configuration

def test_missing_config_lists_names_not_values():
    assert ih.missing_config({}) == [f"{ih.ENABLE}=1"]
    assert ih.missing_config({ih.ENABLE: "0"}) == [f"{ih.ENABLE}=1"]
    env = {ih.ENABLE: "1", ih.KEY: "secret-value"}
    miss = ih.missing_config(env)
    assert ih.URL in miss and ih.RUN_IDS["lstm"] in miss and "secret-value" not in " ".join(miss)
    assert ih.missing_config(FULL) == []


def test_production_variable_names_alone_never_enable_the_test():
    assert ih.missing_config({"SUPABASE_URL": "https://prod", "SUPABASE_SERVICE_KEY": "x"}) == [f"{ih.ENABLE}=1"]


def test_load_config_validates_run_ids_and_hides_secrets():
    cfg = ih.load_config(FULL)
    assert cfg.run_ids == {"lstm": 5, "attention": 6} and cfg.artifact_source == "supabase"
    assert "k-secret" not in repr(cfg) and "test.example" not in repr(cfg)
    for bad in ("0", "-1", "abc", "1.5", ""):
        with pytest.raises((ih.IntegrationConfigError, KeyError)):
            ih.load_config({**FULL, ih.RUN_IDS["lstm"]: bad})
    with pytest.raises(ih.IntegrationConfigError):
        ih.load_config({**FULL, ih.RUN_IDS["attention"]: "5"})        # same id for both types
    with pytest.raises(ih.IntegrationConfigError):
        ih.load_config({**FULL, ih.SOURCE: "disk"})


def test_same_project_as_primary_is_refused_unless_explicit():
    env = {**FULL, ih.PRIMARY_URL: "https://test.example/"}
    with pytest.raises(ih.IntegrationConfigError) as e:
        ih.load_config(env)
    assert "k-secret" not in str(e.value) and "test.example" not in str(e.value)
    assert ih.load_config({**env, ih.ALLOW_SAME_PROJECT: "1"}).run_ids["lstm"] == 5
    assert ih.load_config({**FULL, ih.PRIMARY_URL: "https://other.example"})


def test_local_source_requires_artifact_paths_and_maps_to_forecast_env(tmp_path):
    env = {**FULL, ih.SOURCE: "local"}
    assert set(ih.missing_config(env)) == set(ih.LOCAL_ARTIFACTS.values())
    env.update({ih.LOCAL_ARTIFACTS["lstm"]: "/a/l.pt", ih.LOCAL_ARTIFACTS["attention"]: "/a/a.pt"})
    fe = ih.forecast_env(ih.load_config(env), tmp_path)
    assert fe["FORECAST_LSTM_RUN_ID"] == "5" and fe["FORECAST_ATTENTION_ARTIFACT"] == "/a/a.pt"
    assert fe["FORECAST_ARTIFACT_SOURCE"] == "local"
    sup = ih.forecast_env(ih.load_config(FULL), tmp_path)
    assert sup["FORECAST_ARTIFACT_SOURCE"] == "supabase" and "FORECAST_LSTM_ARTIFACT" not in sup


def test_integration_module_skips_cleanly_when_unconfigured():
    env = {k: v for k, v in __import__("os").environ.items() if not k.startswith("SEQSIGHT_IT_")}
    out = subprocess.run([sys.executable, "-m", "pytest", "tests/test_supabase_integration.py", "-q", "-rs"],
                         cwd=REPO, env=env, capture_output=True, text=True, timeout=120)
    assert out.returncode == 5, out.stdout + out.stderr    # pytest: nothing ran (all skipped), not a pass
    assert "skipped" in out.stdout and "SEQSIGHT_IT_ENABLE=1" in out.stdout and "passed" not in out.stdout


# ------------------------------------------------- exact-row cleanup (fake client)

class FakeClient:
    """Records PostgREST-style delete chains; refuses a delete that has no filters."""

    def __init__(self, fail_on=()):
        self.deletes, self.fail_on = [], set(fail_on)
        self._f = None

    def table(self, name):
        self._f = {"table": name, "filters": {}}
        return self

    def delete(self):
        self._f["op"] = "delete"
        return self

    def eq(self, col, val):
        self._f["filters"][col] = val
        return self

    def execute(self):
        assert self._f["op"] == "delete" and self._f["filters"], "unfiltered delete"
        if self._f["filters"]["id"] in self.fail_on:
            raise RuntimeError("boom with https://secret.example in it")
        self.deletes.append(self._f)
        return None


def test_cleanup_deletes_only_exact_tracked_rows():
    c = FakeClient()
    assert ih.cleanup_predictions(c, [(7, "a" * 64, 5), (9, "b" * 64, 6)]) == []
    assert c.deletes == [
        {"table": "predictions", "op": "delete", "filters": {"id": 7, "input_hash": "a" * 64, "served_by_run_id": 5}},
        {"table": "predictions", "op": "delete", "filters": {"id": 9, "input_hash": "b" * 64, "served_by_run_id": 6}}]
    assert ih.cleanup_predictions(FakeClient(), []) == []                 # nothing tracked -> nothing deleted


def test_cleanup_refuses_invalid_ids_and_continues_after_a_failure():
    c = FakeClient(fail_on={7})
    problems = ih.cleanup_predictions(c, [(7, "a" * 64, 5), (0, "b" * 64, 5), (True, "c" * 64, 5), (8, "d" * 64, 5)])
    assert len(problems) == 3 and [d["filters"]["id"] for d in c.deletes] == [8]
    assert not any("secret.example" in p for p in problems)               # error text is never echoed
    assert not any(d["table"] != "predictions" for d in c.deletes)


# ------------------------------------- tracking + verification against the fake API

@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    d = tmp_path_factory.mktemp("it-arts")
    for kind in MODEL_TYPES:
        make_artifact(d / f"{kind}.pt", kind)
    return d


@pytest.fixture
def served(artifacts):
    repo = FakeRepo()
    for kind, rid in RUN_IDS.items():
        repo.add_run(rid, kind, artifacts / f"{kind}.pt")
    tracked = ih.TrackingRepository(repo)
    with TestClient(create_app(settings_for(artifacts), tracked)) as client:
        yield client, repo, tracked


def _stored_and_run(repo, body):
    stored = next(r for r in repo.predictions if r["id"] == body["prediction_id"])
    return copy.deepcopy(stored), copy.deepcopy(repo.runs[stored["served_by_run_id"]])


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_tracking_and_verification_pass_for_both_model_types(served, model_type):
    client, repo, tracked = served
    hist = ih.sample_history(model_type)
    body = client.post("/predict", json={"model_type": model_type, "history_c": hist}).json()
    assert tracked.created == [(body["prediction_id"], body["input_hash"], RUN_IDS[model_type])]
    stored, run = _stored_and_run(repo, body)
    assert ih.verify_logged_prediction(body, stored, run, model_type=model_type, history_c=hist,
                                       expected_run_id=RUN_IDS[model_type]) == []
    assert (body["attention"] is None) == (model_type == "lstm")
    assert len(ih.sample_history("lstm")) == len(hist) == 72 and ih.sample_history("lstm") != ih.sample_history("attention")


def test_verification_detects_each_kind_of_mismatch(served):
    client, repo, _ = served
    hist = ih.sample_history("attention")
    body = client.post("/predict", json={"model_type": "attention", "history_c": hist}).json()
    stored, run = _stored_and_run(repo, body)
    ok = dict(model_type="attention", history_c=hist, expected_run_id=RUN_IDS["attention"])
    assert ih.verify_logged_prediction(body, stored, run, **ok) == []

    def bad(**mut):
        s = copy.deepcopy(stored)
        s.update(mut)
        return ih.verify_logged_prediction(body, s, run, **ok)

    assert any("forecast" in m for m in bad(forecast=[x + 0.01 for x in stored["forecast"]]))
    assert any("input_hash" in m for m in bad(input_hash="0" * 64))
    assert any("model_type" in m for m in bad(model_type="lstm"))
    assert any("run id" in m for m in bad(served_by_run_id=RUN_IDS["lstm"]))
    assert any("attention" in m for m in bad(attention=stored["attention"][:71]))
    assert any("attention" in m for m in bad(attention=None))
    wrong_run = {**run, "id": RUN_IDS["lstm"], "model_type": "lstm"}
    assert ih.verify_logged_prediction(body, stored, wrong_run, **ok)
    assert ih.verify_logged_prediction(body, stored, run, **{**ok, "history_c": hist[:-1] + [hist[-1] + 1]})
    assert ih.verify_logged_prediction(body, stored, run, **{**ok, "expected_run_id": 999})
    assert ih.verify_logged_prediction(body, stored, run, **ok, expected_data_version="other-version")


def test_verification_requires_lstm_attention_null_and_attention_72x72(served):
    client, repo, _ = served
    hist = ih.sample_history("lstm")
    body = client.post("/predict", json={"model_type": "lstm", "history_c": hist}).json()
    stored, run = _stored_and_run(repo, body)
    ok = dict(model_type="lstm", history_c=hist, expected_run_id=RUN_IDS["lstm"])
    assert ih.verify_logged_prediction(body, {**stored, "attention": [[0.0] * 72] * 72}, run, **ok)
    assert ih.verify_logged_prediction({**body, "attention": [[1.0]]}, stored, run, **ok)
