"""Artifact delivery from private run_artifacts into a verified local cache (mocked, offline)."""
from __future__ import annotations

import base64
import os
import stat
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

from api.artifact_store import ArtifactError, cache_path, ensure_artifact, expected_sha256
from api.forecast_config import ForecastSettings, ModelSetting
from api.forecast_service import ForecastService, file_sha256
from api.main import create_app
from shared import forecast_models as fm
from tests.test_forecast_api import HIST, DATA_VERSION, FakeRepo, make_artifact

IDS = {"lstm": 11, "attention": 12}


class CountingRepo(FakeRepo):
    def __init__(self):
        super().__init__()
        self.get_run_calls = self.artifact_calls = 0
        self.artifact_ids = []

    def get_run(self, run_id):
        self.get_run_calls += 1
        return super().get_run(run_id)

    def get_run_artifact(self, run_id):
        self.artifact_calls += 1
        self.artifact_ids.append(run_id)
        return super().get_run_artifact(run_id)


@pytest.fixture(scope="module")
def arts(tmp_path_factory):
    d = tmp_path_factory.mktemp("src")
    for k in fm.MODEL_TYPES:
        make_artifact(d / f"{k}.pt", k)
    return d


def b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


@pytest.fixture
def repo(arts):
    r = CountingRepo()
    for k, rid in IDS.items():
        r.add_run(rid, k, arts / f"{k}.pt")
        r.b64[rid] = b64(arts / f"{k}.pt")
    return r


def settings(tmp_path, run_ids=IDS, cooldown=0, **kw):
    return ForecastSettings(
        {k: ModelSetting(Path("/does/not/matter.pt"), run_ids.get(k), path_is_default=False) for k in fm.MODEL_TYPES},
        artifact_source="supabase", cache_dir=tmp_path / "cache", retry_cooldown_s=cooldown, **kw)


def run_app(s, repo):
    app = create_app(s, repo)
    with TestClient(app) as c:
        t = getattr(app.state, "startup_thread", None)
        if t:
            t.join(timeout=30)
        yield c


@pytest.fixture
def client(tmp_path, repo):
    yield from run_app(settings(tmp_path), repo)


def post(c, kind="attention"):
    return c.post("/predict", json={"model_type": kind, "history_c": HIST})


# ---------------------------------------------------------------- happy path
def test_downloads_verifies_caches_and_serves(client, repo, arts, tmp_path):
    assert client.get("/healthz").status_code == 200
    assert post(client, "attention").status_code == 200 and post(client, "lstm").status_code == 200
    assert repo.artifact_calls == 2
    for k, rid in IDS.items():
        p = cache_path(tmp_path / "cache", rid, file_sha256(arts / f"{k}.pt"))
        assert p.is_file() and file_sha256(p) == file_sha256(arts / f"{k}.pt")
        assert stat.S_IMODE(p.stat().st_mode) == 0o600
    assert not list((tmp_path / "cache").glob("*.tmp"))
    v = client.get("/version").json()
    assert v["artifact_source"] == "supabase" and v["artifacts_are_development_defaults"] is False
    assert str(tmp_path) not in client.get("/version").text + client.get("/healthz").text


def test_loader_uses_weights_only(tmp_path, repo, monkeypatch):
    seen = []
    real = torch.load
    monkeypatch.setattr(torch, "load", lambda *a, **k: (seen.append(k.get("weights_only")), real(*a, **k))[1])
    svc = ForecastService(settings(tmp_path), repo)
    svc.startup()
    assert seen and all(w is True for w in seen)


def test_warm_cache_avoids_second_download(tmp_path, repo):
    ForecastService(settings(tmp_path), repo).startup()
    first = repo.artifact_calls
    svc = ForecastService(settings(tmp_path), repo)
    svc.startup()
    assert repo.artifact_calls == first and svc.readiness()["ready"]


# ---------------------------------------------------------- cache validation
def test_tampered_cache_is_deleted_and_refetched(tmp_path, repo, arts):
    ForecastService(settings(tmp_path), repo).startup()
    p = cache_path(tmp_path / "cache", 11, file_sha256(arts / "lstm.pt"))
    p.write_bytes(b"tampered")
    calls = repo.artifact_calls
    svc = ForecastService(settings(tmp_path), repo)
    svc.startup()
    assert svc.readiness()["ready"] and repo.artifact_calls == calls + 1
    assert file_sha256(p) == file_sha256(arts / "lstm.pt")


def test_tampered_cache_with_artifact_gone_is_not_trusted(tmp_path, repo, arts):
    ForecastService(settings(tmp_path), repo).startup()
    p = cache_path(tmp_path / "cache", 11, file_sha256(arts / "lstm.pt"))
    p.write_bytes(b"tampered")
    repo.b64.pop(11)
    svc = ForecastService(settings(tmp_path), repo)
    svc.startup()
    assert svc.readiness()["models"]["lstm"]["reason"] == "artifact_not_in_database"
    assert not p.exists()


def test_unusable_cache_dir(tmp_path, repo):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    s = ForecastSettings(settings(tmp_path).models, artifact_source="supabase", cache_dir=blocker / "sub",
                         retry_cooldown_s=0)
    svc = ForecastService(s, repo)
    svc.startup()
    assert svc.readiness()["models"]["lstm"]["reason"] == "artifact_cache_unusable"


# ----------------------------------------------------------------- bad inputs
def reasons(c):
    return {k: m["reason"] for k, m in c.get("/healthz").json()["models"].items()}


def test_hash_mismatch_is_rejected_and_never_cached(tmp_path, repo, arts):
    repo.b64[11] = b64(arts / "attention.pt")            # valid artifact, wrong bytes for run 11
    for c in run_app(settings(tmp_path), repo):
        assert reasons(c)["lstm"] == "artifact_hash_mismatch"
        assert reasons(c)["attention"] is None
        r = post(c, "lstm")
        assert r.status_code == 503 and r.json()["detail"]["code"] == "artifact_hash_mismatch"
        assert post(c, "attention").status_code == 200   # the verified model still serves
    assert not list((tmp_path / "cache").glob("run11-*"))


def test_corrupted_base64_and_truncation(tmp_path, repo):
    repo.b64[11] = "not base64 !!"
    repo.b64[12] = repo.b64[12][:-8]
    for c in run_app(settings(tmp_path), repo):
        assert reasons(c) == {"lstm": "artifact_invalid", "attention": "artifact_hash_mismatch"}


def test_missing_run_and_missing_artifact(tmp_path, repo):
    del repo.runs[11]
    repo.b64.pop(12)
    for c in run_app(settings(tmp_path), repo):
        assert reasons(c) == {"lstm": "run_not_found", "attention": "artifact_not_in_database"}
    assert set(repo.artifact_ids) == {12}                # no artifact lookup for a missing run


def test_run_without_recorded_hash_is_refused(tmp_path, repo):
    repo.runs[11]["hyperparameters"] = {}
    repo.runs[12]["hyperparameters"] = {"artifact_sha256": "XYZ"}
    for c in run_app(settings(tmp_path), repo):
        assert reasons(c) == {"lstm": "artifact_hash_missing_in_run", "attention": "artifact_hash_missing_in_run"}
    assert repo.artifact_calls == 0
    with pytest.raises(ArtifactError):
        expected_sha256({"hyperparameters": None})


def test_artifact_of_wrong_model_type_for_the_run(tmp_path, repo, arts):
    # run 11 claims lstm but its recorded hash/bytes are the attention artifact
    repo.add_run(11, "lstm", arts / "attention.pt")
    repo.b64[11] = b64(arts / "attention.pt")
    for c in run_app(settings(tmp_path), repo):
        assert reasons(c)["lstm"] == "artifact_type_mismatch"


def test_oversized_artifact_row(tmp_path, repo, monkeypatch):
    import api.artifact_store as store
    monkeypatch.setattr(store, "MAX_B64_CHARS", 100)
    for c in run_app(settings(tmp_path), repo):
        assert set(reasons(c).values()) == {"artifact_too_large"}


def test_run_ids_are_never_guessed(tmp_path, repo):
    for c in run_app(settings(tmp_path, run_ids={}), repo):
        assert set(reasons(c).values()) == {"run_id_not_configured"}
        assert c.get("/healthz").status_code == 503
    assert repo.get_run_calls == 0 and repo.artifact_calls == 0
    assert repo.predictions == []


def test_invalid_source_setting_fails_closed(tmp_path, repo):
    s = ForecastSettings.from_env({"FORECAST_ARTIFACT_SOURCE": "s3", "FORECAST_LSTM_RUN_ID": "11"})
    assert s.config_error == "artifact_source_invalid"
    for c in run_app(s, repo):
        assert set(reasons(c).values()) == {"artifact_source_invalid"} and post(c).status_code == 503
    assert ForecastSettings.from_env({"FORECAST_RETRY_COOLDOWN_S": "-1"}).config_error == "retry_cooldown_invalid"
    assert ForecastSettings.from_env({}).artifact_source == "local"


# ------------------------------------------------- startup failure and recovery
def test_database_down_at_startup_then_recovers(tmp_path, repo):
    repo.db_up = False
    for c in run_app(settings(tmp_path), repo):
        assert reasons(c) == {"lstm": "database_unavailable", "attention": "database_unavailable"}
        assert c.get("/livez").status_code == 200 and c.get("/healthz").status_code == 503
        assert post(c).status_code == 503
        repo.db_up = True                                 # recovery without a restart
        assert c.get("/healthz").status_code == 200
        assert post(c).status_code == 200


def test_pending_registration_then_registered_while_running(tmp_path, repo):
    runs = dict(repo.runs)
    repo.runs.clear()
    for c in run_app(settings(tmp_path), repo):
        assert set(reasons(c).values()) == {"run_not_found"} and c.get("/livez").status_code == 200
        repo.runs.update(runs)
        assert c.get("/healthz").status_code == 200 and post(c, "lstm").status_code == 200


def test_retries_are_rate_limited_by_cooldown(tmp_path, repo):
    repo.db_up = False
    for c in run_app(settings(tmp_path, cooldown=3600), repo):
        before = repo.get_run_calls
        for _ in range(5):
            c.get("/healthz")
            post(c)
        assert repo.get_run_calls == before               # no retry storm from health checks
    # and with no cooldown every check retries
    repo.get_run_calls = 0
    for c in run_app(settings(tmp_path, cooldown=0), repo):
        base = repo.get_run_calls
        c.get("/healthz")
        assert repo.get_run_calls > base


def test_livez_is_liveness_only_and_healthz_stays_honest(tmp_path, repo):
    repo.runs.clear()
    for c in run_app(settings(tmp_path), repo):
        assert c.get("/livez").json() == {"status": "alive"}
        h = c.get("/healthz")
        assert h.status_code == 503 and h.json()["status"] == "degraded" and h.json()["ready"] is False


def test_startup_thread_does_not_block_port_open(tmp_path, repo):
    import threading
    gate = threading.Event()
    real = repo.get_run

    def slow(run_id):
        gate.wait(timeout=10)
        return real(run_id)
    repo.get_run = slow
    app = create_app(settings(tmp_path), repo)
    with TestClient(app) as c:                            # entered while startup is still blocked
        assert c.get("/livez").status_code == 200
        assert reasons(c) == {"lstm": "starting", "attention": "starting"}
        assert post(c).json()["detail"]["code"] == "starting"
        gate.set()
        app.state.startup_thread.join(timeout=30)
        assert c.get("/healthz").status_code == 200
