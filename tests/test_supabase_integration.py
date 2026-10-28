"""OPT-IN live test: POST /predict logs a ``predictions`` row joined to the right ``runs`` row.

Skipped unless SEQSIGHT_IT_ENABLE=1 and every required SEQSIGHT_IT_* variable is set
(docs/integration-testing.md). The default suite therefore makes no network call, and a
skip is NOT a verification. It uses an existing, already registered test project:

* it creates no runs and no run_artifacts and edits nothing;
* each model type is exercised once; the only rows it writes are the ``predictions`` rows
  the API itself inserts, and those exact rows (id + input hash + run id) are deleted in a
  ``finally`` block.

Prerequisites (not created here): a separate, fresh Supabase test project initialized with
ONLY db/migrations/002_seq_sight_forecasting.sql (a standalone initialization; 001_init.sql
must not be applied, its tables conflict), plus one legitimately evaluated and registered lstm
run and one attention run in that project (with their ``run_artifacts`` rows unless
``SEQSIGHT_IT_ARTIFACT_SOURCE=local``). Production run ids do not apply to another project.
"""
from __future__ import annotations

import os
import time

import pytest
from fastapi.testclient import TestClient

from api.forecast_config import ForecastSettings
from api.forecast_repo import SupabaseForecastRepository
from api.main import create_app
from shared.forecast_models import MODEL_TYPES
from tests import integration_helpers as ih

pytestmark = pytest.mark.integration

_MISSING = ih.missing_config(os.environ)
if _MISSING:
    pytest.skip("integration not configured; set " + ", ".join(_MISSING) +
                " (see docs/integration-testing.md). Skipped is not verified.", allow_module_level=True)

READY_TIMEOUT_S = 90


@pytest.fixture(scope="module")
def cfg():
    return ih.load_config(os.environ)


@pytest.fixture(scope="module")
def db(cfg):
    return ih.make_client(cfg)


@pytest.fixture(scope="module")
def tracked_repo(cfg):
    return ih.TrackingRepository(SupabaseForecastRepository(lambda: ih.make_client(cfg)))


@pytest.fixture(scope="module")
def api(cfg, tracked_repo, tmp_path_factory):
    settings = ForecastSettings.from_env(ih.forecast_env(cfg, tmp_path_factory.mktemp("it-cache")))
    assert settings.config_error is None, settings.config_error
    app = create_app(settings, tracked_repo)
    with TestClient(app) as client:                    # runs the lifespan (startup + artifact fetch)
        deadline = time.monotonic() + READY_TIMEOUT_S
        while True:
            r = client.get("/healthz")
            if r.status_code == 200:
                break
            if time.monotonic() > deadline:
                reasons = {k: v.get("reason") for k, v in r.json().get("models", {}).items()}
                pytest.fail(f"API not ready after {READY_TIMEOUT_S}s: {reasons}")
            time.sleep(2)
        yield client


@pytest.fixture(scope="module", autouse=True)
def cleanup(tracked_repo, db):
    """Safety net: remove any tracked row a test's own `finally` did not already remove."""
    try:
        yield
    finally:
        problems = ih.cleanup_predictions(db, tracked_repo.created)
        if problems:
            pytest.fail("cleanup problems: " + "; ".join(problems), pytrace=False)


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_predict_logs_row_joined_to_registered_run(api, db, cfg, tracked_repo, model_type):
    history = ih.sample_history(model_type)
    before = len(tracked_repo.created)
    try:
        resp = api.post("/predict", json={"model_type": model_type, "history_c": history})
        assert resp.status_code == 200, resp.json()
        body = resp.json()
        assert len(tracked_repo.created) == before + 1      # the API wrote exactly one row
        assert body["prediction_id"] == tracked_repo.created[-1][0]

        stored, run = ih.fetch_logged(db, body["prediction_id"])
        assert stored is not None, "no predictions row was found for the returned prediction_id"
        assert run is not None, "the prediction references a run that cannot be read"
        mismatches = ih.verify_logged_prediction(body, stored, run, model_type=model_type, history_c=history,
                                                 expected_run_id=cfg.run_ids[model_type])
        assert mismatches == []
    finally:
        mine = tracked_repo.created[before:]                # exactly the rows inserted during this test
        problems = ih.cleanup_predictions(db, mine)
        if not problems:
            del tracked_repo.created[before:]               # removed; the module safety net skips them
        else:
            print("cleanup problems:", "; ".join(problems))  # rows stay tracked; the module fixture reports them
