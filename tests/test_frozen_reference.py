"""Frozen-reference forecasting regression tests.

Expected values come from the stored fixtures in tests/fixtures/ (generated once by
scripts/generate_frozen_reference.py); nothing here regenerates its own expected answers.

* ``test_synthetic_*``: always run. Deterministic UNTRAINED models (fixed seed) guard the serving
  path, scaler and shapes. They are NOT verification of the trained checkpoints.
* ``test_trained_*``: need the git-ignored seed-0 artifacts (artifacts/epoch50/final, or the directory in
  SEQSIGHT_REFERENCE_ARTIFACT_DIR). They are SKIPPED when absent, and a skip is not verification.
"""
from __future__ import annotations

import copy
import os
import re
import tempfile
from pathlib import Path

import numpy as np
import pytest

from api.forecast_service import file_sha256, load_model, run_forecasts
from shared.forecast_models import MODEL_TYPES
from shared.forecasting import Standardizer
from tests import frozen_reference as fr
from tests.test_forecast_api import make_artifact

ARTIFACT_DIR = Path(os.environ.get("SEQSIGHT_REFERENCE_ARTIFACT_DIR")
                    or fr.REPO / "artifacts" / "epoch50" / "final")


def _forecast(loaded):
    f = run_forecasts(loaded, [fr.reference_input()])[0]
    return {"forecast_c": f.forecast_c, "attention": f.attention}


# ------------------------------------------------------ fixture structure (always)

@pytest.mark.parametrize("path", [fr.TRAINED_FIXTURE, fr.SYNTHETIC_FIXTURE], ids=["trained", "synthetic"])
def test_fixture_is_complete_and_input_is_the_documented_one(path):
    fx = fr.load_fixture(path)
    assert fx["format"] == fr.FORMAT and fx["generation"] and fx["torch_version"] and fx["generated_utc"]
    assert fx["input"]["units"] == "degC" and fx["input"]["definition"] == fr.INPUT_DEFINITION
    assert fx["input"]["history_c"] == fr.reference_input() and len(fx["input"]["history_c"]) == 72
    assert fx["tolerances"]["forecast_atol_c"] == fr.FORECAST_ATOL_C and fx["tolerances"]["attention_atol"] == fr.ATTENTION_ATOL
    assert fx["scaler"]["std"] > 0 and fx["data_version"] and set(fx["expected"]) == set(MODEL_TYPES)
    for kind, exp in fx["expected"].items():
        assert len(exp["forecast_c"]) == 24 and np.isfinite(exp["forecast_c"]).all()
        assert ("attention_probe" in exp) == (kind == "attention")
    p = fx["expected"]["attention"]["attention_probe"]
    assert len(p["last_row"]) == len(p["column_means"]) == 72 and sum(p["last_row"]) == pytest.approx(1.0, abs=1e-5)


def test_trained_fixture_pins_the_registered_seed0_artifacts():
    fx = fr.load_fixture(fr.TRAINED_FIXTURE)
    hashes = {k: v["sha256"] for k, v in fx["artifacts"].items()}
    assert all(re.fullmatch(r"[0-9a-f]{64}", h) for h in hashes.values()) and len(set(hashes.values())) == 2
    sql = (fr.REPO / "docs" / "model-comparison.sql").read_text()
    seed0 = dict(re.findall(r"\('(lstm|attention)', 0, '([0-9a-f]{64})'\)", sql))
    assert hashes == seed0                                   # same hashes as the registered runs 1 and 4
    assert fx["data_version"] == "jena-mpi-roof-h1-60138aac7d898b8c" and fx["scaler"]["unit"] == "degC"


def test_synthetic_fixture_is_labelled_as_not_the_trained_models():
    fx = fr.load_fixture(fr.SYNTHETIC_FIXTURE)
    assert fx["kind"] == "synthetic_untrained" and "NOT the trained checkpoints" in fx["model_recipe"]["disclaimer"]
    assert "artifacts" not in fx


def test_comparison_detects_changes_beyond_tolerance_and_accepts_noise():
    fx = fr.load_fixture(fr.SYNTHETIC_FIXTURE)

    def fake(delta=0.0, attn_delta=0.0):
        att = np.full((72, 72), 1 / 72)
        e = fx["expected"]["attention"]["attention_probe"]
        att[-1] = np.array(e["last_row"]) + attn_delta
        att[:, :] = np.where(np.arange(72)[:, None] == 71, att, np.array(e["column_means"]) * 1.0)
        return {"lstm": {"forecast_c": np.array(fx["expected"]["lstm"]["forecast_c"]) + delta, "attention": None},
                "attention": {"forecast_c": np.array(fx["expected"]["attention"]["forecast_c"]) + delta, "attention": att}}

    assert fr.compare({**fx, "expected": {"lstm": fx["expected"]["lstm"]}}, {"lstm": fake(1e-6)["lstm"]}) == []
    assert fr.compare({**fx, "expected": {"lstm": fx["expected"]["lstm"]}}, {"lstm": fake(0.01)["lstm"]})
    bad = {"lstm": {"forecast_c": fx["expected"]["lstm"]["forecast_c"], "attention": [[1.0]]},
           "attention": fake()["attention"]}
    assert any("lstm: attention must be None" in m for m in fr.compare(fx, bad))


# ----------------------------------- synthetic-model regression (always runs)

def test_synthetic_models_match_frozen_forecasts():
    fx = fr.load_fixture(fr.SYNTHETIC_FIXTURE)
    scaler = Standardizer(fx["scaler"]["mean"], fx["scaler"]["std"])
    got = {}
    with tempfile.TemporaryDirectory() as d:
        for kind in MODEL_TYPES:
            p = Path(d) / f"{kind}.pt"
            make_artifact(p, kind, seed=fx["model_recipe"]["seed"], scaler=scaler)
            got[kind] = _forecast(load_model(kind, p))
    assert fr.compare(fx, got) == []
    assert got["lstm"]["attention"] is None and np.asarray(got["attention"]["attention"]).shape == (72, 72)


# ------------------------------- trained-artifact regression (skips when absent)

def _trained_paths(fx):
    paths = {k: ARTIFACT_DIR / v["file"] for k, v in fx["artifacts"].items()}
    missing = [p.name for p in paths.values() if not p.is_file()]
    if missing:
        pytest.skip(f"trained seed-0 artifacts not found in {ARTIFACT_DIR.name}/ ({', '.join(missing)}); "
                    "they are git-ignored. Skipped is NOT verification of the trained checkpoints "
                    "(docs/frozen-reference.md).")
    return paths


def test_trained_artifacts_match_frozen_forecasts():
    fx = fr.load_fixture(fr.TRAINED_FIXTURE)
    paths = _trained_paths(fx)
    for kind, p in paths.items():
        assert file_sha256(p) == fx["artifacts"][kind]["sha256"], f"{p.name} is not the frozen seed-0 artifact"
    got = {}
    for kind, p in paths.items():
        loaded = load_model(kind, p)
        assert (loaded.scaler.mean, loaded.scaler.std) == (fx["scaler"]["mean"], fx["scaler"]["std"])
        assert loaded.meta["data_version"] == fx["data_version"] and loaded.meta["seed"] == 0
        got[kind] = _forecast(loaded)
    assert fr.compare(fx, got) == []
