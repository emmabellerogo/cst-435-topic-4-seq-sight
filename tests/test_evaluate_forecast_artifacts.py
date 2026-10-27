"""Synthetic tests for scripts/evaluate_forecast_artifacts.py (no real data, no network)."""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

import scripts.evaluate_forecast_artifacts as ev
import scripts.register_forecast_runs as reg
from shared import forecast_training as ft
from tests.test_forecast_api import FakeSb, make_artifact
from tests.test_forecast_training import synthetic_data

KEYS = ["lstm_seed0", "attention_seed1"]


@pytest.fixture(scope="module")
def data():
    return synthetic_data()


@pytest.fixture
def art(tmp_path, data):
    for k in KEYS:
        kind, _, seed = k.rpartition("_seed")
        make_artifact(tmp_path / f"{k}.pt", kind, seed=int(seed), scaler=data.scaler)
    return tmp_path


def run(art, data, *extra, loader=None, out=None):
    calls = []

    def default(version):
        calls.append(version)
        return data
    rc = ev.main(["--artifact-dir", str(art), "--keys", *KEYS, *(["--output", str(out)] if out else []), *extra],
                 data_loader=loader or default)
    return rc, calls


def test_dry_run_reads_no_data_and_writes_nothing(art, data, capsys):
    rc, calls = run(art, data)
    assert rc == 0 and calls == []
    assert not (art / "test_eval.json").exists()
    assert "Dry-run only" in capsys.readouterr().out


def test_execute_reports_test_mae_matching_independent_computation(art, data):
    rc, calls = run(art, data, "--execute", "--batch-size", "50")
    assert rc == 0 and calls == ["synthetic-v1"]
    res = json.loads((art / "test_eval.json").read_text())
    assert res["format"] == "seqsight-test-eval-v1" and res["data_version"] == "synthetic-v1"
    assert set(res["test_mae_c"]) == set(KEYS) and set(res["models"]) == set(KEYS)
    test = data.splits["test"]
    for k in KEYS:
        model, scaler, meta = ft.load_artifact(art / f"{k}.pt")
        with torch.no_grad():
            z, _ = model(torch.from_numpy(test.inputs))
        ref = np.abs(scaler.inverse_transform(z.double().numpy()) - scaler.inverse_transform(test.targets)).mean()
        m = res["models"][k]
        assert m["test_mae_c"] == pytest.approx(ref, rel=1e-6)
        assert res["test_mae_c"][k] == m["test_mae_c"]
        assert m["artifact_sha256"] == ev.file_sha256(art / f"{k}.pt") and len(m["weights_sha256"]) == 64
        assert m["n_examples"] == len(test) and len(m["test_mae_by_horizon_c"]) == 24
        assert m["val_mae_c"] == meta["best_val_mae_c"]
    base = ft.evaluate_persistence(test, data.scaler)
    assert res["persistence_test_mae_c"] == pytest.approx(base.mae_c)
    cfg = res["evaluation_config"]
    assert cfg["training"] == "none" and cfg["batch_size"] == 50 and cfg["n_test_windows"] == len(test)


def test_batch_size_does_not_change_result(art, data):
    run(art, data, "--execute", "--batch-size", "7", out=art / "a.json")
    run(art, data, "--execute", "--batch-size", "1024", out=art / "b.json")
    a, b = (json.loads((art / f).read_text())["test_mae_c"] for f in ("a.json", "b.json"))
    assert a == pytest.approx(b, rel=1e-9)


def test_evaluation_never_trains_or_changes_weights(art, data, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("training machinery was invoked")
    monkeypatch.setattr(ft, "train_model", boom)
    monkeypatch.setattr(torch.optim.Adam, "__init__", boom)
    monkeypatch.setattr(torch.optim.Optimizer, "step", boom)
    monkeypatch.setattr(torch.Tensor, "backward", boom)
    before = {k: ev.file_sha256(art / f"{k}.pt") for k in KEYS}
    items = ev.load_fixed(art, KEYS)
    fp = {i["key"]: ev.weights_fingerprint(i["model"]) for i in items}
    ev.evaluate_fixed(items, data)
    for i in items:
        assert ev.weights_fingerprint(i["model"]) == fp[i["key"]]
        assert not i["model"].training
        assert all(p.grad is None for p in i["model"].parameters())
    assert {k: ev.file_sha256(art / f"{k}.pt") for k in KEYS} == before     # files untouched
    assert run(art, data, "--execute")[0] == 0                              # whole CLI path, too


def test_weight_mutation_during_evaluation_is_detected(art, data, monkeypatch):
    items = ev.load_fixed(art, KEYS)
    real = ft.evaluate

    def tampering(model, *a, **k):
        r = real(model, *a, **k)
        with torch.no_grad():
            next(model.parameters()).add_(1e-3)
        return r
    monkeypatch.setattr(ft, "evaluate", tampering)
    with pytest.raises(ev.EvalError, match="weights changed"):
        ev.evaluate_fixed(items, data)


def test_refuses_to_overwrite_existing_results(art, data, capsys):
    assert run(art, data, "--execute")[0] == 0
    first = (art / "test_eval.json").read_text()
    rc, calls = run(art, data, "--execute")
    assert rc == 2 and calls == []                       # refused before touching the corpus
    assert "refusing to overwrite" in capsys.readouterr().err
    assert (art / "test_eval.json").read_text() == first
    with pytest.raises(ev.EvalError):                    # and the writer itself is exclusive
        ev.write_exclusive(art / "test_eval.json", {})


def test_refuses_mismatched_corpus(art, data, capsys):
    class Other:
        data_version, scaler, splits = "other-v1", data.scaler, data.splits
    assert run(art, data, "--execute", loader=lambda v: Other)[0] == 2
    from shared.forecasting import Standardizer

    class Shifted:
        data_version, scaler, splits = "synthetic-v1", Standardizer(data.scaler.mean + 0.5, data.scaler.std), data.splits
    assert run(art, data, "--execute", loader=lambda v: Shifted)[0] == 2
    assert not (art / "test_eval.json").exists()


def test_refuses_metadata_inconsistencies(art, data, tmp_path):
    make_artifact(art / "lstm_seed2.pt", "lstm", seed=0, scaler=data.scaler)       # seed disagrees with name
    with pytest.raises(ev.EvalError, match="file name disagrees"):
        ev.load_fixed(art, ["lstm_seed2"])
    make_artifact(art / "attention_seed0.pt", "lstm", seed=0, scaler=data.scaler)  # type disagrees with name
    with pytest.raises(ev.EvalError, match="file name disagrees"):
        ev.load_fixed(art, ["attention_seed0"])
    from shared.forecasting import Standardizer
    make_artifact(art / "lstm_seed1.pt", "lstm", seed=1, scaler=Standardizer(0.0, 1.0))
    with pytest.raises(ev.EvalError, match="scaler"):
        ev.load_fixed(art, ["lstm_seed0", "lstm_seed1"])
    with pytest.raises(ev.EvalError, match="not found"):
        ev.load_fixed(art, ["lstm_seed9"])


def test_failure_message_hides_exception_details(art, data, capsys):
    def bad(version):
        raise RuntimeError("https://secret-ref.supabase.co SECRET")
    assert run(art, data, "--execute", loader=bad)[0] == 1
    assert "SECRET" not in capsys.readouterr().err


def test_result_feeds_registration_and_checks_artifact_hash(art, data, capsys):
    run(art, data, "--execute")
    result = art / "test_eval.json"
    sb = FakeSb()
    rc = reg.main(["--artifact-dir", str(art), "--keys", *KEYS, "--execute", "--test-mae-file", str(result)],
                  client_factory=lambda: sb)
    assert rc == 0
    saved = json.loads(result.read_text())["test_mae_c"]
    got = {r["model_type"]: r["test_mae_c"] for r in sb.tables["runs"]}
    assert got == {"lstm": saved["lstm_seed0"], "attention": saved["attention_seed1"]}
    # results measured on a different file are rejected
    tampered = json.loads(result.read_text())
    tampered["models"]["lstm_seed0"]["artifact_sha256"] = "0" * 64
    bad = art / "bad.json"
    bad.write_text(json.dumps(tampered))
    sb2 = FakeSb()
    rc = reg.main(["--artifact-dir", str(art), "--keys", *KEYS, "--execute", "--test-mae-file", str(bad)],
                  client_factory=lambda: sb2)
    assert rc == 2 and sb2.tables["runs"] == []
    assert "different artifact" in capsys.readouterr().err
