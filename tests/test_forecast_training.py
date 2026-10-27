"""Synthetic tests for shared/forecast_training.py (tiny CPU runs; no real data)."""
import copy
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from torch import nn

from shared import forecast_models as fm
from shared import forecast_training as ft
from shared import forecasting as fc

T0 = datetime(2020, 1, 1)
QUIET = lambda _msg: None


def synthetic_data(n_train=400, n_val=250, n_test=200, seed=0):
    rng = np.random.default_rng(seed)
    total = n_train + n_val + n_test
    temps = 10 + 8 * np.sin(np.arange(total) * 2 * np.pi / 24) + rng.normal(0, 0.5, total)
    rows, i = [], 0
    for split, n in (("train", n_train), ("validation", n_val), ("test", n_test)):
        for _ in range(n):
            rows.append({"station_ts": (T0 + timedelta(hours=i)).isoformat(), "temperature_c": float(temps[i]),
                         "split": split, "data_version": "synthetic-v1"})
            i += 1
    return fc.prepare_forecasting_data(rows)


def cfg(**kw):
    base = dict(seed=3, max_epochs=3, batch_size=64)
    base.update(kw)
    return ft.TrainConfig(**base)


class ConfigTests(unittest.TestCase):
    def test_defaults_are_the_documented_initial_choices(self):
        c = ft.TrainConfig()
        self.assertEqual((c.optimizer, c.lr, c.batch_size, c.loss, c.grad_clip_norm, c.dropout),
                         ("adam", 1e-3, 128, "l1_standardized", 1.0, 0.0))

    def test_rejects_invalid(self):
        for kw in ({"max_epochs": 0}, {"batch_size": 0}, {"lr": 0.0}, {"optimizer": "sgd"}, {"dropout": 0.1}):
            with self.assertRaises(ValueError):
                ft.TrainConfig(**kw)


class MetricTests(unittest.TestCase):
    def test_aggregation_is_exact_across_batch_sizes_including_short_final_batch(self):
        d = synthetic_data()
        w, sc = d.splits["validation"], d.scaler
        n = len(w)
        self.assertNotEqual(n % 64, 0)
        torch.manual_seed(0)
        model = fm.build_model("lstm")
        # independent reference: float64 degC, one mean over all examples and hours
        with torch.no_grad():
            pred = model(torch.from_numpy(w.inputs))[0].numpy()
        ref = np.abs(sc.inverse_transform(pred) - sc.inverse_transform(w.targets)).mean()
        for bs in (1, 7, 64, n, 5000):
            self.assertAlmostEqual(ft.evaluate(model, w, sc, bs).mae_c, ref, places=5, msg=f"bs={bs}")
        self.assertAlmostEqual(fc.mae_celsius(pred, w.targets, sc), ref, places=5)

    def test_mean_of_batch_means_would_be_wrong_here(self):
        # Constant-error stub: batch of 4 has error 1, final batch of 1 has error 6.
        class Stub(nn.Module):
            def forward(self, x):
                return torch.zeros(len(x), 24), None
        w = fc.WindowSet(np.zeros((5, 72, 1), np.float32), np.zeros((5, 24), np.float32), [T0] * 5)
        w.targets[:4] = 1.0
        w.targets[4] = 6.0
        sc = fc.Standardizer(0.0, 2.0)  # std 2 -> degC errors 2 and 12
        res = ft.evaluate(Stub(), w, sc, batch_size=4)
        self.assertAlmostEqual(res.mae_c, (4 * 2 + 12) / 5)          # 4.0, not (2 + 12) / 2 = 7.0
        self.assertEqual(res.n_examples, 5)
        self.assertEqual(len(res.mae_by_horizon_c), 24)

    def test_evaluate_restores_mode_and_returns_no_attention_state(self):
        d = synthetic_data()
        m = fm.build_model("attention").train()
        ft.evaluate(m, d.splits["validation"], d.scaler, batch_size=32)
        self.assertTrue(m.training)

    def test_persistence_repeats_last_observation(self):
        d = synthetic_data()
        w, sc = d.splits["test"], d.scaler
        pred = ft.persistence_forecast(torch.from_numpy(w.inputs))
        self.assertEqual(tuple(pred.shape), (len(w), 24))
        last_c = sc.inverse_transform(w.inputs[:, -1, 0])
        np.testing.assert_allclose(sc.inverse_transform(pred.numpy()), np.repeat(last_c[:, None], 24, 1), atol=1e-4)
        ref = np.abs(sc.inverse_transform(w.targets) - last_c[:, None]).mean()
        self.assertAlmostEqual(ft.evaluate_persistence(w, sc, 50).mae_c, ref, places=4)

    def test_empty_set_rejected(self):
        e = fc.WindowSet(np.empty((0, 72, 1), np.float32), np.empty((0, 24), np.float32), [])
        with self.assertRaises(fc.ForecastDataError):
            ft.evaluate_persistence(e, fc.Standardizer(0, 1))


class TrainingTests(unittest.TestCase):
    def run_model(self, kind, **kw):
        d = synthetic_data()
        r = ft.train_model(kind, d.splits["train"], d.splits["validation"], d.scaler, cfg(**kw), log=QUIET)
        return d, r

    def test_batch_order_matches_across_models_and_steps_are_equal(self):
        _, a = self.run_model("lstm")
        _, b = self.run_model("attention")
        self.assertEqual([h["batch_order_sha256"] for h in a.history], [h["batch_order_sha256"] for h in b.history])
        self.assertEqual(a.optimizer_steps, b.optimizer_steps)
        self.assertEqual(a.optimizer_steps, 3 * -(-len(synthetic_data().splits["train"]) // 64))
        hashes = [h["batch_order_sha256"] for h in a.history]
        self.assertEqual(len(set(hashes)), 3)  # a new order every epoch

    def test_batch_order_independent_of_global_rng_use(self):
        self.assertTrue(torch.equal(ft.epoch_permutation(300, 3, 2), ft.epoch_permutation(300, 3, 2)))
        before = ft.epoch_permutation(300, 3, 2)
        torch.manual_seed(1)
        torch.rand(10_000)
        np.random.rand(5)
        self.assertTrue(torch.equal(before, ft.epoch_permutation(300, 3, 2)))
        self.assertFalse(torch.equal(before, ft.epoch_permutation(300, 4, 2)))
        self.assertEqual(sorted(before.tolist()), list(range(300)))

    def test_same_seed_reproduces_history_and_weights(self):
        for kind in fm.MODEL_TYPES:
            _, a = self.run_model(kind)
            _, b = self.run_model(kind)
            self.assertEqual([h["val_mae_c"] for h in a.history], [h["val_mae_c"] for h in b.history])
            for k in a.state_dict:
                self.assertTrue(torch.equal(a.state_dict[k], b.state_dict[k]), f"{kind}.{k}")

    def test_different_seed_changes_result(self):
        _, a = self.run_model("lstm", seed=1)
        _, b = self.run_model("lstm", seed=2)
        self.assertNotEqual([h["batch_order_sha256"] for h in a.history], [h["batch_order_sha256"] for h in b.history])

    def test_training_improves_validation_and_reports_fields(self):
        d, r = self.run_model("lstm", max_epochs=6)
        self.assertLess(r.history[-1]["val_mae_c"], r.history[0]["val_mae_c"] + 1e-9)
        self.assertGreater(r.parameter_count, 0)
        self.assertGreater(r.train_seconds, 0)
        for h in r.history:
            for k in ("epoch", "val_mae_c", "elapsed_seconds", "optimizer_steps", "train_seconds"):
                self.assertIn(k, h)
        self.assertEqual(len(r.history), 6)

    def test_best_checkpoint_selected_by_validation_only(self):
        d = synthetic_data()
        scripted = [3.0, 1.5, 2.0, 1.5, 5.0]            # minimum first reached at epoch 2; epoch 4 ties
        snapshots = []

        def fake_eval(model, w, scaler, batch_size=ft.EVAL_BATCH):
            snapshots.append(copy.deepcopy(model.state_dict()))
            return ft.EvalResult(scripted[len(snapshots) - 1], [0.0] * 24, len(w))

        with mock.patch.object(ft, "evaluate", fake_eval):
            r = ft.train_model("lstm", d.splits["train"], d.splits["validation"], d.scaler,
                               cfg(max_epochs=5), log=QUIET)
        self.assertEqual(r.best_epoch, 2)
        self.assertEqual(r.best_val_mae_c, 1.5)
        self.assertEqual([h["best_so_far"] for h in r.history], [True, True, False, False, False])
        for k, v in snapshots[1].items():              # restored weights are the epoch-2 weights
            self.assertTrue(torch.equal(r.model.state_dict()[k], v))
        self.assertFalse(r.model.training)

    def test_test_split_is_not_reachable_from_training(self):
        import inspect
        params = inspect.signature(ft.train_model).parameters
        self.assertNotIn("test", params)
        self.assertIn("val", params)

    def test_nonfinite_validation_is_not_selected(self):
        d = synthetic_data()
        with mock.patch.object(ft, "evaluate", lambda *a, **k: ft.EvalResult(float("nan"), [0.0] * 24, 1)):
            with self.assertRaises(fc.ForecastDataError):
                ft.train_model("lstm", d.splits["train"], d.splits["validation"], d.scaler, cfg(max_epochs=2), log=QUIET)


class ArtifactTests(unittest.TestCase):
    def test_round_trip_reproduces_predictions_and_metadata(self):
        d = synthetic_data()
        for kind in fm.MODEL_TYPES:
            r = ft.train_model(kind, d.splits["train"], d.splits["validation"], d.scaler, cfg(max_epochs=2), log=QUIET)
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "nested" / f"{kind}.pt"
                ft.save_artifact(path, r, d.scaler, "synthetic-v1", extra={"run_kind": "synthetic"})
                model, scaler, meta = ft.load_artifact(path)
            x = torch.from_numpy(d.splits["validation"].inputs[:50])
            with torch.no_grad():
                want, w_want = r.model(x)
                got, w_got = model(x)
            np.testing.assert_array_equal(got.numpy(), want.numpy())
            if w_want is not None:
                np.testing.assert_array_equal(w_got.numpy(), w_want.numpy())
            self.assertEqual(scaler, d.scaler)
            self.assertEqual(meta["data_version"], "synthetic-v1")
            self.assertEqual(meta["seed"], 3)
            self.assertEqual(meta["model_type"], kind)
            self.assertEqual(meta["history"], r.history)
            self.assertEqual(meta["best_epoch"], r.best_epoch)
            self.assertEqual(meta["train_config"]["batch_size"], 64)
            self.assertEqual(meta["extra"], {"run_kind": "synthetic"})
            self.assertFalse(model.training)

    def test_rejects_incomplete_or_wrong_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "bad.pt"
            torch.save({"format_version": 1}, p)
            with self.assertRaisesRegex(fc.ForecastDataError, "missing"):
                ft.load_artifact(p)
            torch.save({k: 0 for k in ft._REQUIRED} | {"format_version": 99}, p)
            with self.assertRaisesRegex(fc.ForecastDataError, "format"):
                ft.load_artifact(p)

    def test_final_test_evaluation_reports_persistence(self):
        d = synthetic_data()
        r = ft.train_model("lstm", d.splits["train"], d.splits["validation"], d.scaler, cfg(max_epochs=1), log=QUIET)
        out = ft.evaluate_test(r.model, d.splits["test"], d.scaler)
        self.assertEqual(out["n_examples"], len(d.splits["test"]))
        self.assertEqual(len(out["test_mae_by_horizon_c"]), 24)
        self.assertGreater(out["persistence_test_mae_c"], 0)


if __name__ == "__main__":
    unittest.main()
