"""Synthetic tests for shared/forecasting.py and shared/forecast_store.py.

No Supabase, network or real data: rows are generated in memory and the database
client is a recording fake.
"""
import json
import unittest
from datetime import datetime, timedelta

import numpy as np

from shared import forecast_store as store
from shared import forecasting as fc

T0 = datetime(2020, 1, 1, 0, 0)
DV = "jena-test-v1"


def row(i, temp, split="train", dv=DV, start=T0):
    return {"station_ts": (start + timedelta(hours=i)).isoformat(), "temperature_c": temp,
            "split": split, "data_version": dv}


def series(n, split="train", start=T0, offset=0, f=lambda i: float(i)):
    return [row(offset + i, f(offset + i), split, start=start) for i in range(n)]


class ValidationTests(unittest.TestCase):
    def test_sorts_and_keeps_naive_timestamps(self):
        rows = series(5)[::-1]
        version, obs = fc.validate_rows(rows)
        self.assertEqual(version, DV)
        self.assertEqual([o.ts for o in obs], [T0 + timedelta(hours=i) for i in range(5)])
        self.assertTrue(all(o.ts.tzinfo is None for o in obs))

    def test_accepts_datetime_objects(self):
        r = row(0, 1.0)
        r["station_ts"] = T0
        self.assertEqual(fc.validate_rows([r])[1][0].ts, T0)

    def bad(self, mutate, msg):
        rows = series(3)
        mutate(rows)
        with self.assertRaisesRegex(fc.ForecastDataError, msg):
            fc.validate_rows(rows)

    def test_missing_and_null_fields(self):
        for f in fc.REQUIRED_FIELDS:
            self.bad(lambda r, f=f: r[1].pop(f), "missing required field")
            self.bad(lambda r, f=f: r[1].__setitem__(f, None), "missing required field")

    def test_nonfinite_and_nonnumeric_temperature(self):
        for v in (float("nan"), float("inf"), float("-inf")):
            self.bad(lambda r, v=v: r[1].__setitem__("temperature_c", v), "not finite")
        for v in ("12.5", True):
            self.bad(lambda r, v=v: r[1].__setitem__("temperature_c", v), "must be a number")

    def test_mixed_and_unexpected_data_version(self):
        self.bad(lambda r: r[2].__setitem__("data_version", "other"), "differs")
        with self.assertRaisesRegex(fc.ForecastDataError, "differs"):
            fc.validate_rows(series(3), expected_data_version="other")

    def test_unknown_split(self):
        self.bad(lambda r: r[0].__setitem__("split", "holdout"), "unknown split")

    def test_timestamp_problems(self):
        self.bad(lambda r: r[0].__setitem__("station_ts", "garbage"), "unparseable")
        self.bad(lambda r: r[0].__setitem__("station_ts", "2020-01-01T00:30:00"), "exact hour")
        self.bad(lambda r: r[0].__setitem__("station_ts", "2020-01-01T00:00:00+01:00"), "time zone")

    def test_duplicates_rejected_even_if_identical(self):
        rows = series(3) + [row(1, 1.0)]
        with self.assertRaisesRegex(fc.ForecastDataError, "duplicate"):
            fc.validate_rows(rows)

    def test_duplicate_across_splits_rejected(self):
        with self.assertRaisesRegex(fc.ForecastDataError, "duplicate"):
            fc.validate_rows([row(0, 1.0, "train"), row(0, 1.0, "test")])

    def test_split_order_must_be_chronological(self):
        rows = [row(0, 1.0, "validation"), row(1, 1.0, "train")]
        with self.assertRaisesRegex(fc.ForecastDataError, "not chronological"):
            fc.validate_rows(rows)
        rows = [row(0, 1.0, "train"), row(1, 1.0, "validation"), row(2, 1.0, "train")]
        with self.assertRaisesRegex(fc.ForecastDataError, "not chronological"):
            fc.validate_rows(rows)

    def test_empty(self):
        with self.assertRaisesRegex(fc.ForecastDataError, "no rows"):
            fc.validate_rows([])


class WindowTests(unittest.TestCase):
    def ts(self, n, start=0):
        return [T0 + timedelta(hours=start + i) for i in range(n)]

    def test_shapes_alignment_and_origins(self):
        n = 100  # 100 - 96 + 1 = 5 windows
        w = fc.make_windows(self.ts(n), np.arange(n, dtype=float))
        self.assertEqual(w.inputs.shape, (5, 72, 1))
        self.assertEqual(w.targets.shape, (5, 24))
        self.assertEqual(len(w), 5)
        for k in range(5):
            np.testing.assert_array_equal(w.inputs[k, :, 0], np.arange(k, k + 72))
            np.testing.assert_array_equal(w.targets[k], np.arange(k + 72, k + 96))
            self.assertEqual(w.origins[k], T0 + timedelta(hours=k + 71))  # last input hour
        self.assertEqual(w.inputs.dtype, np.float32)

    def test_exact_minimum_and_too_short(self):
        self.assertEqual(len(fc.make_windows(self.ts(96), np.zeros(96))), 1)
        short = fc.make_windows(self.ts(95), np.zeros(95))
        self.assertEqual((short.inputs.shape, short.targets.shape), ((0, 72, 1), (0, 24)))

    def test_gap_is_never_crossed(self):
        # run A: 100 h, missing hour, run B: 96 h -> 5 + 1 windows, none spanning the gap.
        a, b = self.ts(100), self.ts(96, start=101)
        vals = np.concatenate([np.arange(100), 1000 + np.arange(96)]).astype(float)
        w = fc.make_windows(a + b, vals)
        self.assertEqual(len(w), 6)
        for k in range(len(w)):
            window = np.concatenate([w.inputs[k, :, 0], w.targets[k]])
            self.assertTrue((window < 1000).all() or (window >= 1000).all())
            self.assertTrue((np.diff(window) == 1).all())
        self.assertEqual(w.origins[-1], T0 + timedelta(hours=101 + 71))

    def test_run_too_short_before_gap_is_dropped(self):
        ts = self.ts(95) + self.ts(96, start=96)
        self.assertEqual(len(fc.make_windows(ts, np.zeros(len(ts)))), 1)

    def test_unsorted_rejected(self):
        ts = self.ts(3)
        with self.assertRaises(fc.ForecastDataError):
            fc.make_windows([ts[1], ts[0], ts[2]], np.zeros(3))


class PipelineTests(unittest.TestCase):
    def make_rows(self):
        # train 0..119 (25 windows), validation 120..239 (25), test 240..359 (25)
        f = lambda i: 10.0 + 0.1 * i + (i % 7)
        return (series(120, "train", f=f) + series(120, "validation", offset=120, f=f)
                + series(120, "test", offset=240, f=f))

    def test_shapes_and_counts(self):
        d = fc.prepare_forecasting_data(self.make_rows())
        self.assertEqual(d.data_version, DV)
        self.assertEqual(d.observation_counts, {"train": 120, "validation": 120, "test": 120})
        for name in fc.SPLIT_ORDER:
            w = d.splits[name]
            self.assertEqual(w.inputs.shape, (25, 72, 1))
            self.assertEqual(w.targets.shape, (25, 24))
            self.assertEqual(len(w.origins), 25)

    def test_split_isolation(self):
        d = fc.prepare_forecasting_data(self.make_rows())
        bounds = {"train": (0, 119), "validation": (120, 239), "test": (240, 359)}
        for name, (lo, hi) in bounds.items():
            w = d.splits[name]
            self.assertGreaterEqual(w.origins[0], T0 + timedelta(hours=lo + 71))
            self.assertLessEqual(w.origins[-1] + timedelta(hours=24), T0 + timedelta(hours=hi))
        # first validation window starts at the first validation hour, not earlier
        raw = d.scaler.inverse_transform(d.splits["validation"].inputs[0, 0, 0])
        self.assertAlmostEqual(float(raw), 10.0 + 0.1 * 120 + (120 % 7), places=3)

    def test_split_with_gap_at_boundary_produces_no_cross_windows(self):
        rows = series(100, "train") + series(100, "validation", offset=100)
        d = fc.prepare_forecasting_data(rows)
        self.assertEqual(len(d.splits["train"]), 5)
        self.assertEqual(len(d.splits["validation"]), 5)
        self.assertEqual(len(d.splits["test"]), 0)

    def test_scaler_fitted_on_training_only(self):
        rows = self.make_rows()
        train = np.array([r["temperature_c"] for r in rows if r["split"] == "train"])
        d = fc.prepare_forecasting_data(rows)
        self.assertAlmostEqual(d.scaler.mean, train.mean(), places=9)
        self.assertAlmostEqual(d.scaler.std, train.std(ddof=0), places=9)
        # changing validation/test values must not move the scaler
        for r in rows:
            if r["split"] != "train":
                r["temperature_c"] += 500.0
        d2 = fc.prepare_forecasting_data(rows)
        self.assertEqual(d2.scaler, d.scaler)
        # ...and the shifted splits are therefore far from zero-mean in scaled space
        self.assertGreater(d2.splits["test"].inputs.mean(), 10)

    def test_train_inputs_are_standardized_and_invertible(self):
        d = fc.prepare_forecasting_data(self.make_rows())
        series_z = np.concatenate([d.splits["train"].inputs[0, :, 0], d.splits["train"].targets[0]])
        raw = d.scaler.inverse_transform(series_z)
        expected = [10.0 + 0.1 * i + (i % 7) for i in range(96)]
        np.testing.assert_allclose(raw, expected, atol=1e-4)  # float32 storage

    def test_missing_train_rejected(self):
        with self.assertRaisesRegex(fc.ForecastDataError, "no training"):
            fc.prepare_forecasting_data(series(100, "validation"))

    def test_constant_training_rejected(self):
        with self.assertRaisesRegex(fc.ForecastDataError, "std"):
            fc.prepare_forecasting_data(series(100, "train", f=lambda i: 5.0))

    def test_invalid_rows_propagate(self):
        rows = self.make_rows()
        rows[3]["temperature_c"] = float("nan")
        with self.assertRaises(fc.ForecastDataError):
            fc.prepare_forecasting_data(rows)

    def test_inference_input_uses_stored_scaler(self):
        d = fc.prepare_forecasting_data(self.make_rows())
        hist = [10.0 + 0.1 * i + (i % 7) for i in range(72)]
        x = fc.prepare_inference_input(hist, fc.Standardizer.from_dict(d.scaler.to_dict()))
        self.assertEqual(x.shape, (1, 72, 1))
        np.testing.assert_allclose(x, d.splits["train"].inputs[:1], atol=1e-5)
        with self.assertRaises(fc.ForecastDataError):
            fc.prepare_inference_input(hist[:71], d.scaler)


class StandardizerTests(unittest.TestCase):
    def test_population_std(self):
        s = fc.Standardizer.fit([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(s.mean, 2.5)
        self.assertAlmostEqual(s.std, np.sqrt(1.25))  # ddof=0, not sqrt(5/3)

    def test_roundtrip(self):
        s = fc.Standardizer(mean=9.3, std=7.7)
        x = np.array([[-20.0, 0.0, 35.5]])
        np.testing.assert_allclose(s.inverse_transform(s.transform(x)), x)

    def test_serialization_roundtrip_through_json(self):
        s = fc.Standardizer(mean=9.3, std=7.7)
        d = json.loads(json.dumps(s.to_dict()))
        self.assertEqual(fc.Standardizer.from_dict(d), s)
        self.assertEqual(d["mean"], 9.3)
        self.assertEqual(d["std"], 7.7)

    def test_rejects_bad_parameters(self):
        for std in (0.0, -1.0, float("nan"), float("inf")):
            with self.assertRaises(fc.ForecastDataError):
                fc.Standardizer(mean=1.0, std=std)
        with self.assertRaises(fc.ForecastDataError):
            fc.Standardizer(mean=float("nan"), std=1.0)
        with self.assertRaises(fc.ForecastDataError):
            fc.Standardizer.from_dict({"mean": 1.0})
        with self.assertRaises(fc.ForecastDataError):
            fc.Standardizer.from_dict({"mean": 1.0, "std": 1.0, "std_ddof": 1})

    def test_fit_rejects_empty_constant_nonfinite(self):
        for bad in ([], [3.0, 3.0, 3.0], [1.0, float("nan")]):
            with self.assertRaises(fc.ForecastDataError):
                fc.Standardizer.fit(bad)

    def test_transform_rejects_nonfinite(self):
        with self.assertRaises(fc.ForecastDataError):
            fc.Standardizer(0.0, 1.0).transform([1.0, float("nan")])


class MaeTests(unittest.TestCase):
    def test_mae_is_in_degrees_celsius(self):
        s = fc.Standardizer(mean=10.0, std=5.0)
        pred_z = np.array([[0.0, 1.0]])
        true_z = np.array([[0.2, 0.6]])
        # |0 - 0.2|*5 = 1.0 and |1 - 0.6|*5 = 2.0 -> mean 1.5 degC
        self.assertAlmostEqual(fc.mae_celsius(pred_z, true_z, s), 1.5)
        self.assertAlmostEqual(fc.mae_degrees([10.0, 15.0], [11.0, 17.0]), 1.5)

    def test_mae_rejects_bad_input(self):
        s = fc.Standardizer(0.0, 1.0)
        with self.assertRaises(fc.ForecastDataError):
            fc.mae_celsius(np.zeros((2, 3)), np.zeros((2, 4)), s)
        with self.assertRaises(fc.ForecastDataError):
            fc.mae_degrees([float("nan")], [1.0])
        with self.assertRaises(fc.ForecastDataError):
            fc.mae_degrees([], [])


# ------------------------------------------------------------------ database

class FakeQuery:
    def __init__(self, client):
        self.client, self.call = client, {}

    def select(self, cols):
        self.call["select"] = cols
        return self

    def eq(self, col, val):
        self.call["eq"] = (col, val)
        return self

    def order(self, col):
        self.call["order"] = col
        return self

    def range(self, a, b):
        self.call["range"] = (a, b)
        return self

    def execute(self):
        self.client.calls.append(self.call)
        col, val = self.call["eq"]
        a, b = self.call["range"]
        data = [r for r in self.client.rows if r[col] == val]
        data.sort(key=lambda r: r[self.call["order"]])
        proj = self.call["select"].split(",")
        page = [{k: r[k] for k in proj} for r in data[a:b + 1]]
        return type("Resp", (), {"data": page})()


class FakeClient:
    def __init__(self, rows):
        self.rows, self.calls, self.tables = rows, [], []

    def table(self, name):
        self.tables.append(name)
        return FakeQuery(self)


class StoreTests(unittest.TestCase):
    def test_paginates_beyond_one_page_with_explicit_columns_and_order(self):
        rows = series(2500)[::-1]  # stored out of order on purpose
        client = FakeClient(rows + [row(0, 99.0, dv="other-version")])
        got = list(store.fetch_corpus_rows(client, DV))
        self.assertEqual(len(got), 2500)
        self.assertEqual(got[0]["station_ts"], T0.isoformat())  # ordered by station_ts
        self.assertEqual([c["range"] for c in client.calls], [(0, 999), (1000, 1999), (2000, 2999)])
        for c in client.calls:
            self.assertEqual(c["select"], "station_ts,temperature_c,split,data_version")
            self.assertEqual(c["order"], "station_ts")
            self.assertEqual(c["eq"], ("data_version", DV))
        self.assertEqual(set(client.tables), {"sequences"})

    def test_exact_multiple_of_page_size_ends_on_empty_page(self):
        client = FakeClient(series(2000))
        self.assertEqual(len(list(store.fetch_corpus_rows(client, DV))), 2000)
        self.assertEqual(len(client.calls), 3)

    def test_small_page_size(self):
        client = FakeClient(series(25))
        self.assertEqual(len(list(store.fetch_corpus_rows(client, DV, page_size=10))), 25)
        self.assertEqual([c["range"] for c in client.calls], [(0, 9), (10, 19), (20, 29)])

    def test_bad_arguments(self):
        client = FakeClient([])
        for kwargs in ({"page_size": 0}, {"page_size": 1001}):
            with self.assertRaises(fc.ForecastDataError):
                list(store.fetch_corpus_rows(client, DV, **kwargs))
        with self.assertRaises(fc.ForecastDataError):
            list(store.fetch_corpus_rows(client, ""))
        self.assertEqual(client.calls, [])

    def test_unknown_version_gives_clear_error(self):
        with self.assertRaisesRegex(fc.ForecastDataError, "no rows"):
            store.load_forecasting_data(FakeClient(series(5)), "missing-version")

    def test_load_forecasting_data_end_to_end(self):
        # 1200 train + 1100 validation + 1000 test rows spans several pages.
        rows = (series(1200, "train") + series(1100, "validation", offset=1200)
                + series(1000, "test", offset=2300))
        d = store.load_forecasting_data(FakeClient(rows[::-1]), DV)
        self.assertEqual(d.observation_counts, {"train": 1200, "validation": 1100, "test": 1000})
        self.assertEqual(len(d.splits["train"]), 1200 - 95)
        self.assertEqual(d.splits["validation"].inputs.shape, (1100 - 95, 72, 1))
        self.assertEqual(d.splits["test"].targets.shape, (1000 - 95, 24))


if __name__ == "__main__":
    unittest.main()
