"""Synthetic tests for db/load.py (stdlib only; no real data, no network, no Supabase)."""
import contextlib
import importlib.util
import io
import socket
import sys
import tempfile
import unittest
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock

_spec = importlib.util.spec_from_file_location(
    "seq_sight_load", Path(__file__).resolve().parents[1] / "db" / "load.py"
)
load = importlib.util.module_from_spec(_spec)
sys.modules["seq_sight_load"] = load  # dataclasses need the module registered
_spec.loader.exec_module(load)
ij = load.ij

T0 = datetime(2020, 1, 1, 0, 0)
STEP = ij.STEP
HEADER = "Date Time,p (mbar),T (degC),Tpot (K)"


def fmt(ts):
    return ts.strftime("%d.%m.%Y %H:%M:%S")


def write_zip(path, rows):
    """rows: iterable of (ts, temp-string)."""
    body = "\n".join(f"{fmt(ts)},990.0,{t},270.0" for ts, t in rows)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(path.stem + ".csv", HEADER + "\n" + body + "\n")


def write_manifest(path, files):
    lines = ["# filename sha256 rows"]
    for p in files:
        n = len(zipfile.ZipFile(p).read(p.stem + ".csv").decode().splitlines()) - 1
        lines.append(f"{p.name}  {ij.sha256(p)}  {n}")
    path.write_text("\n".join(lines) + "\n")


def synthetic_rows(n_hours, missing_hours=(), start=T0):
    """10-minute rows covering hour labels start+1h .. start+n_hours h; temp = hour index.

    Hours in `missing_hours` (labels as hour offsets) lose one of their six readings.
    """
    rows = []
    for k in range(1, n_hours * 6 + 1):
        ts = start + k * STEP
        hour_idx = (k - 1) // 6 + 1
        if hour_idx in missing_hours and k % 6 == 3:
            continue
        rows.append((ts, f"{float(hour_idx % 50):.1f}"))
    return rows


class Workspace(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.raw = self.dir / "raw"
        self.raw.mkdir()
        self.manifest = self.dir / "manifest.txt"

    def make(self, parts):
        """parts: {filename: rows}; returns the written zip paths."""
        files = []
        for name, rows in parts.items():
            p = self.raw / name
            write_zip(p, rows)
            files.append(p)
        write_manifest(self.manifest, files)
        return files


class ManifestVerification(Workspace):
    def test_valid_files_pass_and_version_is_deterministic(self):
        self.make({"mpi_roof_2020a.zip": synthetic_rows(200)})
        a = load.prepare(self.raw, self.manifest)
        b = load.prepare(self.raw, self.manifest)
        self.assertEqual(a.data_version, b.data_version)
        self.assertTrue(a.data_version.startswith("jena-mpi-roof-h1-"))

    def test_data_version_changes_with_source(self):
        self.make({"mpi_roof_2020a.zip": synthetic_rows(200)})
        v1 = load.prepare(self.raw, self.manifest).data_version
        self.make({"mpi_roof_2020a.zip": synthetic_rows(201)})
        v2 = load.prepare(self.raw, self.manifest).data_version
        self.assertNotEqual(v1, v2)

    def test_data_version_changes_with_preparation_setting(self):
        self.make({"mpi_roof_2020a.zip": synthetic_rows(200)})
        manifest = load.read_manifest(self.manifest)
        v1 = load.make_data_version(manifest)
        with mock.patch.object(load, "PREP_VERSION", "2"):
            self.assertNotEqual(v1, load.make_data_version(manifest))

    def test_tampered_file_fails_checksum(self):
        files = self.make({"mpi_roof_2020a.zip": synthetic_rows(200)})
        write_zip(files[0], synthetic_rows(200)[:-1])  # same name, different content
        with self.assertRaisesRegex(load.LoaderError, "SHA-256 mismatch"):
            load.prepare(self.raw, self.manifest)

    def test_missing_and_extra_files_fail(self):
        files = self.make({"mpi_roof_2020a.zip": synthetic_rows(200)})
        write_zip(self.raw / "mpi_roof_2020b.zip", synthetic_rows(10))
        with self.assertRaisesRegex(load.LoaderError, "not-in-manifest"):
            load.prepare(self.raw, self.manifest)
        (self.raw / "mpi_roof_2020b.zip").unlink()
        files[0].unlink()
        with self.assertRaisesRegex(load.LoaderError, "missing"):
            load.prepare(self.raw, self.manifest)

    def test_row_count_mismatch_fails(self):
        self.make({"mpi_roof_2020a.zip": synthetic_rows(200)})
        text = self.manifest.read_text().rstrip("\n")
        self.manifest.write_text(text[: text.rfind(" ")] + " 5\n")
        with self.assertRaisesRegex(load.LoaderError, "row counts differ"):
            load.prepare(self.raw, self.manifest)


class Preparation(Workspace):
    def test_duplicates_collapse_and_conflicts_fail(self):
        rows = synthetic_rows(200)
        self.make({"mpi_roof_2020a.zip": rows + rows[5:15]})  # identical repeats
        p = load.prepare(self.raw, self.manifest)
        self.assertEqual(p.duplicate_extra_rows, 10)
        self.assertEqual(p.valid_hours, 200)
        bad = rows + [(rows[7][0], "99.0")]  # same timestamp, other temperature
        self.make({"mpi_roof_2020a.zip": bad})
        with self.assertRaises(ij.ConflictError):
            load.prepare(self.raw, self.manifest)

    def test_unsorted_input_is_sorted(self):
        rows = synthetic_rows(200)
        self.make({"mpi_roof_2020a.zip": list(reversed(rows))})
        p = load.prepare(self.raw, self.manifest)
        stamps = [r["station_ts"] for r in p.rows]
        self.assertEqual(stamps, sorted(stamps))

    def test_off_grid_is_reported_and_excluded_not_snapped(self):
        rows = synthetic_rows(200)
        rows.append((T0 + timedelta(minutes=35, seconds=30), "77.0"))
        self.make({"mpi_roof_2020a.zip": rows})
        p = load.prepare(self.raw, self.manifest)
        self.assertEqual(p.off_grid, [T0 + timedelta(minutes=35, seconds=30)])
        self.assertTrue(all(r["temperature_c"] < 77 for r in p.rows))

    def test_hour_is_mean_of_six_hour_end_readings_naive_iso(self):
        rows = [(T0 + (k + 1) * STEP, str(float(k))) for k in range(12)]  # 00:10..02:00 -> 0..11
        self.make({"mpi_roof_2020a.zip": rows})
        p = load.prepare(self.raw, self.manifest)
        by_ts = {r["station_ts"]: r["temperature_c"] for r in p.rows}
        self.assertEqual(by_ts["2020-01-01T01:00:00"], sum(range(0, 6)) / 6)  # 00:10..01:00
        self.assertEqual(by_ts["2020-01-01T02:00:00"], sum(range(6, 12)) / 6)
        self.assertNotIn("+", p.rows[0]["station_ts"])  # no offset: naive

    def test_invalid_hours_excluded_without_filling_and_sentinel(self):
        rows = synthetic_rows(200, missing_hours={20})
        rows = [(ts, "-9999.0") if ts == T0 + 30 * 6 * STEP else (ts, t) for ts, t in rows]
        self.make({"mpi_roof_2020a.zip": rows})
        p = load.prepare(self.raw, self.manifest)
        labels = {r["station_ts"] for r in p.rows}
        self.assertNotIn((T0 + timedelta(hours=20)).isoformat(), labels)  # missing reading
        self.assertNotIn((T0 + timedelta(hours=30)).isoformat(), labels)  # sentinel in its six
        self.assertEqual(p.invalid_temp["sentinel_-9999"], 1)
        self.assertEqual(p.valid_hours, p.grid_hours - p.invalid_hours)
        self.assertGreaterEqual(p.invalid_hours, 2)


class Splits(Workspace):
    def _prepared(self, n_hours=1000, missing=()):
        self.make({"mpi_roof_2020a.zip": synthetic_rows(n_hours, missing_hours=missing)})
        return load.prepare(self.raw, self.manifest)

    def test_boundaries_on_full_grid(self):
        p = self._prepared(1000)
        n = p.grid_hours
        b1, b2 = int(n * 0.70), int(n * 0.85)
        s = p.splits
        self.assertEqual(s["train"].grid_hours, b1)
        self.assertEqual(s["validation"].grid_hours, b2 - b1)
        self.assertEqual(s["test"].grid_hours, n - b2)
        self.assertEqual(s["train"].first_label, p.first_hour)
        self.assertEqual(s["validation"].first_label, p.first_hour + timedelta(hours=b1))
        self.assertEqual(s["train"].last_label + timedelta(hours=1), s["validation"].first_label)
        self.assertEqual(s["test"].last_label, p.last_hour)
        self.assertEqual({r["split"] for r in p.rows}, {"train", "validation", "test"})

    def test_rows_land_in_chronological_splits(self):
        p = self._prepared(1000)
        order = [r["split"] for r in p.rows]
        self.assertEqual(order, sorted(order, key={"train": 0, "validation": 1, "test": 2}.get))
        first_val = next(r for r in p.rows if r["split"] == "validation")
        self.assertEqual(first_val["station_ts"], p.splits["validation"].first_label.isoformat())

    def test_gap_keeps_grid_positions_and_blocks_windows(self):
        # 400 hours; remove one reading in hour 150 (inside train when N = 400).
        p = self._prepared(400, missing={150})
        n = p.grid_hours
        self.assertEqual(n, 400)
        # split sizes come from the FULL grid, not from the valid-hour count
        self.assertEqual(p.splits["train"].grid_hours, int(n * 0.70))
        self.assertEqual(p.splits["train"].valid_hours, int(n * 0.70) - 1)
        # windows never span the gap: train hours before the gap and after it are counted apart
        k = int((T0 + timedelta(hours=150) - p.first_hour) / timedelta(hours=1))
        after = int(n * 0.70) - k - 1
        expected = max(0, k - 95) + max(0, after - 95)
        self.assertEqual(p.splits["train"].windows, expected)
        self.assertLess(expected, int(n * 0.70) - 95)

    def test_windows_never_cross_partition_boundary(self):
        p = self._prepared(1000)
        for s in p.splits.values():
            self.assertEqual(s.windows, s.grid_hours - 95)  # no gaps: one run per partition
        self.assertEqual(p.total_windows, sum(s.windows for s in p.splits.values()))


class DryRun(Workspace):
    def test_dry_run_needs_no_credentials_network_or_database_import(self):
        self.make({"mpi_roof_2020a.zip": synthetic_rows(400)})
        buf = io.StringIO()
        with mock.patch.dict("os.environ", {}, clear=True), \
                mock.patch.dict(sys.modules, {"supabase": None, "dotenv": None}), \
                mock.patch.object(socket, "socket", side_effect=AssertionError("network used")), \
                mock.patch.object(socket, "create_connection", side_effect=AssertionError("network used")), \
                contextlib.redirect_stdout(buf):
            rc = load.main(["--raw-dir", str(self.raw), "--manifest", str(self.manifest),
                            "--env-file", str(self.dir / "does-not-exist.env")])
        out = buf.getvalue()
        self.assertEqual(rc, 0)
        self.assertIn("DRY-RUN", out)
        for needle in ("data_version:", "valid hours:", "invalid hours", "train", "validation",
                       "test", "windows(72->24)", "off-grid"):
            self.assertIn(needle, out)

    def test_dry_run_failure_exits_nonzero(self):
        self.make({"mpi_roof_2020a.zip": synthetic_rows(400)})
        self.manifest.write_text(self.manifest.read_text().replace("2020a", "2020z"))
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            rc = load.main(["--raw-dir", str(self.raw), "--manifest", str(self.manifest)])
        self.assertEqual(rc, 2)
        self.assertIn("ERROR", err.getvalue())


class Config(unittest.TestCase):
    URL = "https://abcdefgh.supabase.co"
    KEY = "fake-test-key-" + "x" * 30

    def _env(self, text):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        p = Path(d.name) / ".env"
        p.write_text(text)
        return p

    def test_valid_config_from_env_file(self):
        p = self._env(f"SUPABASE_URL={self.URL}\nSUPABASE_SERVICE_KEY='{self.KEY}'\n")
        c = load.load_config(p, environ={})
        self.assertEqual((c.url, c.key), (self.URL, self.KEY))
        self.assertNotIn(self.KEY, repr(c))

    def test_missing_or_placeholder_values_rejected_without_echo(self):
        for text in ("", "SUPABASE_URL=https://your-project-ref.supabase.co\nSUPABASE_SERVICE_KEY=your-service-role-key\n",
                     f"SUPABASE_URL=http://insecure.example\nSUPABASE_SERVICE_KEY={self.KEY}\n"):
            with self.assertRaises(load.LoaderError) as cm:
                load.load_config(self._env(text), environ={})
            self.assertNotIn(self.KEY, str(cm.exception))

    def test_upload_with_bad_config_stops_before_any_client(self):
        err = io.StringIO()
        with mock.patch.object(load.SupabaseStore, "connect", side_effect=AssertionError("connected")), \
                contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()), \
                mock.patch.dict("os.environ", {}, clear=True):
            rc = load.main(["--upload", "--env-file", "/nonexistent/.env"])
        self.assertEqual(rc, 2)
        self.assertIn("configuration invalid", err.getvalue())

    def test_errors_are_redacted(self):
        c = load.Config(url=self.URL, key=self.KEY)
        self.assertEqual(load._redact(f"failed {self.URL} {self.KEY}", c), "failed *** ***")


class FakeStore:
    """In-memory stand-in for SupabaseStore enforcing UNIQUE (data_version, station_ts)."""

    def __init__(self, fail_on_insert_call=None):
        self.table = {}
        self.insert_calls = []
        self.fail_on = fail_on_insert_call

    def fetch_existing(self, data_version):
        for (dv, ts), r in sorted(self.table.items()):
            if dv == data_version:
                yield {"station_ts": ts, "temperature_c": r["temperature_c"], "split": r["split"]}

    def insert(self, rows):
        self.insert_calls.append(len(rows))
        if self.fail_on == len(self.insert_calls):
            self.fail_on = None
            raise RuntimeError("simulated network failure")
        for r in rows:
            key = (r["data_version"], r["station_ts"])
            if key in self.table:
                raise AssertionError("unique violation: duplicate insert attempted")
            self.table[key] = dict(r)

    def count(self, data_version, split):
        return sum(1 for (dv, _), r in self.table.items() if dv == data_version and r["split"] == split)


class Upload(Workspace):
    def setUp(self):
        super().setUp()
        self.make({"mpi_roof_2020a.zip": synthetic_rows(500, missing_hours={100})})
        self.p = load.prepare(self.raw, self.manifest)
        self.log = []

    def run_upload(self, store, batch=100):
        return load.upload(self.p, store, batch, out=self.log.append)

    def test_fresh_upload_in_bounded_batches(self):
        store = FakeStore()
        res = self.run_upload(store, 100)
        self.assertEqual(res, {"inserted": len(self.p.rows), "skipped": 0})
        self.assertEqual(len(store.table), len(self.p.rows))
        self.assertTrue(all(n <= 100 for n in store.insert_calls))
        self.assertEqual(len(store.insert_calls), -(-len(self.p.rows) // 100))

    def test_rerun_is_idempotent(self):
        store = FakeStore()
        self.run_upload(store)
        calls = len(store.insert_calls)
        res = self.run_upload(store)
        self.assertEqual(res, {"inserted": 0, "skipped": len(self.p.rows)})
        self.assertEqual(len(store.insert_calls), calls)  # nothing re-sent

    def test_resume_after_interruption_creates_no_duplicates(self):
        store = FakeStore(fail_on_insert_call=3)
        with self.assertRaises(RuntimeError):
            self.run_upload(store, 100)
        partial = len(store.table)
        self.assertEqual(partial, 200)
        res = self.run_upload(store, 100)
        self.assertEqual(res["skipped"], partial)
        self.assertEqual(res["inserted"], len(self.p.rows) - partial)
        self.assertEqual(len(store.table), len(self.p.rows))

    def test_conflicting_existing_value_aborts_without_writes(self):
        store = FakeStore()
        self.run_upload(store)
        key = next(iter(store.table))
        store.table[key]["temperature_c"] += 1.0
        before = dict((k, dict(v)) for k, v in store.table.items())
        calls = len(store.insert_calls)
        with self.assertRaises(load.UploadConflict):
            self.run_upload(store)
        self.assertEqual(store.table, before)
        self.assertEqual(len(store.insert_calls), calls)

    def test_conflicting_split_and_unexpected_rows_abort(self):
        store = FakeStore()
        self.run_upload(store)
        key = next(iter(store.table))
        store.table[key]["split"] = "test"
        with self.assertRaises(load.UploadConflict):
            self.run_upload(store)
        store = FakeStore()
        store.table[(self.p.data_version, "1999-01-01T00:00:00")] = {
            "temperature_c": 1.0, "split": "train", "data_version": self.p.data_version,
            "station_ts": "1999-01-01T00:00:00"}
        with self.assertRaises(load.UploadConflict):
            self.run_upload(store)
        self.assertEqual(store.insert_calls, [])

    def test_other_data_versions_are_ignored(self):
        store = FakeStore()
        store.table[("some-older-version", "2020-01-01T01:00:00")] = {
            "temperature_c": 9.9, "split": "train", "data_version": "some-older-version",
            "station_ts": "2020-01-01T01:00:00"}
        self.run_upload(store)
        self.assertEqual(len(store.table), len(self.p.rows) + 1)

    def test_batch_size_bounds(self):
        for bad in (0, -1, load.MAX_BATCH + 1):
            with self.assertRaises(load.LoaderError):
                load.upload(self.p, FakeStore(), bad, out=self.log.append)

    def test_post_upload_verification_detects_missing_rows(self):
        class Lossy(FakeStore):
            def insert(self, rows):
                super().insert(rows[:-1])  # silently drops one row per batch
        with self.assertRaisesRegex(load.LoaderError, "verification failed"):
            self.run_upload(Lossy(), 100)

    def test_rows_contain_only_expected_columns_and_naive_timestamps(self):
        store = FakeStore()
        self.run_upload(store)
        for (dv, ts), r in store.table.items():
            self.assertEqual(set(r), {"data_version", "station_ts", "temperature_c", "split"})
            self.assertRegex(ts, r"^\d{4}-\d\d-\d\dT\d\d:00:00$")


if __name__ == "__main__":
    unittest.main()
