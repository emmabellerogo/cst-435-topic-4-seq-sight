"""Synthetic tests for scripts/inspect_jena.py (stdlib only; no real data needed)."""
import importlib.util
import unittest
from datetime import datetime, timedelta
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "inspect_jena", Path(__file__).resolve().parents[1] / "scripts" / "inspect_jena.py"
)
ij = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ij)

T0 = datetime(2020, 1, 1, 0, 0)


def rec(ts, temp, file="a.zip", order=0):
    return ij.Record(ts, temp, None, file, order)


def series(start, n, temp=1.0):
    return {start + i * ij.STEP: temp for i in range(n)}


class GridCounting(unittest.TestCase):
    def test_off_grid_never_counts_as_present(self):
        stamps = [T0 + i * ij.STEP for i in range(10)] + [T0 + timedelta(minutes=35, seconds=30)]
        g = ij.grid_stats(stamps)
        self.assertEqual((g["expected"], g["present"], g["absent"]), (10, 10, 0))
        self.assertEqual(g["off_grid"], [T0 + timedelta(minutes=35, seconds=30)])

    def test_missing_slots_and_duplicates(self):
        stamps = [T0 + i * ij.STEP for i in (0, 1, 2, 5, 6, 9)] + [T0 + ij.STEP]  # one repeated
        g = ij.grid_stats(stamps)
        self.assertEqual((g["expected"], g["present"], g["absent"]), (10, 6, 4))

    def test_gaps(self):
        stamps = [T0 + i * ij.STEP for i in (0, 1, 4)]
        self.assertEqual(ij.grid_gaps(stamps), [(T0 + ij.STEP, T0 + 4 * ij.STEP, 3 * ij.STEP)])


class DuplicateHandling(unittest.TestCase):
    def test_identical_duplicates_collapse_and_sort(self):
        recs = [rec(T0 + ij.STEP, 2.0), rec(T0, 1.0), rec(T0 + ij.STEP, 2.0, "b.zip")]
        out = ij.collapse_duplicates(recs)
        self.assertEqual(list(out), [T0, T0 + ij.STEP])
        self.assertEqual(out[T0 + ij.STEP], 2.0)

    def test_conflicting_duplicates_fail(self):
        with self.assertRaises(ij.ConflictError):
            ij.collapse_duplicates([rec(T0, 1.0), rec(T0, 1.5)])

    def test_cross_file_vs_within_file(self):
        t1, t2 = T0, T0 + ij.STEP
        recs = [rec(t1, 1.0, "a.zip"), rec(t1, 1.0, "b.zip"), rec(t2, 2.0, "a.zip"), rec(t2, 2.0, "a.zip")]
        cross, within, conflicting = ij.classify_duplicates(recs)
        self.assertEqual((cross, within, conflicting), ([t1], [t2], []))

    def test_conflict_is_reported_by_classifier(self):
        _, within, conflicting = ij.classify_duplicates([rec(T0, 1.0), rec(T0, 2.0)])
        self.assertEqual((within, conflicting), ([T0], [T0]))


class HourlyRule(unittest.TestCase):
    def test_complete_hour_is_mean_of_six(self):
        temps = {T0 + timedelta(minutes=10 * k): float(k) for k in range(7)}  # 00:00..01:00
        valid, first, last = ij.hourly_series(temps)
        self.assertEqual(first, T0 + timedelta(hours=1))
        self.assertEqual(valid, {T0 + timedelta(hours=1): sum(range(1, 7)) / 6})

    def test_missing_slot_or_sentinel_invalidates_hour_without_filling(self):
        temps = series(T0, 13)
        del temps[T0 + timedelta(minutes=70)]  # hour 02:00 loses one record
        temps[T0 + timedelta(minutes=30)] = -9999.0  # hour 01:00 has a sentinel
        valid, _, last = ij.hourly_series(temps)
        self.assertEqual(last, T0 + timedelta(hours=2))
        self.assertEqual(valid, {})

    def test_off_grid_record_is_not_used(self):
        temps = series(T0, 7)
        temps[T0 + timedelta(minutes=30)] = None  # replaced by an invalid on-grid value
        temps[T0 + timedelta(minutes=30, seconds=30)] = 5.0  # off-grid neighbour must not rescue it
        valid, _, _ = ij.hourly_series(temps)
        self.assertEqual(valid, {})


class Windows(unittest.TestCase):
    def test_windows_do_not_span_gaps(self):
        flags = [True] * 100 + [False] + [True] * 96 + [True] * 50
        # runs of 100 -> 5 windows; 146 -> 51 windows
        self.assertEqual(ij.count_windows(flags), (5 + 51, 146))

    def test_short_runs_give_zero(self):
        self.assertEqual(ij.count_windows([True] * 95), (0, 95))

    def test_split_bounds(self):
        self.assertEqual(ij.split_bounds(1000), (700, 850))


if __name__ == "__main__":
    unittest.main()
