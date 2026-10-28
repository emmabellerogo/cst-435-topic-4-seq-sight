"""Selection logic of scripts/curate_forecast_examples.py on synthetic data (no corpus, no artifacts)."""
from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import curate_forecast_examples as cur


def test_pattern_rules():
    t = np.arange(96)
    daily = 10 + 6 * np.sin(2 * np.pi * t / 24)                      # range 12, no trend
    assert cur.pattern_of(daily[:72], daily[72:])[0] == "diurnal"
    flat = np.full(96, 5.0) + 0.5 * np.sin(t)
    assert cur.pattern_of(flat[:72], flat[72:])[0] == "flat"
    ramp = np.concatenate([np.full(72, 2.0), np.full(24, 12.0)])
    assert cur.pattern_of(ramp[:72], ramp[72:])[0] == "shift"
    assert cur.pattern_of(np.linspace(0, 5, 72), np.linspace(5, 8, 24))[0] is None


def _windows(n):
    start = datetime(2020, 1, 1)
    return SimpleNamespace(origins=[start + timedelta(hours=i) for i in range(n)])


def test_failures_respect_threshold_and_spacing():
    w = _windows(1000)
    mae = np.zeros(1000)
    mae[[10, 12, 300, 700, 900]] = [9.0, 9.5, 8.0, 7.0, cur.FAILURE_MAE_THRESHOLD_C - 0.01]
    picked, n_cand = cur.select_failures(w, mae, [])
    assert n_cand == 4 and picked == [12, 300, 700]                   # 10 is within 168 h of 12; 900 is below threshold


def test_failures_avoid_comparison_windows_and_refuse_to_relax():
    w = _windows(1000)
    mae = np.zeros(1000)
    mae[[10, 300, 700]] = 9.0
    with pytest.raises(cur.CurationError):
        cur.select_failures(w, mae, [w.origins[10]])                  # only two spaced candidates remain
