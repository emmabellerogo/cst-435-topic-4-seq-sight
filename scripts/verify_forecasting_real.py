"""Explicit local check of shared.forecasting against the real Jena archives.

Not part of the routine test suite (it needs data/raw/mpi_roof and ~30 s). No
network, no credentials. Run from the repo root with the project venv:

    .venv/bin/python -m scripts.verify_forecasting_real
"""
from __future__ import annotations

import math
import random
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from db import load  # noqa: E402
from shared import forecasting as fc  # noqa: E402

EXPECTED_OBS = {"train": 135438, "validation": 29047, "test": 29063}
EXPECTED_WIN = {"train": 134678, "validation": 28762, "test": 28873}
F32_ATOL = 1e-4  # float32 has ~7 digits; values are O(1) standardized, std ~ 8 degC

failures = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        failures.append(name)


def main() -> int:
    prep = load.prepare()
    print(f"data_version {prep.data_version}; {len(prep.rows):,} rows from db.load.prepare()")
    data = fc.prepare_forecasting_data(prep.rows, expected_data_version=prep.data_version)
    check("data_version carried through", data.data_version == prep.data_version)
    check("observation counts", data.observation_counts == EXPECTED_OBS, str(data.observation_counts))
    check("window counts", {k: len(v) for k, v in data.splits.items()} == EXPECTED_WIN,
          str({k: len(v) for k, v in data.splits.items()}))

    for name, w in data.splits.items():
        n = EXPECTED_WIN[name]
        check(f"{name} shapes", w.inputs.shape == (n, 72, 1) and w.targets.shape == (n, 24),
              f"{w.inputs.shape} {w.targets.shape}")
        check(f"{name} dtype float32", w.inputs.dtype == np.float32 and w.targets.dtype == np.float32)
        check(f"{name} origins match N", len(w.origins) == n == len(w.inputs) == len(w.targets))
        check(f"{name} origins strictly increasing", all(a < b for a, b in zip(w.origins, w.origins[1:])))
        check(f"{name} standardized values finite",
              bool(np.isfinite(w.inputs).all() and np.isfinite(w.targets).all()),
              f"input range [{w.inputs.min():.2f}, {w.inputs.max():.2f}]")

    # Independent scaler check: plain-Python math.fsum over training rows only.
    train = [r["temperature_c"] for r in prep.rows if r["split"] == "train"]
    mean = math.fsum(train) / len(train)
    std = math.sqrt(math.fsum((x - mean) ** 2 for x in train) / len(train))  # population, ddof=0
    check("scaler mean (train only)", abs(data.scaler.mean - mean) < 1e-9, f"{data.scaler.mean:.9f} vs {mean:.9f}")
    check("scaler std (train only, ddof=0)", abs(data.scaler.std - std) < 1e-9, f"{data.scaler.std:.9f} vs {std:.9f}")
    allv = [r["temperature_c"] for r in prep.rows]
    check("scaler differs from all-data statistics (no leakage)", abs(sum(allv) / len(allv) - mean) > 1e-6,
          f"all-data mean {sum(allv) / len(allv):.6f}")
    check("serialization round-trip", fc.Standardizer.from_dict(data.scaler.to_dict()) == data.scaler)

    # Real windows vs. source hourly values, after inverse transform.
    source = {fc._parse_ts(r["station_ts"], "src"): (r["temperature_c"], r["split"]) for r in prep.rows}
    rng = random.Random(0)
    worst = 0.0
    checked = 0
    for name, w in data.splits.items():
        idx = {0, len(w) - 1, *rng.sample(range(len(w)), 200)}
        for i in idx:
            o = w.origins[i]
            times = [o + timedelta(hours=h) for h in range(-71, 25)]
            ok = all(t in source and source[t][1] == name for t in times)
            if not ok:
                check(f"{name} window {i} timestamps exist in-split", False, str(o))
                continue
            want = np.array([source[t][0] for t in times])
            got = data.scaler.inverse_transform(np.concatenate([w.inputs[i, :, 0], w.targets[i]]))
            worst = max(worst, float(np.abs(got - want).max()))
            checked += 1
    check("sampled windows reproduce source hourly values", worst < F32_ATOL and checked > 0,
          f"{checked} windows (first, last, 200 random per split); max abs error {worst:.2e} degC")

    # Independent gap-aware window count straight from the source timestamps.
    for name in fc.SPLIT_ORDER:
        ts = sorted(t for t, (_, sp) in source.items() if sp == name)
        run = count = 0
        prev = None
        for t in ts:
            run = run + 1 if prev is not None and t - prev == timedelta(hours=1) else 1
            count += run >= fc.WINDOW_HOURS
            prev = t
        check(f"{name} independent window count", count == len(data.splits[name]), f"{count:,}")

    print(f"\n{len(failures)} check(s) failed" if failures else "\nall checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
