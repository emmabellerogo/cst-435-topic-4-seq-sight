"""Frozen-reference regression support: the fixed input, fixture I/O and comparison.

Two fixtures live in tests/fixtures/ (see docs/frozen-reference.md):

* ``frozen_reference_trained_seed0.json``: forecasts of the trusted seed-0 trained artifacts
  (git-ignored, so the check that uses it skips when they are absent);
* ``frozen_reference_synthetic.json``: forecasts of deterministic UNTRAINED models built from
  a fixed seed. It runs everywhere, but says nothing about the trained checkpoints.

Expected values are generated once by scripts/generate_frozen_reference.py and then only READ by
tests; no test regenerates its own expected answers.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests" / "fixtures"
TRAINED_FIXTURE = FIXTURES / "frozen_reference_trained_seed0.json"
SYNTHETIC_FIXTURE = FIXTURES / "frozen_reference_synthetic.json"
FORMAT = "seqsight-frozen-reference-v1"

INPUT_DEFINITION = ("temperature_c[h] = round(9.0 + 6.0 * sin(2*pi*(h - 10)/24) - 0.03 * h, 3), h = 0..71 "
                    "(h = 71 is the last observed hour); units: degrees Celsius")
FORECAST_ATOL_C = 1e-4          # absolute tolerance on each forecast hour, degC
ATTENTION_ATOL = 1e-5           # absolute tolerance on stored attention weights
SYNTHETIC_SCALER = {"mean": 9.5, "std": 8.0, "std_ddof": 0, "unit": "degC"}
SYNTHETIC_SEED = 0


def reference_input() -> List[float]:
    return [round(9.0 + 6.0 * math.sin(2 * math.pi * (h - 10) / 24) - 0.03 * h, 3) for h in range(72)]


def load_fixture(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def attention_probe(att: np.ndarray) -> Dict[str, Any]:
    """Small, stable summary of the 72x72 weights: the last row and the column means."""
    a = np.asarray(att, dtype=np.float64)
    return {"last_row": a[-1].tolist(), "column_means": a.mean(axis=0).tolist()}


def compare(fixture: Dict[str, Any], forecasts: Dict[str, Any]) -> List[str]:
    """``forecasts[kind] = {"forecast_c": [...24], "attention": 72x72 or None}`` vs the stored values."""
    bad = []
    tol = fixture["tolerances"]
    for kind, exp in fixture["expected"].items():
        got = forecasts[kind]
        f = np.asarray(got["forecast_c"], dtype=np.float64)
        e = np.asarray(exp["forecast_c"], dtype=np.float64)
        if f.shape != e.shape or not np.allclose(f, e, rtol=0, atol=tol["forecast_atol_c"]):
            worst = float(np.abs(f - e).max()) if f.shape == e.shape else float("nan")
            bad.append(f"{kind}: forecast differs from the frozen values (max abs diff {worst:.3g} degC, "
                       f"tolerance {tol['forecast_atol_c']})")
        if kind == "lstm":
            if got["attention"] is not None:
                bad.append("lstm: attention must be None")
        else:
            p = attention_probe(got["attention"])
            for key in ("last_row", "column_means"):
                if not np.allclose(p[key], exp["attention_probe"][key], rtol=0, atol=tol["attention_atol"]):
                    bad.append(f"{kind}: attention {key} differs from the frozen values")
    return bad
