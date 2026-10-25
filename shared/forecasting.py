"""Pure preprocessing for Seq Sight hourly temperature forecasting.

No database, network or framework imports: it takes corpus rows (as stored in the
Supabase ``sequences`` table) and returns standardized 72 h -> 24 h windows. The
database reader lives in ``shared/forecast_store.py``.

Conventions
-----------
* A row is ``{"station_ts", "temperature_c", "split", "data_version"}``.
  ``station_ts`` is the hour-end label as the station reported it: a NAIVE
  datetime (or its ISO string). It is never converted to another zone.
* A window is 72 consecutive hourly observations (input) followed by the next 24
  consecutive hours (target), stride 1. Windows are built inside each split and
  never span a missing hour or a split boundary.
* The *forecast origin* of a window is the timestamp of its LAST input hour; the
  targets are origin+1 h .. origin+24 h.
* Inputs have shape ``[N, 72, 1]`` and targets ``[N, 24]``, standardized with the
  training-set mean/std. Errors are reported in degrees Celsius by inverting the
  standardization first.
* Standard deviation convention: POPULATION std (``ddof=0``, divide by N), taken
  over the training observations (each valid hour once), not over windows, which
  overlap and would weight hours unevenly.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np

INPUT_HOURS = 72
TARGET_HOURS = 24
WINDOW_HOURS = INPUT_HOURS + TARGET_HOURS
SPLIT_ORDER = ("train", "validation", "test")
REQUIRED_FIELDS = ("station_ts", "temperature_c", "split", "data_version")
STD_DDOF = 0
HOUR = timedelta(hours=1)


class ForecastDataError(ValueError):
    """Corpus rows or scaler parameters violate the documented data contract."""


# --------------------------------------------------------------- validation

@dataclass(frozen=True)
class Observation:
    ts: datetime  # naive station time
    temperature_c: float
    split: str


def _parse_ts(value: Any, where: str) -> datetime:
    if isinstance(value, datetime):
        ts = value
    elif isinstance(value, str):
        try:
            ts = datetime.fromisoformat(value)
        except ValueError:
            raise ForecastDataError(f"{where}: unparseable station_ts {value!r}") from None
    else:
        raise ForecastDataError(f"{where}: station_ts must be a datetime or ISO string, got {type(value).__name__}")
    if ts.tzinfo is not None:
        raise ForecastDataError(f"{where}: station_ts {value!r} carries a time zone; station time must stay naive")
    if (ts.minute, ts.second, ts.microsecond) != (0, 0, 0):
        raise ForecastDataError(f"{where}: station_ts {value!r} is not on an exact hour")
    return ts


def _parse_temp(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ForecastDataError(f"{where}: temperature_c must be a number, got {type(value).__name__}")
    t = float(value)
    if not math.isfinite(t):
        raise ForecastDataError(f"{where}: temperature_c is not finite ({value!r})")
    return t


def validate_rows(rows: Iterable[Mapping[str, Any]],
                  expected_data_version: Optional[str] = None) -> tuple[str, List[Observation]]:
    """Validate rows and return ``(data_version, observations sorted by time)``.

    Rejects (never repairs): missing/null fields, non-finite temperatures,
    unknown splits, mixed or unexpected data versions, non-hourly or tz-aware
    timestamps, duplicate timestamps, and splits that are not chronologically
    ordered (train before validation before test, no interleaving).
    """
    obs: List[Observation] = []
    version: Optional[str] = expected_data_version
    for i, row in enumerate(rows):
        where = f"row {i}"
        for f in REQUIRED_FIELDS:
            if f not in row or row[f] is None:
                raise ForecastDataError(f"{where}: missing required field {f!r}")
        dv = row["data_version"]
        if not isinstance(dv, str) or not dv:
            raise ForecastDataError(f"{where}: data_version must be a non-empty string")
        if version is None:
            version = dv
        elif dv != version:
            raise ForecastDataError(f"{where}: data_version {dv!r} differs from {version!r}")
        split = row["split"]
        if split not in SPLIT_ORDER:
            raise ForecastDataError(f"{where}: unknown split {split!r}")
        obs.append(Observation(_parse_ts(row["station_ts"], where), _parse_temp(row["temperature_c"], where), split))
    if not obs or version is None:
        raise ForecastDataError("no rows" + (f" for data_version {version!r}" if version else ""))

    obs.sort(key=lambda o: o.ts)
    for a, b in zip(obs, obs[1:]):
        if a.ts == b.ts:
            raise ForecastDataError(f"duplicate station_ts {a.ts.isoformat()}")
        if SPLIT_ORDER.index(b.split) < SPLIT_ORDER.index(a.split):
            raise ForecastDataError(
                f"splits are not chronological: {b.split!r} at {b.ts.isoformat()} follows "
                f"{a.split!r} at {a.ts.isoformat()}")
    return version, obs


# --------------------------------------------------------------- standardizer

@dataclass(frozen=True)
class Standardizer:
    """``z = (x - mean) / std`` with parameters fitted on training data only."""

    mean: float
    std: float

    def __post_init__(self) -> None:
        for name in ("mean", "std"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, float, np.integer, np.floating)) or not math.isfinite(v):
                raise ForecastDataError(f"standardizer {name} must be a finite number, got {v!r}")
        if not self.std > 0:
            raise ForecastDataError(f"standardizer std must be > 0, got {self.std!r}")
        object.__setattr__(self, "mean", float(self.mean))
        object.__setattr__(self, "std", float(self.std))

    @classmethod
    def fit(cls, train_temperatures: Iterable[float]) -> "Standardizer":
        """Fit on training observations (one value per valid hour); population std."""
        x = np.asarray(list(train_temperatures) if not isinstance(train_temperatures, np.ndarray)
                       else train_temperatures, dtype=np.float64)
        if x.size == 0:
            raise ForecastDataError("cannot fit a standardizer without training observations")
        if not np.isfinite(x).all():
            raise ForecastDataError("training observations contain non-finite values")
        return cls(mean=float(x.mean()), std=float(x.std(ddof=STD_DDOF)))  # zero/nonfinite std rejected in __post_init__

    def transform(self, degrees_c) -> np.ndarray:
        x = np.asarray(degrees_c, dtype=np.float64)
        if not np.isfinite(x).all():
            raise ForecastDataError("cannot standardize non-finite values")
        return (x - self.mean) / self.std

    def inverse_transform(self, z) -> np.ndarray:
        return np.asarray(z, dtype=np.float64) * self.std + self.mean

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe; ``mean``/``std`` match what ``runs.preprocessing`` requires."""
        return {"mean": self.mean, "std": self.std, "std_ddof": STD_DDOF, "unit": "degC"}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Standardizer":
        for k in ("mean", "std"):
            if k not in d:
                raise ForecastDataError(f"standardizer dict is missing {k!r}")
        if d.get("std_ddof", STD_DDOF) != STD_DDOF:
            raise ForecastDataError(f"unsupported std_ddof {d.get('std_ddof')!r}; expected {STD_DDOF}")
        return cls(d["mean"], d["std"])


def mae_celsius(pred_standardized, target_standardized, scaler: Standardizer) -> float:
    """Mean absolute error in degrees Celsius for standardized predictions/targets."""
    return mae_degrees(scaler.inverse_transform(pred_standardized),
                       scaler.inverse_transform(target_standardized))


def mae_degrees(pred_c, target_c) -> float:
    p, t = np.asarray(pred_c, dtype=np.float64), np.asarray(target_c, dtype=np.float64)
    if p.shape != t.shape:
        raise ForecastDataError(f"shape mismatch: {p.shape} vs {t.shape}")
    if p.size == 0:
        raise ForecastDataError("MAE of an empty array")
    if not (np.isfinite(p).all() and np.isfinite(t).all()):
        raise ForecastDataError("MAE inputs contain non-finite values")
    return float(np.abs(p - t).mean())


# ------------------------------------------------------------------ windows

@dataclass
class WindowSet:
    """Windows for one split. ``origins[i]`` is the last input hour of window i."""

    inputs: np.ndarray    # [N, 72, 1]
    targets: np.ndarray   # [N, 24]
    origins: List[datetime] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.origins)


def _empty(dtype) -> WindowSet:
    return WindowSet(np.empty((0, INPUT_HOURS, 1), dtype), np.empty((0, TARGET_HOURS), dtype), [])


def make_windows(timestamps: List[datetime], values: np.ndarray, dtype=np.float32) -> WindowSet:
    """Stride-1 windows over ONE split's sorted series, never crossing a gap.

    ``timestamps`` must be strictly increasing; ``values`` is aligned with it.
    Output arrays are allocated once and filled run by run (no per-window Python
    objects besides the origin timestamps).
    """
    values = np.asarray(values)
    if values.ndim != 1 or len(values) != len(timestamps):
        raise ForecastDataError("values must be 1-D and aligned with timestamps")
    n = len(timestamps)
    if n == 0:
        return _empty(dtype)
    # Run boundaries: indices where the next hour is not exactly +1 h.
    breaks = [i + 1 for i in range(n - 1) if timestamps[i + 1] - timestamps[i] != HOUR]
    if any(timestamps[i + 1] <= timestamps[i] for i in range(n - 1)):
        raise ForecastDataError("timestamps must be strictly increasing")
    starts = [0] + breaks
    ends = breaks + [n]
    total = sum(max(0, e - s - WINDOW_HOURS + 1) for s, e in zip(starts, ends))
    if total == 0:
        return _empty(dtype)

    inputs = np.empty((total, INPUT_HOURS, 1), dtype)
    targets = np.empty((total, TARGET_HOURS), dtype)
    origins: List[datetime] = []
    k = 0
    for s, e in zip(starts, ends):
        m = e - s - WINDOW_HOURS + 1
        if m <= 0:
            continue
        view = np.lib.stride_tricks.sliding_window_view(values[s:e], WINDOW_HOURS)  # [m, 96], no copy
        inputs[k:k + m, :, 0] = view[:, :INPUT_HOURS]
        targets[k:k + m] = view[:, INPUT_HOURS:]
        origins.extend(timestamps[s + INPUT_HOURS - 1: s + INPUT_HOURS - 1 + m])
        k += m
    return WindowSet(inputs, targets, origins)


@dataclass
class ForecastingData:
    data_version: str
    scaler: Standardizer
    splits: Dict[str, WindowSet]
    observation_counts: Dict[str, int]


def prepare_forecasting_data(rows: Iterable[Mapping[str, Any]],
                             expected_data_version: Optional[str] = None,
                             dtype=np.float32) -> ForecastingData:
    """Validate rows, fit the scaler on train, and build windows for every split."""
    version, obs = validate_rows(rows, expected_data_version)
    by_split: Dict[str, List[Observation]] = {s: [] for s in SPLIT_ORDER}
    for o in obs:
        by_split[o.split].append(o)
    if not by_split["train"]:
        raise ForecastDataError("no training observations to fit the standardizer on")

    # Fit on the training hours BEFORE any overlapping windows exist.
    scaler = Standardizer.fit(np.array([o.temperature_c for o in by_split["train"]], dtype=np.float64))

    splits = {}
    for name, items in by_split.items():
        z = scaler.transform(np.array([o.temperature_c for o in items], dtype=np.float64))
        splits[name] = make_windows([o.ts for o in items], z, dtype)
    return ForecastingData(version, scaler, splits, {s: len(v) for s, v in by_split.items()})


def prepare_inference_input(last_72_temperatures_c, scaler: Standardizer, dtype=np.float32) -> np.ndarray:
    """Standardize one 72-hour history (°C) with the stored training scaler -> [1, 72, 1]."""
    x = np.asarray(last_72_temperatures_c, dtype=np.float64)
    if x.shape != (INPUT_HOURS,):
        raise ForecastDataError(f"expected {INPUT_HOURS} hourly temperatures, got shape {x.shape}")
    return scaler.transform(x).astype(dtype).reshape(1, INPUT_HOURS, 1)
