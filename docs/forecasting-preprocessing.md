# Forecasting Preprocessing

Back to [README](../README.md) · See also [data design](data-design.md) and [Supabase setup](supabase-setup.md).

Shared code that turns the stored hourly corpus into model-ready windows. The
classifier code in `shared/data.py` and `shared/schemas.py` is untouched until its
callers are migrated.

| File | Role |
|------|------|
| [shared/forecasting.py](../shared/forecasting.py) | Pure preprocessing: validation, standardizer, windows, MAE. Needs only `numpy`. |
| [shared/forecast_store.py](../shared/forecast_store.py) | Read-only paginated fetch of one data version. The client is injected; no database library is imported. |
| [tests/test_forecasting.py](../tests/test_forecasting.py) | Synthetic data and a fake client; no network. |

## Usage

```python
from supabase import create_client
from shared.forecast_store import load_forecasting_data

client = create_client(url, service_key)          # credentials handled by the caller
data = load_forecasting_data(client, "jena-mpi-roof-h1-60138aac7d898b8c")

train = data.splits["train"]        # WindowSet
train.inputs   # float32 [N, 72, 1]   standardized, hours t-71 .. t
train.targets  # float32 [N, 24]      standardized, hours t+1 .. t+24
train.origins  # list[datetime]       t = last input hour (naive station time)

params = data.scaler.to_dict()      # {"mean", "std", "std_ddof": 0, "unit": "degC"} -> runs.preprocessing
```

Evaluation and inference reuse the stored parameters:

```python
from shared.forecasting import Standardizer, mae_celsius, prepare_inference_input

scaler = Standardizer.from_dict(params)
mae = mae_celsius(pred_standardized, data.splits["test"].targets, scaler)   # degrees C
x = prepare_inference_input(last_72_hourly_temps_c, scaler)                 # [1, 72, 1]
forecast_c = scaler.inverse_transform(model_output)                         # degrees C
```

`prepare_forecasting_data(rows)` does the same from rows you already hold, with no
database. `dtype` (default `float32`) sets the output array type.

## Rules

- **Input rows** carry `station_ts`, `temperature_c`, `split`, `data_version`.
  Anything missing or null, a non-finite or non-numeric temperature, an unknown
  split, a second data version, a non-hourly or time-zone-aware timestamp, a
  duplicate `station_ts` (even an identical one), or splits that are not
  chronological (train < validation < test, no interleaving) raises
  `ForecastDataError`. Nothing is repaired or silently resolved.
- **Timestamps** stay naive station time. Rows are sorted by time; input order does
  not matter.
- **Windows** are 72 consecutive input hours then 24 consecutive target hours,
  stride 1, built separately inside each split. A missing hour ends a run and a
  split boundary ends a run, so no window spans either. A split with fewer than 96
  consecutive hours yields zero windows (arrays of shape `[0, 72, 1]` / `[0, 24]`).
- **Forecast origin** is the timestamp of the last input hour; targets are
  origin + 1 h .. origin + 24 h.
- **Standardizer** is fitted on the training observations (each valid hour once),
  before any overlapping windows exist, so overlap does not reweight hours. The
  standard deviation is the **population** std (`ddof=0`, divide by N), and the
  convention is stored as `std_ddof` in the serialized form. A zero, negative or
  non-finite std, or an empty/non-finite training set, is rejected. The same
  mean/std is applied to validation, test and inference.
- **MAE** inverts the standardization first, so it is in degrees Celsius.
- **Reader** selects `station_ts,temperature_c,split,data_version` filtered by
  `data_version`, ordered by `station_ts`, in pages of at most 1000 rows. 1000 is
  the PostgREST default response cap, so a larger page would be truncated and look
  like the final page. It stops on the first short page.

## Memory

Windows are filled into preallocated arrays from `sliding_window_view` slices, one
contiguous run at a time. For the real corpus (about 193k windows) float32 inputs
and targets are roughly 55 MB in total. A synthetic full-size run (about 194k
rows) took about 0.5 s with a peak process size of about 200 MB, most of it the
row dicts. Nothing here creates attention matrices or model tensors.

## Real-data verification

[scripts/verify_forecasting_real.py](../scripts/verify_forecasting_real.py) is an
explicit local check, not part of the routine tests (it needs `data/raw/mpi_roof`
and takes about 20 s; no network or credentials):

```bash
.venv/bin/python -m scripts.verify_forecasting_real
```

It runs `db.load.prepare()`, feeds the rows to `prepare_forecasting_data()`, and
checks observation and window counts, shapes, dtypes, finite values, origin
counts, an independent train-only mean/std, sampled windows against the source
hourly values after inverse transform, and an independent gap-aware window count.
Result on 2026-10-24: all checks passed (26 checks).

## Not covered

Not run against the real Supabase project; the reader was only tested with a
mocked client. The default `python3` on this machine has no numpy; the unit tests
ran under `python3.12` and `.venv/bin/python`. pytest is not installed, so the
tests are plain `unittest` (pytest also collects them).
