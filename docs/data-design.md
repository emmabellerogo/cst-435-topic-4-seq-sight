# Seq Sight — Data Design and Forecasting Contract

> **Status: proposed design.** Every choice below is a team design decision, not
> an assignment-mandated setting, and none of it is implemented or verified.
> Items in [Unresolved verification items](#unresolved-verification-items) must
> be checked and recorded before implementation begins.

Back to [README](../README.md).

## Scope

- **Variant A:** hourly temperature forecasting on the Jena Climate dataset.
- **Initial feature:** temperature in °C only (single input feature).
- **Intended use:** an educational demo of attention-based forecasting. Jena
  historical weather is not a validated local campus weather service, and
  forecasts must not be presented as one.

## Forecasting contract

| Item | Proposed value |
|------|----------------|
| Input | 72 consecutive hourly temperature observations |
| Target | The following 24 hourly temperatures, predicted together |
| Input shape | `[batch, 72, 1]` |
| Forecast shape | `[batch, 24]` |
| Loss / metric | MAE reported in °C after inverse-transforming forecasts |

### Timeline example

For a window whose first input hour is `t0` (hours indexed from 0):

```
hour:    t0 ........................ t0+71 | t0+72 ........ t0+95
         |<------- 72-hour input -------->|<-- 24-hour target -->|
shape:            [72, 1]                          [24]
```

The input covers hours `t0 … t0+71`. The target is `t0+72 … t0+95`, entirely
after the input. The next window starts at `t0+1` if stride 1 is used (stride is
not yet decided).

## Splits

- Chronological partitions: **70% train, 15% validation, 15% test**.
- Windows are created **separately within each partition**, so neither inputs
  nor targets cross a partition boundary. (Some observations near boundaries
  therefore belong to no window; this is intended.)
- Exact boundary timestamps are not yet defined; see verification items.

## Preprocessing

- Fit standardization (mean, standard deviation) on **training observations
  only**.
- Reuse those parameters unchanged for validation, test, and inference.
- Inverse-transform forecasts before computing MAE, so errors are in °C.
- Persist the fitted parameters with the data version so inference uses the same
  values.

## Fair comparison

The LSTM/GRU baseline and the self-attention model must use identical data,
windows, preprocessing, and evaluation. Only the model architecture differs.

## Data validity rules

Reject or flag, rather than silently repair:

- Timestamps that are unsorted or duplicated.
- Temperatures that are non-numeric, NaN, or infinite.
- Any window that would span a missing hour. Windows are not created across
  gaps.

How gaps are handled beyond excluding such windows is undecided.

## Attention interpretation

For a single self-attention head over 72 input hours, the attention matrix has
shape `[batch, 72, 72]`. It describes relationships among **input hours**. It is:

- **not** automatically a separate attribution for each of the 24 forecast
  hours (the mapping from this matrix to the forecast depends on the model
  design, e.g. pooling, which is not yet chosen), and
- **not** proof of causality.

UI and reports should describe it as "where the model's attention weights were
concentrated", not "why the temperature will be X".

## Planned database responsibilities

Table and column names are provisional.

| Table | Responsibility |
|-------|----------------|
| `sequences` | Hourly corpus: timestamp, temperature value, split (train/validation/test) |
| `runs` | Model type, seed, hyperparameters, preprocessing/data-version metadata (including standardization parameters), compute-budget information, evaluation metrics (e.g. test MAE) |
| `predictions` | Input hash, 24-hour forecast, attention matrix where applicable, serving run (foreign key to `runs`), creation timestamp |

**Recurrent baseline:** it has no attention matrix. Its `predictions` rows store
the attention field as `NULL` (or equivalent "not applicable"); no weights are
fabricated or substituted. The UI should show "no attention for this model".

## Implementation checklist

All unchecked; none verified.

- [ ] Verify and record the dataset details listed below
- [ ] Decide window stride and gap handling
- [ ] Load the hourly corpus into `sequences` with split labels
- [ ] Implement shared preprocessing (train-only fit, reused for all splits and inference)
- [ ] Implement per-partition window creation; test no window crosses a boundary or a missing hour
- [ ] Test shapes: input `[batch, 72, 1]`, forecast `[batch, 24]`, attention `[batch, 72, 72]`
- [ ] Implement both models against the same data pipeline
- [ ] Persist run metadata and test MAE in °C
- [ ] Persist predictions with `NULL` attention for the baseline

## Unresolved verification items

None of these has been verified; do not fill in from assumption.

- [ ] Dataset acquisition: source, license, version/date retrieved
- [ ] Source timestamp interpretation (time zone, UTC vs. local, daylight-saving behavior)
- [ ] Source sampling frequency
- [ ] Hourly aggregation method (e.g. subsample vs. mean), if the source is finer than hourly
- [ ] Exact split boundary timestamps
- [ ] Missing-data handling: how many gaps exist and what to do with them
- [ ] Whether the 70/15/15 split and the 72→24 windows yield enough windows per partition
- [ ] How the `[batch, 72, 72]` attention maps to the 24-hour forecast (model-head design)
