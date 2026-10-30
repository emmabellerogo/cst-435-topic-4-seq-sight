# Seq Sight — Data Design and Forecasting Contract

> **Status: design record; now implemented.** Every *design choice* below (window sizes, split
> ratios, cleanup rules, tables) is the contributor's decision, not an assignment-mandated
> setting. It is implemented in `shared/forecasting.py`, `db/load.py` and
> `db/migrations/002_seq_sight_forecasting.sql`; wording below such as "proposed" or "planned" is
> the original design language. Items still open are listed under Unresolved verification items.
> **Dataset facts** (marked *Measured*) come from inspecting the downloaded files
> on 2026-10-24; see the [dataset card](dataset-card.md) for evidence and for
> what is source-documented vs. locally measured. Items still open are in
> [Unresolved verification items](#unresolved-verification-items).

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
- **Proposed method:** boundaries on the calendar-hour grid at 70% and 85% of
  N = 193,760 hours (2003-11-24 17:00 – 2026-01-01 00:00 end-of-hour labels).
- **Measured** on the files recorded in the dataset card (stride 1; a window is 96
  consecutive valid hours inside one split):

| Split | Hour labels (inclusive) | Windows |
|-------|-------------------------|--------:|
| Train | 2003-11-24 17:00 – 2019-05-16 00:00 | 134,678 |
| Validation | 2019-05-16 01:00 – 2022-09-08 00:00 | 28,762 |
| Test | 2022-09-08 01:00 – 2026-01-01 00:00 | 28,873 |

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

## Data validity and cleanup rules

*Proposed rules* (checked by `scripts/inspect_jena.py` and its synthetic tests;
no loader exists yet):

1. **Sort** records by timestamp.
2. **Collapse duplicate timestamps only when their temperatures agree; fail on
   conflicting temperatures** (no silent choice).
3. **Off-grid timestamps** (not a `:00/:10/…` minute with zero seconds) are
   excluded from aggregation, never snapped, and reported.
4. Temperatures must be finite numbers; the provider's `-9999` is treated as
   missing (its meaning is undocumented).
5. **Hourly value:** an hour labelled `H:00` is the mean of the six on-grid
   10-minute records stamped `H-50 min … H:00` (source timestamps are
   end-of-period). It is valid only if all six are present and valid; otherwise
   it is an invalid hour.
6. **No interpolation or filling.** Windows contain no invalid hours, so they
   never span a gap.

*Measured on the downloaded data* (not design choices):

- 478 duplicate rows, 0 conflicting (144 from a 2010 archive overlap, 334 repeated
  rows inside archives); 2 off-grid timestamps; 8 out-of-order places.
- 212 of 193,760 hourly-grid hours are invalid (193,548 valid).
- Timestamps are naive station-clock values labelled `MEZ` by the provider. The
  data are consistent with one clock offset all year, but the time zone and DST
  semantics are **not established**, and they are never treated as UTC.

## Attention interpretation

For a single self-attention head over 72 input hours, the attention matrix has
shape `[batch, 72, 72]`. It describes relationships among **input hours**. It is:

- **not** automatically a separate attribution for each of the 24 forecast
  hours (the mapping from this matrix to the forecast depends on the model
  design, e.g. pooling, which is not yet chosen), and
- **not** proof of causality.

UI and reports should describe it as "where the model's attention weights were
concentrated", not "why the temperature will be X".

## Database responsibilities

Implemented in `002_seq_sight_forecasting.sql`; see [supabase-setup.md](supabase-setup.md) for the exact columns.

| Table | Responsibility |
|-------|----------------|
| `sequences` | Hourly corpus: timestamp, temperature value, split (train/validation/test) |
| `runs` | Model type, seed, hyperparameters, preprocessing/data-version metadata (including standardization parameters), compute-budget information, evaluation metrics (e.g. test MAE) |
| `predictions` | Input hash, 24-hour forecast, attention matrix where applicable, serving run (foreign key to `runs`), creation timestamp |

**Recurrent baseline:** it has no attention matrix. Its `predictions` rows store
the attention field as `NULL` (or equivalent "not applicable"); no weights are
fabricated or substituted. The UI should show "no attention for this model".

## Implementation checklist

Checked items are complete and verified; unchecked items are not done.

- [x] Inspect and record the dataset details (see [dataset card](dataset-card.md)); statistics reproduced by `scripts/inspect_jena.py`, with synthetic tests in `tests/test_inspect_jena.py`
- [ ] Decide training window stride (cleanup and gap rules proposed above)
- [ ] Load the hourly corpus into `sequences` with split labels
- [ ] Implement shared preprocessing (train-only fit, reused for all splits and inference)
- [ ] Implement per-partition window creation; test no window crosses a boundary or a missing hour
- [ ] Test shapes: input `[batch, 72, 1]`, forecast `[batch, 24]`, attention `[batch, 72, 72]`
- [ ] Implement both models against the same data pipeline
- [ ] Persist run metadata and test MAE in °C
- [ ] Persist predictions with `NULL` attention for the baseline

## Unresolved verification items

Done (see the [dataset card](dataset-card.md)): source and license (CC BY 4.0 per
the provider's download page), retrieval record and per-file checksums, sampling
frequency (10 minutes), proposed aggregation rule, split boundaries and window
counts (measured).

Still open:

- [ ] Timestamp semantics: `MEZ` is only the provider's label; fixed offset vs.
  DST switching is **not documented and not established** by our checks
- [ ] Meaning of `-9999` (548 temperature records in May 2007) — treated as missing
- [ ] Why the files contain repeated rows and out-of-order records
- [ ] Whether to restrict the date range instead of using all 2003–2026 data
- [ ] Training window stride (stride 1 assumed only for the counts)
- [ ] Whether other "Jena Climate" copies are the same subset
- [ ] How the `[batch, 72, 72]` attention maps to the 24-hour forecast (model-head design)
