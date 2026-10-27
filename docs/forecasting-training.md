# Forecasting Training and Evaluation

Back to [README](../README.md) · See also [models](forecasting-models.md), [preprocessing](forecasting-preprocessing.md), [data design](data-design.md).

Code: [shared/forecast_training.py](../shared/forecast_training.py) (workflow),
[scripts/train_forecast.py](../scripts/train_forecast.py) (CLI),
[tests/test_forecast_training.py](../tests/test_forecast_training.py). The
classifier workflow in `api/training.py` is separate and untouched.

**Status: implemented and tested on synthetic data only. No real-data training has
been run, so there are no results, timings or convergence claims.**

## What is held identical for both models

Data version, scaler, training windows, batch size, the batch order of every epoch,
optimizer, loss, learning rate, gradient clipping and seed. Only the model differs.

| Setting | Initial value | Note |
|---------|---------------|------|
| Optimizer | Adam | initial choice, not tuned |
| Learning rate | 0.001 | initial choice, not tuned |
| Batch size | 128 | initial choice |
| Loss | L1 on standardized values | same quantity MAE measures, up to the constant std |
| Gradient clipping | global norm 1.0 | |
| Dropout / schedule / early stopping | none | |
| Epoch cap | configurable (`--epochs`) | a cap, not evidence of convergence |

These are starting points. Change them only by validation results, identically for
both models, and record the change.

## Seeding and batch order

`seed_everything` seeds Python, NumPy and torch before the model is built. Batch
order comes from a separate `torch.Generator` created per epoch from `(seed, epoch)`
only, so it is unaffected by how many random numbers model initialization consumed
and is the same for both models. Each epoch's SHA-256 of its index sequence is stored
in the history to show this. The final batch of an epoch may be shorter; training
loss is weighted by examples. CPU results are reproducible for the same seed on the
same machine and library versions; cross-machine bit-equality is not claimed.

## Evaluation rules

- **MAE in °C** is the sum of absolute errors over all examples and all 24 hours
  divided once by `N × 24`, never a mean of batch means, so a short final batch is
  weighted correctly. Per-horizon MAE (hours t+1 … t+24) is also returned.
- **Batched evaluation** (1024 windows at a time, eval mode, no gradients) discards
  each batch's attention matrix; nothing is accumulated across the corpus.
- **Selection:** `train_model` receives only training and validation windows (the
  test split cannot reach it). The best checkpoint is the lowest validation MAE in
  °C (ties keep the earlier epoch); its weights are restored at the end.
- **Test:** `evaluate_test` is a separate final call, made once after all
  configurations are fixed. The CLI refuses `--evaluate-test` for smoke runs.
- **Persistence reference:** repeat the last observed temperature for all 24 hours.
  It is reported beside the models (validation always; test only with
  `--evaluate-test`). A model that does not beat it has learned little.

## Artifacts

`save_artifact` writes one `.pt` file with the best `state_dict`, model
architecture config, fitted scaler (`mean`, `std`), data version, seed, training
config, full history (per-epoch validation MAE, elapsed seconds, optimizer steps,
batch-order hashes), best epoch, parameter count and torch version.
`load_artifact` rebuilds the model in eval mode with `torch.load(weights_only=True)`;
a test confirms predictions (and attention weights) are bit-identical after a
round trip. Load only artifacts you produced locally. `artifacts/` is git-ignored.

## Commands (prepared, NOT yet run)

All use `.venv312` from the repo root.

**1. Real-data timing / smoke run (local prepared data, excluded from the final comparison).**
No network or credentials. It takes about 20 s to prepare data, then trains.

```bash
# pipeline sanity + timing on a reduced training set
.venv312/bin/python -m scripts.train_forecast --source local --run-kind smoke \
    --epochs 2 --max-train-windows 20000 --seed 0
# per-epoch timing on the FULL training set (use this to size the budget)
.venv312/bin/python -m scripts.train_forecast --source local --run-kind smoke --epochs 1 --seed 0
```

Output goes to `artifacts/smoke/`. Read `seconds_per_epoch` for each model from the
JSON summary.

**2. Training from Supabase rows (assignment requirement).** Reads
`SUPABASE_URL` / `SUPABASE_SERVICE_KEY` from the git-ignored `.env` (never printed)
and pages the stored rows of one data version. Prerequisite: `.venv312` does not
yet contain the `supabase` package; install `supabase==2.7.4` there first (needs
your approval; not done).

```bash
.venv312/bin/python -m scripts.train_forecast --source supabase --run-kind final \
    --data-version jena-mpi-roof-h1-60138aac7d898b8c --epochs <BUDGET> --seed 0
# only after every configuration is fixed from validation results:
.venv312/bin/python -m scripts.train_forecast --source supabase --run-kind final \
    --data-version jena-mpi-roof-h1-60138aac7d898b8c --epochs <BUDGET> --seed 0 --evaluate-test
```

The second command retrains from scratch with the same seed; it reproduces the
first run on the same machine, so the test number belongs to an already-fixed
configuration. Run the other seeds the same way.

## Training budget recommendation (provisional)

There is no timing yet, so no number can be justified. Work it out from the smoke
timing: measure `seconds_per_epoch` for **both** models on the full training set
(about 134,700 windows, 1,053 optimizer steps per epoch at batch size 128). Then:

1. Pick the epoch cap so the **slower** model fits a wall-clock allowance you can
   accept for every seed (e.g. total = epochs × seconds_per_epoch × seeds).
2. Give both models the same cap, and read the validation curves: if the best
   epoch sits at the cap or the curve is still falling, the run is budget-limited,
   not converged, and must be reported as such.
3. Equal epoch caps give equal optimizer steps and equal data passes, but **not**
   equal compute or wall time (attention is O(T²), the LSTM is sequential); report
   both models' actual seconds per epoch and total. If the budget must be matched
   in time rather than epochs, say so explicitly and keep the data and batch order
   identical.

As a starting guess to be replaced by the measurement: a cap in the tens of epochs
over 2–3 seeds, which is only reasonable if the smoke run shows each epoch takes
well under a minute or two on this CPU.

## Checks run

Synthetic only, in `.venv312` (see the repository test run): batch-order equality
across models and across epochs, order independence from global RNG use,
checkpoint selection (scripted validation values, tie handling, non-finite
rejection, restored weights), exact metric aggregation including a short final batch
and a case where a mean of batch means would be wrong, persistence, same-seed
reproducibility of history and weights, artifact round trip and rejection of
malformed artifacts, and the test split being absent from `train_model`'s inputs.

## Limitations

- Nothing has been trained on the real corpus; no accuracy, convergence or timing
  claims exist.
- Hyperparameters are untuned initial choices; no early stopping or schedule.
- The `--source supabase` path is written but has never been executed.
- Only CPU on this machine has been exercised.
