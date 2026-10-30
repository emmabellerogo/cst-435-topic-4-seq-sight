# Frozen-reference forecasting regression

Back to [README](../README.md) · See also [forecasting API](forecasting-api.md), [models](forecasting-models.md), [integration testing](integration-testing.md).

A frozen-reference test feeds one fixed 72 h input through the serving path and compares the 24 h forecast with **stored** expected values. It catches accidental changes to the models, scaler, preprocessing or inference code. It is not a measure of forecast quality.

## The fixed input

`temperature_c[h] = round(9.0 + 6.0·sin(2π(h − 10)/24) − 0.03·h, 3)` for h = 0…71 (h = 71 is the last observed hour), in °C. It is synthetic: a smooth daily cycle with a slight cooling trend. The 72 values are stored in each fixture, and a test checks they equal the formula.

## Two fixtures (`tests/fixtures/`)

| Fixture | Models | Runs in CI? | What it verifies |
|---|---|---|---|
| `frozen_reference_synthetic.json` | **Untrained** LSTM and attention built with `torch.manual_seed(0)`, scaler mean 9.5 / std 8.0 | **Yes**, everywhere | the serving path, scaler, shapes and determinism. **Not** the trained checkpoints |
| `frozen_reference_trained_seed0.json` | the trusted seed-0 trained artifacts (runs 1 and 4 as reported by the user), pinned by SHA-256 | **No**: needs the git-ignored `artifacts/epoch50/final/*.pt`; **skipped** with an explicit message when they are absent | that the trained checkpoints still produce the frozen forecasts |

Each fixture holds: the input and its units; expected 24 h forecasts in °C for both models; for the attention model, a small probe of the weights (the last row and column means of the 72×72 matrix); scaler and data-version metadata; the generation procedure, torch version and UTC time; the tolerances. The trained fixture also stores both artifact SHA-256 values, which a test compares with the seed-0 hashes in `docs/model-comparison.sql`.

**Tolerances** (absolute): forecast ±1e-4 °C per hour, attention probe ±1e-5. They absorb float32 and platform or torch-build differences; a real change in weights, scaler or code moves forecasts by far more. If a legitimate torch upgrade breaks the synthetic fixture (random initialization can differ between torch versions), review the diff and re-freeze deliberately.

## Generation (once; never overwrites)

```bash
.venv312/bin/python -m scripts.generate_frozen_reference --kind synthetic
.venv312/bin/python -m scripts.generate_frozen_reference --kind trained     # needs the local artifacts
```

The generator runs the input through `api.forecast_service.run_forecasts` (the `/predict` path), refuses an artifact whose hash differs from the pinned one, reads no corpus, trains nothing and uses no network or credentials. It will not replace an existing fixture; to re-freeze, delete the file yourself and review the new diff. The tests only **read** the fixtures; they never regenerate expected values. The two fixtures here were generated on the development machine (torch 2.4.1, CPU) on 2026-10-28.

## Running

```bash
.venv312/bin/python -m pytest tests/test_frozen_reference.py -v -rs
# artifacts elsewhere:
SEQSIGHT_REFERENCE_ARTIFACT_DIR=/path/to/dir .venv312/bin/python -m pytest tests/test_frozen_reference.py -v -rs
```

If an artifact is present but its hash differs from the fixture, the trained test **fails** (it is not a different checkpoint to compare against).

## Status

* Executed locally, with the trained artifacts present: all 7 tests in `tests/test_frozen_reference.py` passed, including `test_trained_artifacts_match_frozen_forecasts`. Run with the artifact directory pointed at a missing path: 6 passed, trained test skipped.
* In GitHub Actions the trained test will be **skipped** (artifacts are git-ignored). Green CI therefore verifies the synthetic fixture and fixture structure, not the trained checkpoints.
