# Curated comparison examples and failure gallery

Back to [README](../README.md) · See also [models](forecasting-models.md), [UI](ui.md), [model card](../MODEL_CARD.md).

`ui/content/examples.json` is produced by `scripts/curate_forecast_examples.py`, which runs locally. It reads the manifest-verified Jena ZIP files and two fixed checkpoints. It never trains, never touches the test split, reads no credentials and makes no network request. It does not contact Supabase, so the run ids below are **as reported by the user** and are not verified by the script.

```bash
.venv312/bin/python -m scripts.curate_forecast_examples --measure-only   # print the selection
.venv312/bin/python -m scripts.curate_forecast_examples                  # write ui/content/examples.json
.venv312/bin/python -m scripts.curate_forecast_examples --check          # re-derive and compare
```

## Fixed inputs

* LSTM **run 1** = `artifacts/epoch50/final/lstm_seed0.pt`, attention **run 4** = `.../attention_seed0.pt`. These were already the seed-0 runs; no checkpoint was chosen from any test or example result. The script pins and checks both SHA-256 values (`de0af505…` and `82c51854…`, which match `test_eval.json`).
* Data version `jena-mpi-roof-h1-60138aac7d898b8c`; the local corpus is verified against `docs/jena-mpi-roof-checksums.txt` and must reproduce that version and the artifacts' frozen training-fitted scaler (mean 9.3898, std 8.2668) within 1e-9. Windows come from `shared.forecasting`; each selected window is re-run through `prepare_inference_input` (the API's path) and must agree with the batch scan.
* As a consistency check, the script's aggregate **validation** MAE reproduces the values recorded in the artifacts: LSTM 1.8335 °C, attention 1.9158 °C (persistence 3.5257 °C). The scan scores validation windows only.

## Selection rules (fixed before the gallery was produced; not tuned)

Spacing for all six windows together: origins at least 168 h apart, so the 96 h spans cannot overlap.

**Comparison examples** are chosen from the *actual temperatures only*; no forecast is consulted. Candidates have origin 00:00. Patterns, tested in order: **diurnal** (range of the target day ≥ 10 °C, range of history ≥ 10 °C, mean difference ≤ 3 °C), **flat** (whole 96 h within 4 °C) and **shift** (target-day mean differs from the last history day by ≥ 6 °C). Within each pattern the window closest to the median pattern statistic is taken. Candidates found: 255 / 8 / 20.

**Failure cases** are validation windows with attention-model MAE ≥ **5.0 °C** (about 2.6 × its validation MAE; 588 windows qualify), taken in descending order of attention MAE subject to the spacing rule. The LSTM's error is reported but not used in selection.

## Disclosure

* The failure gallery **intentionally selects difficult cases**. It does not represent average performance (validation MAE about 1.92 °C for the attention model, 1.83 °C for the LSTM). The three cases are the extreme of 588 qualifying windows, selected by error size.
* Comparison examples are single windows chosen by temperature pattern. They do not rank the models. On one the LSTM is worse than persistence; on another the attention model is better than the LSTM although the LSTM has the lower aggregate MAE.
* Numbers are rounded (4 decimals °C, 6 for attention weights). Every MAE in the file is computed from the stored values.
* Attention weights are the model's 72×72 encoder input-to-input weights for the selected input, stored as measured for all six examples (comparison and failure). They are not per-output attribution and not an explanation of the error. They were not compared with weights on typical windows, so nothing here says whether these patterns are unusual.

## "Confidently wrong": clarification needed

The assignment wording "confidently wrong" presupposes a model that expresses confidence. These models output one deterministic value per hour with no prediction interval and no calibrated confidence, so there is no measured confidence to be wrong about. The gallery therefore documents **large errors from a model that displays no uncertainty**. Concentrated attention and a single deterministic forecast are not confidence. The wording should be confirmed with the instructor, for example whether large-error cases satisfy the requirement or whether a model with a confidence output is expected.

## Selected cases (validation windows; MAE in °C)

| Case | Origin | LSTM (run 1) | Attention (run 4) | Persistence |
|---|---|---|---|---|
| Comparison: strong daily cycle | 2020-04-07 00:00 | 1.110 | 1.007 | 4.688 |
| Comparison: nearly flat | 2020-10-17 00:00 | 0.632 | 0.408 | 0.441 |
| Comparison: large level change | 2021-03-29 00:00 | 3.740 | 2.932 | 4.409 |
| Failure 1 | 2020-05-11 03:00 | 10.180 | 11.284 | 9.693 |
| Failure 2 | 2021-12-24 19:00 | 10.044 | 10.096 | 10.812 |
| Failure 3 | 2021-04-11 17:00 | 10.990 | 9.944 | 14.026 |

Observed in all three failures: a large change in the next 24 hours that the attention model, the LSTM and persistence all missed by more than 9 °C on average (a sharp cooling in each case). This says the errors are not specific to attention. That temperature-only inputs cannot show an approaching weather change is a hypothesis (the models see no other variables), not something these windows test. Discussion text in `ui/content/example_notes.json` labels observation and speculation separately.

## Not done

The gallery (`examples.json`, `example_notes.json`, the script, tests and docs) is a **local, uncommitted change**; nothing has been committed, pushed or deployed. 
Not viewed in a browser. Run ids 1 and 4 are not checked against Supabase. No test-split windows are used. The only unit-tested parts are the selection helpers (synthetic data) and the shape and consistency of the current local `examples.json` (`tests/test_curate_examples.py`, `tests/test_ui_helpers.py`).
