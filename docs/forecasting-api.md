# Forecasting API

Back to [README](../README.md) · See also [models](forecasting-models.md), [training workflow](forecasting-training.md), [Supabase setup](supabase-setup.md).

The FastAPI service in `api/` serves the trained LSTM and self-attention models. The three-cloud layout and the app/CORS scaffolding come from the professor-provided *Attend-It* template; the endpoints and logic here are ours. The Attend-It endpoints (`/train`, `/datasets`, `/predict_sample`, `/classes`, `/runs/{id}`) were removed. `api/db.py`, `api/training.py`, `shared/data.py` and `shared/schemas.py` are template leftovers the API no longer uses. The Streamlit UI has been adapted to this API; see [ui.md](ui.md).

## Layout

| File | Role |
|---|---|
| `api/main.py` | `create_app(settings, repo)`: endpoints only |
| `api/forecast_service.py` | artifact loading, registration checks, inference, canonical hash (no SQL, no web framework) |
| `api/forecast_repo.py` | `ForecastRepository` interface + Supabase implementation (lazy client, sanitized errors) |
| `api/forecast_config.py` | environment configuration |
| `shared/forecast_schemas.py` | request/response models and limits |
| `shared/forecast_registration.py`, `scripts/register_forecast_runs.py` | run registration (dry-run by default) |

## Configuration

| Variable | Meaning | Default |
|---|---|---|
| `FORECAST_LSTM_ARTIFACT` / `FORECAST_ATTENTION_ARTIFACT` | artifact path (relative paths resolve from the repo root) | `artifacts/epoch50/final/{lstm,attention}_seed0.pt` |
| `FORECAST_LSTM_RUN_ID` / `FORECAST_ATTENTION_RUN_ID` | `runs.id` registered for that artifact | **none** (required) |
| `SUPABASE_URL`, `SUPABASE_SERVICE_KEY` | database access (server side only) | none |
| `ALLOWED_ORIGINS` | CORS | `*` |

The default artifacts are **seed-0 development defaults, not a final deployment selection**; `/version` reports `artifacts_are_development_defaults`. Artifacts are loaded once at startup with `weights_only=True`; load only files you produced.

## Serving rule: no registered run, no predictions

Each served artifact must be mapped to a database run **by configuration**; run ids are never guessed or created by the API. A model type becomes ready only when its artifact loads and the configured `runs` row matches it: `model_type`, `data_version`, `seed`, scaler (`preprocessing.mean/std`), `val_mae_c`, and `hyperparameters.artifact_sha256` (SHA-256 of the exact artifact file, written by the registration command). Anything else leaves that type not ready, with a reason code: `artifact_not_found`, `artifact_invalid`, `artifact_type_mismatch`, `run_id_not_configured`, `run_id_invalid`, `run_not_found`, `run_mismatch_{model_type,data_version,seed,scaler,val_mae,artifact}`, `database_unavailable`. Verification is retried on `/healthz` and on requests, so a database outage at startup heals itself. A not-ready type returns **503** with `{"detail": {"code", "message"}}` (no paths); the other type keeps serving.

## Endpoints

| | |
|---|---|
| `POST /predict` | `{"model_type": "lstm"\|"attention", "history_c": [72 numbers]}` |
| `POST /predict_batch` | `{"model_type", "histories_c": [[72 numbers], ...]}`, **1 to 32 histories** |
| `GET /predictions?limit=20&offset=0&include_attention=false` | newest first; `limit` 1–100, `offset` 0–10 000; `has_more` instead of a total |
| `GET /runs?limit=20&offset=0` | persisted runs, newest first; `served_by_this_api` marks the served ones |
| `GET /healthz` | 200 only if the database answers and both models are ready; otherwise 503 with per-model reasons |
| `GET /version` | API/git/torch versions, limits, hash format, per-model run id, seed, data version, validation MAE, artifact hash prefix; no paths or credentials |

**Validation** (422): exactly 72 values; each a JSON number (booleans, strings and null are rejected, not coerced); finite; `|v| <= 1000` (a numerical guard against float32 overflow, not a physical-plausibility check); unknown fields rejected. Validation error bodies omit the offending input.

**Response** (`/predict`): `prediction_id`, `model_type`, `served_by_run_id`, `input_hash`, `forecast_c` (24 values, t+1 … t+24, °C) and `attention` (72×72 for `attention`, `null` for `lstm`). Attention rows are the encoder's input-to-input weights, not a per-output-hour attribution (see [models](forecasting-models.md)).

**Inference**: standardize with the artifact's own scaler, model in eval mode under `torch.inference_mode`, inverse-transform to °C, reject non-finite output. A batch is one forward pass; its results can differ from single calls by float rounding (about 1e-5).

## Audit logging and failure behavior

Every forecast is written to `predictions` (input hash, forecast, attention, model type, `served_by_run_id`) **before** it is returned.

* `/predict`: if the write fails, the forecast is withheld and the response is **503 `prediction_not_audited`**. A forecast is never returned as served without an audit row.
* `/predict_batch`: validation is all-or-nothing (422, nothing logged). After inference each item is logged independently. **200** all audited; **207** some were not (those items have `status: "not_audited"`, no prediction, an error string; the others are valid and logged); **503** none could be audited. No rollback of rows already written.
* `/predictions`, `/runs`: database failure gives 503 `database_unavailable`.

**Canonical input hash** (`seqsight-input-v1`): SHA-256 hex of the UTF-8 text `seqsight-input-v1|degC|72|v0,v1,…,v71`, each value a float64 written with Python `repr`, `-0.0` as `0.0`. `5` and `5.0` hash the same; the model type is not part of it. Raw inputs are never stored.

## Run registration (not executed)

```bash
.venv312/bin/python scripts/register_forecast_runs.py            # dry-run (default): local files only
```

Prints, for the six `artifacts/epoch50/final` artifacts, the `runs` fields (hyperparameters, scaler, compute budget, validation MAE, artifact hash) and the size of the `run_artifacts` payload.

**Blocked on the final test evaluation.** `runs.test_mae_c` is `NOT NULL` and no test MAE exists yet (the artifacts hold validation metrics only). The command therefore reports `test_mae_c=MISSING`, and `--execute` refuses to run without `--test-mae-file`. It never inserts zero, a validation figure or any placeholder.

### Test evaluation (not run)

`scripts/evaluate_forecast_artifacts.py` scores the six fixed checkpoints on the test split. It loads trusted artifacts, never trains, fingerprints the weights before and after (aborting if they change), and reports all six without ranking or selecting. It is a dry-run by default (local files only). With `--execute` it reads the corpus through `shared/forecast_store.py`, refuses unless the corpus reproduces every artifact's data version and scaler, and creates `artifacts/epoch50/final/test_eval.json` exclusively: an existing file is never overwritten and there is no force flag. The file holds test MAE and persistence MAE (overall and per horizon), each artifact's file and weights SHA-256, the data version, and the evaluation configuration. Run it once, after the models to serve have been decided from validation results.

```bash
.venv312/bin/python scripts/evaluate_forecast_artifacts.py             # dry-run
.venv312/bin/python scripts/evaluate_forecast_artifacts.py --execute   # real test evaluation, once
.venv312/bin/python scripts/register_forecast_runs.py --execute --test-mae-file artifacts/epoch50/final/test_eval.json
```

The registration command accepts that file directly and rejects it if any artifact's SHA-256 or the data version differs from the artifact being registered (a plain `{"lstm_seed0": <degC>, ...}` JSON still works). It skips artifacts already registered (same SHA-256), and deletes a just-created run if its `run_artifacts` insert fails (best effort; an orphan is reported). Then set `FORECAST_LSTM_RUN_ID` and `FORECAST_ATTENTION_RUN_ID` to the printed ids of the chosen runs.

## Local setup

```bash
.venv312/bin/python -m pip install -r api/requirements.txt   # pinned; torch/numpy untouched
.venv312/bin/python -m pytest tests/test_forecast_api.py
# with real registration and credentials in .env, later:
FORECAST_LSTM_RUN_ID=<id> FORECAST_ATTENTION_RUN_ID=<id> .venv312/bin/python -m uvicorn api.main:app
```

Without run ids the server starts, `/healthz` returns 503 and prediction endpoints return 503 `run_id_not_configured`.

## Tested vs. remaining

**Tested** (`tests/test_forecast_api.py`, synthetic untrained artifacts, in-memory fake repository, a fake Supabase client): validation, response shapes, scaler/eval-mode inference, hash format, logging and run linkage, batch success/422/207/503, pagination bounds, readiness reasons and recovery, registration mismatches, `/version` leaking no paths, registration planning/refusal/idempotency/cleanup, repository laziness and error sanitizing. The real seed-0 artifacts were also loaded and run once locally.

**Not tested / remaining:** anything against a real Supabase (query shapes were checked only against a recording stub; the `predictions` JSON/foreign-key checks are untested end to end); the registration `--execute` path against the real database; executing the test evaluation (the script exists and is tested on synthetic data only); choosing the deployment models; getting artifacts onto Render (`artifacts/` is git-ignored, and `run_artifacts.model_b64` is stored but not read by the API); `render.yaml` still describes Attend-It; concurrency and latency under load; response size of large batches (32 attention items ≈ 1.6 MB).
