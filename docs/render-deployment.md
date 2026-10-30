# Render deployment (deployed; working state recorded by Emma)

Back to [README](../README.md) · See also [forecasting API](forecasting-api.md), [Streamlit UI](ui.md), [model card](../MODEL_CARD.md).

> **Status.** The service is deployed at <https://seq-sight-api.onrender.com> (confirmed). Emma recorded that run registration, artifact download from Supabase, serving and prediction logging work, and that the opt-in live integration test passed on 2026-10-28 (see [integration testing](integration-testing.md)). Cold-start wake time was not measured. Sections below that say "will" or "expected" describe the design; items marked unverified remain unverified.

The `render.yaml` structure was adapted from the professor-provided *Attend-It* template; Seq Sight's configuration is our own. The Streamlit UI is a separate deployment and is not part of this service.

## Open question (not resolved here)

The class question is still open: **what "same compute budget" and "convergence" mean for this assignment.** The six existing runs share epochs (50), optimizer steps (52,650), batch size and batch order, which is one defensible reading of a matched budget, but nothing in this deployment work decides it. Registration copies the budget facts into `runs.compute_budget`; it does not assert they satisfy the rubric. Resolve with the instructor before the final comparison is written up.

## Sources used for the configuration

* Blueprint spec: <https://render.com/docs/blueprint-spec>: field names `runtime`, `buildCommand`, `startCommand`, `healthCheckPath` (must start with `/`), `autoDeployTrigger`, `envVars` with `sync: false`.
* Blueprint JSON schema: <https://render.com/schema/render.yaml.json>: `plan` accepts `free`; `autoDeployTrigger` is `off | commit | checksPass`; `envVars` items are `key` plus `value`, `sync`, `generateValue`.
* Python version: <https://render.com/docs/python-version>: `PYTHON_VERSION` env var needs a fully qualified version (any release from 3.7.3); it takes precedence over a `.python-version` file.
* Health checks: <https://render.com/docs/health-checks>: HTTP GET, 2xx/3xx is healthy, 4xx/5xx unhealthy, 5 s timeout, checks every few seconds; a new deploy must pass within 15 minutes or it is cancelled; a running instance failing for 15 s stops receiving traffic and after 60 s is restarted.
* Free tier: <https://render.com/docs/free>: spins down after 15 minutes without inbound traffic; waking "takes about one minute"; the filesystem is ephemeral; 750 free instance hours per month; the page gives no memory or CPU figures.
* CPU PyTorch: <https://pytorch.org/get-started/previous-versions/>: `pip install torch==2.4.1 ... --index-url https://download.pytorch.org/whl/cpu`.

These pages were read on 2026-10-27; Render changes its docs, so re-check before deploying. The docs list `3.14.8` as Render's current default Python; `render.yaml` pins `3.12.14` explicitly so the default cannot change under us.

## What `render.yaml` does

| Setting | Value | Why |
|---|---|---|
| `runtime` | `python` | native Python runtime |
| `PYTHON_VERSION` | `3.12.14` | fully qualified, as Render requires. The local `.venv312` interpreter is 3.12.14 and every test in this repo ran on it; torch 2.4.1 and numpy 1.26.4 publish cp312 wheels. That Render accepts this exact patch is **unverified** (the docs say any release ≥ 3.7.3 is accepted) |
| `buildCommand` | CPU torch first, then `pip install -r api/requirements.txt` | see "Dependency size" |
| `startCommand` | `uvicorn api.main:app --host 0.0.0.0 --port $PORT` | Render supplies `PORT` |
| `healthCheckPath` | `/livez` | see "Health checks" |
| `autoDeployTrigger` | `"off"` (quoted: bare `off` is a YAML boolean) | no accidental deploys; deploy manually |
| `plan` | `free` | kept from the template; **not** required, see "Risks" |

Secrets and run ids are **not** in the file. `SUPABASE_URL` and `SUPABASE_SERVICE_KEY` are declared `sync: false` (Render asks for them once, at Blueprint creation). The two run ids are added in the dashboard after registration. The `render.yaml` syntax was parsed locally with PyYAML; it has not been validated by Render.

## Environment variables (dashboard)

| Name | Set where | Value |
|---|---|---|
| `SUPABASE_URL` | dashboard (prompted) | `https://<project-ref>.supabase.co` |
| `SUPABASE_SERVICE_KEY` | dashboard (prompted) | service-role key. Server side only; never in the UI, Streamlit secrets or Git |
| `FORECAST_LSTM_RUN_ID` | dashboard, **after registration** | the `runs.id` printed by the registration command for the chosen LSTM |
| `FORECAST_ATTENTION_RUN_ID` | dashboard, **after registration** | likewise for attention |
| `FORECAST_ARTIFACT_SOURCE` | `render.yaml` | `supabase` |
| `FORECAST_ARTIFACT_CACHE_DIR` | `render.yaml` | `/tmp/seqsight-artifacts` |
| `FORECAST_RETRY_COOLDOWN_S` | `render.yaml` | `30` |
| `ALLOWED_ORIGINS` | `render.yaml` | `*`; the Streamlit app calls the API from its server, so tighten if a browser origin ever calls it directly |

Run ids are never guessed or chosen automatically: unset means "not ready, `run_id_not_configured`".

## Artifact flow (supabase source)

1. Registration (separate and explicit; reported by the user as done, see [forecasting-api.md](forecasting-api.md)) inserts a `runs` row whose `hyperparameters.artifact_sha256` is the SHA-256 of the artifact file, and a `run_artifacts` row with the file base64-encoded. `run_artifacts` is not readable with the anon key.
2. At startup the service reads the run row for each configured run id, takes the expected SHA-256 from it, and (unless a valid cached copy exists) reads `run_artifacts.model_b64`.
3. The bytes are decoded strictly and hashed **before** anything is written. A mismatch discards them (`artifact_hash_mismatch`). A verified file is written atomically (temp file, mode 0600, rename) to the cache. A cached file is re-hashed on every use; a mismatching one is deleted and fetched again.
4. The file is then loaded by the existing `weights_only=True` loader, and the run row is checked against the artifact (model type, data version, seed, scaler, validation MAE, artifact hash). Only then is that model type ready.

Failures carry reason codes only (no paths, URLs or database messages): `run_id_not_configured`, `run_id_invalid`, `run_not_found`, `artifact_hash_missing_in_run`, `artifact_not_in_database`, `artifact_hash_mismatch`, `artifact_invalid`, `artifact_too_large`, `artifact_cache_unusable`, `artifact_type_mismatch`, `run_mismatch_*`, `database_unavailable`, `starting`, `artifact_source_invalid`, `retry_cooldown_invalid`. Each model type is independent: one failing does not stop the other serving.

**Network and retry policy.** Every Supabase call has a 20 s timeout. Startup runs in a background thread so the port opens immediately; until it finishes, `/healthz` reports `starting` and predictions return 503 `starting`. Failed or pending types are retried lazily on `/healthz` and prediction requests, at most once per `FORECAST_RETRY_COOLDOWN_S` (default 30 s), so Render's frequent health checks cannot hammer the database. There is no retry loop and no background poller. The cache is ephemeral (free tier): every cold start re-downloads two files of roughly 60 KB each (base64 ≈ 59 KB and 66 KB for the current artifacts).

## Health checks before registration

Two endpoints, deliberately different:

* **`/livez`** returns `{"status":"alive"}` whenever the process runs. It is *liveness only* and says nothing about models. This is what `healthCheckPath` points at.
* **`/healthz`** is the readiness truth: HTTP 200 only when the database answers **and** both models are loaded, registered and consistent; otherwise **503** with per-model reason codes. There is no success response for an unready service. Use it to monitor, and have the UI "Check API status" button show it.

Why the platform check is `/livez`: if Render's check hit `/healthz`, a deploy made before registration would never become healthy and Render would cancel it after 15 minutes; worse, any later Supabase outage would make a *running* instance fail for 60 s and be restarted in a loop that cannot fix a database problem. With `/livez`, the first deploy succeeds with the service honestly **not ready** (`/healthz` 503, predictions 503 `run_id_not_configured`). The cost: Render will route traffic to an instance that cannot yet predict, and a failed model load will not trigger a restart; watch `/healthz`. If you prefer strict gating, change `healthCheckPath` to `/healthz` and accept those behaviors.

## Cold start and recovery

On the free plan the service spins down after 15 minutes idle and wakes on the next request; Render documents about one minute (not measured here). On wake: process starts, torch and the app import, the port opens, the startup thread fetches and verifies the artifacts (one run lookup and one artifact read per model, each ≤ 20 s). During this window the UI sees timeouts or 502/503 (it reports a possible cold start and offers a bounded "Check API status" poll), then `/healthz` 503 `starting`, then 200. If the database was unreachable at start, models stay `database_unavailable` and recover on the first `/healthz` or prediction after the cooldown, without a restart. If a run is registered or its env var is set *after* deploy, adding the run id variables is expected to restart the service (Render normally applies environment changes by redeploying or restarting; that was not confirmed in the pages read), after which the pending models load.

## Deployment steps (as planned; Emma recorded that the deployed flow works)

Prerequisites: models trained (done); **final test evaluation** run once (done: local `test_eval.json`); models to serve chosen from *validation* results; registration executed (`scripts/register_forecast_runs.py --execute --test-mae-file ...`; per Emma) and its run ids noted (Emma recorded LSTM run 1 and attention run 4).

1. Review `render.yaml`; push the branch you intend to deploy (this work is uncommitted).
2. Render dashboard → **New → Blueprint** → select the repository and branch → Render reads `render.yaml`.
3. When prompted, enter `SUPABASE_URL` and `SUPABASE_SERVICE_KEY` (service-role). Do not paste them anywhere else.
4. Create the Blueprint; the first deploy builds (CPU torch, then requirements) and starts. Expect `/livez` 200 and `/healthz` 503 `run_id_not_configured` until step 5.
5. Service → **Environment** → add `FORECAST_LSTM_RUN_ID` and `FORECAST_ATTENTION_RUN_ID` with the registered ids → save (expect a restart; see above).
6. Check, in order: `GET /livez` → 200; `GET /healthz` → 200 with both models `ready`; `GET /version` → expected run ids, `artifact_source: supabase`, correct artifact hash prefixes; one `POST /predict` per model → 200 with a `prediction_id`; confirm the row appears in `predictions` with the right `served_by_run_id`.
7. Set the UI's Streamlit secret `API_URL` to the service URL (separate deployment; see [ui.md](ui.md)).

## Rollback

* **Bad deploy:** In the dashboard's deploy history, redeploy or roll back to the previous successful deploy (exact menu names not verified against the current dashboard). (Auto-deploy is off, so only deploys you triggered exist.)
* **Wrong or bad model:** change `FORECAST_*_RUN_ID` back to the previous registered ids, or remove one to make that type fail closed (503) instead of serving. Registered runs are never edited or deleted by the API; a run with predictions cannot be deleted (schema `ON DELETE RESTRICT`).
* **Emergency stop:** Service → **Suspend**. The UI then reports connection errors or a cold start; no predictions are served or logged.
* **Cache doubts:** the cache is re-verified on every use and lives on an ephemeral disk; a restart clears it.

## Risks and unmeasured items

* **Dependency size.** PyPI's Linux x86_64 wheel of `torch==2.4.1` declares NVIDIA CUDA packages and `triton` as dependencies (seen in its metadata), i.e. hundreds of MB to gigabytes of install we do not need. The build command installs the `+cpu` build first. That `--index-url` resolution (the CPU index must also serve torch's small dependencies) and the resulting install size are **unmeasured**; if the CPU index lacks a dependency the build fails visibly, and the fallback is `--extra-index-url` instead (pip then prefers `2.4.1+cpu` over `2.4.1`). Pins in `api/requirements.txt` are unchanged.
* **Memory.** The free plan's memory is not stated on the pages read (the Blueprint spec names a `0.5c-512mb` default for plans without `plan`; whether that describes `free` was not verified). Locally on macOS, importing torch and loading both models peaked at about 177 MB resident; Linux, the web stack, Supabase client and request handling add to that, and a 72×72 attention response per request is small. Whether it fits Render's limit is **unknown** until measured on Render. A paid plan is not required by this work; it is an option only if measurement shows a problem.
* **Cold start.** Not measured. Components: Render's documented ~1 minute wake, Python imports (torch import took about 1.3 s locally), and up to four sequential bounded Supabase calls.
* **Free-tier behavior.** Render may restart a free service at any time (docs); each restart repeats the download. Instance hours are limited to 750/month.
* **Not tested anywhere:** Render build, `PYTHON_VERSION=3.12.14` acceptance, CPU-index install, live artifact download from `run_artifacts`, real `predictions` inserts, free-tier memory.
