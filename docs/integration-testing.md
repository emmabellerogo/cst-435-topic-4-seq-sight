# Integration test and CI

Back to [README](../README.md) · See also [forecasting API](forecasting-api.md), [Supabase setup](supabase-setup.md), [Render deployment](render-deployment.md).

## Status (read this first)

| Item | Status |
|---|---|
| Offline suite (`pytest`) | **Run locally** in `.venv312`: passes (one expected skip, the live test). No network. |
| Offline tests of the integration helpers (config gating, exact-row cleanup, response/row/run comparison) | **Run locally**, against the in-memory repository and synthetic artifacts |
| `tests/test_supabase_integration.py` against a real Supabase project | **NOT run.** It is skipped without configuration. A skip is not a verification; nothing here shows the live path works |
| `ci.yml` (offline CI) | **Reported green by the user** (not independently observed by Claude) |
| `integration.yml` (manual live job) | **NOT run.** YAML was parsed locally only; the `supabase-integration` Environment and its secrets are the user's to create |
| `docs/model-comparison.sql` | **Executed by the user** in the production project; outputs transcribed in [results/](results/README.md). Claude only ran static checks (read-only, scoped hashes) |

The user separately reports that the deployed Streamlit → Render → Supabase flow works (manual checks); that is not a result of this test, and green offline CI is not live verification either.

## What the integration test does

`POST /predict` for each registered model type, through the real `create_app` and `SupabaseForecastRepository` (the same code path as production), then reads the stored rows back and compares:

* response vs stored `predictions` row: `prediction_id` = `id`, `input_hash` (also recomputed from the sent history), `model_type`, `served_by_run_id` = the configured run id, and `forecast_c` = `forecast` (24 values);
* the referenced `runs` row (read by `served_by_run_id`): same id, same `model_type`; the database foreign key `(served_by_run_id, model_type) → runs(id, model_type)` also enforces this;
* LSTM: response `attention` is null and stored `attention` is NULL; attention model: both are finite 72×72 with rows summing to 1 and equal to each other.

It uses a deterministic synthetic 72 h history (a connectivity/logging fixture, not a forecast-quality test). It **creates no runs and no `run_artifacts` and edits no artifacts**; the only rows it writes are the `predictions` rows the API inserts.

## Prerequisites (documented, not created by the test)

1. A **separate, fresh Supabase test project** initialized with **only** [`db/migrations/002_seq_sight_forecasting.sql`](../db/migrations/002_seq_sight_forecasting.sql). `002` is a standalone initialization that replaces Artzi's `001_init.sql`; **do not apply `001` to this project** (their `runs` and `run_artifacts` definitions conflict, and `002` refuses to run if `001`'s tables exist). `001` stays unchanged in the repository for reference only. See [Supabase setup](supabase-setup.md). The test refuses a URL equal to `SUPABASE_URL` in the environment unless you set `SEQSIGHT_IT_ALLOW_PRIMARY_PROJECT=1`.
2. **A new project starts empty, so it needs its own registered runs before the live test can run.** That means one `lstm` and one `attention` artifact that were legitimately trained and evaluated (the one-time test evaluation, producing a real `test_eval.json`), registered into *that* project with `scripts/register_forecast_runs.py --execute --test-mae-file <real test_eval.json>`, with their `run_artifacts` rows (the default `SEQSIGHT_IT_ARTIFACT_SOURCE=supabase` downloads them exactly as Render does). Registration writes only `runs` and `run_artifacts`, and the API never reads `sequences`, so the Jena corpus does not have to be loaded into the test project for this test. Never insert fabricated or placeholder metrics, and never reuse the production project's run ids: run ids are per project, so use the ids that registration prints for the test project. The existing production runs are not touched and cannot be used from another project. Alternatively set `SEQSIGHT_IT_ARTIFACT_SOURCE=local` plus `SEQSIGHT_IT_LSTM_ARTIFACT` / `SEQSIGHT_IT_ATTENTION_ARTIFACT` paths to the local `.pt` files whose hashes are recorded on those runs.
3. The service-role key of that test project (never the production key).

## Environment variables

| Variable | Required | Meaning |
|---|---|---|
| `SEQSIGHT_IT_ENABLE` | yes | must be `1`; the explicit opt-in |
| `SEQSIGHT_IT_SUPABASE_URL` | yes | test project URL |
| `SEQSIGHT_IT_SUPABASE_SERVICE_KEY` | yes | test project service-role key |
| `SEQSIGHT_IT_LSTM_RUN_ID` | yes | `runs.id` of the registered lstm run (positive integer; no default) |
| `SEQSIGHT_IT_ATTENTION_RUN_ID` | yes | `runs.id` of the registered attention run (must differ) |
| `SEQSIGHT_IT_ARTIFACT_SOURCE` | no | `supabase` (default) or `local` |
| `SEQSIGHT_IT_LSTM_ARTIFACT`, `SEQSIGHT_IT_ATTENTION_ARTIFACT` | only for `local` | artifact paths |
| `SEQSIGHT_IT_ALLOW_PRIMARY_PROJECT` | no | `1` to accept the same project as `SUPABASE_URL` (not recommended) |

The test never reads `.env`, and the production names `SUPABASE_URL` / `SUPABASE_SERVICE_KEY` alone never enable it. Failure messages carry run ids and mismatch descriptions, not the URL or key.

## Commands

Default suite (offline; the integration test shows as skipped):

```bash
.venv312/bin/python -m pytest -q -rs
```

Only the integration file, unconfigured (expected: all skipped, which pytest reports with **exit code 5**, not a pass):

```bash
.venv312/bin/python -m pytest tests/test_supabase_integration.py -q -rs
```

Live run against the test project, from your own shell (set the variables yourself; do not paste keys into commands that are logged):

```bash
export SEQSIGHT_IT_ENABLE=1
export SEQSIGHT_IT_SUPABASE_URL='https://<test-project>.supabase.co'
export SEQSIGHT_IT_SUPABASE_SERVICE_KEY='<test service-role key>'
export SEQSIGHT_IT_LSTM_RUN_ID=<id>
export SEQSIGHT_IT_ATTENTION_RUN_ID=<id>
.venv312/bin/python -m pytest tests/test_supabase_integration.py -m integration -v --tb=short
```

Expected on success: 2 passed (`lstm`, `attention`). The first request may wait up to 90 s for the API to load artifacts from the project; a timeout fails with the per-model reason codes.

## Cleanup behavior

The API inserts the rows; a wrapper records `(id, input_hash, served_by_run_id)` of each insert it performs. In each test's `finally`, and again in a module-level safety net, those rows are deleted one at a time with `delete().eq("id").eq("input_hash").eq("served_by_run_id")`. No truncate, no range or `like` filter, no delete by table or by time. A delete that fails is reported (`prediction <id> could not be deleted; remove it manually`) and the run fails. Limits: if the API wrote a row but the process died before the wrapper saw the response, that row is not tracked and stays; look for rows whose `input_hash` equals the SHA-256 printed by `canonical_input_hash` for the synthetic history. Do not enable `SEQSIGHT_IT_ALLOW_PRIMARY_PROJECT` casually: a run there deletes only its own rows, but it still writes to the production audit table.

## GitHub Actions

* **`ci.yml`** (push to `main`, and all pull requests including forks): Python 3.12; installs CPU torch `2.4.1` from `https://download.pytorch.org/whl/cpu` first (the PyPI Linux wheel pulls CUDA packages), then `requirements-dev.txt` (pins unchanged); runs `pytest -q -rs`. `permissions: contents: read`; no secrets; `SEQSIGHT_IT_ENABLE=0`. Expected skips: the live integration module, and the SQL-hash check against the git-ignored local `test_eval.json`.
* **`integration.yml`** (manual `workflow_dispatch` only, ref must be `main`, a typed `run` confirmation): uses the GitHub Environment `supabase-integration`. Create it under Settings → Environments, add required reviewers, restrict to `main`, and store `SEQSIGHT_IT_SUPABASE_URL` and `SEQSIGHT_IT_SUPABASE_SERVICE_KEY` as environment **secrets** and the two run ids as environment **variables** (`SEQSIGHT_IT_LSTM_RUN_ID`, `SEQSIGHT_IT_ATTENTION_RUN_ID`). There is no `pull_request`/`push` trigger, so fork PRs never receive these secrets; GitHub also masks secret values in logs. A `concurrency` group prevents overlapping runs on one test project.
* Actions are referenced by major-version tag (`actions/checkout@v4`, `actions/setup-python@v5`); pinning to commit SHAs is stricter and was not done because the SHAs could not be verified offline.

Start the manual run from the repository's Actions tab → "Supabase integration (manual)" → Run workflow, or `gh workflow run integration.yml -f confirm=run`.

## README comparison query

`docs/model-comparison.sql` holds three read-only statements for the Supabase SQL editor: (1) per-model run count, mean test MAE and sample SD (also validation), (2) per-seed details, (3) a completeness/duplicate check. They are filtered to data version `jena-mpi-roof-h1-60138aac7d898b8c` and the six registered artifact hashes with matching model type and seed. No output is recorded in the repository; the README table stays "TBD" until you run it.
