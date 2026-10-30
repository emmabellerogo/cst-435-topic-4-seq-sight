# CLC -Product 4: “Seq-Sight”: A Sequence-Modeling Service with Attention Explanations

_Short name used in this repository: **Seq Sight** (hourly temperature forecasting with attention explanations)._

> **Status: implemented and deployed; final human review of the report and cards pending.** Emma
> Rogoveanu completed this project individually, starting from Professor Artzi's Attend-It starter and
> using documented AI assistance. Her recorded checks: GitHub Actions CI is green, the deployed
> Streamlit → Render → Supabase flow works (forecasts, attention heatmaps, stored-attention drilldown,
> prediction history, persisted run metrics), and the opt-in live Supabase integration test passed
> from her local terminal on 2026-10-28 (2 passed in 5.36 s;
> [details](docs/integration-testing.md#live-run-record-user-executed)). The video link is in the
> deliverables table below; its access and duration have not yet been verified.

## Project scope

The project is **Variant A: hourly temperature forecasting** on the **Jena Climate**
dataset, with a **24-hour forecast horizon**. It compares an LSTM baseline
against a model with a hand-written scaled-dot-product self-attention
block, and exposes attention weights so users can see where the encoder's attention fell
(input-to-input weights, not a causal explanation of a forecast).

**Customer story**

> As a campus outdoor-events planner, I want a 24-hour temperature forecast and
> a view of which past hours influenced it, so that I can plan events while
> understanding the forecast's limitations.

**Design docs:** [Data design and forecasting contract](docs/data-design.md) · [Dataset card (Jena Climate)](docs/dataset-card.md) · [Supabase setup (migration 002, applied)](docs/supabase-setup.md) · [Forecasting preprocessing](docs/forecasting-preprocessing.md) · [Forecasting models](docs/forecasting-models.md) · [Training workflow](docs/forecasting-training.md) · [Training review (50-epoch runs)](docs/training-review.md) · [Forecasting API](docs/forecasting-api.md) · [Streamlit UI](docs/ui.md) · [Render deployment](docs/render-deployment.md) · [Integration test and CI](docs/integration-testing.md) · [Curated examples](docs/forecasting-examples.md) · [Model-comparison SQL results](docs/results/README.md)

## Starter template vs. Seq Sight

| | Existing: Attend-It starter | Seq Sight (this project) |
|---|---|---|
| Origin | Professor-provided template, added in commit `dae6272` | The contributor's assignment work, with documented AI assistance (see below) |
| Task | Synthetic token-sequence classification | Hourly temperature forecasting (Jena Climate, 24 h horizon) |
| Model | LSTM + additive (Bahdanau) attention | Separate LSTM baseline + hand-written scaled-dot-product self-attention model |
| Status | Starter code; its classification parts are no longer used | Implemented (see checklist for what is verified) |

**Attribution:** the three-cloud structure (Streamlit UI ⇄ FastAPI API ⇄
Supabase) and the starter scaffolding come from the professor-provided
*Attend-It* template for this course. The scaffolding was adapted; the
Seq Sight data, models, API contract, UI, and documentation are the contributor's
assignment work (with the AI assistance recorded below) unless noted otherwise.
Starter files not yet replaced (for example `api/db.py`, `api/training.py`,
`db/seed.py`, `db/migrations/001_init.sql`) may still describe or implement
Attend-It behavior and are not used by Seq Sight.

## Architecture

```
Streamlit (UI, thin client)  ⇄  FastAPI on Render (models, PyTorch)  ⇄  Supabase (Postgres)
```

The UI holds no ML code and never writes to Supabase. During serving, the API is the only component that writes (one `predictions` record per forecast). The administrative loader (`db/load.py`) and registration script (`scripts/register_forecast_runs.py`) also write, during setup, using the server-side key.

## Deployment links

| Tier | Platform | URL |
|------|----------|-----|
| Code | GitHub | [github.com/emmabellerogo/cst-435-topic-4-seq-sight](https://github.com/emmabellerogo/cst-435-topic-4-seq-sight) |
| UI | Streamlit Community Cloud | [cst-435-topic-4-seq-sight-pb5jwfwvdotwudcv9fuifb.streamlit.app](https://cst-435-topic-4-seq-sight-pb5jwfwvdotwudcv9fuifb.streamlit.app/) (reported working) |
| API | Render | [https://seq-sight-api.onrender.com](https://seq-sight-api.onrender.com) (confirmed) |
| Data | Supabase | Project URL `https://vrcdwzypcqkpxetunnxm.supabase.co`, project reference `vrcdwzypcqkpxetunnxm` (confirmed; no keys recorded) |

## Data persisted / not persisted

- **Stored in Supabase:** the Jena training corpus (`sequences`), run metrics and hyperparameters (`runs`), model artifacts (`run_artifacts`), and a prediction audit record for every served forecast (`predictions`): input hash, 24 forecast values, attention matrix (attention model only), serving run id and timestamp.
- **Not stored in the predictions table:** the submitted raw 72-hour temperature history (only its SHA-256 hash), and any user identifier. A hash does not make an input private, because it can be recomputed from a guessed input.
- **Publicly readable (intended anon policy):** `runs` and `predictions`, including forecasts and attention records. Do not submit confidential data.
- **Private:** model artifacts (`run_artifacts`) and the corpus (`sequences`) have no anon access; only server-side keys (the API, and the administrative loader and registration scripts) read or write them.
- Hosting-platform logs (Render, Streamlit, Supabase) were not inspected and no claim is made about them.

Details: [model card, Persistence and privacy](MODEL_CARD.md#persistence-and-privacy).

## Implementation checklist

Ticked only where the repository or Emma's recorded results support it. "Offline" = the pytest suite run locally. CI runs the offline suite only: it never contacts Supabase, and it skips both the live integration test and the trained-artifact frozen-reference regression (the artifacts are git-ignored). Those two passed separately: the live Supabase integration test from Emma's terminal on 2026-10-28 (2 passed in 5.36 s; see [integration-testing.md](docs/integration-testing.md#live-run-record-user-executed)), and the trained-artifact regression locally. Not claimed: the manual GitHub `integration.yml` job (not used for the live run), a repeat live run, and load or cold-start timing. Detailed verification provenance is in the linked docs.

**Data and preprocessing**
- [x] Load Jena Climate hourly readings into Supabase using `db/load.py` (corpus upload and split counts checked by Emma; training read the rows from Supabase per the artifacts' recorded source)
- [x] Define reproducible input windows and chronological train/validation/test splits; fit preprocessing on training data only (offline tests; [preprocessing](docs/forecasting-preprocessing.md))
- [x] Create `sequences`, `runs`, and `predictions` tables; `predictions` stores input hash, forecast, attention weights, serving run ID, and creation timestamp (schema applied; prediction logging confirmed by the live integration test and the deployed prediction history)

**Models and comparison**
- [x] Implement a separate PyTorch LSTM baseline
- [x] Implement a second model with a hand-written scaled-dot-product self-attention block (no `torch.nn.MultiheadAttention`)
- [x] Compare models with identical preprocessing, matched seeds and compute budgets, and persisted test MAE; generate the README comparison table with SQL. *Qualifications:* the budget is matched in epochs and optimizer steps (50 epochs = 52,650 steps per run, same windows, same batch order), **not** measured or equalized compute: the two architectures do different amounts of work per step, and wall-clock time was recorded but not controlled ([training review §6](docs/training-review.md)). Convergence is not claimed (continued, decelerating improvement was observed). Professor Artzi approved this training approach (per Emma; no written record in the repository). The table below comes from SQL Emma ran.

**API**
- [x] `POST /predict`, `POST /predict_batch`, `GET /predictions`, `GET /runs`, `GET /healthz`, `GET /version` (offline-tested; `/predict`, history and run metrics checked on the deployment)
- [x] Input-length and dtype validation on prediction endpoints (offline tests)

**UI and explanations**
- [x] Six Streamlit tabs: Concepts, Try It, Baseline vs. Attention, Recent Predictions, Failure Gallery, Model Card (offline headless render; deployed app, including stored-attention drilldown, checked by Emma)
- [x] Worked numerical attention example explaining √d_k (computed live; offline-tested)
- [x] Matplotlib attention heatmaps for at least three examples (three comparison and three failure examples, measured on validation windows; [details](docs/forecasting-examples.md))
- [ ] Three confidently wrong attention-model forecasts, with discussion. Three large-error cases are documented, but the models show no confidence, so "confidently wrong" is not literally measurable; the wording needs confirming with the instructor ([clarification](docs/forecasting-examples.md))

**Tests**
- [x] Input-length validation
- [x] Attention-matrix shape
- [x] Frozen reference input ([details](docs/frozen-reference.md)): a fixed synthetic 72 h input with stored expected forecasts. The check against the trained seed-0 artifacts passed locally but is skipped in CI (the artifacts are git-ignored); a separate untrained-model fixture runs in CI and is not verification of the trained checkpoints
- [x] Prediction logging linked to a run (offline with an in-memory repository, and the live integration test against Supabase, run by Emma 2026-10-28: both model types verified against the stored `predictions` row and referenced `runs` row, test rows cleaned up)
- [x] Hand-written attention block (compared against a NumPy reference)
- [x] Shared preprocessing (stored training scaler applied at inference; offline tests)

**Deployment and documentation**
- [x] Deploy Streamlit, Render, and Supabase (working per Emma's checks; identifiers in the table above)
- [x] Add CI (`.github/workflows/ci.yml`, offline tests; green)
- [x] Handle API cold starts gracefully in the UI (implemented and offline-tested; actual wake time not measured)
- [ ] Complete model card and dataset card (drafted; final review pending)
- [x] Explicit data-persistence/privacy statement ([model card](MODEL_CARD.md))
- [x] Uncertainty and risk disclosures ([model card](MODEL_CARD.md))

**Submission**
- [ ] The contributor's individual 3–5 minute video and 750–1000 word engineering report, with the links filled in below (report: AI-assisted draft saved, **awaiting Emma's final review**; video: link added, access and duration not yet verified)

## Model comparison

Final test MAE in °C, from the SQL in [docs/model-comparison.sql](docs/model-comparison.sql). Emma ran the query in the Supabase SQL editor; the outputs are stored in [docs/results/](docs/results/README.md). Each model has 3 runs (seeds 0, 1, 2; LSTM run ids 1 to 3, attention run ids 4 to 6), all trained for the same 50 epochs / 52,650 optimizer steps. Data version `jena-mpi-roof-h1-60138aac7d898b8c`.

| Model | Runs | Mean test MAE (°C) | SD across seeds (°C) | Mean validation MAE (°C) | SD across seeds (°C) |
|-------|-----:|-------------------:|---------------------:|-------------------------:|---------------------:|
| LSTM baseline | 3 | 1.8689 | 0.0070 | 1.8387 | 0.0045 |
| Self-attention | 3 | 1.9452 | 0.0122 | 1.9070 | 0.0154 |

The SD is the sample standard deviation (n − 1) of the per-run MAE **across the three training seeds**. It measures how much the trained models vary with the seed. It is not prediction uncertainty, not a confidence interval, and says nothing about how sure a model is about a forecast; with three seeds it is itself imprecise. A persistence baseline (repeat the last temperature) has test MAE 3.4150 °C (local evaluation file, not part of the SQL).

On this test split the LSTM's mean MAE is 0.0763 °C lower than attention's (1.9452 − 1.8689), and every LSTM seed was below every attention seed on validation. Both models are far better than persistence. This is a result for these untuned settings, one station and one training budget; it does not show that attention is a worse architecture in general, and the budget was matched in steps, not in compute. Professor Artzi approved the existing training approach (per Emma; no written record is in the repository).

## Contributor and individual deliverables

This project was completed by one person, Emma Rogoveanu, working alone over about a week. There is no partner, team or division of work to describe.

| Contributor | Video (3–5 min) | Engineering report (750–1000 words) | Contribution evidence |
|-------------|-----------------|--------------------------------------|-----------------------|
| Emma Rogoveanu | [Emma Rogoveanu — Product Demonstration](https://www.loom.com/share/2c4aebe3ddad42808a0758bc4bda2c57) (access and duration not yet verified) | [docs/engineering-report.md](docs/engineering-report.md) (AI-assisted; **final review by Emma pending**) | This README and the AI-assistance log; commit history; [docs/results/](docs/results/README.md) |

The engineering report and the video script were AI-assisted. Emma supplied her personal reflection and recorded and presented the video herself. Detailed assistance records are in [`ai-documentation/`](ai-documentation/).

### Contribution summary

- Emma Rogoveanu completed Seq Sight individually, using Professor Artzi's Attend-It starter and documented AI assistance. There is no partner or division of work.
- Scope: she chose Variant A (Jena Climate, 72 h input, 24 h forecast) and the customer story.
- Starter: the Attend-It template supplied the three-cloud structure and scaffolding; the Seq Sight data, models, API, UI and documentation replaced its classification parts.
- Data: she applied schema `002`, uploaded the corpus to Supabase, and checked access and split counts.
- Models: an LSTM baseline and a hand-written scaled-dot-product attention model, six 50-epoch runs (three seeds each); Professor Artzi approved the training approach (per Emma).
- Deployment and results: she ran the comparison SQL, registered the runs, deployed Streamlit, Render and Supabase, and checked the deployed flow. She ran the live integration test from her terminal (2 passed, 2026-10-28) and CI is green.
- AI assistance: used for code, tests, workflows, documentation, the engineering report and the video script (records under `ai-documentation/`). The report includes Emma's personal reflection; she recorded and presented the video.
- Still to do by Emma: final review of the report, model card and dataset card; check that the video link opens for viewers who are not signed in and that its duration is 3–5 minutes; confirm the "confidently wrong" wording with the instructor.

## Project tasks: who did what

Detailed assistance records are in the logs under [`ai-documentation/`](ai-documentation/); this table is a summary.

| Task | Professor-provided starter | Emma | AI assistance |
|---|---|---|---|
| Three-cloud structure, starter schema `001_init.sql`, Attend-It code | Provided | Assignment owner; adapted the starter | Helped adapt |
| Variant, dataset, 24 h horizon, scope | | Chose Variant A (Jena Climate) | Helped review requirements |
| Corpus preparation, schema `002`, upload | | Applied schema, uploaded corpus, checked access and split counts | Drafted loader, preprocessing code, schema, docs |
| Models, training workflow, six 50-epoch runs | | Ran training; Professor Artzi approved the approach (per Emma) | Drafted model and training code and training review |
| Registration, API, Streamlit UI, deployment | Template scaffolding | Registered runs, configured and checked the deployment | Drafted API, registration and UI code and deployment docs |
| Comparison table | | Ran the SQL in Supabase | Wrote the query and documented the results |
| Curated examples and failure gallery | | Review pending ("confidently wrong" wording to confirm with the instructor) | Wrote selection script, measured cases, drafted discussion |
| Tests, CI, live integration test, frozen-reference test | | Ran CI and the live integration test | Wrote the tests, workflows and helpers |
| Model card, dataset card, README | | Final review pending | Drafted |
| Engineering report | | Supplied project evidence and her personal reflection; final review pending | AI-assisted drafting |
| Video | | Recorded and presented; link added, access and duration not yet verified | AI-assisted script |

## AI-assistance log

AI assistance helped review requirements and draft code, tests, documentation, the engineering report and the video script, in the sessions recorded under [`ai-documentation/`](ai-documentation/). Emma must still complete her final review of the report and the model and dataset cards before submission.
