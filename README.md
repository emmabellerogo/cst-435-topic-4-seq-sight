# CLC -Product 4: “Seq-Sight”: A Sequence-Modeling Service with Attention Explanations

_Short name used in this repository: **Seq Sight** (hourly temperature forecasting with attention explanations)._

> **Status: implemented and deployed; individual submission materials still pending.** This is a
> solo project. The user reports (not independently observed by Claude) that GitHub Actions CI is
> green, that the deployed Streamlit → Render → Supabase flow works (forecasts, attention heatmaps,
> stored-attention drilldown, prediction history, persisted run metrics), and that the opt-in live
> Supabase integration test passed on 2026-10-28 (2 passed in 5.36 s, run from a local terminal, not
> GitHub Actions; [details](docs/integration-testing.md#live-run-record-user-executed)). An
> AI-assisted draft of the engineering report is in
> [docs/engineering-report.md](docs/engineering-report.md) and needs the contributor's review. The
> video link is not yet in the repository.

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

**Design docs:** [Data design and forecasting contract (proposed)](docs/data-design.md) · [Dataset card (Jena Climate)](docs/dataset-card.md) · [Supabase setup (migration for review)](docs/supabase-setup.md) · [Forecasting preprocessing](docs/forecasting-preprocessing.md) · [Forecasting models](docs/forecasting-models.md) · [Training workflow](docs/forecasting-training.md) · [Training review (50-epoch runs)](docs/training-review.md) · [Forecasting API](docs/forecasting-api.md) · [Streamlit UI](docs/ui.md) · [Render deployment](docs/render-deployment.md) · [Integration test and CI](docs/integration-testing.md) · [Curated examples](docs/forecasting-examples.md) · [Model-comparison SQL results](docs/results/README.md)

## Starter template vs. Seq Sight

| | Existing: Attend-It starter | Planned: Seq Sight |
|---|---|---|
| Origin | Professor-provided template, added in commit `dae6272` | The contributor's assignment work, with documented AI assistance (see below) |
| Task | Synthetic token-sequence classification | Hourly temperature forecasting (Jena Climate, 24 h horizon) |
| Model | LSTM + additive (Bahdanau) attention | Separate LSTM/GRU baseline + hand-written scaled-dot-product self-attention model |
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

The UI holds no ML code; the API is the only writer to Supabase.

## Deployment links

| Tier | Platform | URL |
|------|----------|-----|
| UI | Streamlit Community Cloud | [cst-435-topic-4-seq-sight-pb5jwfwvdotwudcv9fuifb.streamlit.app](https://cst-435-topic-4-seq-sight-pb5jwfwvdotwudcv9fuifb.streamlit.app/) (user-provided; reported working) |
| API | Render | [https://seq-sight-api.onrender.com](https://seq-sight-api.onrender.com) (user-confirmed) |
| Data | Supabase | Project URL `https://vrcdwzypcqkpxetunnxm.supabase.co`, project reference `vrcdwzypcqkpxetunnxm` (user-confirmed; no keys recorded) |

## Implementation checklist

Ticked only where the repository or the user's reported results support it. "Offline" = the pytest suite run locally (CI is reported green by the user; CI never contacts Supabase and skips the live test). "User-reported" = checks the user ran themselves and reported, not independently observed by Claude. The live Supabase integration test (`tests/test_supabase_integration.py`) is a **separate, user-executed** run on 2026-10-28: 2 passed in 5.36 s (see [integration-testing.md](docs/integration-testing.md#live-run-record-user-executed)). Optional checks that were not reported (for example a browser walkthrough by Claude, the manual GitHub `integration.yml` job, which was not used for the live run, and load or cold-start timing) are not claimed.

**Data and preprocessing**
- [x] Load Jena Climate hourly readings into Supabase using `db/load.py` (corpus upload verified by the user; training read the rows from Supabase per the artifacts' recorded source)
- [x] Define reproducible input windows and chronological train/validation/test splits; fit preprocessing on training data only (offline tests; [preprocessing](docs/forecasting-preprocessing.md))
- [x] Create `sequences`, `runs`, and `predictions` tables; `predictions` stores input hash, forecast, attention weights, serving run ID, and creation timestamp (schema applied and prediction history working: user-reported; no automated live check)

**Models and comparison**
- [x] Implement a separate PyTorch LSTM or GRU baseline (LSTM)
- [x] Implement a second model with a hand-written scaled-dot-product self-attention block (no `torch.nn.MultiheadAttention`)
- [x] Compare models with identical preprocessing, matched seeds and compute budgets, and persisted test MAE; generate the README comparison table with SQL. *Qualifications:* the budget is matched in epochs and optimizer steps (50 epochs = 52,650 steps per run, same windows, same batch order), **not** measured or equalized compute: the two architectures do different amounts of work per step, and wall-clock time was recorded but not controlled ([training review §6](docs/training-review.md)). Convergence is not claimed (continued, decelerating improvement was observed). Artzi's approval of this training approach is user-reported. The table below comes from SQL the user ran.

**API**
- [x] `POST /predict`, `POST /predict_batch`, `GET /predictions`, `GET /runs`, `GET /healthz`, `GET /version` (offline-tested; `/predict`, history and run metrics user-reported working when deployed)
- [x] Input-length and dtype validation on prediction endpoints (offline tests)

**UI and explanations**
- [x] Six Streamlit tabs: Concepts, Try It, Baseline vs. Attention, Recent Predictions, Failure Gallery, Model Card (offline headless render; deployed app, including stored-attention drilldown, user-reported working)
- [x] Worked numerical attention example explaining √d_k (computed live; offline-tested)
- [x] Matplotlib attention heatmaps for at least three examples (three comparison and three failure examples, measured on validation windows; [details](docs/forecasting-examples.md))
- [ ] Three confidently wrong attention-model forecasts, with discussion. Three large-error cases are documented, but the models show no confidence, so "confidently wrong" is not literally measurable; the wording needs confirming with the instructor ([clarification](docs/forecasting-examples.md))

**Tests**
- [x] Input-length validation
- [x] Attention-matrix shape
- [x] Frozen reference input ([details](docs/frozen-reference.md)): a fixed synthetic 72 h input with stored expected forecasts. The check against the trained seed-0 artifacts passed locally but is skipped in CI (the artifacts are git-ignored); a separate untrained-model fixture runs in CI and is not verification of the trained checkpoints. Not yet seen in GitHub Actions
- [x] Prediction logging linked to a run (offline with an in-memory repository, and the live integration test against Supabase, user-executed 2026-10-28: both model types verified against the stored `predictions` row and referenced `runs` row, test rows cleaned up)
- [x] Hand-written attention block (compared against a NumPy reference)
- [x] Shared preprocessing (stored training scaler applied at inference; offline tests)

**Deployment and documentation**
- [x] Deploy Streamlit, Render, and Supabase (user-reported working; Streamlit, Render and Supabase identifiers in the table above, all user-confirmed)
- [x] Add CI (`.github/workflows/ci.yml`, offline tests; reported green by the user)
- [x] Handle API cold starts gracefully in the UI (implemented and offline-tested; actual wake time not measured)
- [ ] Complete model card and dataset card (drafted; final review pending)
- [x] Explicit data-persistence/privacy statement ([model card](MODEL_CARD.md))
- [x] Uncertainty and risk disclosures ([model card](MODEL_CARD.md))

**Submission**
- [ ] The contributor's individual 3–5 minute video and 750–1000 word engineering report, with the links filled in below (report: AI-assisted draft saved, **awaiting the contributor's review**; video: pending)

## Model comparison

Final test MAE in °C, from the SQL in [docs/model-comparison.sql](docs/model-comparison.sql). The query was **run by the user** in the Supabase SQL editor and the outputs were supplied in chat; they are stored in [docs/results/](docs/results/README.md) (Claude did not query Supabase). Each model has 3 runs (seeds 0, 1, 2; LSTM run ids 1 to 3, attention run ids 4 to 6), all trained for the same 50 epochs / 52,650 optimizer steps. Data version `jena-mpi-roof-h1-60138aac7d898b8c`.

| Model | Runs | Mean test MAE (°C) | SD across seeds (°C) | Mean validation MAE (°C) | SD across seeds (°C) |
|-------|-----:|-------------------:|---------------------:|-------------------------:|---------------------:|
| LSTM baseline | 3 | 1.8689 | 0.0070 | 1.8387 | 0.0045 |
| Self-attention | 3 | 1.9452 | 0.0122 | 1.9070 | 0.0154 |

The SD is the sample standard deviation (n − 1) of the per-run MAE **across the three training seeds**. It measures how much the trained models vary with the seed. It is not prediction uncertainty, not a confidence interval, and says nothing about how sure a model is about a forecast; with three seeds it is itself imprecise. A persistence baseline (repeat the last temperature) has test MAE 3.4150 °C (local evaluation file, not part of the SQL).

On this test split the LSTM's mean MAE is 0.0763 °C lower than attention's (1.9452 − 1.8689), and every LSTM seed was below every attention seed on validation. Both models are far better than persistence. This is a result for these untuned settings, one station and one training budget; it does not show that attention is a worse architecture in general, and the budget was matched in steps, not in compute. Artzi approved the existing training approach (**user-reported**; no written record is in the repository).

## Contributor and individual deliverables

This project was completed by one person, Emma Rogoveanu, working alone over about a week (user-stated). There is no partner, team or division of work to describe.

| Contributor | Video (3–5 min) | Engineering report (750–1000 words) | Contribution evidence |
|-------------|-----------------|--------------------------------------|-----------------------|
| Emma Rogoveanu | _Link pending_ | [docs/engineering-report.md](docs/engineering-report.md) (**AI-assisted draft, awaiting the contributor's review**) | This README and the AI-assistance log; commit history; [docs/results/](docs/results/README.md) |

The engineering report is an **AI-assisted draft**: Claude Code drafted it from the contributor's project evidence and the contributor's own personal worldview reflection, and the contributor must review it for accuracy and confirm that it expresses their views before it is submitted as their own. No video exists yet; none has been drafted here.

## Project tasks: who did what

Only what the repository or the user's reports record is stated. Anything not listed is not claimed.

| Task | Professor-provided starter | Contributor | AI assistance (Claude Code) |
|---|---|---|---|
| Three-cloud structure (Streamlit, FastAPI, Supabase), starter schema `001_init.sql`, Attend-It classification code | Provided | Assignment owner; adaptation details not itemized here | Helped adapt |
| Choosing the variant, dataset, 24 h horizon and project scope | | Scope as recorded in this README (Variant A, Jena Climate) | Helped review requirements |
| Corpus preparation, schema `002`, upload to Supabase | | Reports applying the schema and uploading the corpus, and verified access and split counts (user-reported) | Drafted loader, preprocessing code, schema and documentation |
| LSTM and hand-written attention models, training workflow, six 50-epoch runs | | Reports that Artzi approved the training approach; the six runs and the one-time test evaluation exist as local artifacts | Drafted model and training code and the training review |
| Registration, API, Streamlit UI, deployment | Template scaffolding | Reports registered runs and a working Streamlit → Render → Supabase deployment (user-reported) | Drafted API, registration and UI code and deployment docs |
| Comparison table from persisted runs | | Executed the SQL in Supabase and supplied the outputs | Wrote the query and documented the results |
| Curated examples and failure gallery | | Pending review (the "confidently wrong" wording is still to be confirmed with the instructor) | Wrote the selection script, measured the cases, drafted the discussion |
| Tests, CI, live integration test, frozen-reference test | | Reports CI green and ran the live integration test from a local terminal (2 passed, 2026-10-28; user-reported) | Wrote the tests, workflows and helpers |
| Model card, dataset card, README | | To review and finalize before submission | Drafted |
| Individual engineering report | | Supplied the project evidence and the personal reflection; **review and approval pending** | Drafted the report from that material |
| Individual video | | **Contributor's own; pending** | None |

The exact split of authorship within the AI-assisted rows is for the contributor to state in their report; it is not asserted here.

## AI-assistance log

AI (Claude Code) helped review requirements and draft code, tests and documentation in the sessions recorded under [`ai-documentation/`](ai-documentation/). A human must still review all AI-assisted content for accuracy before submission. AI assisted in drafting the engineering report (see [docs/engineering-report.md](docs/engineering-report.md)), which is not final until the contributor reviews it; it has not drafted the video.
