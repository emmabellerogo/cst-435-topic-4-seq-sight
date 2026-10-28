# Seq Sight — Temperature Forecasting with Attention Explanations

> **Status: implemented and deployed; submission materials still pending.** The user reports the
> deployed Streamlit → Render → Supabase flow works end to end (forecasts, attention heatmaps,
> prediction history, persisted run metrics); that is manual, user-reported evidence that was not
> independently observed. The automated live Supabase integration test has **not** been run.
> Reports, videos and member details are not yet in the repository.

## Project scope

We chose **Variant A: hourly temperature forecasting** on the **Jena Climate**
dataset, with a **24-hour forecast horizon**. We will compare an LSTM/GRU
baseline against a model with a hand-written scaled-dot-product self-attention
block, and expose attention weights so users can see which past hours influenced
a forecast.

**Customer story**

> As a campus outdoor-events planner, I want a 24-hour temperature forecast and
> a view of which past hours influenced it, so that I can plan events while
> understanding the forecast's limitations.

**Design docs:** [Data design and forecasting contract (proposed)](docs/data-design.md) · [Dataset card (Jena Climate)](docs/dataset-card.md) · [Supabase setup (migration for review)](docs/supabase-setup.md) · [Forecasting preprocessing](docs/forecasting-preprocessing.md) · [Forecasting models](docs/forecasting-models.md) · [Training workflow](docs/forecasting-training.md) · [Training review (50-epoch runs)](docs/training-review.md) · [Forecasting API](docs/forecasting-api.md) · [Streamlit UI](docs/ui.md) · [Render deployment](docs/render-deployment.md) · [Integration test and CI](docs/integration-testing.md) · [Curated examples](docs/forecasting-examples.md) · [Model-comparison SQL results](docs/results/README.md)

## Starter template vs. Seq Sight

| | Existing: Attend-It starter | Planned: Seq Sight |
|---|---|---|
| Origin | Professor-provided template, added in commit `dae6272` | Our implementation for this assignment |
| Task | Synthetic token-sequence classification | Hourly temperature forecasting (Jena Climate, 24 h horizon) |
| Model | LSTM + additive (Bahdanau) attention | Separate LSTM/GRU baseline + hand-written scaled-dot-product self-attention model |
| Status | Starter code; its classification parts are no longer used | Implemented (see checklist for what is verified) |

**Attribution:** the three-cloud structure (Streamlit UI ⇄ FastAPI API ⇄
Supabase) and the starter scaffolding come from the professor-provided
*Attend-It* template for this course. We are adapting that scaffolding; the
Seq Sight data, models, API contract, UI, and documentation are our own work
unless noted otherwise. Starter files not yet replaced may still describe or
implement Attend-It behavior.

## Architecture

```
Streamlit (UI, thin client)  ⇄  FastAPI on Render (models, PyTorch)  ⇄  Supabase (Postgres)
```

The UI holds no ML code; the API is the only writer to Supabase.

## Deployment links

| Tier | Platform | URL |
|------|----------|-----|
| UI | Streamlit Community Cloud | [cst-435-topic-4-seq-sight-pb5jwfwvdotwudcv9fuifb.streamlit.app](https://cst-435-topic-4-seq-sight-pb5jwfwvdotwudcv9fuifb.streamlit.app/) (user-provided; reported working) |
| API | Render | _Pending: URL not recorded in the repository_ |
| Data | Supabase | _Pending: project reference not recorded (and it need not be public)_ |

## Implementation checklist

Ticked only where the repository holds evidence. "Offline" = the local pytest suite (passes locally; GitHub Actions CI is reported green by the user). "User-reported" = manual live checks by the user that were not independently observed. The automated live Supabase integration test (`tests/test_supabase_integration.py`) is **not run**; offline tests and CI do not exercise Supabase.

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
- [x] Six Streamlit tabs: Concepts, Try It, Baseline vs. Attention, Recent Predictions, Failure Gallery, Model Card (offline headless render; deployed app user-reported working)
- [x] Worked numerical attention example explaining √d_k (computed live; offline-tested)
- [x] Matplotlib attention heatmaps for at least three examples (three comparison and three failure examples, measured on validation windows; [details](docs/forecasting-examples.md))
- [ ] Three confidently wrong attention-model forecasts, with discussion. Three large-error cases are documented, but the models show no confidence, so "confidently wrong" is not literally measurable; the wording needs confirming with the instructor ([clarification](docs/forecasting-examples.md))

**Tests**
- [x] Input-length validation
- [x] Attention-matrix shape
- [x] Frozen reference input ([details](docs/frozen-reference.md)): a fixed synthetic 72 h input with stored expected forecasts. The check against the trained seed-0 artifacts passed locally but is skipped in CI (the artifacts are git-ignored); a separate untrained-model fixture runs in CI and is not verification of the trained checkpoints. Not yet seen in GitHub Actions
- [ ] Prediction logging linked to a run (covered offline with an in-memory repository only; the live test is written and **not run**)
- [x] Hand-written attention block (compared against a NumPy reference)
- [x] Shared preprocessing (stored training scaler applied at inference; offline tests)

**Deployment and documentation**
- [x] Deploy Streamlit, Render, and Supabase (user-reported working; Streamlit URL above; Render URL and Supabase reference pending)
- [x] Add CI (`.github/workflows/ci.yml`, offline tests; reported green by the user)
- [x] Handle API cold starts gracefully in the UI (implemented and offline-tested; actual wake time not measured)
- [ ] Complete model card and dataset card (drafted; final review pending)
- [x] Explicit data-persistence/privacy statement ([model card](MODEL_CARD.md))
- [x] Uncertainty and risk disclosures ([model card](MODEL_CARD.md))

**Submission**
- [ ] Each member's individual 3–5 minute video, 750–1000 word engineering report, and contribution evidence linked below

## Model comparison

Final test MAE in °C, from the SQL in [docs/model-comparison.sql](docs/model-comparison.sql). The query was **run by the user** in the Supabase SQL editor and the outputs were supplied to us; they are stored in [docs/results/](docs/results/README.md) (we did not query Supabase). Each model has 3 runs (seeds 0, 1, 2; LSTM run ids 1 to 3, attention run ids 4 to 6), all trained for the same 50 epochs / 52,650 optimizer steps. Data version `jena-mpi-roof-h1-60138aac7d898b8c`.

| Model | Runs | Mean test MAE (°C) | SD across seeds (°C) | Mean validation MAE (°C) | SD across seeds (°C) |
|-------|-----:|-------------------:|---------------------:|-------------------------:|---------------------:|
| LSTM baseline | 3 | 1.8689 | 0.0070 | 1.8387 | 0.0045 |
| Self-attention | 3 | 1.9452 | 0.0122 | 1.9070 | 0.0154 |

The SD is the sample standard deviation (n − 1) of the per-run MAE **across the three training seeds**. It measures how much the trained models vary with the seed. It is not prediction uncertainty, not a confidence interval, and says nothing about how sure a model is about a forecast; with three seeds it is itself imprecise. A persistence baseline (repeat the last temperature) has test MAE 3.4150 °C (local evaluation file, not part of the SQL).

On this test split the LSTM's mean MAE is 0.0763 °C lower than attention's (1.9452 − 1.8689), and every LSTM seed was below every attention seed on validation. Both models are far better than persistence. This is a result for these untuned settings, one station and one training budget; it does not show that attention is a worse architecture in general, and the budget was matched in steps, not in compute. Artzi approved the existing training approach (**user-reported**; no written record is in the repository).

## Member deliverables

| Member | Video | Engineering report | Contribution evidence |
|--------|-------|--------------------|-----------------------|
| _TBD_ | _TBD_ | _TBD_ | _TBD_ |

## AI-assistance log

AI (Claude Code) helped review requirements and draft this documentation. A
human must still review all AI-assisted content for accuracy before submission.
