# Seq Sight — Temperature Forecasting with Attention Explanations

> **Status: planning / early development.** Seq Sight is not implemented yet.
> No forecasting, training, API, or deployment works today. Everything under
> "Planned" below is a goal, not a result.

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

**Design docs:** [Data design and forecasting contract (proposed)](docs/data-design.md) · [Dataset card (Jena Climate)](docs/dataset-card.md) · [Supabase setup (migration for review)](docs/supabase-setup.md) · [Forecasting preprocessing](docs/forecasting-preprocessing.md) · [Forecasting models](docs/forecasting-models.md) · [Training workflow](docs/forecasting-training.md) · [Training review (50-epoch runs)](docs/training-review.md) · [Forecasting API](docs/forecasting-api.md)

## What exists today vs. what is planned

| | Existing: Attend-It starter | Planned: Seq Sight |
|---|---|---|
| Origin | Professor-provided template, added in commit `dae6272` | Our implementation for this assignment |
| Task | Synthetic token-sequence classification | Hourly temperature forecasting (Jena Climate, 24 h horizon) |
| Model | LSTM + additive (Bahdanau) attention | Separate LSTM/GRU baseline + hand-written scaled-dot-product self-attention model |
| Status | Starter code, not yet adapted | Not started |

**Attribution:** the three-cloud structure (Streamlit UI ⇄ FastAPI API ⇄
Supabase) and the starter scaffolding come from the professor-provided
*Attend-It* template for this course. We are adapting that scaffolding; the
Seq Sight data, models, API contract, UI, and documentation are our own work
unless noted otherwise. Starter files not yet replaced may still describe or
implement Attend-It behavior.

## Planned architecture

```
Streamlit (UI, thin client)  ⇄  FastAPI on Render (models, PyTorch)  ⇄  Supabase (Postgres)
```

The UI holds no ML code; the API is the only writer to Supabase.

## Deployment links (placeholders — not deployed)

| Tier | Platform | URL |
|------|----------|-----|
| UI | Streamlit Community Cloud | _TBD_ |
| API | Render | _TBD_ |
| Data | Supabase | _TBD_ |

## Implementation checklist

Nothing below has been verified complete.

**Data and preprocessing**
- [ ] Load Jena Climate hourly readings into Supabase using `db/load.py`
- [ ] Define reproducible input windows and chronological train/validation/test splits; fit preprocessing on training data only
- [ ] Create `sequences`, `runs`, and `predictions` tables; `predictions` stores input hash, forecast, attention weights, serving run ID, and creation timestamp

**Models and comparison**
- [ ] Implement a separate PyTorch LSTM or GRU baseline
- [ ] Implement a second model with a hand-written scaled-dot-product self-attention block (no `torch.nn.MultiheadAttention`)
- [ ] Compare models with identical preprocessing, matched seeds and compute budgets, and persisted test MAE; generate the README comparison table with SQL

**API**
- [ ] `POST /predict`, `POST /predict_batch`, `GET /predictions`, `GET /runs`, `GET /healthz`, `GET /version`
- [ ] Input-length and dtype validation on prediction endpoints

**UI and explanations**
- [ ] Six Streamlit tabs: Concepts, Try It, Baseline vs. Attention, Recent Predictions, Failure Gallery, Model Card
- [ ] Worked numerical attention example explaining √d_k
- [ ] Matplotlib attention heatmaps for at least three examples
- [ ] Three confidently wrong attention-model forecasts, with discussion

**Tests**
- [ ] Input-length validation
- [ ] Attention-matrix shape
- [ ] Frozen reference input
- [ ] Prediction logging linked to a run
- [ ] Hand-written attention block
- [ ] Shared preprocessing

**Deployment and documentation**
- [ ] Deploy Streamlit, Render, and Supabase
- [ ] Add CI
- [ ] Handle API cold starts gracefully in the UI
- [ ] Complete model card and dataset card
- [ ] Explicit data-persistence/privacy statement
- [ ] Uncertainty and risk disclosures

**Submission**
- [ ] Each member's individual 3–5 minute video, 750–1000 word engineering report, and contribution evidence linked below

## Model comparison

_Not yet available._ This table will be generated from persisted test MAE via SQL.

| Model | Test MAE |
|-------|----------|
| LSTM/GRU baseline | TBD |
| Self-attention | TBD |

## Member deliverables

| Member | Video | Engineering report | Contribution evidence |
|--------|-------|--------------------|-----------------------|
| _TBD_ | _TBD_ | _TBD_ | _TBD_ |

## AI-assistance log

AI (Claude Code) helped review requirements and draft this documentation. A
human must still review all AI-assisted content for accuracy before submission.
