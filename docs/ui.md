# Streamlit UI

Back to [README](../README.md) · See also [forecasting API](forecasting-api.md), [model card](../MODEL_CARD.md).

`ui/app.py` is a thin client over the [forecasting API](forecasting-api.md). The three-cloud structure and the scaffolding were adapted from the professor-provided *Attend-It* template; the tabs and logic are ours. It contains no PyTorch or model code.

## Tabs

| Tab | Source of data |
|---|---|
| Concepts | static text; the scaling example is computed live with NumPy |
| Try It | `POST /predict` (exactly 72 numbers, `lstm` or `attention`); forecast chart, and for attention a 72×72 matplotlib heatmap labelled as input-to-input, not causal or per-output attribution. Sample buttons are labelled synthetic demonstrations |
| Baseline vs. Attention | `runs` via the anon key: validation and test MAE shown in separate columns, plus per-model mean/SD. The three example slots read `ui/content/examples.json` |
| Recent Predictions | `predictions` via the anon key: sort, page, select one row. The attention matrix is fetched only for the selected row, on request |
| Failure Gallery | `ui/content/examples.json`; pending until three measured cases exist. No confidence figures are shown |
| Model Card | renders [MODEL_CARD.md](../MODEL_CARD.md) |

## Behavior worth knowing

* All API calls go through `ForecastApi` in `ui/forecast_client.py`: connect timeout 5 s, read timeout 60 s (15 s for status). 422 is shown with the API's field messages; 503 "no registered model" is explained as pending registration rather than as a failure; timeouts and 502/503/504 without an API body are reported as a possible cold start; connection errors never echo URLs. "Check API status" polls `/healthz` for up to 90 s.
* Until the models are registered and the API has run ids (see [forecasting API](forecasting-api.md)), the Try It tab can only show the pending explanation. Recent Predictions and Baseline vs. Attention stay empty until rows exist.
* The UI refuses a `SUPABASE_ANON_KEY` that is detectably a service-role or secret key. Never put the service-role key in Streamlit secrets.
* `predictions` and `runs` are publicly readable with the anon key by design (see [Supabase setup](supabase-setup.md)).

## Local setup

```bash
.venv312/bin/python -m pip install -r ui/requirements.txt          # pinned
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # repo root; then fill in; git-ignored
.venv312/bin/python -m streamlit run ui/app.py
```

Run these from the repository root: Streamlit reads `.streamlit/secrets.toml` from the directory it is launched in. (The old `ui/.streamlit/secrets.toml.example` is now a keyless stub pointing here.) Secrets (placeholders only in the example file): `API_URL`, `SUPABASE_URL`, `SUPABASE_ANON_KEY`.

Dependencies: `matplotlib==3.9.2` was added. `pyarrow==17.0.0` is a transitive pin: Streamlit 1.38 accepts any pyarrow, and pyarrow 26 does not import under the NumPy 1.26.4 pinned for the API/dev environment (Streamlit's tables then crash). `streamlit run` was not exercised against a browser here; see below.

## Adding the real examples

Edit `ui/content/examples.json` only with measured cases; the schema is in [ui/content/README.md](../ui/content/README.md). Incomplete entries are ignored and reported on the page.

## Tested vs. remaining

**Tested** (`tests/test_ui_helpers.py`, no network): input parsing, request/response schema handling, malformed-response rejection, 422/503/cold-start/timeout/connection handling, health polling bounds, Supabase query shapes (attention never in the list query), anon-key guard, run summary, example validation, scaling-example numbers, figure construction, and a headless Streamlit `AppTest` that renders all six tabs with no secrets.

**Not tested:** a real browser session; any live API or Supabase call; rendering with real data; Streamlit Community Cloud deployment; how long the free-tier API takes to wake up.

**Content still missing:** three measured comparison examples, three measured failure cases with discussion, test MAE (final evaluation pending), 3–5 minute videos and reports. README checklist items are deliberately left unchecked.
