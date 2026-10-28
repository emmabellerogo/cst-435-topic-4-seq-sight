# Model Card — Seq Sight

> Status: draft for review. Test-set results do not exist yet; nothing below reports a test figure. Facts marked *(validation)* come from the validation split only. The user verified the Supabase schema, access permissions, corpus upload and split counts. Model registration, artifact retrieval, forecast serving and prediction logging have not been tested against live Supabase. The UI was opened locally; deployed services remain untested.

## Model details
- **Task:** forecast hourly air temperature (°C) for the next **24 hours** from the previous **72 hours**.
- **Models (trained separately, same data, scaler, windows and batch order):**
  - **LSTM baseline:** 1-layer LSTM (hidden 32) → final state → two-layer head. 8,152 parameters.
  - **Self-attention:** input projection + sinusoidal positions → one head of hand-written scaled dot-product self-attention, `softmax(QKᵀ/√d_k)V`, with residual + LayerNorm → last-hour and mean pooling → the same head pattern, with different input widths (64 vs. 32). 9,016 parameters.
- **Training:** Adam, lr 1e-3, batch 128, L1 loss on standardized values, gradient clipping 1.0, 50 epochs, checkpoint chosen by validation MAE. Three seeds per model. Untuned initial settings.
- **Framework:** PyTorch, run only in the FastAPI service. The UI contains no model code.
- **Owner:** the Seq Sight team (add names). Adapted from the professor-provided Attend-It template for the three-cloud structure; the data, models and logic are our own.

## Intended use
- A teaching and analysis demonstration of recurrent versus attention-based forecasting, with the attention matrix shown for inspection.
- **Not for:** real event or safety decisions, agriculture, energy trading, any use where a wrong temperature has consequences, or any location other than the training station without separate validation.

## Data
- Hourly means built from the 10-minute air-temperature (`T`) readings of the Max Planck Institute for Biogeochemistry **Beutenberg campus roof station** in Jena, Germany (`mpi_roof`), licensed CC-BY-4.0. See [dataset card](docs/dataset-card.md).
- Chronological split 70 / 15 / 15 %: 134,678 training, 28,762 validation, 28,873 test windows. The scaler (mean 9.39 °C, std 8.27 °C) is fitted on training data only.
- Hours with any missing reading are dropped, never interpolated; windows never span a gap. Station timestamps are used as reported; their time-zone/DST semantics are unverified.

## Metrics
- Metric: mean absolute error in °C over all windows and all 24 hours.
- *(validation)* persistence reference 3.53 °C; LSTM about 1.84 °C and self-attention about 1.91 °C (means of three seeds), so at this training budget the LSTM was lower on every seed. This is a validation result for one period and untuned settings, not a general ranking.
- **Status:** training and validation are complete for the documented 50-epoch experiments (three seeds per model). The **final test evaluation has not been run**.
- **Test MAE: pending.** It is computed once by a separate evaluation script and then stored in the `runs` table; the app's Baseline vs. Attention tab reads it from there.

## Uncertainty
- The models output a single value per hour. **There are no prediction intervals and no calibrated confidence.** Error by forecast horizon has not yet been reported in a reproducible, saved form, so no per-horizon figures are given here; individual errors can be much larger than any average.
- Attention weights are **not** a confidence measure: sharply concentrated attention does not mean a forecast is more likely to be right.

## What the attention matrix is
Entry (i, j) says how much the encoder's representation of input hour *i* draws on input hour *j*. It describes **input-to-input** relationships inside the encoder. It is not a causal explanation and not an attribution of any of the 24 forecast hours to input hours.

## Limitations
- One station, one site type (a roof in a small German city), one climate. **Historical Jena data does not validate forecasts for a local campus or any other place.** Local microclimate, instrument height, season mix and climate trends can all differ.
- Trained and evaluated on station history from 2003-11 to 2026-01, one validation period, untuned hyperparameters, three seeds.
- Forecasts use temperature history only: no pressure, humidity, wind, cloud or numerical weather prediction, so fronts and weather changes are largely unforeseeable.
- Inputs must be real hourly means in °C; the models do not detect wrong units, gaps or sensor faults.

## Persistence and privacy
- Every served forecast is stored in Supabase `predictions`: the **SHA-256 hash of the input** (not the input), the 24 forecast values, the 72×72 attention matrix for the attention model, the model type, the serving run id and a timestamp. No user identifier, IP address or raw input is stored.
- **That table is publicly readable** with the project's anon key, which ships with the app. A hash does not make an input private: it can be recomputed from a guessed input. Do not enter data that must stay confidential.
- **Current state:** model weights are saved only as local files on the development machine. Registering runs and artifacts in Supabase (`runs`, `run_artifacts`) is pending, so run registration and artifact storage have not been tested against live Supabase. (The schema, permissions and corpus upload were verified by the user.)
- Design: weights go in `run_artifacts`, which the anon key cannot read, and the service-role key is used only by the API server.

## Ethical considerations
- Present forecasts as rough guidance with the error figures above, never as a guarantee. State plainly that the evidence comes from a different location and period than any intended use.
