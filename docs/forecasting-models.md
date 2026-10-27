# Forecasting Models

Back to [README](../README.md) · See also [training workflow](forecasting-training.md), [data design](data-design.md) and [forecasting preprocessing](forecasting-preprocessing.md).

Code: [shared/forecast_models.py](../shared/forecast_models.py). Tests:
[tests/test_forecast_models.py](../tests/test_forecast_models.py). The classifier in
`api/training.py` is untouched until its callers are migrated. Nothing here has
been trained on the real corpus.

## Environment

PyTorch work uses a separate Python 3.12 virtual environment, `.venv312/`
(git-ignored), created because the older `.venv` runs Python 3.14, for which
`torch==2.4.1` has no wheel. `.venv` is unchanged.

| Package | Version in `.venv312` |
|---------|-----------------------|
| Python | 3.12.14 |
| torch | 2.4.1 (CPU) |
| numpy | 1.26.4 |
| pytest | 8.3.3 |

Created with `python3.12 -m venv .venv312` and
`.venv312/bin/python -m pip install --only-binary=:all: torch==2.4.1 numpy==1.26.4 pytest==8.3.3`.
Activate with `source .venv312/bin/activate`, or call `.venv312/bin/python` directly:

```bash
.venv312/bin/python -m pytest tests/test_forecast_models.py tests/test_forecasting.py -q
```

`fastapi`, `supabase` and the other tier packages are **not** installed there, so
the API/UI tests do not run in it. The repo's dependency pins were not changed;
the Python version for Render is still to be decided.

## Contract

Both models take float32 `[batch, 72, 1]` (standardized, e.g. `WindowSet.inputs`
via `torch.from_numpy`) and return `(forecast, attention)`:

| Model | `forecast` | `attention` |
|-------|-----------|-------------|
| `LSTMForecaster` (`model_type="lstm"`) | `[batch, 24]` standardized | `None` (no attention weights exist) |
| `AttentionForecaster` (`model_type="attention"`) | `[batch, 24]` standardized | `[batch, 72, 72]`, rows sum to 1 |

Inverse-transform forecasts with the stored `Standardizer` before reporting °C.
Wrong shape or dtype raises `ValueError`. Only the 72 known history hours are
inputs; no target value enters either model, so full (non-causal) attention over
the history is legitimate. `build_model(model_type, **config)` and each model's
`config()` give the values for `runs.hyperparameters`.

## Architectures (initial sizes, local-CPU friendly)

- **Baseline:** `LSTM(1→32, 1 layer)`, final hidden state → head.
- **Attention:** `Linear(1→32)` input projection, fixed sinusoidal positional
  encoding, one hand-written single-head scaled dot-product self-attention block
  (`Q,K,V` linear projections, `QKᵀ/√d_k`, softmax over keys; no
  `nn.MultiheadAttention`, `nn.Transformer` or functional SDPA), residual +
  LayerNorm, then pooling and the head.
- **Head (same layer pattern, different input width):** `Linear(in→64) → ReLU →
  Linear(64→24)`, producing all 24 hours at once (direct multi-horizon; no feeding
  forecasts back in). `in` is 32 for the LSTM (final hidden state) and 64 for
  attention (concatenated pooling), so the first head layer has different
  parameters; the heads are not identical.
- **Pooling:** concatenation of the last hour's contextualized vector and the mean
  over all 72 contextualized vectors (`[batch, 64]`). Each row after attention
  already mixes in information from all 72 hours. The last row is the position the
  forecast directly continues from; the mean is an order-free summary of the
  window. The head maps both to the 24 outputs.
- No dropout or other regularization yet; add it only if validation shows overfitting.

Parameter counts (trainable; positional encoding is a fixed buffer):
**LSTM 8,152; attention 9,016.** These are similar by design, but equal parameter
counts do not mean equal compute: attention is O(T²) per layer and parallel over
time, the LSTM is O(T) and sequential. Compare measured training time.

## What the attention matrix means

`attention[b, i, j]` is how much the contextualized representation of input hour
*i* draws on input hour *j*. It shows **input-to-input** relationships inside the
encoder. It is **not** per-output attribution: pooling collapses the 72 positions
before the head, so none of the 24 forecast hours has its own attention row, and
the matrix does not say which input hours drove forecast hour *t+k*. The UI and
reports should say "where the encoder's attention weights were", per
[data design](data-design.md#attention-interpretation). Row 72 (the last hour) is
the one most directly feeding the last-hour pooling path, but even that is not an
attribution.

## Proposed fair-training controls (not yet implemented)

Identical for both models, recorded in `runs`:

- **Data:** same data version, windows and train-fitted standardizer
  ([preprocessing](forecasting-preprocessing.md)); validation for model selection.
- **Seeds:** the same list of seeds for each model (e.g. 3 seeds), set for Python,
  NumPy and torch; report mean and spread, not the best seed.
- **Batch size, maximum epochs, optimizer, learning rate, loss:** one shared batch
  size and epoch cap; Adam with the same learning rate (or the same small
  validation-selected grid for both); the same loss (L1 on standardized values, to
  match the MAE metric) and gradient-clipping policy.
- **Selection:** best epoch and any hyperparameters are chosen by **validation**
  MAE in °C only. **Test MAE is computed once at the end and never used to select
  a model, epoch, seed or hyperparameter.**
- **Compute reporting:** record parameter count, wall-clock training time per run
  and per epoch, hardware (CPU), and epochs actually run. Training time will be
  reported after real training, not estimated here.

## Checks run

In `.venv312` (Python 3.12.14, torch 2.4.1): 20 synthetic tests pass: shapes,
return contract, finite outputs, input validation, attention row sums and
non-negativity, scaled dot-product against an independent NumPy float64 reference
(including one recomputed from the model's own weights), the √d_k scaling,
positional-encoding formula and order sensitivity, gradient flow to every
parameter and to the input, a tiny synthetic overfit, batch independence in eval
mode, `state_dict` save/load equivalence, and config round-trip. No real data was
used and nothing was trained beyond 60 steps on random tensors.

## Limitations

- Untrained: no accuracy claim of any kind, and no timing measurements yet.
- Single head, single layer, sinusoidal encoding; other designs are not compared.
- Gradient clipping, dropout and the learning-rate choice are undecided.
- Determinism across machines and devices is not guaranteed; only CPU on this
  machine has been exercised.
