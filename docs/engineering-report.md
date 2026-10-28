> **Status: AI-assisted draft, not yet reviewed.** This report was drafted with AI assistance (Claude Code) from the contributor's project evidence and the contributor's personal worldview reflection. It is **not final** until the contributor has reviewed it for accuracy and confirmed that it expresses their own views. This note is not part of the report body or its word count. See the [README](../README.md).

# CLC -Product 4: “Seq-Sight”: A Sequence-Modeling Service with Attention Explanations: Engineering Analysis and Reflection

**By Emma Rogoveanu**

## Product and Implementation

Seq Sight is an educational temperature-forecasting service that predicts the next 24 hourly temperatures from the previous 72 hours. Its customer story centers on a campus outdoor-events planner who needs a forecast and an understandable view of the model’s behavior. However, the product uses historical readings from Jena, Germany, and is not validated for operational decisions at another location.

I completed this project without a partner, adapting Professor Artzi’s Attend-It starter into a forecasting application. I used AI assistance for implementation, debugging, tests, and documentation, while personally running commands, managing commits, configuring cloud services, executing training, and checking results. The starter supplied the three-cloud structure: Streamlit presents the interface, FastAPI on Render performs inference, and Supabase stores the corpus, model runs, artifacts, and prediction audit trail.

Preparing reliable data was a major engineering task. The pipeline verifies 43 source archives against checksums, collapses identical duplicate readings, excludes off-grid timestamps, and treats missing or invalid readings as gaps. Six valid 10-minute readings produce each hourly mean. No interpolation fills missing hours. The resulting corpus contains 193,548 valid hourly observations.

The data is divided chronologically into training, validation, and test partitions. Each example contains 72 input hours followed by 24 target hours, entirely within one partition and without crossing gaps. Standardization uses only training observations, preventing information from later periods from influencing preprocessing.

## Architecture and Comparative Results

The recurrent baseline is a single-layer LSTM with 32 hidden units. It processes observations sequentially and uses its final hidden state to produce a forecast. The second model projects observations into a 32-dimensional representation, adds sinusoidal positional encoding, and applies a hand-written scaled-dot-product self-attention block. It combines the last contextualized representation with the mean representation before its forecast head. The models share a prediction-head pattern, although their input widths differ.

Both models used Adam, a learning rate of 0.001, batch size 128, standardized L1 loss, gradient clipping, and 50 epochs. Three matched seeds and identical per-epoch batch orders supported comparison. Checkpoints were selected using validation MAE only. Final test evaluation occurred after the checkpoints were fixed.

The first concrete difference concerns forecasting accuracy. Across three seeds, mean test MAE was 1.8689°C for the LSTM and 1.9452°C for attention. Their sample standard deviations were 0.0070°C and 0.0122°C respectively. Both improved on the persistence reference of 3.4150°C, but the LSTM had lower error on every seed.

The LSTM’s sequential state provides an inductive bias for ordered temporal patterns, which may help summarize recurring temperature behavior. Attention can directly relate distant historical hours, but its pooled representations compress those relationships before forecasting. These structural differences offer plausible explanations for the observed accuracy gap; the experiment does not isolate them as proven causes.

The second difference concerns execution. Attention completed the recorded training runs faster than the LSTM. Attention’s parallel matrix operations offer a possible explanation for its shorter recorded training times, while LSTM updates proceed sequentially. However, execution conditions were not controlled, so this experiment cannot establish the cause of the timing difference. The models received matched epochs and optimizer steps, not demonstrated equal computational cost. Professor Artzi approved the approach, but that approval does not establish proven convergence.

## Explainability and Verification

The interface visualizes a 72-by-72 attention matrix for historical inputs. Each row describes how one input representation draws information from other input hours. Because pooling precedes the forecast head, the matrix is not a separate attribution for each predicted hour. It is neither a causal explanation nor a confidence measure.

Three comparison examples and three deliberately selected failure cases make model behavior concrete. The failure cases involve large errors during sharp temperature drops, and the LSTM also performs poorly on those cases. Their selection is disclosed so users do not mistake the gallery for average performance.

Verification includes input validation, attention-shape checks, frozen-reference regression tests, and green offline CI. A separate live Supabase test passed for both architectures, verifying that predictions were stored and linked to the correct runs. Test-created prediction rows were then removed. I also checked the deployed forecast and stored-heatmap flow.

## Risk, Privacy, and Stewardship

A confidently presented but incorrect forecast could affect event attendees and workers, particularly people vulnerable to heat or cold. The product therefore needs clear educational-use warnings, disclosure of its geographical limitations, and guidance to consult authoritative forecasts for actual planning. Future improvements could include validated prediction intervals, unfamiliar-input warnings, and a documented fallback forecast.

Training observations are environmental measurements. User-submitted histories are handled differently: the application stores their hashes, forecasts, attention matrices, serving-run identifiers, and timestamps rather than raw inputs. Prediction records are publicly readable through the intended anonymous-access policy. A hash does not guarantee privacy, so users must avoid confidential submissions.

Proverbs 27:1 teaches, “Do not boast about tomorrow, for you do not know what a day may bring” (ESV). Although we can forecast tomorrow, we are not God and cannot know everything it will bring. My faith rests in the Lord who sustains us, rather than in a model’s apparent certainty.

This belief calls for humility and honest communication. Seq Sight shows evaluation errors and failures, but its single-value forecasts hide the range of plausible outcomes because prediction intervals are absent. Responsible stewardship means acknowledging that limitation. Human knowledge can support planning, but it remains incomplete; the Lord is enduring.
