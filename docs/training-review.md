# Training Review: Six 50-Epoch Matched-Step Runs

Back to [README](../README.md) · See also [training workflow](forecasting-training.md), [models](forecasting-models.md), [preprocessing](forecasting-preprocessing.md).

Reviewed 2026-10-25; **revision 2**. Analysis only: nothing was retrained, the test split was not evaluated, no credentials were read, no Supabase connection was made, and no model or training code was changed. Section 8 lists options for further work; **none has been run**.

> **Update 2026-10-28 (later events; the analysis below is unchanged).** The one-time test evaluation of these six artifacts was subsequently run and saved to the git-ignored local file `artifacts/epoch50/final/test_eval.json`; that answers the open question in *Scope* below. The user reports that runs and artifacts were then registered in Supabase and the deployed app works; this was not independently observed. Seed-0 LSTM (run 1) and seed-0 attention (run 4) are the ones used for the curated examples ([forecasting-examples.md](forecasting-examples.md)).

## What changed from revision 1

Revision 1 called these runs a "pilot", said "none has converged", recommended rerunning everything with a new protocol, and attributed timing differences to machine load. Those points are withdrawn or reworded:

- The six runs are **valid matched-epoch, matched-step experiments** and are kept as they are. They were run with the documented initial settings and were not tuned, so nothing about them was selected on validation or test results. Nothing here requires discarding or repeating them.
- "No model has converged" was too categorical. The evidence shows continued, decelerating improvement; whether that meets a convergence criterion depends on a criterion nobody has defined (section 4).
- The cause of the timing differences is not known and is not asserted (section 6).
- The earlier 150-epoch protocol, stopping rule and tuning grid were our own methodological preferences, not requirements (section 1). They are now only one option among several.

## Scope

- Subject: `artifacts/epoch50/final/{lstm,attention}_seed{0,1,2}.pt`.
- **Every error figure is a validation result in °C. None is a test result.** The artifacts contain no test metrics. Whether `--evaluate-test` was ever run cannot be told from them (the CLI prints test results and does not save them); please confirm that no test number has been viewed, or disclose it.
- Each artifact records `source: supabase`, so the training rows were read from Supabase. That is not the same as persisting runs: no repository code writes forecasting `runs` or `predictions` rows, the artifacts are git-ignored local files, and I have no evidence that any run or artifact is stored in Supabase.
- Method: artifacts were loaded with `torch.load(weights_only=True)` in `.venv312` (torch 2.4.1). Checks that ran models used locally prepared data (`db.load.prepare()`, no network) and the validation split only. Scratch scripts were kept outside the repository.

## 1. Assignment requirements versus our own preferences

I cannot see the original assignment or rubric, only what the repository and earlier sessions recorded. Please check the rubric wording itself.

| Item | Source | Status |
|---|---|---|
| Models compared with identical preprocessing, **matched seeds and compute budgets**, and persisted test MAE; README table generated with SQL | [README checklist](../README.md), transcribed from the assignment | **Requirement** |
| `runs` stores compute-budget information; test MAE persisted | README and [Supabase setup](supabase-setup.md) ("README requirement") | **Requirement** that the information is stored. Storing it as a JSON object is our choice |
| Training reads corpus rows from Supabase | task statement | **Requirement**; the six runs satisfy it (`source: supabase`) |
| "Train to convergence under the same compute budget" | your task description to me; the word "convergence" does not appear in the repository docs | **Possibly a requirement**; check the rubric text |
| What "convergence" means (a threshold, a patience, a plateau) | not defined anywhere | **Undefined** |
| What unit the "compute budget" is in (epochs, steps, arithmetic, seconds) | not defined anywhere | **Undefined** |
| Epoch cap 50 / 150, 5-epoch smoothing, patience 20, 0.005 °C threshold, tuning grid, learning-rate decay, interleaving the models, idle-machine runs | revisions 1 and 2 of this review | **Our preferences**, not requirements |
| Adam, lr 0.001, batch 128, clip 1.0, L1 loss, no dropout, validation-based selection | [training workflow](forecasting-training.md) | **Our design choices** |

A budget defined as "50 epochs = 52,650 optimizer steps at batch size 128 on the same windows in the same order" is a coherent, documentable definition of a matched budget. Whether the rubric accepts it, and whether "convergence" can be satisfied by showing deceleration instead of a criterion, is not something I can determine.

## 2. Artifact verification

All six artifacts load, use format 1 and torch 2.4.1, and agree on everything that should be shared:

| Field | Value (identical across all six unless noted) | Check |
|---|---|---|
| Data version | `jena-mpi-roof-h1-60138aac7d898b8c` | matches the uploaded and verified version and the local `db.load.prepare()` output |
| Scaler | mean 9.389809174677712, std 8.26676825284948, `std_ddof` 0, °C | equals the scaler refit from the local data, exactly |
| Training config | Adam, lr 0.001, batch 128, clip norm 1.0, L1 on standardized values, dropout 0.0, `max_epochs` 50 | matches the documented initial choices |
| Training windows | 134,678 | matches the verified window count; no truncation |
| Run kind / source | `final` / `supabase` | |
| Seeds | `seed` and `train_config.seed` agree with the filename in every file | |
| Model config | LSTM: hidden 32, head 64. Attention: `d_model` 32, head 64 | 8,152 and 9,016 parameters |
| Epochs and steps | 50 history rows; 52,650 steps = 50 × 1,053 | 1,053 steps per epoch (134,678 / 128, last batch of 22 included); cumulative steps are exactly 1,053 × epoch in every row |
| Best epoch | saved `best_epoch` equals the argmin of the history in all six; saved best MAE equals that minimum | consistent |
| Elapsed time | `train_seconds` equals the final `elapsed_seconds` and the sum of per-epoch train + validation seconds | validation time is **included** |

Checks beyond metadata:

- **The saved weights are the best-validation weights.** Re-scoring each checkpoint on the validation split (28,762 windows) reproduced the saved best MAE with a difference of 0 in all six cases.
- **Training is exactly reproducible on this machine.** The earlier 20-epoch run (`artifacts/final/`) has validation MAE and batch-order hashes bit-identical to the first 20 epochs of the matching 50-epoch run, and the 1-epoch smoke run matches epoch 1, for both models (seed 0). There is no schedule and the batch order is a stateless function of `(seed, epoch)`, so a longer run reproduces a shorter one exactly.

**Not recorded:** hardware, CPU thread count, OS, wall-clock start time, git commit (only the torch version is stored); optimizer state; RNG state; weights from the **last** epoch (only the best-validation weights are stored); the persistence reference and per-horizon errors (printed, not saved); the validation window count. No contradictions were found.

## 3. Observed validation results and ranking

| Model | Seed | Best val MAE (°C) | Best epoch | Elapsed (m:ss) | Median train s/epoch | Parameters |
|---|--:|--:|--:|--:|--:|--:|
| lstm | 0 | 1.8335 | 45 | 18:44 | 19.3 | 8,152 |
| lstm | 1 | 1.8420 | 46 | 16:24 | 17.8 | 8,152 |
| lstm | 2 | 1.8407 | 49 | 16:44 | 10.9 | 8,152 |
| attention | 0 | 1.9158 | 50 | 9:08 | 8.2 | 9,016 |
| attention | 1 | 1.9159 | 49 | 8:19 | 7.8 | 9,016 |
| attention | 2 | 1.8892 | 50 | 3:47 | 4.2 | 9,016 |

| Model | Mean best val MAE (°C) | Sample SD (n=3) | Min–max | Best epochs | Mean elapsed (m:ss) | SD elapsed (s) |
|---|--:|--:|--:|--|--:|--:|
| lstm | 1.8387 | 0.0045 | 1.8335–1.8420 | 45, 46, 49 | 17:17 | 76 |
| attention | 1.9070 | 0.0154 | 1.8892–1.9159 | 50, 49, 50 | 7:05 | 173 |

Per-seed gap (attention − LSTM): +0.0823, +0.0739, +0.0485; mean +0.0683, SD 0.0176.

All seeds are included for both models. Context: a persistence forecast (repeat the last observed temperature for 24 hours) has validation MAE **3.526 °C**, so both models learned real structure. By forecast hour the LSTM's validation MAE rises from about 0.42 °C at +1 h to about 2.40 °C at +24 h; attention's from 0.46–0.48 °C to 2.42–2.45 °C. The gap between the models is concentrated at short horizons and nearly vanishes at +24 h.

**Observed ranking at the 50-epoch, 52,650-step budget:** the LSTM has the lower validation MAE on every seed, and every LSTM seed (1.8335–1.8420 °C) is below every attention seed (1.8892–1.9159 °C). The mean gap is 0.068 °C (per-seed 0.049 to 0.082).

**Limitations of that ranking:**

- It is a ranking *at this budget*. Neither model was shown to have stopped improving (section 4), and the attention model is still improving faster, so it says nothing certain about the ranking at a larger budget.
- Three seeds per model. Complete separation of 3 versus 3 is the strongest result a rank test can give, and its exact one-sided probability under "no difference" is 1/20 = 0.05 (two-sided 0.10): suggestive, not strong. Seeds vary initialization and batch order only, not the data, the period or the architecture details.
- One validation period (2019-05 to 2022-09), one set of hyperparameters (untuned for either model), one attention design (single head, last+mean pooling) and one LSTM size. Differences may reflect these choices as much as the architectures. No diagnosis of why was attempted.
- Each checkpoint is the minimum of 50 noisy epoch values, which flatters noisier curves slightly (attention's are noisier, so if anything this favours attention).
- The gap (0.068 °C) is about the same size as epoch-to-epoch fluctuation within a single run (0.016–0.036 °C range over the last ten epochs) and larger than the LSTM seed SD (0.0045 °C).
- Validation results only; the test ranking may differ.

## 4. Continued improvement versus "converged"

Validation MAE (°C) at selected epochs:

| Epoch | lstm s0 | lstm s1 | lstm s2 | atte s0 | atte s1 | atte s2 |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 2.1076 | 2.0865 | 2.1372 | 2.2451 | 2.2140 | 2.2540 |
| 5 | 1.9629 | 1.9788 | 1.9610 | 2.0620 | 2.0747 | 2.1329 |
| 10 | 1.9206 | 1.9278 | 1.9071 | 2.0420 | 2.0702 | 2.0027 |
| 15 | 1.8959 | 1.8986 | 1.8848 | 1.9818 | 1.9958 | 1.9580 |
| 20 | 1.8684 | 1.8770 | 1.8838 | 1.9634 | 1.9725 | 1.9397 |
| 25 | 1.8571 | 1.8643 | 1.8665 | 1.9617 | 1.9460 | 1.9357 |
| 30 | 1.8571 | 1.8625 | 1.8768 | 1.9519 | 1.9369 | 1.9152 |
| 35 | 1.8362 | 1.8530 | 1.8580 | 1.9370 | 1.9456 | 1.9116 |
| 40 | 1.8470 | 1.8598 | 1.8518 | 1.9292 | 1.9421 | 1.9051 |
| 45 | 1.8335 | 1.8449 | 1.8456 | 1.9271 | 1.9467 | 1.9121 |
| 50 | 1.8345 | 1.8447 | 1.8476 | 1.9158 | 1.9213 | 1.8892 |

Epochs 41–50 in full (bold = the saved best epoch):

| Epoch | lstm s0 | lstm s1 | lstm s2 | atte s0 | atte s1 | atte s2 |
|--:|--:|--:|--:|--:|--:|--:|
| 41 | 1.8342 | 1.8640 | 1.8583 | 1.9415 | 1.9248 | 1.9064 |
| 42 | 1.8358 | 1.8554 | 1.8451 | 1.9301 | 1.9203 | 1.9024 |
| 43 | 1.8389 | 1.8578 | 1.8492 | 1.9302 | 1.9208 | 1.9007 |
| 44 | 1.8494 | 1.8627 | 1.8464 | 1.9405 | 1.9371 | 1.9253 |
| 45 | **1.8335** | 1.8449 | 1.8456 | 1.9271 | 1.9467 | 1.9121 |
| 46 | 1.8348 | **1.8420** | 1.8569 | 1.9249 | 1.9461 | 1.9011 |
| 47 | 1.8486 | 1.8457 | 1.8487 | 1.9249 | 1.9220 | 1.9244 |
| 48 | 1.8376 | 1.8441 | 1.8526 | 1.9247 | 1.9186 | 1.8939 |
| 49 | 1.8387 | 1.8507 | **1.8407** | 1.9314 | **1.9159** | 1.8894 |
| 50 | 1.8345 | 1.8447 | 1.8476 | **1.9158** | 1.9213 | **1.8892** |

| Run | Epochs 41–50 range (°C) | SD | Rises (of 10 steps) | Largest 1-epoch rise | Mean 41–45 | Mean 46–50 | Train L1 → train MAE ≈ °C (ep 50) |
|---|--:|--:|--:|--:|--:|--:|--:|
| lstm s0 | 0.0158 | 0.0058 | 6 | +0.0138 | 1.8384 | 1.8388 | 1.760 |
| lstm s1 | 0.0220 | 0.0082 | 5 | +0.0066 | 1.8569 | 1.8454 | 1.758 |
| lstm s2 | 0.0176 | 0.0054 | 5 | +0.0112 | 1.8489 | 1.8493 | 1.758 |
| attention s0 | 0.0257 | 0.0076 | 5 | +0.0123 | 1.9339 | 1.9243 | 1.882 |
| attention s1 | 0.0308 | 0.0115 | 4 | +0.0163 | 1.9299 | 1.9248 | 1.878 |
| attention s2 | 0.0360 | 0.0129 | 3 | +0.0246 | 1.9094 | 1.8996 | 1.864 |

(The last column is the training loss averaged over the epoch, converted to °C. It is computed while the weights change, so it is a slightly pessimistic estimate of training error, not a separate evaluation.)

Gain per 10-epoch block (mean validation MAE of the block):

| Run | Mean ep 1–10 | 11–20 | 21–30 | 31–40 | 41–50 | Drop 31–40 → 41–50 | Drop 21–30 → 31–40 | Ratio (last/prev) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| lstm s0 | 1.9780 | 1.8906 | 1.8684 | 1.8434 | 1.8386 | 0.0048 | 0.0250 | 0.19 |
| lstm s1 | 1.9786 | 1.8953 | 1.8696 | 1.8547 | 1.8512 | 0.0035 | 0.0149 | 0.23 |
| lstm s2 | 1.9773 | 1.8905 | 1.8747 | 1.8584 | 1.8491 | 0.0093 | 0.0164 | 0.57 |
| attention s0 | 2.0960 | 1.9863 | 1.9561 | 1.9400 | 1.9291 | 0.0109 | 0.0161 | 0.67 |
| attention s1 | 2.0928 | 1.9929 | 1.9542 | 1.9480 | 1.9274 | 0.0206 | 0.0062 | 3.33 |
| attention s2 | 2.1142 | 1.9657 | 1.9345 | 1.9161 | 1.9045 | 0.0116 | 0.0185 | 0.63 |

**What the evidence supports**

- **Continued improvement is observed in all six runs.** The best 5-epoch-average value fell by 0.020 to 0.029 °C (about 1.1–1.5%) between epochs 1–30 and 31–50, against smoothing noise of 0.0016–0.0045 °C. Training loss was still falling at epoch 50.
- **The improvement is decelerating, unevenly.** For the LSTM the last 10-epoch block gained 0.0035–0.0093 °C (0.19–0.50% of the error), 0.19–0.57 times the previous block's gain. For attention it gained 0.011–0.021 °C (0.57–1.07%), and in one run (seed 1) the gain was larger than the previous block's, which is within the noise of these curves.
- **Best epochs:** LSTM 45, 46, 49; attention 50, 49, 50. A best epoch at or next to the cap means the cap, not a rise in validation error, ended the run.
- **No sign of overfitting.** Validation did not rise in any run, and training error (about 1.76 °C LSTM, 1.86–1.88 °C attention) is close to validation error. Why the models plateau at these levels was not diagnosed; capacity, optimization and the train/validation period shift are all untested explanations.

**What it does not support.** It does not establish that any model has converged, and it does not establish that none has. "Converged" has no definition in the assignment as recorded. Under a lenient criterion such as "less than 0.5% change over the last ten epochs", the LSTM runs would qualify (0.19–0.50%) and the attention runs would not. Under a strict one, none qualifies; our own candidate rule (best 5-epoch average improved by less than 0.005 °C over 20 epochs) is not met by any run, with improvements of 0.020–0.029 °C. That rule was chosen by me and checked against these same histories, so it is a description, not a validated criterion. A plateau being approached is not proof of convergence, and a still-falling curve is not proof that further training would change a conclusion.

Illustration of what is at stake for the ranking: attention's last-block gain (0.011–0.021 °C) is 16–30% of the current mean gap. If such gains continued for several more blocks they would narrow the gap; whether they would close it is unknown, and the observed deceleration points the other way.

## 5. Matched batch order, and what was matched

For every seed the per-epoch SHA-256 of the batch index sequence is **identical between the LSTM and attention run in all 50 epochs**, and the 50 hashes within a run are all distinct. I recomputed them from `epoch_permutation(134678, seed, epoch)` and they equal the stored values.

| Matched between the architectures (verified) | Not matched |
|---|---|
| Data version, scaler, training and validation windows | Architecture, parameter count (8,152 vs 9,016), arithmetic per step |
| Seed value, batch order of every epoch, steps per epoch (1,053) and in total (52,650) | Weight initialization (same seed, different layers, so different draws) |
| Optimizer, learning rate, batch size, clipping, loss, dropout, epoch cap, selection rule | Wall-clock time and whatever conditions the machine was under (section 6) |

## 6. Compute: what equal steps do and do not establish

Equal epochs and equal optimizer steps establish a **matched optimization budget**: the same number of parameter updates on the same data in the same order. They do **not** establish equal compute.

- **Arithmetic differs.** From the architecture sizes (my approximate count, ignoring layer norm, softmax and elementwise work): about 0.31 M multiply-accumulates per window forward for the LSTM and about 0.56 M for attention, so roughly 1.8× more for attention. Parameter counts (within 11%) say little about either.
- **Measured time differs, and is not a controlled benchmark.** The runs below produced bit-identical results to their longer counterparts yet took different times:

| Run (seed 0) | Median train seconds per epoch |
|---|--:|
| LSTM, 20-epoch run | 28.8 |
| LSTM, 50-epoch run | 19.3 |
| Attention, 20-epoch run | 10.5 |
| Attention, 50-epoch run | 8.2 |

  Across the 50-epoch runs the median epoch time was 10.9–19.3 s (LSTM) and 4.2–8.2 s (attention). Within LSTM seed 2, the first five epochs took 32, 29, 25, 44 and 85 s and later epochs about 11 s. The runs were executed one after another, not interleaved. **The cause of these differences is not recorded in the artifacts and was not determined;** I make no claim about it. Because it is unknown, elapsed times cannot be read as a property of the architectures.
- **What can be said about time.** In all four same-seed pairs the LSTM epoch was slower than the attention epoch (by 2.3–2.7×). That is consistent in direction, but each pair ran at a different time, so the factor is an indication, not a measurement. Mean elapsed time was 17:17 (LSTM) and 7:05 (attention), including per-epoch validation.
- Therefore the honest description of the budget is "50 epochs / 52,650 steps each, with measured wall-clock reported separately and not controlled".

## 7. Retrospective comparison under a shared elapsed-time cutoff

**Is it possible?** Partly, for validation metrics only. Every history row stores the cumulative `elapsed_seconds` and the validation MAE, so for any cutoff T one can read each run's best validation MAE among the epochs finished by T. This needs no weights.

Best validation MAE (°C) among epochs finished by a cutoff (epochs completed in brackets):

| Cutoff (s elapsed) | lstm s0 | lstm s1 | lstm s2 | atte s0 | atte s1 | atte s2 |
|--:|--:|--:|--:|--:|--:|--:|
| 100 | 1.9680 (ep 4) | 1.9788 (ep 5) | 2.0092 (ep 3) | 2.0000 (ep 11) | 2.0185 (ep 11) | 1.9374 (ep 22) |
| 200 | 1.9215 (ep 8) | 1.9278 (ep 10) | 1.9695 (ep 4) | 1.9519 (ep 22) | 1.9529 (ep 23) | 1.9007 (ep 44) |
| 300 | 1.9084 (ep 11) | 1.8846 (ep 15) | 1.9610 (ep 5) | 1.9375 (ep 30) | 1.9365 (ep 35) | 1.8892 (ep 50, finished) |
| 400 | 1.8888 (ep 15) | 1.8770 (ep 20) | 1.9282 (ep 8) | 1.9275 (ep 36) | 1.9248 (ep 41) | 1.8892 (ep 50, finished) |
| 500 | 1.8777 (ep 19) | 1.8643 (ep 25) | 1.9071 (ep 10) | 1.9275 (ep 44) | 1.9159 (ep 50, finished) | 1.8892 (ep 50, finished) |
| 600 | 1.8652 (ep 24) | 1.8618 (ep 30) | 1.8918 (ep 14) | 1.9158 (ep 50, finished) | 1.9159 (ep 50, finished) | 1.8892 (ep 50, finished) |

Pairing each seed by the attention run's total time (so attention is complete and the LSTM is cut off):

| Seed | Cutoff = attention run's total time (s) | LSTM epochs done | LSTM best val so far | Attention epochs | Attention best val | Attention − LSTM |
|--:|--:|--:|--:|--:|--:|--:|
| 0 | 548 | 21 | 1.8684 | 50 | 1.9158 | +0.0475 |
| 1 | 499 | 25 | 1.8643 | 50 | 1.9159 | +0.0517 |
| 2 | 227 | 4 | 1.9695 | 50 | 1.8892 | -0.0803 |

Mean +0.0063.

**What this shows, and its limits**

- **The result depends on which timings you use.** For seeds 0 and 1 the LSTM is better at the cutoff (+0.048 and +0.052 °C for attention). For seed 2, whose attention run happened to be the fastest (227 s), the LSTM had completed only 4 epochs at the cutoff and attention wins by 0.080 °C. The mean (+0.006 °C) is dominated by that one pairing. Since timing was uncontrolled and its cause is unknown, a time-matched ranking from these runs would partly reflect when each run happened to execute.
- **It cannot manufacture a prospective experiment.** The cutoff is chosen after seeing the curves (a researcher degree of freedom), the speed of each run was not controlled, and no run was designed with a time budget.
- **Intermediate weights mostly do not exist.** Only the best-validation weights are stored. A validation number at a cutoff is readable, but a test MAE, predictions, or attention maps at that cutoff cannot be produced unless the best epoch at the cutoff happens to be the saved one.
- **Coverage is limited.** Attention finished 50 epochs in 227–548 s; beyond that there is no attention information, so cutoffs above about 550 s compare a finished attention run with a partial LSTM run only.
- **Appropriate use:** a descriptive supplementary table, clearly labelled retrospective, validation-only and timing-uncontrolled. It cannot support a claim of matched compute.

## 8. Options for additional training

### 8.1 Exact resume versus starting from saved weights

**Exact resume of the existing runs is not possible.** Each artifact holds only the best-validation `state_dict` plus config, scaler and history. It lacks the last-epoch weights (they differ from the saved ones for four of six runs; for attention seeds 0 and 2 the best epoch is the last), Adam's moment estimates and step counter, and any RNG state. `train_model` also has no resume or initial-weights argument, so any continuation needs new code, which was not written. The training loop uses no other random operations (no dropout; batch order is a function of `(seed, epoch)`), so a future exact resume would need to save last-epoch weights, optimizer state and the epoch counter, and it should be tested for equivalence with uninterrupted training before being trusted. I have not verified that claim for this code.

**Starting a new stage from the saved best weights** (a warm start) is possible with a small code addition, but it is **not equivalent to uninterrupted training**:

- *Adam state reset.* A fresh Adam has zero moments and step 0. Its first update is about `lr · sign(gradient)` for every parameter, which is larger for many parameters than the steady-state steps of the stage-1 optimizer. The second-moment estimate takes on the order of 1,000 steps (about one epoch) to rebuild. A short transient is therefore expected, possibly a brief validation bump before recovery. Its size and its effect on the final result are unknown and would have to be measured.
- *Weights from the best epoch, not the last.* The restart point is epoch 45–49 (LSTM) or 49–50 (attention), not a common epoch, and the epochs after it were discarded. A convention is needed for which batch-order epochs the new stage uses (for example 51 onward for both models).
- *Accounting.* Total steps become stage 1 (52,650, of which a few epochs' work was discarded) plus stage 2.

A warm-started stage is a defined protocol in its own right ("50 epochs, then a restart from the validation-best weights with fresh Adam for N more"), and applied identically to both models it supports a comparison under *that* protocol. It must not be described as "50 + N epochs of uninterrupted training".

**A fresh rerun to a larger cap** is the other route. It reproduces epochs 1–50 exactly (verified in section 2), so it also acts as a check on the saved runs, and it needs no code change; it costs the first 50 epochs again.

### 8.2 Options compared

Time estimates use the per-epoch times observed in the six runs, including per-epoch validation: about 11.5–20.5 s (LSTM) and 4.5–8.9 s (attention). The 20-epoch run was slower than all of them (about 30 s and 11 s per epoch), so the upper figures allow for that. Estimates are from six runs and could be off by 2×.

| Option | Extra time | What it can legitimately establish | What it cannot |
|---|---|---|---|
| **A. Keep the six runs; disclose.** Report the 50-epoch/52,650-step matched-budget results, the validation curves and deceleration figures, measured and approximate-arithmetic costs, the retrospective table labelled as such, and the unmet convergence and compute-matching evidence | none | The validation ranking at the stated budget; reproducibility; that the budget was matched in epochs, steps, data and order | That either model converged; equal compute; ranking at a larger budget. **Rubric risk:** if the rubric requires demonstrated convergence or a compute budget not defined in epochs/steps, this is a visible gap that disclosure mitigates but does not remove |
| **B. A plus the retrospective time-cutoff table (section 7)** | minutes of analysis | Descriptive validation-only picture under a shared elapsed time, and how sensitive it is to timing | A controlled compute match; anything needing intermediate weights |
| **C. Smallest extra experiment: one seed (0), both models, fresh run to 100 epochs, new directory; existing artifacts untouched** | about 0.5–1.2 h (LSTM 19–50 min, attention 8–19 min) | Direct evidence of whether validation MAE keeps falling past epoch 50 for both models on one seed; whether the seed-0 ranking holds at 100 epochs; a built-in check that epochs 1–50 reproduce the saved history | Seed-to-seed variability at 100 epochs; that 100 epochs is converged; anything about the other seeds |
| **D. Warm-start probe: one seed, both models, 50 more epochs from the saved best weights with fresh Adam (needs a small code addition)** | about 15–35 min plus the code change | Whether more training lowers validation MAE under a restart protocol | Equivalence to uninterrupted training (Adam reset); needs code written and tested first |
| **E. Three seeds, both models, fresh to 100 epochs** | about 1.3–2.5 h, up to about 3.5 h | C for all three seeds, with seed spread at 100 epochs | Convergence beyond 100 epochs |
| **F. The revision-1 protocol (150 epochs, three seeds, stopping rule, tuning grid)** | about 2–4 h, plus about 2–3.7 h for a tuning allowance | A full prospective protocol with stopping criterion | Nothing the existing runs do not already cover for the matched-step claim; it duplicates them; not recommended unless the rubric clearly demands it |

### 8.3 Recommendation (prepared only; nothing executed)

1. **Retain the six runs** as the primary matched-step result. No concrete reason exists to discard or repeat them.
2. **Choose between A/B and C according to the rubric.** If it explicitly requires evidence of convergence, run **Option C**, the smallest experiment that adds real evidence. If it only requires matched seeds and compute budgets, **A with B** is defensible with full disclosure.
3. **If running C, fix the interpretation rules before looking at results**, for example: report the 100-epoch curves for both models; call the 50-epoch result "near plateau" only if the best 100-epoch validation MAE improves on the best 50-epoch value by less than a stated amount (say 0.01 °C) for both models; otherwise report both as "still improving at 100 epochs" and keep the six 50-epoch runs as the primary result. Either outcome is reported.
4. **Command, prepared and not run** (writes to a new directory, so nothing existing is overwritten; no `--evaluate-test`):

```bash
.venv312/bin/python -m scripts.train_forecast --source local --run-kind final \
    --epochs 100 --seed 0 --out-dir artifacts/probe100
```

   `--source local` needs no credentials, no network and no extra package, and gave bit-identical results to the Supabase-sourced runs in the earlier smoke comparison. `--source supabase` would keep the "training reads Supabase rows" property for the probe but needs the `supabase` package in `.venv312` (not installed) and the `.env` credentials. Run the models one after the other on an otherwise idle machine, and note the hardware and time, since none of that is stored automatically.
5. **Keep the test split reserved** until the set of runs to be reported is fixed; evaluate it once, after that.

## 9. Limitations

- Validation results only; no test results exist or were examined.
- Three seeds per model; spreads and the gap are rough estimates.
- Timing evidence is uncontrolled and its causes are unknown.
- Reproducibility was verified on one machine with one torch version; a different machine or thread count was not tested.
- The retrospective analysis uses a post hoc cutoff and cannot recover intermediate weights.
- No learning-curve plots: matplotlib is not installed in `.venv312` or `.venv`, and nothing was installed. The tables carry the same data.
- Runs and artifacts have not been persisted to Supabase as far as any evidence shows.

## Decisions needed from you

1. **Rubric wording:** does it require demonstrated convergence, and in what unit is the compute budget defined (epochs/steps, time, arithmetic)? This decides between A/B and C.
2. If C: seed 0 only, `--source local` or `--source supabase`, and the pre-stated interpretation threshold.
3. Whether to report the retrospective time-cutoff table (B) and how to label it.
4. Whether any warm-start or exact-resume code (D, or resume support for future runs) is wanted at all.
5. Confirm no test result has been viewed so far.
6. When to persist runs and artifacts to Supabase.
