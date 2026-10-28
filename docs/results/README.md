# Reference results from `docs/model-comparison.sql`

**Provenance: user-executed.** The user ran the three statements in [`../model-comparison.sql`](../model-comparison.sql) in the Supabase SQL editor of the project that holds the registered runs and supplied the outputs below. Claude did not query Supabase and could not check these values against the database; the files are a transcription of what the user reported. Date of execution and project reference: not recorded (pending, to be added by the user if wanted).

| File | Statement | Content |
|---|---|---|
| [model-comparison-query1.csv](model-comparison-query1.csv) | 1, per-model summary | 2 rows (attention, lstm), transcribed as supplied, unrounded |
| [model-comparison-query2.csv](model-comparison-query2.csv) | 2, per-seed details | 6 rows: run id, test and validation MAE, artifact hash prefix and `created_at` per model and seed (LSTM seeds 0/1/2 are runs 1/2/3; attention seeds 0/1/2 are runs 4/5/6), transcribed as supplied |
| [model-comparison-query3.csv](model-comparison-query3.csv) | 3, completeness check | 6 rows, each with `runs_found = 1` and one run id: no missing runs and no duplicates |

All three are complete outputs as the user pasted them into the chat; nothing was dropped, reordered by us, or filled in from another source. `created_at` values are as returned (UTC, 2026-10-28 06:51).

Filter: data version `jena-mpi-roof-h1-60138aac7d898b8c` and the six artifact SHA-256 values listed in the SQL. MAE is mean absolute error in °C. `test_mae_c` comes from the one-time final test evaluation; no model was selected from it.

**What the SD is.** `sd_*` is the sample standard deviation (n − 1, n = 3) of the per-run MAE across the three training seeds. It describes seed-to-seed variation of the trained models' aggregate error. It is not prediction uncertainty, not a confidence interval for a forecast or for the true error, and not a measure of how sure the model is about any forecast; with three seeds the SD itself is imprecise.

As a consistency check (done by Claude on the supplied text, not against the database): the per-seed test MAEs in query 2 equal the values in the local, git-ignored `artifacts/epoch50/final/test_eval.json` (separate evidence from the evaluation script) to the precision supplied, and the query 1 means and sample SDs follow from the six query 2 values.
