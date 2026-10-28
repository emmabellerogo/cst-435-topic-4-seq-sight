# Curated examples (`examples.json`)

`examples.json` is **generated** by `scripts/curate_forecast_examples.py` (rules and disclosure: [docs/forecasting-examples.md](../../docs/forecasting-examples.md)). Do not hand-write numbers. The hand-written part is `example_notes.json` (interpretation, keyed by case key); the script appends it to an auto-generated "Measured" paragraph. Regenerate with `python -m scripts.curate_forecast_examples`; verify with `--check`.

All entries are **validation** windows. The failure cases are intentionally the hardest windows for the attention model and do not represent average performance.

`comparison_examples` (Baseline vs. Attention tab), up to three:
`title`, `origin` (the window's forecast origin, station time), `history_c` (72), `actual_c` (24), `lstm_predicted_c` (24), `attention_predicted_c` (24), `lstm_run_id`, `attention_run_id`, `discussion`, and optionally `attention` (72x72, the attention artifact's measured encoder weights for that history; the generator always writes it).

`failure_cases` (Failure Gallery tab), up to three:
`title`, `origin`, `model_type`, `run_id`, `history_c` (72), `actual_c` (24), `predicted_c` (24), `discussion`, and optionally `attention` (72x72).

Entries missing a required field are ignored and reported on the tab. The UI ignores unknown fields; the generator adds: `key`, `pattern`, `*_mae_c`, `persistence_mae_c`, `*_artifact_sha256` / `artifact_sha256`, `input_provenance`, `lstm_predicted_c` and `lstm_mae_c` (failure cases), `max_abs_error_c`, `attention_summary`; and a top-level `_meta` block (rules, thresholds, checkpoints, selection audit). The UI shows the MAE and hash-prefix captions when present.
