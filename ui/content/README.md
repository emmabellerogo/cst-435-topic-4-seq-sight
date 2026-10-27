# Curated examples (`examples.json`)

The UI shows nothing in the example sections until real, measured entries are added here. Do not hand-write numbers. Each entry must come from a persisted run's forecast on a real test or validation window, with the evaluation that produced it recorded.

`comparison_examples` (Baseline vs. Attention tab), up to three:
`title`, `origin` (the window's forecast origin, station time), `history_c` (72), `actual_c` (24), `lstm_predicted_c` (24), `attention_predicted_c` (24), `lstm_run_id`, `attention_run_id`, `discussion`.

`failure_cases` (Failure Gallery tab), up to three:
`title`, `origin`, `model_type`, `run_id`, `history_c` (72), `actual_c` (24), `predicted_c` (24), `discussion`, and optionally `attention` (72x72).

Entries missing a field are ignored and reported on the tab. The UI does not treat a case as documented until it validates.
