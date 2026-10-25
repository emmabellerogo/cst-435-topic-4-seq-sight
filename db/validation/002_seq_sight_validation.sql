-- 002_seq_sight_validation.sql
-- Rollback-only behavior checks for 002_seq_sight_forecasting.sql.
--
-- *** NEVER RUN THIS AGAINST THE REAL SUPABASE PROJECT. ***
-- Run it only in a DISPOSABLE database (e.g. a throwaway local PostgreSQL with
-- the Supabase roles `anon`, `authenticated`, `service_role` created, after
-- applying 002). It inserts fixture rows and always ends with ROLLBACK, but it
-- still must not touch a database whose contents matter. It refuses to start if
-- any of the four tables already has rows.
--
-- Every check prints NOTICE 'PASS ...'. Any unexpected result raises an
-- exception ('FAIL ...') and aborts the script; if that happens, run ROLLBACK.
-- As of writing, this script has NOT been executed anywhere.

begin;

do $$
begin
    if exists (select 1 from sequences) or exists (select 1 from runs)
       or exists (select 1 from run_artifacts) or exists (select 1 from predictions) then
        raise exception 'Refusing to run: tables are not empty. Use an empty disposable database.';
    end if;
end
$$;

-- Helpers (temp objects, discarded at rollback).
create function pg_temp.expect_ok(label text, stmt text) returns void
language plpgsql as $$
begin
    execute stmt;
    raise notice 'PASS (accepted): %', label;
end
$$;

-- Runs stmt; passes only if it fails with exactly the expected SQLSTATE.
-- 23514 check_violation, 23502 not_null_violation, 23503 foreign_key_violation,
-- 23505 unique_violation, 42501 insufficient_privilege
create function pg_temp.expect_fail(label text, stmt text, expected text) returns void
language plpgsql as $$
begin
    begin
        execute stmt;
    exception when others then
        if sqlstate = expected then
            raise notice 'PASS (rejected, %): %', sqlstate, label;
            return;
        end if;
        raise exception 'FAIL: % was rejected with SQLSTATE % (expected %): %',
                        label, sqlstate, expected, sqlerrm;
    end;
    raise exception 'FAIL: % was ACCEPTED but must be rejected', label;
end
$$;

-- Valid JSON fixtures.
create temp table fx as
select (select jsonb_agg(i * 0.5) from generate_series(1, 24) i) as good_forecast,
       (select jsonb_agg((select jsonb_agg(1.0 / 72) from generate_series(1, 72)))
        from generate_series(1, 72)) as good_attn;

-- run insert with SQL fragments for the variable columns
create function pg_temp.run_sql(model text, prep text, val text, test text) returns text
language sql as $$
    select format($f$insert into runs (model_type, data_version, seed, hyperparameters,
        preprocessing, compute_budget, val_mae_c, test_mae_c)
        values (%s, 'v-test', 900, '{}', %s, '{}', %s, %s)$f$, model, prep, val, test)
$$;

-- prediction insert with SQL fragments
create function pg_temp.pred_sql(hash text, fc text, attn text, model text, run text) returns text
language sql as $$
    select format($f$insert into predictions (input_hash, forecast, attention, model_type, served_by_run_id)
        values (%s, %s, %s, %s, %s)$f$, hash, fc, attn, model, run)
$$;

-- ===========================================================================
-- sequences
-- ===========================================================================
select pg_temp.expect_ok('sequences: valid row',
  $$insert into sequences (data_version, station_ts, temperature_c, split)
    values ('v-test', '2020-03-29 02:00:00', 5.25, 'train')$$);

do $$
begin
    -- 02:00 on the last Sunday of March is a DST gap in Europe/Berlin; a naive
    -- timestamp must come back unchanged, whatever the session time zone is.
    set local timezone = 'UTC';
    if (select station_ts::text from sequences where data_version = 'v-test') <> '2020-03-29 02:00:00' then
        raise exception 'FAIL: naive station_ts was altered';
    end if;
    raise notice 'PASS: station_ts preserved as naive timestamp';
end
$$;

select pg_temp.expect_fail('sequences: duplicate (data_version, station_ts)',
  $$insert into sequences (data_version, station_ts, temperature_c, split)
    values ('v-test', '2020-03-29 02:00:00', 6, 'train')$$, '23505');
select pg_temp.expect_ok('sequences: same timestamp, different data_version',
  $$insert into sequences (data_version, station_ts, temperature_c, split)
    values ('v-test-2', '2020-03-29 02:00:00', 6, 'validation')$$);
select pg_temp.expect_fail('sequences: timestamp not on the hour',
  $$insert into sequences (data_version, station_ts, temperature_c, split)
    values ('v-test', '2020-03-29 03:10:00', 6, 'train')$$, '23514');
select pg_temp.expect_fail('sequences: invalid split',
  $$insert into sequences (data_version, station_ts, temperature_c, split)
    values ('v-test', '2020-03-29 04:00:00', 6, 'dev')$$, '23514');
select pg_temp.expect_fail('sequences: -9999 sentinel',
  $$insert into sequences (data_version, station_ts, temperature_c, split)
    values ('v-test', '2020-03-29 04:00:00', -9999, 'train')$$, '23514');
select pg_temp.expect_fail('sequences: NaN temperature',
  $$insert into sequences (data_version, station_ts, temperature_c, split)
    values ('v-test', '2020-03-29 04:00:00', 'NaN', 'train')$$, '23514');
select pg_temp.expect_fail('sequences: empty data_version',
  $$insert into sequences (data_version, station_ts, temperature_c, split)
    values ('', '2020-03-29 04:00:00', 6, 'train')$$, '23514');
select pg_temp.expect_fail('sequences: NULL temperature',
  $$insert into sequences (data_version, station_ts, temperature_c, split)
    values ('v-test', '2020-03-29 04:00:00', null, 'train')$$, '23502');

-- ===========================================================================
-- runs: valid rows
-- ===========================================================================
select pg_temp.expect_ok('runs: valid lstm',
  pg_temp.run_sql($$'lstm'$$, $$'{"mean": 9.5, "std": 8.1}'$$, '1.5', '1.7'));
select pg_temp.expect_ok('runs: valid attention, zero MAE allowed',
  pg_temp.run_sql($$'attention'$$, $$'{"mean": 9.5, "std": 8.1, "window": 72}'$$, '0', '1.6'));

-- runs: preprocessing
select pg_temp.expect_fail('runs: preprocessing missing mean',
  pg_temp.run_sql($$'lstm'$$, $$'{"std": 8.1}'$$, '1', '1'), '23514');
select pg_temp.expect_fail('runs: preprocessing missing std',
  pg_temp.run_sql($$'lstm'$$, $$'{"mean": 9.5}'$$, '1', '1'), '23514');
select pg_temp.expect_fail('runs: preprocessing empty object',
  pg_temp.run_sql($$'lstm'$$, $$'{}'$$, '1', '1'), '23514');
select pg_temp.expect_fail('runs: preprocessing mean is JSON null',
  pg_temp.run_sql($$'lstm'$$, $$'{"mean": null, "std": 8.1}'$$, '1', '1'), '23514');
select pg_temp.expect_fail('runs: preprocessing std is JSON null',
  pg_temp.run_sql($$'lstm'$$, $$'{"mean": 9.5, "std": null}'$$, '1', '1'), '23514');
select pg_temp.expect_fail('runs: preprocessing std is a string',
  pg_temp.run_sql($$'lstm'$$, $$'{"mean": 9.5, "std": "8.1"}'$$, '1', '1'), '23514');
select pg_temp.expect_fail('runs: preprocessing mean is a string',
  pg_temp.run_sql($$'lstm'$$, $$'{"mean": "9.5", "std": 8.1}'$$, '1', '1'), '23514');
select pg_temp.expect_fail('runs: preprocessing std = 0',
  pg_temp.run_sql($$'lstm'$$, $$'{"mean": 9.5, "std": 0}'$$, '1', '1'), '23514');
select pg_temp.expect_fail('runs: preprocessing std < 0',
  pg_temp.run_sql($$'lstm'$$, $$'{"mean": 9.5, "std": -1}'$$, '1', '1'), '23514');
select pg_temp.expect_fail('runs: preprocessing is an array',
  pg_temp.run_sql($$'lstm'$$, $$'[9.5, 8.1]'$$, '1', '1'), '23514');
select pg_temp.expect_fail('runs: preprocessing is JSON null',
  pg_temp.run_sql($$'lstm'$$, $$'null'::jsonb$$, '1', '1'), '23514');
select pg_temp.expect_fail('runs: preprocessing is SQL NULL',
  pg_temp.run_sql($$'lstm'$$, 'null', '1', '1'), '23502');

-- runs: metrics (each of val_mae_c and test_mae_c)
select pg_temp.expect_fail('runs: val_mae_c NaN',
  pg_temp.run_sql($$'lstm'$$, $$'{"mean":1,"std":1}'$$, $$'NaN'::float8$$, '1'), '23514');
select pg_temp.expect_fail('runs: val_mae_c +Infinity',
  pg_temp.run_sql($$'lstm'$$, $$'{"mean":1,"std":1}'$$, $$'Infinity'::float8$$, '1'), '23514');
select pg_temp.expect_fail('runs: val_mae_c -Infinity',
  pg_temp.run_sql($$'lstm'$$, $$'{"mean":1,"std":1}'$$, $$'-Infinity'::float8$$, '1'), '23514');
select pg_temp.expect_fail('runs: val_mae_c negative',
  pg_temp.run_sql($$'lstm'$$, $$'{"mean":1,"std":1}'$$, '-0.1', '1'), '23514');
select pg_temp.expect_fail('runs: test_mae_c NaN',
  pg_temp.run_sql($$'lstm'$$, $$'{"mean":1,"std":1}'$$, '1', $$'NaN'::float8$$), '23514');
select pg_temp.expect_fail('runs: test_mae_c +Infinity',
  pg_temp.run_sql($$'lstm'$$, $$'{"mean":1,"std":1}'$$, '1', $$'Infinity'::float8$$), '23514');
select pg_temp.expect_fail('runs: test_mae_c -Infinity',
  pg_temp.run_sql($$'lstm'$$, $$'{"mean":1,"std":1}'$$, '1', $$'-Infinity'::float8$$), '23514');
select pg_temp.expect_fail('runs: test_mae_c negative',
  pg_temp.run_sql($$'lstm'$$, $$'{"mean":1,"std":1}'$$, '1', '-0.1'), '23514');
select pg_temp.expect_fail('runs: val_mae_c NULL',
  pg_temp.run_sql($$'lstm'$$, $$'{"mean":1,"std":1}'$$, 'null', '1'), '23502');

-- runs: other columns
select pg_temp.expect_fail('runs: model_type outside lstm/attention',
  pg_temp.run_sql($$'gru'$$, $$'{"mean":1,"std":1}'$$, '1', '1'), '23514');
select pg_temp.expect_fail('runs: hyperparameters is an array',
  $$insert into runs (model_type, data_version, seed, hyperparameters, preprocessing, compute_budget, val_mae_c, test_mae_c)
    values ('lstm', 'v-test', 901, '[]', '{"mean":1,"std":1}', '{}', 1, 1)$$, '23514');
select pg_temp.expect_fail('runs: compute_budget is a string',
  $$insert into runs (model_type, data_version, seed, hyperparameters, preprocessing, compute_budget, val_mae_c, test_mae_c)
    values ('lstm', 'v-test', 901, '{}', '{"mean":1,"std":1}', '"big"', 1, 1)$$, '23514');

-- run_artifacts
select pg_temp.expect_ok('run_artifacts: valid',
  $$insert into run_artifacts (run_id, model_b64)
    select id, 'AAAA' from runs where model_type = 'lstm'$$);

-- ===========================================================================
-- predictions
-- ===========================================================================
-- fragments
create temp table frag as select
    $$repeat('a', 64)$$                                              as hash,
    $$(select good_forecast from fx)$$                               as fc,
    $$(select good_attn from fx)$$                                   as attn,
    $$(select id from runs where model_type = 'lstm')$$              as lstm_run,
    $$(select id from runs where model_type = 'attention')$$         as attn_run;

select pg_temp.expect_ok('predictions: valid lstm, attention NULL',
  (select pg_temp.pred_sql(hash, fc, 'null', $$'lstm'$$, lstm_run) from frag));
select pg_temp.expect_ok('predictions: valid attention with 72x72 matrix',
  (select pg_temp.pred_sql(hash, fc, attn, $$'attention'$$, attn_run) from frag));

-- forecast shape
select pg_temp.expect_fail('predictions: forecast has 23 values',
  (select pg_temp.pred_sql(hash, $$(select jsonb_agg(i) from generate_series(1, 23) i)$$, 'null', $$'lstm'$$, lstm_run) from frag), '23514');
select pg_temp.expect_fail('predictions: forecast has 25 values',
  (select pg_temp.pred_sql(hash, $$(select jsonb_agg(i) from generate_series(1, 25) i)$$, 'null', $$'lstm'$$, lstm_run) from frag), '23514');
select pg_temp.expect_fail('predictions: forecast contains a string',
  (select pg_temp.pred_sql(hash, $$jsonb_set((select good_forecast from fx), '{0}', '"x"')$$, 'null', $$'lstm'$$, lstm_run) from frag), '23514');
select pg_temp.expect_fail('predictions: forecast contains JSON null',
  (select pg_temp.pred_sql(hash, $$jsonb_set((select good_forecast from fx), '{0}', 'null'::jsonb)$$, 'null', $$'lstm'$$, lstm_run) from frag), '23514');
select pg_temp.expect_fail('predictions: forecast is an object',
  (select pg_temp.pred_sql(hash, $$'{"a": 1}'::jsonb$$, 'null', $$'lstm'$$, lstm_run) from frag), '23514');
select pg_temp.expect_fail('predictions: forecast is JSON null',
  (select pg_temp.pred_sql(hash, $$'null'::jsonb$$, 'null', $$'lstm'$$, lstm_run) from frag), '23514');
select pg_temp.expect_fail('predictions: forecast is SQL NULL',
  (select pg_temp.pred_sql(hash, 'null', 'null', $$'lstm'$$, lstm_run) from frag), '23502');

-- input hash
select pg_temp.expect_fail('predictions: input_hash too short',
  (select pg_temp.pred_sql($$repeat('a', 63)$$, fc, 'null', $$'lstm'$$, lstm_run) from frag), '23514');
select pg_temp.expect_fail('predictions: input_hash uppercase',
  (select pg_temp.pred_sql($$repeat('A', 64)$$, fc, 'null', $$'lstm'$$, lstm_run) from frag), '23514');

-- attention shape
select pg_temp.expect_fail('predictions: attention has 71 rows',
  (select pg_temp.pred_sql(hash, fc,
     $$(select jsonb_agg(r) from jsonb_array_elements((select good_attn from fx)) with ordinality t(r, n) where n <= 71)$$,
     $$'attention'$$, attn_run) from frag), '23514');
select pg_temp.expect_fail('predictions: attention row with 71 columns',
  (select pg_temp.pred_sql(hash, fc,
     $$jsonb_set((select good_attn from fx), '{0}', (select jsonb_agg(0.1) from generate_series(1, 71)))$$,
     $$'attention'$$, attn_run) from frag), '23514');
select pg_temp.expect_fail('predictions: attention cell is a string',
  (select pg_temp.pred_sql(hash, fc,
     $$jsonb_set((select good_attn from fx), '{3,5}', '"x"')$$, $$'attention'$$, attn_run) from frag), '23514');
select pg_temp.expect_fail('predictions: attention cell is JSON null',
  (select pg_temp.pred_sql(hash, fc,
     $$jsonb_set((select good_attn from fx), '{3,5}', 'null'::jsonb)$$, $$'attention'$$, attn_run) from frag), '23514');
select pg_temp.expect_fail('predictions: attention is a flat array of 72 numbers',
  (select pg_temp.pred_sql(hash, fc,
     $$(select jsonb_agg(0.1) from generate_series(1, 72))$$, $$'attention'$$, attn_run) from frag), '23514');
select pg_temp.expect_fail('predictions: attention is JSON null (not SQL NULL)',
  (select pg_temp.pred_sql(hash, fc, $$'null'::jsonb$$, $$'attention'$$, attn_run) from frag), '23514');

-- model type / attention consistency
select pg_temp.expect_fail('predictions: lstm with an attention matrix',
  (select pg_temp.pred_sql(hash, fc, attn, $$'lstm'$$, lstm_run) from frag), '23514');
select pg_temp.expect_fail('predictions: attention model without a matrix',
  (select pg_temp.pred_sql(hash, fc, 'null', $$'attention'$$, attn_run) from frag), '23514');
select pg_temp.expect_fail('predictions: model_type disagrees with the run (attention row, lstm run)',
  (select pg_temp.pred_sql(hash, fc, attn, $$'attention'$$, lstm_run) from frag), '23503');
select pg_temp.expect_fail('predictions: model_type disagrees with the run (lstm row, attention run)',
  (select pg_temp.pred_sql(hash, fc, 'null', $$'lstm'$$, attn_run) from frag), '23503');
select pg_temp.expect_fail('predictions: nonexistent run',
  (select pg_temp.pred_sql(hash, fc, 'null', $$'lstm'$$, '-1') from frag), '23503');
select pg_temp.expect_fail('predictions: served_by_run_id NULL',
  (select pg_temp.pred_sql(hash, fc, 'null', $$'lstm'$$, 'null') from frag), '23502');
select pg_temp.expect_fail('predictions: model_type outside lstm/attention',
  (select pg_temp.pred_sql(hash, fc, 'null', $$'gru'$$, lstm_run) from frag), '23514');

-- ===========================================================================
-- deletion behavior
-- ===========================================================================
select pg_temp.expect_fail('delete: run referenced by a prediction is blocked (lstm)',
  $$delete from runs where model_type = 'lstm'$$, '23503');
select pg_temp.expect_fail('delete: run referenced by a prediction is blocked (attention)',
  $$delete from runs where model_type = 'attention'$$, '23503');

do $$
begin
    if (select count(*) from predictions) <> 2 then
        raise exception 'FAIL: prediction history changed after blocked deletes';
    end if;
    raise notice 'PASS: prediction history intact after blocked deletes';
end
$$;

-- an unreferenced run can be deleted and takes its artifact with it
insert into runs (model_type, data_version, seed, hyperparameters, preprocessing, compute_budget, val_mae_c, test_mae_c)
values ('lstm', 'v-test', 902, '{}', '{"mean":1,"std":1}', '{}', 1, 1);
insert into run_artifacts (run_id, model_b64) select id, 'BBBB' from runs where seed = 902;

do $$
begin
    delete from runs where seed = 902;
    if exists (select 1 from run_artifacts where model_b64 = 'BBBB') then
        raise exception 'FAIL: artifact survived deletion of its run';
    end if;
    raise notice 'PASS: unreferenced run deleted, artifact cascaded';
end
$$;

-- ===========================================================================
-- anonymous role: read runs/predictions only; everything else denied.
-- (Requires the Supabase roles; fails with "role anon does not exist" otherwise.)
-- ===========================================================================
set local role anon;

do $$
begin
    if (select count(*) from runs) <> 2 or (select count(*) from predictions) <> 2 then
        raise exception 'FAIL: anon cannot read runs/predictions through the policies';
    end if;
    raise notice 'PASS: anon can read runs and predictions';
end
$$;

do $$
declare
    stmt text;
begin
    foreach stmt in array array[
        'select * from sequences',
        'select * from run_artifacts',
        'insert into runs (model_type) values (''lstm'')',
        'update runs set seed = 0',
        'delete from runs',
        'truncate runs',
        'insert into predictions (input_hash) values (''x'')',
        'update predictions set model_type = ''lstm''',
        'delete from predictions',
        'truncate predictions',
        'insert into sequences (data_version) values (''x'')',
        'truncate sequences',
        'truncate run_artifacts'
    ] loop
        begin
            execute stmt;
            raise exception 'FAIL: anon was allowed: %', stmt;
        exception when insufficient_privilege then
            raise notice 'PASS (anon denied, 42501): %', stmt;
        end;
    end loop;
end
$$;

reset role;

-- Nothing above is kept.
rollback;
