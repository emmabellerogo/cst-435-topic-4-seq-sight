-- 002_seq_sight_forecasting.sql
-- Seq Sight: hourly temperature forecasting (72 h input -> 24 h forecast).
--
-- STANDALONE INITIALIZATION for a FRESH Supabase project. It REPLACES
-- 001_init.sql (the Attend-It classification schema); it is not an increment on
-- top of it. Apply this file ONLY. 001 creates tables named `runs` and
-- `run_artifacts` with different columns, so applying both would conflict.
-- This file deliberately uses plain CREATE (no IF NOT EXISTS) and refuses to
-- run if 001's tables are present, so it can never silently reuse the wrong
-- `runs`. It is one transaction: everything is created or nothing is.
--
-- Apply in the Supabase dashboard: SQL Editor -> New query -> paste -> Run.
-- See docs/supabase-setup.md for what differs from 001 and how to verify.

begin;

do $$
begin
    if to_regclass('public.datasets') is not null
       or to_regclass('public.sequence_metadata') is not null
       or to_regclass('public.runs') is not null then
        raise exception
            'Existing Attend-It (001_init.sql) or Seq Sight tables found. '
            '002 is a standalone initialization for a fresh project; do not apply 001 first.';
    end if;
end
$$;

-- ---------------------------------------------------------------------------
-- sequences: the hourly Jena corpus. Server-side only (no anon access).
-- station_ts is the hour-end label exactly as the station reported it: a NAIVE
-- timestamp (WITHOUT time zone). Its zone/DST semantics are unverified (see
-- docs/dataset-card.md), so it is never converted to UTC. Only valid hours are
-- stored; invalid hours are absent (no interpolation).
-- ---------------------------------------------------------------------------
create table sequences (
    id             bigint generated always as identity primary key,
    data_version   text        not null check (data_version <> ''),
    station_ts     timestamp without time zone not null
                   check (station_ts = date_trunc('hour', station_ts)),
    temperature_c  double precision not null
                   check (temperature_c between -90 and 60),  -- also rejects the -9999 sentinel
    split          text        not null check (split in ('train', 'validation', 'test')),
    unique (data_version, station_ts)
);

-- ---------------------------------------------------------------------------
-- runs: one row per finished training/evaluation job. Anon-readable, so its
-- contents must be safe to publish (no hostnames, paths, secrets).
-- preprocessing holds the train-only standardization (mean, std in degC) plus
-- any other metadata; hyperparameters and compute_budget are free-form objects.
-- ---------------------------------------------------------------------------
create table runs (
    id               bigint generated always as identity primary key,
    model_type       text    not null check (model_type in ('lstm', 'attention')),
    data_version     text    not null check (data_version <> ''),
    seed             integer not null,
    hyperparameters  jsonb   not null check (jsonb_typeof(hyperparameters) = 'object'),
    -- coalesce(..., false): a missing key makes `->` return SQL NULL, and a CHECK
    -- treats NULL as a pass. Missing keys, JSON nulls, non-numbers and std <= 0
    -- must all fail, so any NULL result is forced to false.
    preprocessing    jsonb   not null check (coalesce(
                         jsonb_typeof(preprocessing) = 'object'
                         and jsonb_typeof(preprocessing -> 'mean') = 'number'
                         and jsonb_typeof(preprocessing -> 'std')  = 'number'
                         and (preprocessing -> 'std') > to_jsonb(0),
                         false)),
    compute_budget   jsonb   not null check (jsonb_typeof(compute_budget) = 'object'),
    -- finite and >= 0: NaN sorts above Infinity in PostgreSQL, so `< Infinity`
    -- rejects NaN and +Infinity; `>= 0` rejects -Infinity and negatives.
    val_mae_c        double precision not null
                     check (val_mae_c  >= 0 and val_mae_c  < 'Infinity'::double precision),
    test_mae_c       double precision not null
                     check (test_mae_c >= 0 and test_mae_c < 'Infinity'::double precision),
    created_at       timestamptz not null default now(),
    unique (id, model_type)   -- target of predictions' (run, model_type) foreign key
);

-- ---------------------------------------------------------------------------
-- run_artifacts: the fitted model, base64-encoded. Not anon-readable.
-- (Same shape as 001.)
-- ---------------------------------------------------------------------------
create table run_artifacts (
    run_id      bigint      primary key references runs (id) on delete cascade,
    model_b64   text        not null,
    created_at  timestamptz not null default now()
);

-- ---------------------------------------------------------------------------
-- predictions: one row per served forecast: SHA-256 of the input (never the
-- input), 24-value forecast, nullable 72x72 attention matrix, serving run.
-- A run that has predictions cannot be deleted (ON DELETE RESTRICT), so the
-- audit link is never lost or cleared.
-- model_type is repeated here only so the database can enforce "the lstm
-- baseline has NULL attention" (the composite foreign key makes it match the
-- run's real type). JSON checks cover structure (array lengths, numeric
-- elements); softmax row sums are not checked.
-- ---------------------------------------------------------------------------
create table predictions (
    id                bigint generated always as identity primary key,
    input_hash        text    not null check (input_hash ~ '^[0-9a-f]{64}$'),
    forecast          jsonb   not null check (
                          case
                              when jsonb_typeof(forecast) <> 'array' then false
                              when jsonb_array_length(forecast) <> 24 then false
                              else not jsonb_path_exists(
                                  forecast, 'strict $[*] ? (@.type() != "number")')
                          end),
    attention         jsonb   check (attention is null or
                          case
                              when jsonb_typeof(attention) <> 'array' then false
                              when jsonb_array_length(attention) <> 72 then false
                              when jsonb_path_exists(attention,
                                  'strict $[*] ? (@.type() != "array" || @.size() != 72)') then false
                              else not jsonb_path_exists(
                                  attention, 'strict $[*][*] ? (@.type() != "number")')
                          end),
    model_type        text    not null check (model_type in ('lstm', 'attention')),
    served_by_run_id  bigint  not null,
    created_at        timestamptz not null default now(),
    check ((model_type = 'lstm' and attention is null)
        or (model_type = 'attention' and attention is not null)),
    foreign key (served_by_run_id, model_type)
        references runs (id, model_type) on delete restrict
);

create index idx_runs_created_at               on runs (created_at desc);
create index idx_predictions_served_by_run_id  on predictions (served_by_run_id);
create index idx_predictions_created_at        on predictions (created_at desc);

-- ---------------------------------------------------------------------------
-- Row Level Security on every table. Anon may SELECT runs and predictions (the
-- UI history). sequences and run_artifacts have no policy, so anon sees
-- nothing there. The service-role key bypasses RLS: all writes are server-side.
-- ---------------------------------------------------------------------------
alter table sequences     enable row level security;
alter table runs          enable row level security;
alter table run_artifacts enable row level security;
alter table predictions   enable row level security;

create policy "anon can read runs"
    on runs for select
    to anon
    using (true);

create policy "anon can read predictions"
    on predictions for select
    to anon
    using (true);

-- ---------------------------------------------------------------------------
-- Privileges. Supabase's default privileges give anon/authenticated ALL on new
-- tables, including TRUNCATE, which RLS does not govern. Strip that, then grant
-- back only SELECT on the two public tables. (001 relies on RLS alone.)
-- ---------------------------------------------------------------------------
revoke all on table sequences, runs, run_artifacts, predictions from anon, authenticated;
revoke all on all sequences in schema public from anon, authenticated;
grant select on table runs, predictions to anon;

commit;
