# Supabase Setup — Seq Sight Forecasting Schema

Back to [README](../README.md) · See also [data design](data-design.md).

## Status

| Item | Status | Evidence |
|------|--------|----------|
| `002` applied to the Supabase project | **Applied by Emma** | Recorded by Emma; not re-checked from this repository |
| RLS enabled, anon read-only grants, the two `SELECT` policies | **Verified by Emma** | Recorded by Emma; no live connection from this repository |
| Corpus uploaded to `sequences` | **Uploaded and verified by Emma** | Data version `jena-mpi-roof-h1-60138aac7d898b8c`: train 135,438 rows (2003-11-24 17:00 – 2019-05-16 00:00), validation 29,047 (2019-05-16 01:00 – 2022-09-08 00:00), test 29,063 (2022-09-08 01:00 – 2026-01-01 00:00). These match the local dry-run counts |
| `db/load.py` dry-run on the real local files | **Executed locally** | Manifest and checksums verified; counts match the dataset card (see below) |
| `tests/test_load.py` (synthetic preparation, splits, gaps, config, mocked upload/resume) | **Executed locally**: 43 tests pass together with `tests/test_inspect_jena.py` | `python3 -m unittest tests.test_load tests.test_inspect_jena` |
| `db/load.py --upload` against real Supabase | **Run by Emma** (outcome above) | Automated tests use an in-memory fake store |
| `shared/forecasting.py`, `shared/forecast_store.py` (preprocessing, windows, paginated reader) | **Executed locally** against synthetic data and a mocked client | `python3 -m unittest tests.test_forecasting`; never run against real Supabase or the real corpus. See [forecasting preprocessing](forecasting-preprocessing.md) |
| `db/validation/002_seq_sight_validation.sql` (constraint/FK/JSON behavior) | **Never run** | Needs a disposable local PostgreSQL, which is not installed here |
| Constraint behavior of `002` (JSON `jsonpath` checks, composite FK, NaN rules) | **Untested** | Same reason |

The applied migration `002` must not be edited. Any future schema change goes in
a new migration file (`003_…`).

## Which file to apply

On the fresh Supabase project, **one** file was to be applied (Emma applied `002`):

| File | Apply? | Role |
|------|--------|------|
| [db/migrations/002_seq_sight_forecasting.sql](../db/migrations/002_seq_sight_forecasting.sql) | **Yes, only this** | **Standalone initialization** of the Seq Sight schema. It replaces `001`; it is *not* an increment on top of it. |
| [db/migrations/001_init.sql](../db/migrations/001_init.sql) | **No** | Professor-provided Attend-It classification schema, preserved unchanged for reference. |

`001` and `002` both define `runs` and `run_artifacts` with different columns,
so they must never both be applied. To make a mistake fail loudly rather than
leave a conflicting schema, `002` uses plain `CREATE` (no `IF NOT EXISTS`) and
starts with a guard that aborts if `datasets`, `sequence_metadata` or `runs`
already exist. It runs in one transaction, so a failure creates nothing. The
numeric prefix only reflects file order in the repo; nothing applies these files
automatically (there is no `supabase/` CLI folder). If a project ever had `001`
applied, it must be reset before applying `002`.

How: Supabase dashboard → SQL Editor → New query → paste `002` → Run.

## Expected tables

| Table | Purpose | Anonymous access |
|-------|---------|------------------|
| `sequences` | Hourly corpus: `data_version`, `station_ts`, `temperature_c`, `split` | None |
| `runs` | One row per finished run: `model_type`, `data_version`, `seed`, `hyperparameters`, `preprocessing`, `compute_budget`, `val_mae_c`, `test_mae_c`, `created_at` | `SELECT` |
| `run_artifacts` | Fitted model (`model_b64`) per run | None |
| `predictions` | `input_hash`, `forecast` (24 values), `attention` (72×72 or NULL), `model_type`, `served_by_run_id`, `created_at` | `SELECT` |

## Differences from Artzi's original `001_init.sql`

**What counts as "required".** The only requirements I can point to are those
stated in the [README](../README.md) (scope, checklist) — I have not seen the
assignment handout itself. [Data design](data-design.md) states that its
windows, splits and tables are *team decisions, not assignment-mandated*. The
"Basis" column uses three labels:

- **README requirement**: stated in the README's scope or checklist.
- **Starter pattern**: kept from Artzi's `001`.
- **Our choice**: a team implementation decision, including everything in
  [data design](data-design.md).

| Area | `001` (Attend-It) | `002` (Seq Sight) | Basis |
|------|-------------------|-------------------|-------|
| Tables | `datasets`, `runs`, `run_artifacts`, `sequence_metadata` | `sequences`, `runs`, `run_artifacts`, `predictions` | **README requirement**: it names `sequences`, `runs`, `predictions`. `run_artifacts` is a **starter pattern**. |
| `predictions` columns | `sequence_metadata`: hash, length, class, confidence | input hash, forecast, attention, serving run id, created_at | **README requirement** (checklist lists exactly these). Storing a hash, not the raw input, is a **starter pattern**. |
| Corpus | `datasets` holds generation parameters only | `sequences` stores the hourly corpus with `data_version`, `split` | **README requirement** (load the hourly readings with train/validation/test split). `data_version` as a plain text label is **our choice**. |
| Timestamps | n/a | `station_ts timestamp without time zone`, on the hour, unique per `(data_version, station_ts)` | **Our choice**, following the data-design/dataset-card position that the time zone is unverified and never converted to UTC. Hour alignment and uniqueness are also ours. |
| Temperature range | n/a | −90…60 °C (NaN also fails); rejects the `-9999` sentinel | **Our choice** |
| `runs` identity | `dataset_id` FK | `model_type` (`lstm`/`attention`), `data_version`, `seed` | `model_type`, `seed`: **README requirement** (compare models with matched seeds). `data_version`: **our choice**. |
| `runs` hyperparameters | `lr`, `batch_size`, `epochs` columns | `hyperparameters` JSON object | Persisting them is a **starter pattern**; JSON instead of columns is **our choice** (the two models differ). |
| `runs` compute budget | none | `compute_budget` JSON object | **README requirement** (matched compute budgets); storing it as JSON is **our choice**. |
| `runs` metrics | `accuracy`, `macro_f1` | `val_mae_c`, `test_mae_c`, finite and ≥ 0 | Persisted **test MAE**: **README requirement**. Validation MAE and the finiteness rule are **our choice**. |
| `runs.preprocessing` | none | JSON object that must contain numeric `mean` and `std > 0` | **Our choice**. Persisting preprocessing with the run is our implementation of the data-design plan (train-only standardization), not something the README mandates by itself. |
| 72-hour input / 24-value forecast | n/a | Not columns on `runs`; `predictions.forecast` must be an array of exactly 24 numbers, `attention` 72 rows × 72 numbers | **Our choice.** The 24-hour horizon is the project scope in the README, but the 72-hour input window and the array-of-24 representation are our design (data design is explicitly "proposed"). |
| Baseline attention NULL | n/a | `model_type` on `predictions`; `CHECK` lstm ⇒ NULL, attention ⇒ required; composite FK to `runs(id, model_type)` | The *behavior* (no fabricated attention for the baseline) is stated in data design, which is itself proposed. Enforcing it **in the database** is **our choice**. |
| JSON validation | none | Inline `CHECK`s | **Our choice** |
| Run link | `run_id … on delete cascade` | `served_by_run_id not null … on delete restrict` | **Our choice** (audit linkage: referenced runs cannot be deleted). |
| Public reads | policy `anon can read runs` | policies on `runs` **and** `predictions` | The UI history tab: **README requirement**. Policy naming is a **starter pattern**. |
| Privileges | RLS only, default grants | Revoke all from `anon`/`authenticated`; grant `SELECT` on `runs`, `predictions` | **Our choice** (security hardening) |
| Index naming | `idx_<table>_<col>` | same | **Starter pattern** |

### Constraint details

- **`runs.preprocessing`** is rejected unless it is an object whose `mean` and
  `std` are both JSON numbers and `std > 0`. A missing key, a JSON `null`, a
  string, or `std <= 0` all fail. The expression is wrapped in `coalesce(…,
  false)` because a missing key makes `->` return SQL `NULL`, and a `CHECK`
  *passes* on `NULL`; without the wrapper an object lacking `mean` or `std`
  would have been accepted. Pure SQL `NULL` is rejected by `NOT NULL`.
- **`val_mae_c` / `test_mae_c`** must satisfy `>= 0 and < Infinity`. PostgreSQL
  sorts `NaN` above `Infinity`, so `NaN` and `+Infinity` fail the upper bound,
  and `-Infinity` and negatives fail the lower bound.
- The same NaN ordering is why `temperature_c between -90 and 60` rejects `NaN`.

### Choices reviewed and simplified

- **Custom JSON validator functions: removed.** The first draft used helper SQL
  functions with revoked execute rights. Plain inline `CHECK`s do the same job
  with nothing extra to expose. They check structure only (array lengths,
  numeric elements); softmax row sums are not validated, and the API must still
  validate its own inputs.
- **Composite foreign key: kept, but only because it enforces the baseline-NULL
  rule.** Without it, the database would have no way to know whether a row is the
  lstm baseline, and that rule would rest on API code alone. Cost: a
  `model_type` column on `predictions` that repeats the run's, plus
  `unique (id, model_type)` on `runs`. If you prefer the simpler shape, drop
  `model_type` and the composite FK, use `foreign key (served_by_run_id)
  references runs (id) on delete restrict`, and enforce baseline-NULL in the API
  and tests. Nothing else depends on it.
- **Column-level grants: removed.** Table-level `SELECT` on the two public
  tables plus the RLS policies is enough; `runs` and `predictions` contain no
  column that shouldn't be public. A column added later *would* be public, so
  review additions to those two tables.
- **Revoking default grants: kept.** Supabase grants new tables to `anon` and
  `authenticated`, including `TRUNCATE`, which RLS does not restrict. Without
  the revoke, an anon client with the public key could truncate `sequences` or
  `run_artifacts`. This is a security measure, not decoration.
- **Deletion: `RESTRICT` instead of `SET NULL`.** `served_by_run_id` is `NOT NULL`
  and the database refuses to delete a run that predictions point to. This also
  removes the PostgreSQL 15 requirement of `SET NULL (column)`.
- **Dropped as unnecessary:** `input_hours`/`forecast_hours` constant columns,
  JSON size caps, regex checks on `data_version`, the extra
  `(data_version, split, station_ts)` index, an `input_hash` index, and the
  `bytea`/`format`/`sha256` artifact columns from the first draft. Add them back
  in a later migration if a measured need appears. The unique constraint on
  `(data_version, station_ts)` already provides an index for ordered corpus reads.

## Public-access boundary

Only the service role (the FastAPI server) can write; it bypasses RLS.

- RLS is enabled on all four tables.
- `anon` can `SELECT` all rows of `runs` and `predictions` (policies plus
  grants) and has no other privilege on anything. `sequences` and
  `run_artifacts` have no policy and no grants, so the anon key sees nothing
  there. `authenticated` has no access at all.
- **Everything in `runs` and `predictions` is publicly readable by anyone with
  the project URL and anon key**, both of which ship in the Streamlit app. That
  includes `hyperparameters`, `preprocessing`, `compute_budget`, `forecast`,
  `attention` and `input_hash`. **A hash does not make a forecast private**: it
  can be recomputed from a guessed input, and the forecast itself is stored in
  clear. Do not put hostnames, file paths, tokens, or personal data in the JSON
  columns, and do not store inputs users would expect to stay private.
- Model weights are only in `run_artifacts`, never in `runs`.

## Loading the corpus (`db/load.py`)

`db/load.py` reuses the verified preparation functions in
[scripts/inspect_jena.py](../scripts/inspect_jena.py) (it does not reimplement
them) and adds manifest verification, per-hour split assignment, the data-version
id and the upload.

**Dry-run (default).** Needs no credentials and no network, imports no database
library, writes nothing. From the repo root, with the ZIPs in `data/raw/mpi_roof/`:

```bash
python3 -m db.load
```

It verifies the 43 archives (file set, SHA-256, row counts) against
[jena-mpi-roof-checksums.txt](jena-mpi-roof-checksums.txt), then applies the
cleanup rules in [data design](data-design.md) and prints source counts,
duplicates and off-grid timestamps, valid/invalid hours, split boundaries,
per-split row counts and 72→24 window counts, and whether they match the
[dataset card](dataset-card.md). Any mismatch or conflicting duplicate exits
with status 2. Dry-run takes about 20 seconds.

Real local dry-run result (executed 2026-10-24 on this machine): 1,162,360 rows;
193,760 grid hours, 193,548 valid, 212 invalid; train 135,438 / validation
29,047 / test 29,063 rows; 134,678 / 28,762 / 28,873 windows; matches the
dataset card.

**Data version.** `jena-mpi-roof-h1-<16 hex>` is a hash of the per-file SHA-256s
plus every preparation setting (hour rule, 72/24, sentinel, split fractions,
`PREP_VERSION`). Changing the source files or any rule changes the id, so old and
new corpora can coexist under `UNIQUE (data_version, station_ts)`. Current value:
`jena-mpi-roof-h1-60138aac7d898b8c`.

**Upload (run by the user; see Status).** Prerequisites: `pip install supabase`
(pinned in [api/requirements.txt](../api/requirements.txt)), and a git-ignored
`.env` made from [.env.example](../.env.example) with the real project URL and
service-role key. Then:

```bash
python3 -m db.load --upload                 # default batches of 1000
python3 -m db.load --upload --batch-size 500
```

Behavior:

- Configuration is validated first (https URL, key present and not a
  placeholder) and values are never printed; error text redacts them.
- It reads the rows already stored for this `data_version` and compares them
  with the prepared data. Identical rows are skipped; only missing rows are
  inserted, with plain `INSERT` in batches (1–5000 rows), never an upsert.
- If a stored row for this data version differs in temperature (beyond 1e-9,
  which only absorbs float round-tripping) or split, or is not in the prepared
  data, it aborts with no changes. It never updates or deletes.
- An interrupted upload is resumed by running the same command again.
- Afterwards it checks the per-split row counts in the database.

`station_ts` is sent as a naive ISO string (`2020-01-01T01:00:00`) with no offset.

## Behavior tests (not yet run)

[db/validation/002_seq_sight_validation.sql](../db/validation/002_seq_sight_validation.sql)
is a **rollback-only** script: it inserts valid rows, then checks that bad rows
fail with the expected error code, and always ends with `ROLLBACK`. It covers
missing/null/non-numeric/zero/negative preprocessing values, NaN, infinite and
negative MAE, malformed forecast and attention shapes, baseline-with-attention
and attention-without-matrix, model type that disagrees with the run, deletion
of a referenced run, naive-timestamp preservation, and the anonymous role's
read-only access.

**Never run it against the real Supabase project.** It is meant for a
disposable local PostgreSQL that has `002` applied and the `anon`,
`authenticated` and `service_role` roles created. It refuses to start if any
table has rows. **It has not been executed yet, so the behavior of `002` is
still untested.**

## Read-only verification SQL

Run these after applying `002`. They only read catalogs.

**1. Tables and columns**

```sql
select table_name, ordinal_position, column_name, data_type, is_nullable, column_default
from information_schema.columns
where table_schema = 'public'
  and table_name in ('sequences', 'runs', 'run_artifacts', 'predictions')
order by table_name, ordinal_position;
```

Expected: only those four tables in `public` (no `datasets`, `sequence_metadata`);
`sequences.station_ts` is `timestamp without time zone`; the only nullable
column is `predictions.attention`.

**2. Constraints and foreign keys**

```sql
select conrelid::regclass as table_name, conname,
       case contype when 'p' then 'primary key' when 'f' then 'foreign key'
                    when 'u' then 'unique' when 'c' then 'check' end as kind,
       pg_get_constraintdef(oid) as definition
from pg_constraint
where connamespace = 'public'::regnamespace
  and conrelid::regclass::text in ('sequences', 'runs', 'run_artifacts', 'predictions')
order by conrelid::regclass::text, contype, conname;
```

Expected foreign keys: `run_artifacts.run_id → runs(id) ON DELETE CASCADE` and
`predictions (served_by_run_id, model_type) → runs (id, model_type) ON DELETE RESTRICT`.

**3. Indexes**

```sql
select tablename, indexname, indexdef
from pg_indexes
where schemaname = 'public'
  and tablename in ('sequences', 'runs', 'run_artifacts', 'predictions')
order by tablename, indexname;
```

**4. Row level security**

```sql
select c.relname, c.relrowsecurity as rls_enabled, c.relforcerowsecurity as rls_forced
from pg_class c
where c.relnamespace = 'public'::regnamespace
  and c.relname in ('sequences', 'runs', 'run_artifacts', 'predictions')
order by c.relname;
```

Expected: `rls_enabled = true` on all four (`rls_forced` false).

**5. Policies**

```sql
select tablename, policyname, roles, cmd, qual, with_check
from pg_policies
where schemaname = 'public'
order by tablename, policyname;
```

Expected: exactly two rows, `runs` and `predictions`, each `{anon}`, `SELECT`.

**6. Table grants**

```sql
select table_name, grantee, privilege_type
from information_schema.role_table_grants
where table_schema = 'public'
  and table_name in ('sequences', 'runs', 'run_artifacts', 'predictions')
  and grantee in ('anon', 'authenticated', 'service_role')
order by table_name, grantee, privilege_type;
```

Expected: `anon` has only `SELECT`, only on `runs` and `predictions`;
`authenticated` has no rows. (`service_role` appears through Supabase's default
privileges and bypasses RLS.)

**7. Column and sequence grants for anon**

```sql
select 'column' as kind, table_name || '.' || column_name as object, privilege_type
from information_schema.column_privileges
where table_schema = 'public' and grantee in ('anon', 'authenticated')
  and privilege_type <> 'SELECT'
union all
select 'sequence', object_name, privilege_type
from information_schema.role_usage_grants
where object_schema = 'public' and object_type = 'SEQUENCE'
  and grantee in ('anon', 'authenticated');
```

Expected: no rows (no write privilege on any column; no sequence access).

## Starter application is not compatible yet

The Attend-It code still targets the obsolete schema and will fail against a
database that only has `002`:

- [api/db.py](../api/db.py) uses `datasets`, `sequence_metadata`, and the old
  `runs` columns (`dataset_id`, `lr`, `accuracy`, `macro_f1`, …); `ping()`
  queries `datasets`, so `/healthz` would report the database unreachable. Its
  `run_artifacts.model_b64` usage still matches.
- [db/seed.py](../db/seed.py), [api/main.py](../api/main.py),
  [api/training.py](../api/training.py), [ui/app.py](../ui/app.py) and the tests
  (`tests/test_schema.py`, `tests/test_supabase_roundtrip.py`) assume the
  classification task.
- [.streamlit/secrets.toml.example](../.streamlit/secrets.toml.example)
  says the anon key reads only `runs`; it now also reads `predictions`.
- [render.yaml](../render.yaml) still names the service `attend-it-api`.

None of these have been changed. The loader `db/load.py` is new and independent of them.

## Configuration

Copy [.env.example](../.env.example) to `.env` for the backend (placeholders
only; `.env` and `secrets.toml` files are git-ignored). The service-role key is
server-only; the UI gets just the project URL and the public anon key.
