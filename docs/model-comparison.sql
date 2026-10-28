-- Seq Sight: README model comparison from PERSISTED runs. READ-ONLY (select statements only).
--
-- Not executed by the repository's tools and no output is recorded here. Run it yourself in the
-- Supabase SQL editor (paste ONE statement at a time; the editor shows only the last result) or
-- with psql. Fill the README table from what it returns.
--
-- Scope: runs of data_version 'jena-mpi-roof-h1-60138aac7d898b8c' whose
-- hyperparameters.artifact_sha256 is one of the six artifacts in artifacts/epoch50/final (hashes as
-- recorded in the local test_eval.json). model_type and seed must also match the expected pair, so a
-- run registered under the wrong type or seed is excluded rather than averaged in.
-- test_mae_c is the one-time final test evaluation; this query neither selects nor ranks models.
-- Sample standard deviation (n - 1) is NULL when a model has a single run.

-- ===== 1. Per-model summary (README table) =================================================
with expected (model_type, seed, artifact_sha256) as (
  values
    ('lstm', 0, 'de0af50530a4a4d9eddf33fa3ccfd492dd467268fd95e62816a8277b5115aa69'),
    ('lstm', 1, 'a628f4361a6471c4db9416d4227ca203baeb59e736ca81495fc9a30df0ba2109'),
    ('lstm', 2, 'b731633de1406f52b4c093a886ebc57e225f8ba909c19ccc430dcd3a859a4acc'),
    ('attention', 0, '82c51854a590124317c76dc973d03bdd68a0ebe799a347abebd4669ad7048234'),
    ('attention', 1, '6e5a88f34c70184cd325d8ad492fd79e4f21156b2450525100c7f33de4e906c9'),
    ('attention', 2, '8170cfcf232aecd115476e620ad062ab64c53b72a7b393a98c54262de300eb70')
),
matched as (                      -- earliest run per registered artifact hash (duplicates are flagged by query 3)
  select distinct on (r.hyperparameters ->> 'artifact_sha256')
         r.id, r.model_type, r.seed, r.val_mae_c, r.test_mae_c, r.created_at,
         r.hyperparameters ->> 'artifact_sha256' as artifact_sha256
  from runs r
  join expected e
    on  r.hyperparameters ->> 'artifact_sha256' = e.artifact_sha256
   and  r.model_type = e.model_type
   and  r.seed       = e.seed
  where r.data_version = 'jena-mpi-roof-h1-60138aac7d898b8c'
  order by r.hyperparameters ->> 'artifact_sha256', r.id
)
select model_type,
       count(*)                          as runs,
       avg(test_mae_c)                   as mean_test_mae_c,
       stddev_samp(test_mae_c)           as sd_test_mae_c,
       avg(val_mae_c)                    as mean_val_mae_c,
       stddev_samp(val_mae_c)            as sd_val_mae_c
from matched
group by model_type
order by model_type;

-- ===== 2. Per-seed details =================================================================
with expected (model_type, seed, artifact_sha256) as (
  values
    ('lstm', 0, 'de0af50530a4a4d9eddf33fa3ccfd492dd467268fd95e62816a8277b5115aa69'),
    ('lstm', 1, 'a628f4361a6471c4db9416d4227ca203baeb59e736ca81495fc9a30df0ba2109'),
    ('lstm', 2, 'b731633de1406f52b4c093a886ebc57e225f8ba909c19ccc430dcd3a859a4acc'),
    ('attention', 0, '82c51854a590124317c76dc973d03bdd68a0ebe799a347abebd4669ad7048234'),
    ('attention', 1, '6e5a88f34c70184cd325d8ad492fd79e4f21156b2450525100c7f33de4e906c9'),
    ('attention', 2, '8170cfcf232aecd115476e620ad062ab64c53b72a7b393a98c54262de300eb70')
),
matched as (                      -- earliest run per registered artifact hash (duplicates are flagged by query 3)
  select distinct on (r.hyperparameters ->> 'artifact_sha256')
         r.id, r.model_type, r.seed, r.val_mae_c, r.test_mae_c, r.created_at,
         r.hyperparameters ->> 'artifact_sha256' as artifact_sha256
  from runs r
  join expected e
    on  r.hyperparameters ->> 'artifact_sha256' = e.artifact_sha256
   and  r.model_type = e.model_type
   and  r.seed       = e.seed
  where r.data_version = 'jena-mpi-roof-h1-60138aac7d898b8c'
  order by r.hyperparameters ->> 'artifact_sha256', r.id
)
select model_type, seed, id as run_id, test_mae_c, val_mae_c,
       left(artifact_sha256, 12) as artifact_sha256_prefix, created_at
from matched
order by model_type, seed;

-- ===== 3. Completeness / duplicate check (expect 6 rows, runs_found = 1 for each) ==========
with expected (model_type, seed, artifact_sha256) as (
  values
    ('lstm', 0, 'de0af50530a4a4d9eddf33fa3ccfd492dd467268fd95e62816a8277b5115aa69'),
    ('lstm', 1, 'a628f4361a6471c4db9416d4227ca203baeb59e736ca81495fc9a30df0ba2109'),
    ('lstm', 2, 'b731633de1406f52b4c093a886ebc57e225f8ba909c19ccc430dcd3a859a4acc'),
    ('attention', 0, '82c51854a590124317c76dc973d03bdd68a0ebe799a347abebd4669ad7048234'),
    ('attention', 1, '6e5a88f34c70184cd325d8ad492fd79e4f21156b2450525100c7f33de4e906c9'),
    ('attention', 2, '8170cfcf232aecd115476e620ad062ab64c53b72a7b393a98c54262de300eb70')
)
select e.model_type, e.seed, left(e.artifact_sha256, 12) as artifact_sha256_prefix,
       count(r.id)                  as runs_found,     -- 0 = not registered / wrong type, seed or data version; >1 = duplicate
       array_agg(r.id order by r.id) filter (where r.id is not null) as run_ids
from expected e
left join runs r
  on  r.hyperparameters ->> 'artifact_sha256' = e.artifact_sha256
 and  r.model_type = e.model_type
 and  r.seed       = e.seed
 and  r.data_version = 'jena-mpi-roof-h1-60138aac7d898b8c'
group by e.model_type, e.seed, e.artifact_sha256
order by e.model_type, e.seed;
