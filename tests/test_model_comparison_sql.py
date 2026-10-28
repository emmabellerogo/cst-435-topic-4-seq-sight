"""Static checks on docs/model-comparison.sql. The SQL is PostgreSQL-specific and is NOT executed here."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SQL = (REPO / "docs" / "model-comparison.sql").read_text(encoding="utf-8")
CODE = "\n".join(re.sub(r"--.*$", "", line) for line in SQL.splitlines())
DATA_VERSION = "jena-mpi-roof-h1-60138aac7d898b8c"


def test_sql_is_read_only():
    words = set(re.findall(r"[a-z_]+", CODE.lower()))
    forbidden = {"insert", "update", "delete", "drop", "alter", "truncate", "create", "grant", "revoke",
                 "copy", "call", "do", "execute", "vacuum", "set"}
    assert not (words & forbidden), words & forbidden
    statements = [s.strip() for s in CODE.split(";") if s.strip()]
    assert len(statements) == 3 and all(re.match(r"(with|select)\b", s, re.I) for s in statements)


def test_sql_scopes_to_the_documented_version_and_six_distinct_hashes():
    assert CODE.count(f"'{DATA_VERSION}'") == 3
    pairs = re.findall(r"\('(lstm|attention)', (\d), '([0-9a-f]{64})'\)", CODE)
    per_statement = len(pairs) // 3
    assert per_statement == 6 and pairs[:6] == pairs[6:12] == pairs[12:]
    assert len({h for _, _, h in pairs[:6]}) == 6
    assert sorted((t, int(s)) for t, s, _ in pairs[:6]) == [(t, s) for t in ("attention", "lstm") for s in range(3)]


def test_sql_reports_required_statistics():
    low = CODE.lower()
    for needle in ("count(*)", "avg(test_mae_c)", "stddev_samp(test_mae_c)", "group by model_type", "seed"):
        assert needle in low


def test_sql_hashes_match_the_local_test_evaluation_when_present():
    f = REPO / "artifacts" / "epoch50" / "final" / "test_eval.json"
    if not f.is_file():
        pytest.skip("artifacts/ is git-ignored; local test_eval.json not present")
    models = json.loads(f.read_text())["models"]
    pairs = {(t, int(s)): h for t, s, h in re.findall(r"\('(lstm|attention)', (\d), '([0-9a-f]{64})'\)", CODE)}
    assert pairs == {(m["model_type"], m["seed"]): m["artifact_sha256"] for m in models.values()}
