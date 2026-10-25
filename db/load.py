"""Prepare the hourly Jena corpus and (optionally) load it into `sequences`.

Dry-run is the default: it reads the local ZIP files, verifies them against the
checksum manifest, applies the documented preparation rules and prints a report.
It needs no credentials, makes no network requests and writes nothing.

    python3 -m db.load                      # dry-run (default)
    python3 -m db.load --upload             # later: load into Supabase

The preparation logic is NOT reimplemented here: it is imported from
scripts/inspect_jena.py, which produced the numbers in docs/dataset-card.md.
This module only adds manifest verification, split assignment per hour, the
data-version identifier, and the upload.

Preparation rules (docs/data-design.md): sort; collapse identical duplicates;
fail on conflicting duplicates; exclude and report off-grid records; hour H:00 is
the mean of the six on-grid readings stamped H-50 min .. H:00 and is invalid
unless all six are valid (no interpolation); splits are chronological, cut on the
FULL hourly grid at 70% / 85%; timestamps stay naive station-clock values.

--upload reads SUPABASE_URL and SUPABASE_SERVICE_KEY from a git-ignored .env
(or the environment), never prints them, inserts only missing rows in bounded
batches, and aborts instead of overwriting anything that differs.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional

REPO = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = REPO / "data" / "raw" / "mpi_roof"
DEFAULT_MANIFEST = REPO / "docs" / "jena-mpi-roof-checksums.txt"
DEFAULT_ENV = REPO / ".env"

# Bump when any preparation rule changes, so the data version changes with it.
PREP_VERSION = "1"
SPLIT_FRACTIONS = {"train": 0.70, "validation": 0.15, "test": 0.15}
MAX_BATCH = 5000
DEFAULT_BATCH = 1000
TEMP_TOLERANCE = 1e-9  # only absorbs float text round-tripping; real differences exceed it

# Figures recorded in docs/dataset-card.md, used only for a "matches" line.
DOCUMENTED = {
    "grid_hours": 193_760,
    "valid_hours": 193_548,
    "windows": {"train": 134_678, "validation": 28_762, "test": 28_873},
}


def _load_inspect():
    path = REPO / "scripts" / "inspect_jena.py"
    spec = importlib.util.spec_from_file_location("inspect_jena", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("inspect_jena", mod)
    spec.loader.exec_module(mod)
    return mod


ij = _load_inspect()


class LoaderError(Exception):
    """A verification or configuration failure; the message is safe to print."""


class UploadConflict(LoaderError):
    """The database already holds rows that differ from the prepared data."""


# ------------------------------------------------------------------ manifest

def read_manifest(path: Path) -> Dict[str, dict]:
    """{filename: {"sha256": str, "rows": int}} from the checksum manifest."""
    out: Dict[str, dict] = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 3 or len(parts[1]) != 64:
            raise LoaderError(f"malformed manifest line: {line[:80]!r}")
        out[parts[0]] = {"sha256": parts[1], "rows": int(parts[2])}
    if not out:
        raise LoaderError(f"manifest {path} lists no files")
    return out


def verify_files(raw_dir: Path, manifest: Dict[str, dict]) -> List[Path]:
    """Check file set and SHA-256 of every archive against the manifest."""
    files = sorted(Path(raw_dir).glob("mpi_roof_*.zip"))
    present = {p.name for p in files}
    missing, extra = sorted(set(manifest) - present), sorted(present - set(manifest))
    if missing or extra:
        raise LoaderError(f"files differ from manifest: missing={missing} not-in-manifest={extra}")
    bad = [p.name for p in files if ij.sha256(p) != manifest[p.name]["sha256"]]
    if bad:
        raise LoaderError(f"SHA-256 mismatch against manifest: {bad}")
    return files


def make_data_version(manifest: Dict[str, dict]) -> str:
    """Deterministic id from the source checksums plus every preparation setting."""
    settings = {
        "prep_version": PREP_VERSION,
        "input_hours": ij.INPUT_H,
        "target_hours": ij.TARGET_H,
        "hour_rule": "mean of six on-grid 10-min readings stamped H-50min..H:00, all required",
        "sentinels": sorted(ij.SENTINELS),
        "split_fractions": SPLIT_FRACTIONS,
        "split_basis": "full hourly grid, floor(0.70*N) and floor(0.85*N)",
    }
    canon = json.dumps(
        {"files": {k: manifest[k]["sha256"] for k in sorted(manifest)}, "settings": settings},
        sort_keys=True,
    )
    return "jena-mpi-roof-h1-" + hashlib.sha256(canon.encode()).hexdigest()[:16]


# --------------------------------------------------------------- preparation

@dataclass
class SplitInfo:
    name: str
    first_label: datetime
    last_label: datetime
    grid_hours: int
    valid_hours: int
    windows: int
    longest_run: int


@dataclass
class Prepared:
    data_version: str
    n_files: int
    n_records: int
    per_file_rows: Dict[str, int]
    unique_timestamps: int
    duplicate_extra_rows: int
    duplicate_cross_file: int
    duplicate_within_file: int
    off_grid: list
    invalid_temp: Counter
    first_hour: datetime
    last_hour: datetime
    grid_hours: int
    valid_hours: int
    splits: Dict[str, SplitInfo]
    rows: List[dict] = field(repr=False)

    @property
    def invalid_hours(self) -> int:
        return self.grid_hours - self.valid_hours

    @property
    def total_windows(self) -> int:
        return sum(s.windows for s in self.splits.values())


def assign_split(index: int, b1: int, b2: int) -> str:
    return "train" if index < b1 else "validation" if index < b2 else "test"


def prepare_records(records, data_version: str, per_file_rows: Dict[str, int], n_files: int) -> Prepared:
    """Apply the documented rules to parsed records (also used with synthetic data)."""
    cross, within, conflicting = ij.classify_duplicates(records)
    if conflicting:
        raise ij.ConflictError(
            f"{len(conflicting)} timestamps with conflicting temperatures, first: {conflicting[0]}")
    groups = ij.group_by_timestamp(records)
    temps = ij.collapse_duplicates(records)  # sorted; raises ConflictError on conflicts
    if not temps:
        raise LoaderError("no records")

    invalid = Counter()
    for v in temps.values():
        if v is None:
            invalid["unparseable"] += 1
        elif v != v or v in (float("inf"), float("-inf")):
            invalid["nonfinite"] += 1
        elif v in ij.SENTINELS:
            invalid["sentinel_-9999"] += 1

    off_grid = ij.grid_stats(list(temps))["off_grid"]
    valid, first_h, last_h = ij.hourly_series(temps)
    if not valid:
        raise LoaderError("no valid hours")
    hour = timedelta(hours=1)
    n = int((last_h - first_h) / hour) + 1
    b1, b2 = ij.split_bounds(n)

    rows = [
        {
            "data_version": data_version,
            "station_ts": h.isoformat(),  # naive: no offset, never converted
            "temperature_c": valid[h],
            "split": assign_split(int((h - first_h) / hour), b1, b2),
        }
        for h in sorted(valid)
    ]

    splits = {}
    for name, (a, b) in {"train": (0, b1), "validation": (b1, b2), "test": (b2, n)}.items():
        flags = [(first_h + i * hour) in valid for i in range(a, b)]
        windows, longest = ij.count_windows(flags)
        splits[name] = SplitInfo(name, first_h + a * hour, first_h + (b - 1) * hour,
                                 b - a, sum(flags), windows, longest)

    return Prepared(
        data_version=data_version, n_files=n_files, n_records=len(records),
        per_file_rows=per_file_rows, unique_timestamps=len(temps),
        duplicate_extra_rows=sum(len(g) - 1 for g in groups.values()),
        duplicate_cross_file=len(cross), duplicate_within_file=len(within),
        off_grid=off_grid, invalid_temp=invalid, first_hour=first_h, last_hour=last_h,
        grid_hours=n, valid_hours=len(valid), splits=splits, rows=rows)


def prepare(raw_dir: Path = DEFAULT_RAW_DIR, manifest_path: Path = DEFAULT_MANIFEST) -> Prepared:
    manifest = read_manifest(manifest_path)
    files = verify_files(raw_dir, manifest)
    records, _headers, bad_ts, per_file = ij.load_records(files)
    if bad_ts:
        raise LoaderError(f"{bad_ts} rows with unparseable timestamps")
    wrong = {k: (per_file[k], manifest[k]["rows"]) for k in manifest if per_file[k] != manifest[k]["rows"]}
    if wrong:
        raise LoaderError(f"row counts differ from manifest (found, expected): {wrong}")
    return prepare_records(records, make_data_version(manifest), per_file, len(files))


def batches(rows: List[dict], size: int) -> Iterator[List[dict]]:
    for i in range(0, len(rows), size):
        yield rows[i:i + size]


# -------------------------------------------------------------------- report

def report(p: Prepared, out: Callable[[str], None] = print) -> None:
    out(f"data_version: {p.data_version}")
    out(f"source: {p.n_files} archives (SHA-256 and row counts match the manifest), {p.n_records:,} data rows")
    out(f"  unique timestamps: {p.unique_timestamps:,}")
    out(f"  duplicate rows collapsed: {p.duplicate_extra_rows:,} "
        f"(timestamps duplicated across archives: {p.duplicate_cross_file}, within one archive: "
        f"{p.duplicate_within_file}); conflicting duplicates: 0")
    out(f"  off-grid timestamps excluded: {len(p.off_grid)} {[str(t) for t in p.off_grid]}")
    out(f"  invalid temperature values (unique timestamps): {dict(p.invalid_temp) or 'none'}")
    out(f"hourly grid {p.first_hour} .. {p.last_hour}: {p.grid_hours:,} hours")
    out(f"  valid hours: {p.valid_hours:,}   invalid hours (excluded, not filled): {p.invalid_hours:,}")
    out("splits (hour-end labels, naive station time; cut on the full grid at 70% / 85%):")
    for s in p.splits.values():
        out(f"  {s.name:<10} {s.first_label} .. {s.last_label}  grid_hours={s.grid_hours:,} "
            f"rows={s.valid_hours:,} windows(72->24)={s.windows:,} longest_run={s.longest_run:,}")
    out(f"rows to upload: {len(p.rows):,}   total valid 72->24 windows: {p.total_windows:,}")
    diffs = []
    if p.grid_hours != DOCUMENTED["grid_hours"]:
        diffs.append(f"grid_hours {p.grid_hours} != {DOCUMENTED['grid_hours']}")
    if p.valid_hours != DOCUMENTED["valid_hours"]:
        diffs.append(f"valid_hours {p.valid_hours} != {DOCUMENTED['valid_hours']}")
    for name, w in DOCUMENTED["windows"].items():
        if p.splits[name].windows != w:
            diffs.append(f"{name} windows {p.splits[name].windows} != {w}")
    out("matches docs/dataset-card.md: " + ("yes" if not diffs else "NO - " + "; ".join(diffs)))


# -------------------------------------------------------------------- config

@dataclass(frozen=True)
class Config:
    url: str
    key: str = field(repr=False)

    def secrets(self) -> List[str]:
        return [self.url, self.key]


def parse_env_file(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not Path(path).is_file():
        return out
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
            v = v[1:-1]
        out[k.strip()] = v
    return out


def load_config(env_file: Path = DEFAULT_ENV, environ: Optional[dict] = None) -> Config:
    """Validate credentials without ever echoing their values."""
    environ = os.environ if environ is None else environ
    file_vals = parse_env_file(env_file)
    url = environ.get("SUPABASE_URL") or file_vals.get("SUPABASE_URL", "")
    key = environ.get("SUPABASE_SERVICE_KEY") or file_vals.get("SUPABASE_SERVICE_KEY", "")
    problems = []
    if not url:
        problems.append("SUPABASE_URL is not set")
    elif not (url.startswith("https://") and "." in url[8:] and "your-project-ref" not in url):
        problems.append("SUPABASE_URL must be an https:// project URL, not the placeholder")
    if not key:
        problems.append("SUPABASE_SERVICE_KEY is not set")
    elif key.startswith("your-") or any(c.isspace() for c in key) or len(key) < 20:
        problems.append("SUPABASE_SERVICE_KEY looks like a placeholder or is malformed")
    if problems:
        raise LoaderError("configuration invalid (" + str(env_file) + "): " + "; ".join(problems))
    return Config(url=url.rstrip("/"), key=key)


# -------------------------------------------------------------------- upload

class SupabaseStore:
    """Thin wrapper over the supabase client for the `sequences` table."""

    TABLE = "sequences"
    PAGE = 1000

    def __init__(self, client):
        self.client = client

    @classmethod
    def connect(cls, config: Config) -> "SupabaseStore":
        from supabase import create_client  # imported only for --upload
        return cls(create_client(config.url, config.key))

    def fetch_existing(self, data_version: str) -> Iterator[dict]:
        start = 0
        while True:
            rows = (self.client.table(self.TABLE).select("station_ts,temperature_c,split")
                    .eq("data_version", data_version).order("station_ts")
                    .range(start, start + self.PAGE - 1).execute().data)
            yield from rows
            if len(rows) < self.PAGE:
                return
            start += self.PAGE

    def insert(self, rows: List[dict]) -> None:
        self.client.table(self.TABLE).insert(rows).execute()  # plain insert: never overwrites

    def count(self, data_version: str, split: str) -> int:
        resp = (self.client.table(self.TABLE).select("id", count="exact")
                .eq("data_version", data_version).eq("split", split).limit(1).execute())
        return resp.count


def _ts(value) -> datetime:
    return value if isinstance(value, datetime) else datetime.fromisoformat(str(value))


def plan_resume(existing: Iterable[dict], rows: List[dict]) -> List[dict]:
    """Rows still to insert. Raises UploadConflict if anything stored differs."""
    local = {_ts(r["station_ts"]): r for r in rows}
    stored = {}
    for e in existing:
        stored[_ts(e["station_ts"])] = e
    unexpected = [t for t in stored if t not in local]
    different = [t for t, e in stored.items() if t in local and (
        e["split"] != local[t]["split"]
        or abs(float(e["temperature_c"]) - local[t]["temperature_c"]) > TEMP_TOLERANCE)]
    if unexpected or different:
        raise UploadConflict(
            f"database already holds rows for this data_version that do not match the prepared data "
            f"(unexpected timestamps: {len(unexpected)}, differing values/splits: {len(different)}); "
            f"nothing was changed. Resolve manually; the loader never overwrites or deletes.")
    return [r for r in rows if _ts(r["station_ts"]) not in stored]


def upload(p: Prepared, store, batch_size: int = DEFAULT_BATCH,
           out: Callable[[str], None] = print) -> dict:
    if not 1 <= batch_size <= MAX_BATCH:
        raise LoaderError(f"--batch-size must be between 1 and {MAX_BATCH}")
    todo = plan_resume(store.fetch_existing(p.data_version), p.rows)
    skipped = len(p.rows) - len(todo)
    out(f"already present and identical: {skipped:,}; to insert: {len(todo):,}")
    done = 0
    for i, chunk in enumerate(batches(todo, batch_size), 1):
        store.insert(chunk)
        done += len(chunk)
        out(f"  batch {i}: inserted {done:,}/{len(todo):,}")
    for name, s in p.splits.items():
        got = store.count(p.data_version, name)
        if got != s.valid_hours:
            raise LoaderError(f"verification failed for split {name}: database has {got}, expected {s.valid_hours}")
    out("verified per-split row counts in the database match the prepared data")
    return {"inserted": done, "skipped": skipped}


# ----------------------------------------------------------------------- cli

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Prepare (dry-run, default) or upload the Jena hourly corpus.")
    ap.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--upload", action="store_true",
                    help="write to Supabase (needs .env credentials); default is a no-network dry-run")
    ap.add_argument("--env-file", type=Path, default=DEFAULT_ENV)
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config = None
    try:
        if args.upload:  # validate before doing any work, without printing secrets
            if not 1 <= args.batch_size <= MAX_BATCH:
                raise LoaderError(f"--batch-size must be between 1 and {MAX_BATCH}")
            config = load_config(args.env_file)
            print("configuration valid (credentials not shown)")
        prepared = prepare(args.raw_dir, args.manifest)
        report(prepared)
        if not args.upload:
            print("DRY-RUN: no credentials used, no network, nothing written. Use --upload to load.")
            return 0
        upload(prepared, SupabaseStore.connect(config), args.batch_size)
        return 0
    except (LoaderError, ij.ConflictError) as e:
        print(f"ERROR: {_redact(str(e), config)}", file=sys.stderr)
    except Exception as e:  # e.g. network/client errors during upload
        print(f"ERROR: {type(e).__name__}: {_redact(str(e), config)}", file=sys.stderr)
    return 2


def _redact(text: str, config: Optional[Config]) -> str:
    for s in (config.secrets() if config else []):
        text = text.replace(s, "***")
    return text


if __name__ == "__main__":
    sys.exit(main())
