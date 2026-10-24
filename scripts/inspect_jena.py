"""Inspect the raw Jena (MPI-BGC Beutenberg `mpi_roof`) ZIP files; stdlib only.

Reads CSV members straight out of the ZIPs as text (nothing in the download is
executed) and prints the statistics recorded in docs/dataset-card.md.

    python3 -I scripts/inspect_jena.py [data/raw/mpi_roof] [--write-manifest docs/jena-mpi-roof-checksums.txt]

Cleanup rules (also documented in docs/data-design.md):
  1. sort records by timestamp;
  2. collapse duplicate timestamps only when their temperatures agree;
  3. fail (ConflictError) when duplicates disagree;
  4. off-grid timestamps (not on a :00/:10/... minute with zero seconds) are
     excluded from the hourly aggregation, never snapped, and reported;
  5. a missing/non-finite/-9999 temperature makes its hour invalid; no filling.
"""
import csv
import hashlib
import io
import math
import sys
import zipfile
from collections import Counter, defaultdict, namedtuple
from datetime import datetime, timedelta
from pathlib import Path

STEP = timedelta(minutes=10)
INPUT_H, TARGET_H = 72, 24
SENTINELS = {-9999.0}

# One raw CSV data row. `temp` is None when unparseable; `order` is the position
# in file-concatenation order (used to report out-of-order records).
Record = namedtuple("Record", "ts temp swdr file order")


class ConflictError(ValueError):
    """Duplicate timestamps whose temperatures disagree."""


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_ts(text):
    """German date format; the files carry seconds even though the docs' example does not."""
    for fmt in ("%d.%m.%Y %H:%M:%S", "%d.%m.%Y %H:%M"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    return None


def is_valid_temp(v):
    return v is not None and math.isfinite(v) and v not in SENTINELS


def on_grid(ts):
    return ts.second == 0 and ts.microsecond == 0 and ts.minute % 10 == 0


# ---------------------------------------------------------------- reading

def read_zip(path):
    """Yield (member, header, rows) per CSV member; rows are lists of strings."""
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if name.endswith("/") or not name.lower().endswith((".csv", ".txt")):
                print(f"  NOTE non-CSV member ignored in {path.name}: {name}")
                continue
            text = io.TextIOWrapper(z.open(name), encoding="utf-8-sig", errors="replace", newline="")
            r = csv.reader(text)
            yield name, next(r), [row for row in r if row]


def load_records(files):
    """Return (records, header_layouts, unparseable_timestamps, per_file_rows)."""
    records, headers, bad_ts, per_file = [], Counter(), 0, {}
    for p in files:
        n0 = len(records)
        for _member, header, rows in read_zip(p):
            headers[tuple(header)] += 1
            ti = header.index("T (degC)")
            si = next((i for i, c in enumerate(header) if c.startswith("SWDR")), None)
            for row in rows:
                ts = parse_ts(row[0].strip())
                if ts is None:
                    bad_ts += 1
                    continue
                try:
                    temp = float(row[ti])
                except ValueError:
                    temp = None
                try:
                    sw = float(row[si]) if si is not None else None
                except ValueError:
                    sw = None
                records.append(Record(ts, temp, sw, p.name, len(records)))
        per_file[p.name] = len(records) - n0
    return records, headers, bad_ts, per_file


# --------------------------------------------------------------- cleaning

def group_by_timestamp(records):
    groups = defaultdict(list)
    for r in records:
        groups[r.ts].append(r)
    return groups


def classify_duplicates(records):
    """Duplicate-timestamp groups -> (cross_file, within_file, conflicting) lists of ts.

    cross_file: copies come from different archives (file overlap);
    within_file: all copies are in one archive (repeated rows);
    conflicting: copies whose temperatures disagree (reported under either kind too).
    """
    cross, within, conflicting = [], [], []
    for ts, grp in sorted(group_by_timestamp(records).items()):
        if len(grp) < 2:
            continue
        (cross if len({r.file for r in grp}) > 1 else within).append(ts)
        if len({r.temp for r in grp}) > 1:
            conflicting.append(ts)
    return cross, within, conflicting


def collapse_duplicates(records):
    """Sorted {ts: temp}; duplicates collapse only if temperatures agree, else ConflictError."""
    out = {}
    for ts, grp in sorted(group_by_timestamp(records).items()):
        temps = {r.temp for r in grp}
        if len(temps) > 1:
            raise ConflictError(f"{ts}: conflicting temperatures {sorted(temps, key=str)}")
        out[ts] = grp[0].temp
    return out


# ------------------------------------------------------------------- grid

def grid_stats(timestamps):
    """Slot accounting on the 10-minute grid.

    Only unique ON-GRID timestamps count as present. Off-grid timestamps are
    returned separately and never count toward present slots.
    """
    unique = set(timestamps)
    grid = sorted(t for t in unique if on_grid(t))
    off = sorted(t for t in unique if not on_grid(t))
    if not grid:
        return {"expected": 0, "present": 0, "absent": 0, "off_grid": off, "first": None, "last": None}
    expected = int((grid[-1] - grid[0]) / STEP) + 1
    return {"expected": expected, "present": len(grid), "absent": expected - len(grid),
            "off_grid": off, "first": grid[0], "last": grid[-1]}


def grid_gaps(timestamps):
    """(previous, next, delta) for consecutive ON-GRID timestamps more than one step apart."""
    grid = sorted({t for t in timestamps if on_grid(t)})
    return [(a, b, b - a) for a, b in zip(grid, grid[1:]) if b - a != STEP]


# ----------------------------------------------------------------- hourly

def hourly_series(temps):
    """Hourly means from {ts: temp}; returns (valid {hour: mean}, first_hour, last_hour).

    Hour label H:00 = mean of the six on-grid records stamped H-50min..H:00
    (timestamps are end-of-period). An hour is valid only if all six exist and are
    valid temperatures; otherwise it is absent from `valid` (never filled).
    The first hour is the first label whose six records could all exist.
    """
    grid = sorted(t for t in temps if on_grid(t))
    if not grid:
        return {}, None, None
    first = grid[0] + timedelta(minutes=50)
    first_h = first.replace(minute=0, second=0) + (timedelta(hours=1) if first.minute else timedelta(0))
    last_h = grid[-1].replace(minute=0, second=0)
    valid, h = {}, first_h
    while h <= last_h:
        vals = [temps.get(h - timedelta(minutes=10 * k)) for k in range(6)]
        if all(is_valid_temp(v) for v in vals):
            valid[h] = sum(vals) / 6
        h += timedelta(hours=1)
    return valid, first_h, last_h


def split_bounds(n_hours, train=0.70, val=0.15):
    return int(n_hours * train), int(n_hours * (train + val))


def count_windows(valid_flags, need=INPUT_H + TARGET_H):
    """Stride-1 windows of `need` consecutive True flags (list of bool for one partition)."""
    windows = run = longest = 0
    for ok in valid_flags:
        if ok:
            run += 1
        else:
            windows += max(0, run - need + 1)
            longest, run = max(longest, run), 0
    windows += max(0, run - need + 1)
    return windows, max(longest, run)


# --------------------------------------------------------------- reporting

def last_sunday(year, month):
    d = (datetime(year, month + 1, 1) if month < 12 else datetime(year + 1, 1, 1)) - timedelta(days=1)
    while d.weekday() != 6:
        d -= timedelta(days=1)
    return d.date()


def write_manifest(path, files, per_file_rows):
    lines = ["# filename  sha256  data_rows   (generated by scripts/inspect_jena.py from the local files)"]
    for p in files:
        lines.append(f"{p.name}  {sha256(p)}  {per_file_rows[p.name]}")
    Path(path).write_text("\n".join(lines) + "\n")


def main(argv):
    manifest = argv[argv.index("--write-manifest") + 1] if "--write-manifest" in argv else None
    args = [a for a in argv if not a.startswith("--") and a != manifest]
    raw = Path(args[0] if args else "data/raw/mpi_roof")
    files = sorted(raw.glob("mpi_roof_*.zip"))
    print(f"files: {len(files)}")
    records, headers, bad_ts, per_file = load_records(files)
    if manifest:
        write_manifest(manifest, files, per_file)
        print("manifest written:", manifest)
    print("distinct header layouts:", len(headers))
    for h, c in headers.items():
        print(f"  x{c}: {len(h)} cols; first 3: {h[:3]}")
    print("unparseable timestamps:", bad_ts)
    print("total data rows:", len(records))

    nonmono = [(a, b) for a, b in zip(records, records[1:]) if b.ts <= a.ts]
    print("non-increasing steps in file order:", len(nonmono))
    for a, b in nonmono:
        print(f"   {a.file} {a.ts} -> {b.file} {b.ts}")

    # ---- raw duplicates
    cross, within, conflicting = classify_duplicates(records)
    groups = group_by_timestamp(records)
    extra = sum(len(g) - 1 for g in groups.values())
    print(f"duplicate rows beyond first occurrence: {extra}")
    print(f"  timestamps duplicated across archives: {len(cross)} "
          f"({sum(len(groups[t]) - 1 for t in cross)} extra rows)")
    print(f"  timestamps duplicated within one archive: {len(within)} "
          f"({sum(len(groups[t]) - 1 for t in within)} extra rows)")
    print(f"  timestamps with CONFLICTING temperatures: {len(conflicting)}")
    by_day = Counter(t.date() for t in within)
    print("  within-archive duplicate days (top 10):", by_day.most_common(10))
    temps = collapse_duplicates(records)  # raises ConflictError on conflicts
    stamps = sorted(temps)
    print("unique timestamps:", len(stamps), "range:", stamps[0], "->", stamps[-1])

    # ---- temperature validity (unique timestamps)
    bad = Counter()
    for v in temps.values():
        if v is None:
            bad["unparseable"] += 1
        elif not math.isfinite(v):
            bad["nonfinite"] += 1
        elif v in SENTINELS:
            bad["sentinel_-9999"] += 1
    good = [v for v in temps.values() if is_valid_temp(v)]
    print("T invalid:", dict(bad), "| finite range:", f"{min(good):.2f}..{max(good):.2f}", "| outside -40..45:",
          sum(1 for v in good if v < -40 or v > 45))
    sent_ts = sorted(t for t, v in temps.items() if v in SENTINELS)
    if sent_ts:
        print("  sentinel span:", sent_ts[0], "->", sent_ts[-1])

    # ---- grid accounting (unique on-grid timestamps only)
    g = grid_stats(stamps)
    print(f"on-grid 10-min slots: expected={g['expected']} present={g['present']} absent={g['absent']} "
          f"(grid {g['first']} -> {g['last']})")
    print(f"off-grid timestamps (reported separately, excluded from hourly aggregation): {len(g['off_grid'])}")
    for t in g["off_grid"]:
        near = t.replace(second=0, minute=t.minute - t.minute % 10)
        print(f"   {t}  (its floor slot {near} on-grid present: {near in temps})")
    gaps = grid_gaps(stamps)
    print("on-grid gaps (>10 min):", len(gaps), "| largest:")
    for a, b, d in sorted(gaps, key=lambda x: x[2], reverse=True)[:6]:
        print("  ", a, "->", b, d)

    # ---- clock-behaviour evidence from RAW records (not deduplicated)
    print("raw records stamped 02:00-02:50 on last-Sunday Mar/Oct (raw rows, unique timestamps; 6/6 = continuous):")
    for y in range(stamps[0].year, stamps[-1].year + 1):
        row = []
        for m in (3, 10):
            day = last_sunday(y, m)
            if not (stamps[0].date() < day < stamps[-1].date()):
                row.append(f"{day.isoformat()}: outside data range")
                continue
            sel = [r.ts for r in records if r.ts.date() == day and r.ts.hour == 2]
            row.append(f"{day.isoformat()}: {len(sel)}/{len(set(sel))}")
        print("  ", y, "Mar", row[0], "| Oct", row[1])
    change_days = {last_sunday(y, m) for y in range(stamps[0].year, stamps[-1].year + 1) for m in (3, 10)}
    dup_on_change_days = [t for t in cross + within if t.date() in change_days]
    print("duplicated timestamps falling on a last-Sunday Mar/Oct day:", dup_on_change_days or "none")
    swdr = defaultdict(list)
    for r in records:
        if r.swdr is not None and r.swdr != -9999.0 and r.ts.minute == 0 and r.ts.second == 0 \
                and r.ts.month in (1, 6, 7, 12):
            swdr[(r.ts.month in (6, 7), r.ts.hour)].append(r.swdr)
    for summer in (False, True):
        means = {h: sum(v) / len(v) for (s, h), v in swdr.items() if s == summer}
        peak = max(means, key=means.get)
        print(f"peak mean SWDR hour label ({'Jun/Jul' if summer else 'Dec/Jan'}): {peak}:00 ({means[peak]:.0f} W/m2)")

    # ---- hourly series, splits, windows
    valid, first_h, last_h = hourly_series(temps)
    n = int((last_h - first_h) / timedelta(hours=1)) + 1
    print(f"hourly grid {first_h} -> {last_h}: {n} hours; valid={len(valid)}; invalid={n - len(valid)}")
    b1, b2 = split_bounds(n)
    for name, (a, b) in {"train": (0, b1), "validation": (b1, b2), "test": (b2, n)}.items():
        hrs = [first_h + timedelta(hours=i) for i in range(a, b)]
        flags = [h in valid for h in hrs]
        w, longest = count_windows(flags)
        print(f"  {name}: {hrs[0]} .. {hrs[-1]} grid_hours={b - a} valid_hours={sum(flags)} "
              f"windows={w} longest_run={longest}")


if __name__ == "__main__":
    main(sys.argv[1:])
