# Dataset Card — Jena Weather Station Data (Beutenberg `mpi_roof`)

Back to [README](../README.md) · See also [data design](data-design.md).

Facts are labelled by origin:

- **[Source]** stated in the provider's own documentation.
- **[Measured]** observed by us on the files we downloaded, using
  [scripts/inspect_jena.py](../scripts/inspect_jena.py) (stdlib only).
- **[Undocumented]** not stated by the provider; we do not assume it.

## Source

Provider: Max Planck Institute for Biogeochemistry (MPI-BGC), Jena. The
station-page contact listed by the provider is Tarek El-Madany.

| What | Link |
|------|------|
| Weather station landing page | <https://www.bgc-jena.mpg.de/wetter/> |
| Download page (files + terms of use) | <https://www.bgc-jena.mpg.de/wetter/weather_data.html> |
| Station documentation (English PDF) | <https://www.bgc-jena.mpg.de/wetter/Weatherstation.pdf> |
| Example file | <https://www.bgc-jena.mpg.de/wetter/data/mpi_roof_2009a.zip> (same pattern for all files) |

We use the **Beutenberg campus roof station, file prefix `mpi_roof`**. The
provider also publishes a Saaleaue station (`mpi_saale`) and soil beds
(`MPI_Soil`); we do not use them. We did not verify any other copy or mirror of
the data, and we have not confirmed that this is the same subset as other
"Jena Climate" copies found elsewhere.

## Verified from the source documentation

- **[Source]** Files are ZIP archives of CSV files, one per half-year until 2023
  and one per full year from 2024. The download page lists `mpi_roof` files from
  `mpi_roof_2003b.zip` (24.11.2003 – 31.12.2003) through `mpi_roof_2025.zip`, plus
  a rolling `mpi_roof.zip` ("01.01.2026 up to today").
- **[Source]** Sensors are scanned every 10 seconds; the datalogger stores
  **10-minute averages, totals and maxima**.
- **[Source]** CSV conventions: comma-separated, dot decimal, German
  `dd.mm.yyyy` dates. The header column for time is labelled `(MEZ)`.
- **[Source]** "The timestamp represents the end of the averaging period."
- **[Source]** `T` is **air temperature in °C** (header `T (degC)`); other
  columns include pressure, potential temperature, dew point, relative
  humidity, wind, rain, radiation, CO2.
- **[Source]** The logger clock is synchronised to a time server via the
  collecting computer.
- **[Source]** The download page states the terms of use as **Creative Commons
  CC-BY-4.0** ("Terms of Use (as per Creative Commons CC-BY-4.0)"). Using it
  requires attribution to the provider. The page gives no specific citation
  format.

## Retrieval record

- Retrieved: **2026-10-24**, from the URLs above, 43 `mpi_roof_*.zip` files
  (`2003b`, `2004a` … `2023b`, `2024`, `2025`). The rolling current-year
  `mpi_roof.zip` was **deliberately not used** because it changes daily.
- Stored in `data/raw/mpi_roof/` (git-ignored; the corpus is not committed).
- Per-file manifest (filename, full SHA-256, data-row count on every line):
  [jena-mpi-roof-checksums.txt](jena-mpi-roof-checksums.txt), generated from the
  local files by `scripts/inspect_jena.py --write-manifest`. We independently
  re-checked all 43 lines with `shasum -a 256` and `unzip -p … | wc` row counts;
  row counts sum to 1,162,360.
- Digest of the full checksum list (`shasum -a 256 *.zip | shasum -a 256`):
  `0e3d8cd57dbc6f4ad85c3fffd5a97de318da77d03bc2c4a1dc2434e026da599e`
- All 43 archives passed `unzip -t`. The scripts read CSV members as text; no
  code from the download was executed.
- Provider documentation snapshots (also local only):
  `Weatherstation.pdf` SHA-256 `d8e9e728…14a08e2`,
  `weather_data.html` SHA-256 `66e62b59…8846a9f`. The PDF's embedded creation
  date is 2025-02-05.

## Measured on the downloaded files

| Property | Result |
|----------|--------|
| Rows (all files) | 1,162,360 |
| Unique timestamps | 1,161,882 |
| First / last timestamp | 2003-11-24 16:00:00 / 2026-01-01 00:00:00 |
| Columns | 21 in 9 files, 22 in 34 files; `Date Time`, `p (mbar)`, `T (degC)` are the first three in every file |
| Timestamp format | `dd.mm.yyyy HH:MM:SS` (the provider's example omits seconds) |
| Header encoding | Not valid UTF-8 (non-ASCII bytes in unit labels); we decode with replacement and match columns by ASCII names |
| Sampling | 10 minutes, on the grid `:00, :10, …, :50` |
| Off-grid timestamps | 2 unique timestamps: `2004-12-07 02:00:30` and `2005-01-28 20:10:20`. Their on-grid slots (`02:00:00`, `20:10:00`) are also present as separate records |
| Duplicate rows | 478 extra rows, none with conflicting temperature. **144** are a one-day overlap between `mpi_roof_2010a` and `mpi_roof_2010b` (archives share 2010-07-01 00:10 – 2010-07-02 00:00). **334** are repeated rows inside one archive, concentrated on 2014-03-20/21 (183), 2023-10-20/21 (143) and a few single dates in 2020 and 2022 |
| Out-of-order records | 8 places where a timestamp does not increase (`2010a`→`2010b`, `2014a`, `2020a`, `2022a` ×3, `2022b`, `2023b`); 4 of them are an immediate repeat of the same timestamp, 3 step back by 20 minutes to 1 day, and 1 is the 2010 archive overlap |
| `T` unparseable / NaN / infinite | 0 |
| `T = -9999` sentinel | **548** records, all in one episode: 2007-05-17 15:10 – 2007-05-21 10:20 |
| `T` range (excluding sentinel) | −23.01 °C … 37.95 °C; none outside −40…45 °C |
| On-grid 10-minute slots | Expected 1,162,561 (first to last on-grid timestamp); **present 1,161,880** (unique, on-grid only); **absent 681**. Off-grid timestamps are not counted as present |
| Gaps between on-grid records | 10 gaps larger than 10 min. Largest: 2016-10-25 10:30 → 10-28 12:50 (3 d 2 h 20 min); 2014-09-24 17:00 → 09-25 09:00 (16 h); 2022-08-10 20:30 → 08-11 10:00 (13.5 h); 2018-11-07 07:30 → 14:50 (7 h 20 min) |

The `-9999` code is not defined in the provider documentation we read **[Undocumented]**;
we treat it as a missing-value marker because it is physically impossible for
air temperature.

## Time zone and daylight saving

- **[Source]** The time column is labelled **MEZ** (Mitteleuropäische Zeit, Central
  European Time).
- **[Undocumented]** The documentation does not say whether the clock is a fixed
  UTC+1 all year or switches to summer time, and it does not state a UTC offset.
  **We do not assume the timestamps are UTC.** The provider has not confirmed
  anything beyond the "MEZ" label.
- **[Measured, raw records]** The check uses the original, non-deduplicated rows.
  For the last Sunday of March and of October in every year with data (2004–2025,
  44 days), exactly six raw rows (one per `02:00`–`02:50` slot) exist on each
  day — no skipped hour in March and no repeated hour (which would give about 12
  raw rows or repeated timestamps) in October. None of the 478 duplicated rows
  falls on those days; they come from file overlap (144) and repeated rows
  inside archives (334), on unrelated dates. So the duplicates are **not**
  evidence about clock behaviour.
- **[Measured]** Peak mean short-wave radiation (`SWDR`) on the hour falls at the
  `12:00` label in both Dec/Jan and Jun/Jul, i.e. no one-hour seasonal shift.
- **What this does and does not establish:** the data are consistent with a
  clock that keeps one offset all year. They cannot establish the provider's
  intended semantics, nor rule out that a logger silently dropped or overwrote
  data at clock changes, nor give the absolute UTC offset. Treat the time zone
  and DST behaviour as **unverified**.
- **Working decision (proposed):** treat timestamps as naive station-clock labels on
  a continuous axis; never convert to UTC or another zone.

## Cleanup and hourly preparation (proposed rules; script-checked on this data)

The rules below are *proposed* design decisions. The script
(`scripts/inspect_jena.py`) implements them to produce the measured counts, and
`tests/test_inspect_jena.py` covers them with synthetic data. No loader exists yet.

1. **Sort** all records by timestamp (8 out-of-order places exist).
2. **Collapse duplicate timestamps only when their temperatures agree.** If they
   disagree, **fail** (`ConflictError`) rather than choose one. Measured: 478
   duplicate rows, 0 conflicts, so all collapse.
3. **Off-grid observations** (timestamps not on `:00/:10/…` with zero seconds) are
   **excluded** from aggregation — not snapped or rounded — and listed. Measured: 2.
   Both have their on-grid slot present, so excluding them loses nothing.
4. Source is 10-minute **end-of-period** averages. An hour labelled `H:00`
   (covering `(H-1h, H]`) is the **mean of the six on-grid temperatures stamped
   `H-50 min … H:00`**.
5. The hour is **valid only if all six are present and finite** and none is the
   `-9999` sentinel; otherwise it is an **invalid hour**. No interpolation,
   forward-fill or imputation.
6. Windows never contain an invalid hour (so they never span a gap).

Measured: the hourly grid from 2003-11-24 17:00 (first label with six possible
records) to 2026-01-01 00:00 has **193,760** hours: **193,548 valid** and **212
invalid**.

## Chronological split and window counts (measured, stride 1)

Method (proposed): boundaries on the **calendar-hour grid** at `floor(0.70·N)` and
`floor(0.85·N)`, N = 193,760. Windows (72 input + 24 target = 96 consecutive
valid hours) are built inside each partition. Hour labels are end-of-hour.

| Split | Hour labels (inclusive) | Grid hours | Valid hours | Windows | Longest valid run (h) |
|-------|-------------------------|-----------:|------------:|--------:|----------------------:|
| Train | 2003-11-24 17:00 – 2019-05-16 00:00 | 135,632 | 135,438 | 134,678 | 31,582 |
| Validation | 2019-05-16 01:00 – 2022-09-08 00:00 | 29,064 | 29,047 | 28,762 | 19,281 |
| Test | 2022-09-08 01:00 – 2026-01-01 00:00 | 29,064 | 29,063 | 28,873 | 23,582 |

Total windows: 192,313. These numbers are tied to the checksum manifest; rerun
the script if the files change. Stride-1 windows overlap almost entirely, so the
number of statistically independent test samples is far smaller than the window
count.

## Intended use

Educational: comparing a recurrent baseline with a self-attention model for
24-hour-ahead temperature forecasting, and illustrating attention weights, in a
course project.

## Limitations and out-of-scope uses

- **Not a validated local forecasting service.** The data is one rooftop station
  in Jena, Germany. It says nothing about any campus weather and the models are
  not validated for operational decisions.
- Single variable (temperature); no weather-system inputs, so long-horizon skill
  will be limited. Forecast errors can be large in rapidly changing weather.
- Station and sensor changes, siting, and long-term climate trends over 22
  years mean the later test period may differ from the training period.
- Quality control by the provider is described only as "checked for plausibility"
  **[Source]**; we did not independently validate sensor accuracy.
- Timestamp zone/DST semantics are inferred, not documented (above).
- Attribution is required under CC BY 4.0: *Max Planck Institute for
  Biogeochemistry, Jena — Weather Station Beutenberg (`mpi_roof`),
  <https://www.bgc-jena.mpg.de/wetter/>*.

## Privacy and persistence

The data is environmental measurements and contains no personal data.
User-submitted forecast inputs are a separate matter; see the project's privacy
statement (see the persistence and privacy section of the [model card](../MODEL_CARD.md)).

## Remaining open items

- [ ] Time zone / DST semantics of `MEZ` are **unverified** (see above); ask the
  provider or find documentation.
- [ ] Meaning of `-9999` (548 records in May 2007) is undocumented; treated as missing.
- [ ] Why the provider's files contain repeated rows and 8 out-of-order places
  (e.g. 2014-03-20/21 and 2023-10-20/21 repeats) is unknown.
- [ ] Whether to use all years (2003–2026, as counted above) or restrict the range.
- [ ] Whether other "Jena Climate" copies are the same subset is unverified.
- [ ] Window stride for training; stride 1 is assumed only for the counts.
