"""Read one data version of the Jena corpus from Supabase (read-only).

Kept apart from the pure preprocessing in ``shared/forecasting.py``. The client
is injected, so this module imports no database library and tests can pass a
mock. It uses only ``table().select().eq().order().range().execute()``, which the
pinned supabase-py (2.7.4, see api/requirements.txt) supports.
"""
from __future__ import annotations

from typing import Iterator, List

import numpy as np

from shared.forecasting import ForecastDataError, ForecastingData, prepare_forecasting_data

TABLE = "sequences"
COLUMNS = "station_ts,temperature_c,split,data_version"
# PostgREST caps a response at 1000 rows by default (Supabase "Max rows"), so a
# larger page would be silently truncated and look like the last page.
MAX_PAGE = 1000


def fetch_corpus_rows(client, data_version: str, page_size: int = MAX_PAGE) -> Iterator[dict]:
    """Yield every stored row of ``data_version``, ordered by ``station_ts``.

    Pages with an explicit column list and a deterministic order
    (``station_ts`` is unique within a data version). Stops at the first short
    page, or an empty one when the row count is an exact multiple of the page size.
    """
    if not isinstance(data_version, str) or not data_version:
        raise ForecastDataError("data_version must be a non-empty string")
    if not 1 <= page_size <= MAX_PAGE:
        raise ForecastDataError(f"page_size must be between 1 and {MAX_PAGE}")
    start = 0
    while True:
        rows: List[dict] = (client.table(TABLE).select(COLUMNS)
                            .eq("data_version", data_version).order("station_ts")
                            .range(start, start + page_size - 1).execute().data)
        yield from rows
        if len(rows) < page_size:
            return
        start += page_size


def load_forecasting_data(client, data_version: str, page_size: int = MAX_PAGE,
                          dtype=np.float32) -> ForecastingData:
    """Fetch ``data_version`` and prepare scaled windows for all splits."""
    return prepare_forecasting_data(fetch_corpus_rows(client, data_version, page_size),
                                    expected_data_version=data_version, dtype=dtype)
