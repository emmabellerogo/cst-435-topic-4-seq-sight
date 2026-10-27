"""Database access for the forecasting API, behind a small interface.

``ForecastRepository`` is the only thing the service and endpoints know about, so
tests pass an in-memory fake. ``SupabaseForecastRepository`` is the real
implementation; it builds its client lazily on first use (nothing connects at
import time) and re-raises every failure as ``RepositoryError`` with a generic
message, so credentials or URLs in a library error never reach a response.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional, Protocol

RUN_COLUMNS = ("id,model_type,data_version,seed,hyperparameters,preprocessing,"
               "compute_budget,val_mae_c,test_mae_c,created_at")
PREDICTION_COLUMNS = "id,created_at,input_hash,model_type,served_by_run_id,forecast"


class RepositoryError(RuntimeError):
    """Any database failure. The message is safe to log; it carries no secrets."""


class ForecastRepository(Protocol):
    def ping(self) -> bool: ...

    def get_run(self, run_id: int) -> Optional[Dict[str, Any]]: ...

    def list_runs(self, limit: int, offset: int) -> List[Dict[str, Any]]: ...

    def insert_prediction(self, *, input_hash: str, forecast: List[float],
                          attention: Optional[List[List[float]]], model_type: str,
                          served_by_run_id: int) -> Dict[str, Any]:
        """Insert one row and return it (at least ``id``). Raise on any failure."""
        ...

    def list_predictions(self, limit: int, offset: int, include_attention: bool) -> List[Dict[str, Any]]:
        """Newest first (created_at desc, id desc)."""
        ...


def _default_client():
    from supabase import create_client
    return create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])


class SupabaseForecastRepository:
    def __init__(self, client_factory: Callable[[], Any] = _default_client):
        self._factory = client_factory
        self._client = None

    def _c(self):
        if self._client is None:
            self._client = self._factory()
        return self._client

    def _run(self, what: str, fn):
        try:
            return fn(self._c())
        except Exception as e:  # noqa: BLE001 - deliberately generic, see module docstring
            raise RepositoryError(f"{what} failed ({type(e).__name__})") from None

    def ping(self) -> bool:
        try:
            self._run("ping", lambda c: c.table("runs").select("id").limit(1).execute())
            return True
        except RepositoryError:
            return False

    def get_run(self, run_id: int) -> Optional[Dict[str, Any]]:
        data = self._run("get_run", lambda c: c.table("runs").select(RUN_COLUMNS)
                         .eq("id", run_id).limit(1).execute().data)
        return data[0] if data else None

    def list_runs(self, limit: int, offset: int) -> List[Dict[str, Any]]:
        return self._run("list_runs", lambda c: c.table("runs").select(RUN_COLUMNS)
                         .order("created_at", desc=True).order("id", desc=True)
                         .range(offset, offset + limit - 1).execute().data)

    def insert_prediction(self, *, input_hash, forecast, attention, model_type, served_by_run_id):
        row = {"input_hash": input_hash, "forecast": forecast, "attention": attention,
               "model_type": model_type, "served_by_run_id": served_by_run_id}
        data = self._run("insert_prediction", lambda c: c.table("predictions").insert(row).execute().data)
        if not data:
            raise RepositoryError("insert_prediction returned no row")
        return data[0]

    def list_predictions(self, limit, offset, include_attention):
        cols = PREDICTION_COLUMNS + (",attention" if include_attention else "")
        return self._run("list_predictions", lambda c: c.table("predictions").select(cols)
                         .order("created_at", desc=True).order("id", desc=True)
                         .range(offset, offset + limit - 1).execute().data)
