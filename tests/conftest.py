"""Shared pytest helpers.

The forecasting API tests build their own app with synthetic artifacts and a fake
repository (tests/test_forecast_api.py); nothing here touches the network.
"""
from __future__ import annotations

import os


def has_supabase_creds() -> bool:
    return bool(os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_SERVICE_KEY"))
