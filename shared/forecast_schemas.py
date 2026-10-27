"""Pydantic request/response models for the Seq Sight forecasting API.

Strict on purpose: a temperature must be a real JSON number. Booleans and numeric
strings are rejected (never coerced), the history must be exactly 72 values, and
every value must be finite and within ``+-MAX_ABS_TEMPERATURE_C``.

The bound is a numerical guard, not a claim about physical plausibility: it keeps
standardized values far inside float32 range so that a finite request cannot
overflow into a non-finite forecast.
"""
from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, field_validator

from shared.forecast_models import INPUT_HOURS, TARGET_HOURS

MAX_BATCH_SIZE = 32          # /predict_batch; each audited row stores a 72x72 matrix
MAX_ABS_TEMPERATURE_C = 1000.0
MAX_PAGE_SIZE = 100
MAX_OFFSET = 10_000

ModelType = Literal["lstm", "attention"]
Temperature = Union[StrictInt, StrictFloat]  # StrictInt/StrictFloat both reject bool and str


def _check_history(values: List[Union[int, float]]) -> List[float]:
    out: List[float] = []
    for i, v in enumerate(values):
        try:
            f = float(v)
        except OverflowError:
            raise ValueError(f"value at index {i} is too large")
        if not math.isfinite(f):
            raise ValueError(f"value at index {i} is not finite")
        if abs(f) > MAX_ABS_TEMPERATURE_C:
            raise ValueError(f"value at index {i} is outside +-{MAX_ABS_TEMPERATURE_C:g} degC")
        out.append(f)
    return out


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model_type: ModelType
    history_c: List[Temperature] = Field(
        ..., min_length=INPUT_HOURS, max_length=INPUT_HOURS,
        description="The last 72 hourly temperatures in degC, oldest first.")

    @field_validator("history_c")
    @classmethod
    def _finite(cls, v):
        return _check_history(v)


class BatchPredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model_type: ModelType
    histories_c: List[List[Temperature]] = Field(
        ..., min_length=1, max_length=MAX_BATCH_SIZE,
        description=f"1..{MAX_BATCH_SIZE} histories of exactly {INPUT_HOURS} degC values each.")

    @field_validator("histories_c")
    @classmethod
    def _each_history(cls, v):
        out = []
        for n, history in enumerate(v):
            if len(history) != INPUT_HOURS:
                raise ValueError(f"histories_c[{n}] has {len(history)} values; expected {INPUT_HOURS}")
            try:
                out.append(_check_history(history))
            except ValueError as e:
                raise ValueError(f"histories_c[{n}]: {e}")
        return out


class PredictResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    prediction_id: int = Field(..., description="Primary key of the audit row in `predictions`.")
    model_type: ModelType
    served_by_run_id: int
    input_hash: str = Field(..., description="Canonical SHA-256 of the input (see api/forecast_service.py).")
    forecast_c: List[float] = Field(..., min_length=TARGET_HOURS, max_length=TARGET_HOURS,
                                    description="Hours t+1 .. t+24 in degC.")
    attention: Optional[List[List[float]]] = Field(
        None, description="72x72 encoder self-attention (input hour i over input hour j); null for lstm.")


class BatchItem(BaseModel):
    index: int
    status: Literal["ok", "not_audited"]
    prediction: Optional[PredictResponse] = None
    error: Optional[str] = None


class BatchPredictResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_type: ModelType
    served_by_run_id: int
    n_requested: int
    n_audited: int
    n_not_audited: int
    results: List[BatchItem]


class PredictionRecord(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    id: int
    created_at: Optional[datetime] = None
    input_hash: str
    model_type: ModelType
    served_by_run_id: int
    forecast_c: List[float]
    attention: Optional[List[List[float]]] = None


class PredictionPage(BaseModel):
    items: List[PredictionRecord]
    limit: int
    offset: int
    has_more: bool


class RunRecord(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    id: int
    model_type: ModelType
    data_version: str
    seed: int
    hyperparameters: Dict[str, Any]
    preprocessing: Dict[str, Any]
    compute_budget: Dict[str, Any]
    val_mae_c: float
    test_mae_c: float
    created_at: Optional[datetime] = None
    served_by_this_api: bool = False


class RunPage(BaseModel):
    items: List[RunRecord]
    limit: int
    offset: int
    has_more: bool


class ModelHealth(BaseModel):
    artifact_loaded: bool
    run_registered: bool = Field(..., description="A run id is configured and matches persisted run metadata.")
    ready: bool
    reason: Optional[str] = Field(None, description="Machine-readable code; never a path or secret.")


class Health(BaseModel):
    status: Literal["ok", "degraded"]
    ready: bool
    database: bool
    models: Dict[str, ModelHealth]


class ServedModelVersion(BaseModel):
    ready: bool
    serving_run_id: Optional[int] = None
    seed: Optional[int] = None
    data_version: Optional[str] = None
    best_epoch: Optional[int] = None
    best_val_mae_c: Optional[float] = None
    parameter_count: Optional[int] = None
    artifact_sha256_prefix: Optional[str] = None
    trained_with_torch: Optional[str] = None


class Version(BaseModel):
    api_version: str
    git_sha: str
    torch_version: str
    input_hours: int
    forecast_hours: int
    max_batch_size: int
    input_hash_format: str
    artifacts_are_development_defaults: bool
    models: Dict[str, ServedModelVersion]
