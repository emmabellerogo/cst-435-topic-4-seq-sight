"""FastAPI model service -- Cloud #2 (deployed on Render.com).

Seq Sight: 24 h temperature forecasting from a 72 h history, with an LSTM
baseline and a hand-written self-attention model. The three-cloud structure
(Streamlit UI <-> this API <-> Supabase) and the app/CORS scaffolding were
adapted from the professor-provided Attend-It template; the endpoints, models
and logic here are Seq Sight's own. The Attend-It classification endpoints were
replaced; ``api/db.py`` and ``api/training.py`` are template leftovers that this
module no longer uses.

The API is the only writer to Supabase. Inference lives in
``api/forecast_service.py`` and database access behind
``api/forecast_repo.py``; ``create_app`` takes both as arguments so tests can
inject synthetic artifacts and a fake repository.

Audit rule: a forecast is returned only after its ``predictions`` row was
written. If the write fails the forecast is withheld and the response says so.
"""
from __future__ import annotations

import os
import threading
from contextlib import asynccontextmanager
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.forecast_config import ForecastSettings
from api.forecast_repo import ForecastRepository, SupabaseForecastRepository
from api.forecast_service import (API_VERSION, HASH_FORMAT, ForecastService, ServeError, git_sha,
                                  run_forecasts)
from shared.forecast_models import INPUT_HOURS, TARGET_HOURS
from shared.forecast_schemas import (MAX_BATCH_SIZE, MAX_OFFSET, MAX_PAGE_SIZE, BatchItem,
                                     BatchPredictRequest, BatchPredictResponse, Health, ModelHealth,
                                     PredictionPage, PredictionRecord, PredictRequest,
                                     PredictResponse, RunPage, RunRecord, Version)


def _http(e: ServeError) -> HTTPException:
    return HTTPException(status_code=e.status, detail={"code": e.code, "message": e.message})


def _db_unavailable() -> HTTPException:
    return HTTPException(status_code=503, detail={"code": "database_unavailable",
                                                  "message": "the database is unavailable"})


def create_app(settings: Optional[ForecastSettings] = None,
               repo: Optional[ForecastRepository] = None) -> FastAPI:
    settings = settings or ForecastSettings.from_env()
    repo = repo or SupabaseForecastRepository()
    service = ForecastService(settings, repo)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if settings.artifact_source == "supabase":
            # Network fetches are bounded but can take a while: open the port first so the
            # platform sees the process alive; /healthz reports "starting" until this finishes.
            t = threading.Thread(target=service.startup, name="seqsight-startup", daemon=True)
            _app.state.startup_thread = t
            t.start()
        else:
            service.startup()   # local files only; never raises: failures become readiness reasons
        yield

    app = FastAPI(
        title="Seq Sight API",
        description="24 h temperature forecasts (LSTM baseline and self-attention) with audit logging.",
        version=API_VERSION,
        lifespan=lifespan,
    )
    app.state.service = service
    app.state.repo = repo

    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.environ.get("ALLOWED_ORIGINS", "*").split(","),
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_request, exc: RequestValidationError):
        # FastAPI's default body echoes each offending `input`; a NaN/Infinity there cannot be
        # serialized as JSON (the 422 would become a 500). Report location and message only.
        errors = [{"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
        return JSONResponse(status_code=422, content={"detail": errors})

    def _audited(model_type: str, run_id: int, fc) -> PredictResponse:
        """Write the audit row, then build the response. Raises on write failure."""
        row = repo.insert_prediction(input_hash=fc.input_hash, forecast=fc.forecast_c,
                                     attention=fc.attention, model_type=model_type,
                                     served_by_run_id=run_id)
        return PredictResponse(prediction_id=row["id"], model_type=model_type, served_by_run_id=run_id,
                               input_hash=fc.input_hash, forecast_c=fc.forecast_c, attention=fc.attention)

    # ------------------------------------------------------------ prediction
    @app.post("/predict", response_model=PredictResponse, tags=["prediction"])
    def predict(req: PredictRequest) -> PredictResponse:
        """Forecast the next 24 h from exactly 72 hourly temperatures (degC)."""
        try:
            loaded = service.require_ready(req.model_type)
            fc = run_forecasts(loaded, [req.history_c])[0]
        except ServeError as e:
            raise _http(e)
        try:
            return _audited(req.model_type, service.run_id(req.model_type), fc)
        except Exception:  # noqa: BLE001 - any logging failure
            raise HTTPException(status_code=503, detail={
                "code": "prediction_not_audited",
                "message": "the forecast could not be recorded, so it was withheld"})

    @app.post("/predict_batch", response_model=BatchPredictResponse, tags=["prediction"],
              responses={207: {"model": BatchPredictResponse}, 503: {"model": BatchPredictResponse}})
    def predict_batch(req: BatchPredictRequest, response: Response) -> BatchPredictResponse:
        """Forecast 1..32 histories (MAX_BATCH_SIZE) with one model.

        Validation is all-or-nothing (any invalid history -> 422, nothing logged).
        After inference each forecast is audited independently. 200: every item
        audited. 207: some were not (their forecasts are withheld, ``status`` is
        ``not_audited``). 503: none could be audited.
        """
        try:
            loaded = service.require_ready(req.model_type)
            forecasts = run_forecasts(loaded, req.histories_c)
        except ServeError as e:
            raise _http(e)
        run_id = service.run_id(req.model_type)
        results = []
        for i, fc in enumerate(forecasts):
            try:
                results.append(BatchItem(index=i, status="ok",
                                         prediction=_audited(req.model_type, run_id, fc)))
            except Exception:  # noqa: BLE001
                results.append(BatchItem(index=i, status="not_audited",
                                         error="the forecast could not be recorded, so it was withheld"))
        n_ok = sum(r.status == "ok" for r in results)
        response.status_code = 200 if n_ok == len(results) else (207 if n_ok else 503)
        return BatchPredictResponse(model_type=req.model_type, served_by_run_id=run_id,
                                    n_requested=len(results), n_audited=n_ok,
                                    n_not_audited=len(results) - n_ok, results=results)

    # --------------------------------------------------------------- history
    @app.get("/predictions", response_model=PredictionPage, tags=["prediction"])
    def predictions(limit: int = Query(20, ge=1, le=MAX_PAGE_SIZE),
                    offset: int = Query(0, ge=0, le=MAX_OFFSET),
                    include_attention: bool = Query(False, description="72x72 per row; large.")
                    ) -> PredictionPage:
        """Recent audited predictions, newest first."""
        try:
            rows = repo.list_predictions(limit + 1, offset, include_attention)
        except Exception:  # noqa: BLE001
            raise _db_unavailable()
        items = [PredictionRecord(id=r["id"], created_at=r.get("created_at"), input_hash=r["input_hash"],
                                  model_type=r["model_type"], served_by_run_id=r["served_by_run_id"],
                                  forecast_c=r["forecast"], attention=r.get("attention") if include_attention else None)
                 for r in rows[:limit]]
        return PredictionPage(items=items, limit=limit, offset=offset, has_more=len(rows) > limit)

    @app.get("/runs", response_model=RunPage, tags=["training"])
    def runs(limit: int = Query(20, ge=1, le=MAX_PAGE_SIZE),
             offset: int = Query(0, ge=0, le=MAX_OFFSET)) -> RunPage:
        """Persisted runs, newest first. ``served_by_this_api`` marks the ones being served."""
        try:
            rows = repo.list_runs(limit + 1, offset)
        except Exception:  # noqa: BLE001
            raise _db_unavailable()
        serving = service.served_run_ids()
        items = [RunRecord(**{k: r[k] for k in RunRecord.model_fields if k in r},
                           served_by_this_api=r["id"] in serving) for r in rows[:limit]]
        return RunPage(items=items, limit=limit, offset=offset, has_more=len(rows) > limit)

    # ------------------------------------------------------------------- ops
    @app.get("/livez", tags=["ops"])
    def livez() -> dict:
        """Process liveness ONLY (used by the Render health check). It says nothing about whether any
        model is registered or serving: that is /healthz, which returns 503 until it is true."""
        return {"status": "alive"}

    @app.get("/healthz", response_model=Health, tags=["ops"],
             responses={503: {"model": Health}})
    def healthz(response: Response) -> Health:
        """Ready only if the database answers and every model is loaded and registered."""
        r = service.readiness()
        ok = r["ready"] and r["database"]
        response.status_code = 200 if ok else 503
        return Health(status="ok" if ok else "degraded", ready=r["ready"], database=r["database"],
                      models={k: ModelHealth(**v) for k, v in r["models"].items()})

    @app.get("/version", response_model=Version, tags=["ops"])
    def version() -> Version:
        return Version(api_version=API_VERSION, git_sha=git_sha(), torch_version=torch.__version__,
                       input_hours=INPUT_HOURS, forecast_hours=TARGET_HOURS, max_batch_size=MAX_BATCH_SIZE,
                       input_hash_format=HASH_FORMAT,
                       artifacts_are_development_defaults=settings.artifacts_are_development_defaults,
                       artifact_source=settings.artifact_source,
                       models=service.version_info())

    return app


app = create_app()
