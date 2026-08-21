from __future__ import annotations

from typing import Dict, Optional

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from .schemas import (
    TrainingJobCreate,
    TrainingJobList,
    TrainingJobLogs,
    TrainingJobResponse,
    TrainingJobSubmitted,
)
from .service import TrainingService


def create_app(service: Optional[TrainingService] = None) -> FastAPI:
    training_service = service or TrainingService.from_env()
    app = FastAPI(title="DP Training API")
    app.state.training_service = training_service

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    def _startup() -> None:
        app.state.training_service.start()

    @app.on_event("shutdown")
    def _shutdown() -> None:
        app.state.training_service.stop()

    @app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.post("/training-jobs", response_model=TrainingJobSubmitted, status_code=202)
    def submit_training_job(request: TrainingJobCreate) -> TrainingJobSubmitted:
        job = app.state.training_service.submit_job(request)
        return TrainingJobSubmitted(job_id=job.job_id, status=job.status)

    @app.get("/training-jobs", response_model=TrainingJobList)
    def list_training_jobs(
        limit: int = Query(default=100, ge=1, le=500),
    ) -> TrainingJobList:
        jobs = app.state.training_service.list_jobs(limit=limit)
        return TrainingJobList(jobs=list(jobs))

    @app.get("/training-jobs/{job_id}", response_model=TrainingJobResponse)
    def get_training_job(job_id: str) -> TrainingJobResponse:
        return app.state.training_service.get_job(job_id)

    @app.get("/training-jobs/{job_id}/logs", response_model=TrainingJobLogs)
    def get_training_logs(
        job_id: str,
        tail: int = Query(default=200, ge=1, le=5000),
    ) -> TrainingJobLogs:
        return app.state.training_service.get_logs(job_id, tail=tail)

    @app.get("/training-jobs/{job_id}/events")
    def training_events(job_id: str) -> StreamingResponse:
        return StreamingResponse(
            app.state.training_service.event_stream(job_id),
            media_type="text/event-stream",
        )

    @app.post("/training-jobs/{job_id}/cancel", response_model=TrainingJobResponse)
    def cancel_training_job(job_id: str) -> TrainingJobResponse:
        return app.state.training_service.cancel_job(job_id)

    return app


app = create_app()

