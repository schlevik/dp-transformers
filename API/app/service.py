from __future__ import annotations

import json
import queue
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional

from fastapi import HTTPException

from .config import Settings
from .runner import LocalSubprocessRunner, TrainingRunner
from .schemas import (
    JobStatus,
    TrainingJobCreate,
    TrainingJobLogs,
    TrainingJobResponse,
    model_to_dict,
)
from .store import JobRecord, JobStore, utc_now


_SAFE_OUTPUT_NAME = re.compile(r"^[A-Za-z0-9._-]+$")


class TrainingService:
    def __init__(
        self,
        *,
        settings: Settings,
        store: Optional[JobStore] = None,
        runner: Optional[TrainingRunner] = None,
        autostart: bool = False,
    ):
        self.settings = settings
        self.store = store or JobStore(settings.database_path)
        self.runner = runner or LocalSubprocessRunner(settings)
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None
        if autostart:
            self.start()

    @classmethod
    def from_env(cls, autostart: bool = False) -> "TrainingService":
        return cls(settings=Settings.from_env(), autostart=autostart)

    def start(self) -> None:
        if self._worker_thread and self._worker_thread.is_alive():
            return
        self._stop_event.clear()
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="training-job-worker",
            daemon=True,
        )
        self._worker_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._queue.put("")
        if self._worker_thread:
            self._worker_thread.join(timeout=5)

    def submit_job(self, request: TrainingJobCreate) -> TrainingJobResponse:
        request = self._with_config_defaults(request)
        job_id = uuid.uuid4().hex
        train_file = self._validate_train_file(request.train_file)
        output_dir = self._resolve_output_dir(request, job_id)
        log_path = self.settings.runtime_root / "logs" / f"{job_id}.log"

        request_payload = model_to_dict(request)
        request_payload["train_file"] = str(train_file)
        request.train_file = str(train_file)

        record = self.store.create_job(
            job_id=job_id,
            request=request_payload,
            output_dir=output_dir,
            log_path=log_path,
        )
        self._queue.put(job_id)
        return TrainingJobResponse(**record.to_response_dict())

    def get_job(self, job_id: str) -> TrainingJobResponse:
        return TrainingJobResponse(**self._get_record_or_404(job_id).to_response_dict())

    def list_jobs(self, limit: int = 100) -> Iterable[TrainingJobResponse]:
        return [
            TrainingJobResponse(**record.to_response_dict())
            for record in self.store.list_jobs(limit=limit)
        ]

    def get_logs(self, job_id: str, tail: int = 200) -> TrainingJobLogs:
        record = self._get_record_or_404(job_id)
        tail = max(1, min(tail, 5000))
        lines = _tail_lines(record.log_path, tail)
        return TrainingJobLogs(
            job_id=record.job_id,
            log_path=str(record.log_path),
            lines=lines,
        )

    def cancel_job(self, job_id: str) -> TrainingJobResponse:
        record = self._get_record_or_404(job_id)
        if record.status in {
            JobStatus.succeeded,
            JobStatus.failed,
            JobStatus.cancelled,
        }:
            return TrainingJobResponse(**record.to_response_dict())

        if record.pid:
            self.runner.cancel(record.pid)

        record = self.store.update_job(
            job_id,
            status=JobStatus.cancelled,
            completed_at=utc_now() if record.status == JobStatus.queued else None,
            error="Cancelled by request",
        )
        return TrainingJobResponse(**record.to_response_dict())

    def event_stream(self, job_id: str) -> Generator[str, None, None]:
        last_status: Optional[str] = None
        last_size = 0
        while True:
            record = self._get_record_or_404(job_id)
            if record.status.value != last_status:
                last_status = record.status.value
                yield _sse("status", record.to_response_dict())

            if record.log_path.exists():
                with record.log_path.open("r", encoding="utf-8", errors="replace") as fh:
                    fh.seek(last_size)
                    new_data = fh.read()
                    last_size = fh.tell()
                for line in new_data.splitlines():
                    yield _sse("log", {"message": line})

            if record.status in {
                JobStatus.succeeded,
                JobStatus.failed,
                JobStatus.cancelled,
            }:
                yield _sse("done", record.to_response_dict())
                break

            time.sleep(1)

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                job_id = self._queue.get(timeout=1)
            except queue.Empty:
                continue

            if not job_id:
                continue

            try:
                self._run_job(job_id)
            finally:
                self._queue.task_done()

    def _run_job(self, job_id: str) -> None:
        record = self.store.get_job(job_id)
        if record is None or record.status == JobStatus.cancelled:
            return

        request = TrainingJobCreate(**record.request)
        try:
            process = self.runner.start(
                job_id=job_id,
                request=request,
                output_dir=record.output_dir,
                log_path=record.log_path,
            )
            self.store.update_job(
                job_id,
                status=JobStatus.running,
                started_at=utc_now(),
                pid=process.pid,
            )
            exit_code = process.wait()
            current = self.store.get_job(job_id)
            if current and current.status == JobStatus.cancelled:
                self.store.update_job(
                    job_id,
                    exit_code=exit_code,
                    completed_at=current.completed_at or utc_now(),
                )
                return

            final_model_path = record.output_dir / "final"
            metrics = _read_metrics(record.output_dir)
            if exit_code == 0:
                self.store.update_job(
                    job_id,
                    status=JobStatus.succeeded,
                    completed_at=utc_now(),
                    exit_code=exit_code,
                    final_model_path=final_model_path,
                    metrics=metrics,
                )
            else:
                self.store.update_job(
                    job_id,
                    status=JobStatus.failed,
                    completed_at=utc_now(),
                    exit_code=exit_code,
                    error=f"Training process exited with code {exit_code}",
                    metrics=metrics,
                )
        except Exception as exc:
            self.store.update_job(
                job_id,
                status=JobStatus.failed,
                completed_at=utc_now(),
                error=f"{exc.__class__.__name__}: {exc}",
            )

    def _with_config_defaults(self, request: TrainingJobCreate) -> TrainingJobCreate:
        payload = model_to_dict(request)
        if payload.get("model_name") == _field_default("model_name"):
            payload["model_name"] = self.settings.default_model_name
        if payload.get("sequence_len") == _field_default("sequence_len"):
            payload["sequence_len"] = self.settings.default_sequence_len
        if payload.get("gpu_device") == _field_default("gpu_device"):
            payload["gpu_device"] = self.settings.default_gpu_device
        return TrainingJobCreate(**payload)

    def _validate_train_file(self, train_file: str) -> Path:
        path = Path(train_file).expanduser().resolve()
        if path.suffix != ".jsonl":
            raise HTTPException(status_code=400, detail="train_file must be a .jsonl file")
        if not path.exists() or not path.is_file():
            raise HTTPException(status_code=400, detail="train_file does not exist")
        if not _is_relative_to(path, self.settings.data_root):
            raise HTTPException(
                status_code=400,
                detail=f"train_file must be under {self.settings.data_root}",
            )
        return path

    def _resolve_output_dir(self, request: TrainingJobCreate, job_id: str) -> Path:
        if request.output_name and request.output_dir:
            raise HTTPException(
                status_code=400,
                detail="Use either output_name or output_dir, not both",
            )

        if request.output_name:
            if not _SAFE_OUTPUT_NAME.match(request.output_name):
                raise HTTPException(
                    status_code=400,
                    detail="output_name may contain only letters, numbers, '.', '_' and '-'",
                )
            output_dir = self.settings.output_root / request.output_name
        elif request.output_dir:
            output_dir = Path(request.output_dir).expanduser().resolve()
        else:
            output_dir = self.settings.output_root / job_id

        output_dir = output_dir.resolve()
        if not _is_relative_to(output_dir, self.settings.output_root):
            raise HTTPException(
                status_code=400,
                detail=f"output_dir must be under {self.settings.output_root}",
            )
        if output_dir.exists() and any(output_dir.iterdir()):
            raise HTTPException(
                status_code=400,
                detail="output_dir already exists and is not empty",
            )
        if output_dir.exists() and not self.settings.allow_existing_empty_output_dir:
            raise HTTPException(status_code=400, detail="output_dir already exists")
        return output_dir

    def _get_record_or_404(self, job_id: str) -> JobRecord:
        record = self.store.get_job(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="training job not found")
        return record


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _tail_lines(path: Path, limit: int) -> List[str]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()
    return [line.rstrip("\n") for line in lines[-limit:]]


def _read_metrics(output_dir: Path) -> Optional[Dict[str, Any]]:
    for metrics_path in (
        output_dir / "train_results.json",
        output_dir / "all_results.json",
    ):
        if metrics_path.exists():
            with metrics_path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
    return None


def _sse(event: str, payload: Dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, default=str)}\n\n"


def _field_default(name: str) -> Any:
    fields = getattr(TrainingJobCreate, "model_fields", None)
    if fields is None:
        fields = getattr(TrainingJobCreate, "__fields__")
    return fields[name].default

