from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from .schemas import JobStatus


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class JobRecord:
    job_id: str
    status: JobStatus
    created_at: str
    updated_at: str
    output_dir: Path
    log_path: Path
    request: Dict[str, Any]
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    pid: Optional[int] = None
    exit_code: Optional[int] = None
    final_model_path: Optional[Path] = None
    error: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None

    def to_response_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "pid": self.pid,
            "exit_code": self.exit_code,
            "output_dir": str(self.output_dir),
            "log_path": str(self.log_path),
            "final_model_path": str(self.final_model_path)
            if self.final_model_path
            else None,
            "error": self.error,
            "metrics": self.metrics,
            "request": self.request,
        }


class JobStore:
    def __init__(self, database_path: Path):
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_schema()

    def create_job(
        self,
        *,
        job_id: str,
        request: Dict[str, Any],
        output_dir: Path,
        log_path: Path,
    ) -> JobRecord:
        now = utc_now()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO training_jobs (
                    job_id, status, created_at, updated_at, request_json,
                    output_dir, log_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    JobStatus.queued.value,
                    now,
                    now,
                    json.dumps(request, sort_keys=True),
                    str(output_dir),
                    str(log_path),
                ),
            )
        return self.get_job(job_id)

    def get_job(self, job_id: str) -> Optional[JobRecord]:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM training_jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        return self._row_to_record(row) if row else None

    def list_jobs(self, limit: int = 100) -> Iterable[JobRecord]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM training_jobs
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def update_job(self, job_id: str, **fields: Any) -> JobRecord:
        if not fields:
            job = self.get_job(job_id)
            if job is None:
                raise KeyError(job_id)
            return job

        fields["updated_at"] = utc_now()
        serialized = {}
        for key, value in fields.items():
            column = "metrics_json" if key == "metrics" else key
            serialized[column] = self._serialize_value(column, value)
        assignments = ", ".join(f"{key} = ?" for key in serialized)
        values = list(serialized.values()) + [job_id]

        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                f"UPDATE training_jobs SET {assignments} WHERE job_id = ?", values
            )
            if cursor.rowcount == 0:
                raise KeyError(job_id)

        job = self.get_job(job_id)
        if job is None:
            raise KeyError(job_id)
        return job

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.database_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS training_jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    request_json TEXT NOT NULL,
                    output_dir TEXT NOT NULL,
                    log_path TEXT NOT NULL,
                    final_model_path TEXT,
                    pid INTEGER,
                    exit_code INTEGER,
                    error TEXT,
                    metrics_json TEXT
                )
                """
            )

    @staticmethod
    def _serialize_value(key: str, value: Any) -> Any:
        if isinstance(value, JobStatus):
            return value.value
        if isinstance(value, Path):
            return str(value)
        if key == "metrics" or key == "metrics_json":
            return json.dumps(value, sort_keys=True) if value is not None else None
        return value

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> JobRecord:
        metrics = json.loads(row["metrics_json"]) if row["metrics_json"] else None
        return JobRecord(
            job_id=row["job_id"],
            status=JobStatus(row["status"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            pid=row["pid"],
            exit_code=row["exit_code"],
            output_dir=Path(row["output_dir"]),
            log_path=Path(row["log_path"]),
            final_model_path=Path(row["final_model_path"])
            if row["final_model_path"]
            else None,
            error=row["error"],
            metrics=metrics,
            request=json.loads(row["request_json"]),
        )

