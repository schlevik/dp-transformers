from __future__ import annotations

import sys
import time
from pathlib import Path

from app.config import Settings
from app.schemas import TrainingJobCreate
from app.service import TrainingService


def make_settings(tmp_path: Path, training_script: Path) -> Settings:
    data_root = tmp_path / "data"
    output_root = tmp_path / "outputs"
    runtime_root = tmp_path / "runtime"
    data_root.mkdir()
    output_root.mkdir()
    return Settings(
        repo_root=tmp_path,
        research_dir=tmp_path,
        training_script=training_script,
        data_root=data_root,
        output_root=output_root,
        runtime_root=runtime_root,
        database_path=runtime_root / "jobs.sqlite3",
        python_executable=sys.executable,
    )


def test_worker_runs_fake_training_command(tmp_path: Path) -> None:
    fake_script = tmp_path / "fake_train.py"
    fake_script.write_text(
        """
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", required=True)
args, _ = parser.parse_known_args()
output_dir = Path(args.output_dir)
(output_dir / "final").mkdir(parents=True)
(output_dir / "train_results.json").write_text(json.dumps({"train_loss": 0.25}))
print("fake training complete")
""".strip(),
        encoding="utf-8",
    )
    settings = make_settings(tmp_path, fake_script)
    train_file = settings.data_root / "train.jsonl"
    train_file.write_text("{}\n", encoding="utf-8")
    service = TrainingService(settings=settings)

    service.start()
    try:
        submitted = service.submit_job(
            TrainingJobCreate(
                train_file=str(train_file),
                target_epsilon=2,
                output_name="fake-job",
            )
        )
        job = _wait_for_terminal_status(service, submitted.job_id)
    finally:
        service.stop()

    assert job.status == "succeeded"
    assert job.final_model_path == str(settings.output_root / "fake-job" / "final")
    assert job.metrics == {"train_loss": 0.25}
    assert "fake training complete" in "\n".join(
        service.get_logs(submitted.job_id).lines
    )


def _wait_for_terminal_status(
    service: TrainingService, job_id: str, timeout_seconds: float = 5
):
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        job = service.get_job(job_id)
        if job.status in {"succeeded", "failed", "cancelled"}:
            return job
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish")

