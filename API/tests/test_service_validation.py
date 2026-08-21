from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import HTTPException

from app.config import Settings
from app.schemas import TrainingJobCreate
from app.service import TrainingService


def make_settings(tmp_path: Path) -> Settings:
    data_root = tmp_path / "data"
    output_root = tmp_path / "outputs"
    runtime_root = tmp_path / "runtime"
    data_root.mkdir()
    output_root.mkdir()
    return Settings(
        repo_root=tmp_path,
        research_dir=tmp_path,
        training_script=tmp_path / "fine-tune-dp.py",
        data_root=data_root,
        output_root=output_root,
        runtime_root=runtime_root,
        database_path=runtime_root / "jobs.sqlite3",
        python_executable="python",
    )


def test_submit_job_validates_and_persists_safe_paths(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    train_file = settings.data_root / "train.jsonl"
    train_file.write_text('{"text":"hello","label":["granted"]}\n', encoding="utf-8")
    service = TrainingService(settings=settings)

    job = service.submit_job(
        TrainingJobCreate(
            train_file=str(train_file),
            target_epsilon=2,
            output_name="safe-output",
        )
    )

    saved = service.get_job(job.job_id)
    assert saved.status == "queued"
    assert saved.output_dir == str(settings.output_root / "safe-output")
    assert saved.request["train_file"] == str(train_file)


def test_rejects_dataset_outside_data_root(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    outside = tmp_path / "outside.jsonl"
    outside.write_text("{}\n", encoding="utf-8")
    service = TrainingService(settings=settings)

    with pytest.raises(HTTPException) as exc:
        service.submit_job(
            TrainingJobCreate(train_file=str(outside), target_epsilon=2)
        )

    assert exc.value.status_code == 400
    assert "under" in exc.value.detail


def test_rejects_output_path_traversal(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    train_file = settings.data_root / "train.jsonl"
    train_file.write_text("{}\n", encoding="utf-8")
    service = TrainingService(settings=settings)

    with pytest.raises(HTTPException) as exc:
        service.submit_job(
            TrainingJobCreate(
                train_file=str(train_file),
                target_epsilon=2,
                output_dir=str(tmp_path / "not-allowed"),
            )
        )

    assert exc.value.status_code == 400
    assert "output_dir must be under" in exc.value.detail


def test_rejects_non_empty_output_directory(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    train_file = settings.data_root / "train.jsonl"
    train_file.write_text("{}\n", encoding="utf-8")
    output_dir = settings.output_root / "existing"
    output_dir.mkdir()
    (output_dir / "file.txt").write_text("already used", encoding="utf-8")
    service = TrainingService(settings=settings)

    with pytest.raises(HTTPException) as exc:
        service.submit_job(
            TrainingJobCreate(
                train_file=str(train_file),
                target_epsilon=2,
                output_dir=str(output_dir),
            )
        )

    assert exc.value.status_code == 400
    assert "not empty" in exc.value.detail

