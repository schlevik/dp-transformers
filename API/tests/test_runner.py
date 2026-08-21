from __future__ import annotations

from pathlib import Path

from app.config import Settings
from app.runner import LocalSubprocessRunner
from app.schemas import TrainingJobCreate


def make_settings(tmp_path: Path) -> Settings:
    return Settings(
        repo_root=tmp_path,
        research_dir=tmp_path,
        training_script=tmp_path / "fine-tune-dp.py",
        data_root=tmp_path / "data",
        output_root=tmp_path / "outputs",
        runtime_root=tmp_path / "runtime",
        database_path=tmp_path / "runtime" / "jobs.sqlite3",
        python_executable="python",
    )


def test_build_command_matches_single_dp_training_run(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    runner = LocalSubprocessRunner(settings)
    request = TrainingJobCreate(
        train_file=str(settings.data_root / "train.jsonl"),
        target_epsilon=2,
        gpu_device="3",
        output_name="demo",
    )

    command = runner.build_command(request, settings.output_root / "demo")

    assert command[:2] == ["python", str(settings.training_script)]
    assert command[command.index("--train_file") + 1] == str(
        settings.data_root / "train.jsonl"
    )
    assert command[command.index("--target_epsilon") + 1] == "2.0"
    assert command[command.index("--output_dir") + 1] == str(
        settings.output_root / "demo"
    )
    assert "--save_strategy" in command
    assert "--report_to" in command


def test_build_command_adds_lora_flags(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    runner = LocalSubprocessRunner(settings)
    request = TrainingJobCreate(
        train_file=str(settings.data_root / "train.jsonl"),
        target_epsilon=1,
        enable_lora=True,
        target_modules=["q_proj", "v_proj"],
    )

    command = runner.build_command(request, settings.output_root / "lora")

    assert command[command.index("--enable_lora") + 1] == "True"
    assert command[command.index("--target_modules") + 1] == '["q_proj", "v_proj"]'

