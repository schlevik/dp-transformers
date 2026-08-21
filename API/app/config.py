from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path


def _bool_from_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    repo_root: Path
    research_dir: Path
    training_script: Path
    data_root: Path
    output_root: Path
    runtime_root: Path
    database_path: Path
    python_executable: str
    default_model_name: str = "meta-llama/Llama-3.2-1B"
    default_sequence_len: int = 3072
    default_gpu_device: str = "0"
    allow_existing_empty_output_dir: bool = True

    @classmethod
    def from_env(cls) -> "Settings":
        repo_root = Path(__file__).resolve().parents[2]
        research_dir = Path(
            os.getenv(
                "API_RESEARCH_DIR",
                repo_root / "research" / "synthetic-text-generation-with-DP",
            )
        ).resolve()
        runtime_root = Path(
            os.getenv("API_RUNTIME_ROOT", repo_root / "API" / "runtime")
        ).resolve()
        output_root = Path(
            os.getenv("API_OUTPUT_ROOT", runtime_root / "outputs")
        ).resolve()

        return cls(
            repo_root=repo_root,
            research_dir=research_dir,
            training_script=Path(
                os.getenv("API_TRAINING_SCRIPT", research_dir / "fine-tune-dp.py")
            ).resolve(),
            data_root=Path(os.getenv("API_DATA_ROOT", repo_root)).resolve(),
            output_root=output_root,
            runtime_root=runtime_root,
            database_path=Path(
                os.getenv("API_DATABASE_PATH", runtime_root / "training_jobs.sqlite3")
            ).resolve(),
            python_executable=os.getenv("API_TRAINING_PYTHON", sys.executable),
            default_model_name=os.getenv(
                "API_DEFAULT_MODEL_NAME", "meta-llama/Llama-3.2-1B"
            ),
            default_sequence_len=int(os.getenv("API_DEFAULT_SEQUENCE_LEN", "3072")),
            default_gpu_device=os.getenv("API_DEFAULT_GPU_DEVICE", "0"),
            allow_existing_empty_output_dir=_bool_from_env(
                "API_ALLOW_EXISTING_EMPTY_OUTPUT_DIR", True
            ),
        )

