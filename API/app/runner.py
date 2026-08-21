from __future__ import annotations

import json
import os
import signal
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import Settings
from .schemas import TrainingJobCreate


def _bool_arg(value: bool) -> str:
    return "True" if value else "False"


class TrainingRunner(ABC):
    @abstractmethod
    def start(
        self,
        *,
        job_id: str,
        request: TrainingJobCreate,
        output_dir: Path,
        log_path: Path,
    ) -> subprocess.Popen:
        raise NotImplementedError

    @abstractmethod
    def cancel(self, pid: int) -> None:
        raise NotImplementedError


class LocalSubprocessRunner(TrainingRunner):
    def __init__(self, settings: Settings):
        self.settings = settings

    def build_command(
        self, request: TrainingJobCreate, output_dir: Path
    ) -> List[str]:
        command = [
            self.settings.python_executable,
            str(self.settings.training_script),
            "--output_dir",
            str(output_dir),
            "--model_name",
            request.model_name,
            "--train_file",
            request.train_file,
            "--sequence_len",
            str(request.sequence_len),
            "--per_device_train_batch_size",
            str(request.per_device_train_batch_size),
            "--gradient_accumulation_steps",
            str(request.gradient_accumulation_steps),
            "--log_level",
            "info",
            "--per_device_eval_batch_size",
            str(request.per_device_eval_batch_size),
            "--eval_accumulation_steps",
            str(request.eval_accumulation_steps),
            "--seed",
            str(request.seed),
            "--prediction_loss_only",
            _bool_arg(request.prediction_loss_only),
            "--target_epsilon",
            str(request.target_epsilon),
            "--per_sample_max_grad_norm",
            str(request.per_sample_max_grad_norm),
            "--weight_decay",
            str(request.weight_decay),
            "--remove_unused_columns",
            _bool_arg(request.remove_unused_columns),
            "--num_train_epochs",
            str(request.num_train_epochs),
            "--logging_steps",
            str(request.logging_steps),
            "--max_grad_norm",
            "0",
            "--lr_scheduler_type",
            request.lr_scheduler_type,
            "--learning_rate",
            str(request.learning_rate),
            "--disable_tqdm",
            _bool_arg(request.disable_tqdm),
            "--dataloader_num_workers",
            str(request.dataloader_num_workers),
            "--label_names",
            request.label_names,
            "--save_safetensors",
            _bool_arg(request.save_safetensors),
            "--save_strategy",
            "steps",
            "--save_total_limit",
            str(request.save_total_limit),
            "--save_steps",
            str(request.save_steps),
            "--report_to",
            request.report_to,
            "--tf32",
            _bool_arg(request.tf32),
            "--bf16",
            _bool_arg(request.bf16),
        ]

        if request.enable_lora:
            command.extend(
                [
                    "--enable_lora",
                    "True",
                    "--lora_dim",
                    str(request.lora_dim),
                    "--lora_alpha",
                    str(request.lora_alpha),
                    "--lora_dropout",
                    str(request.lora_dropout),
                ]
            )
            if request.target_modules:
                command.extend(["--target_modules", json.dumps(request.target_modules)])

        return command

    def start(
        self,
        *,
        job_id: str,
        request: TrainingJobCreate,
        output_dir: Path,
        log_path: Path,
    ) -> subprocess.Popen:
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command = self.build_command(request, output_dir)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = request.gpu_device
        env["API_TRAINING_JOB_ID"] = job_id

        log_file = log_path.open("ab")
        try:
            process = subprocess.Popen(
                command,
                cwd=self.settings.research_dir,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
        except Exception:
            log_file.close()
            raise

        return _PopenWithLogFile(process, log_file)

    def cancel(self, pid: int) -> None:
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            return


class _PopenWithLogFile:
    def __init__(self, process: subprocess.Popen, log_file):
        self._process = process
        self._log_file = log_file

    @property
    def pid(self) -> int:
        return self._process.pid

    @property
    def returncode(self) -> Optional[int]:
        return self._process.returncode

    def wait(self) -> int:
        try:
            return self._process.wait()
        finally:
            self._log_file.close()

