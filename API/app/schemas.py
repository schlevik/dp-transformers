from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    cancelled = "cancelled"


class TrainingJobCreate(BaseModel):
    train_file: str = Field(..., description="Path to a JSONL training file")
    target_epsilon: float = Field(..., gt=0)
    gpu_device: str = Field(default="0", description="CUDA_VISIBLE_DEVICES value")
    output_name: Optional[str] = Field(
        default=None,
        description="Safe name for a directory under the configured output root",
    )
    output_dir: Optional[str] = Field(
        default=None,
        description="Explicit output directory under the configured output root",
    )

    model_name: str = "meta-llama/Llama-3.2-1B"
    sequence_len: int = Field(default=3072, gt=0)
    per_device_train_batch_size: int = Field(default=1, gt=0)
    gradient_accumulation_steps: int = Field(default=4, gt=0)
    per_device_eval_batch_size: int = Field(default=2, gt=0)
    eval_accumulation_steps: int = Field(default=1, gt=0)
    seed: int = 42
    per_sample_max_grad_norm: float = Field(default=1.0, gt=0)
    weight_decay: float = Field(default=0.01, ge=0)
    num_train_epochs: float = Field(default=5, gt=0)
    logging_steps: int = Field(default=5, gt=0)
    learning_rate: float = Field(default=1e-4, gt=0)
    lr_scheduler_type: str = "cosine"
    dataloader_num_workers: int = Field(default=2, ge=0)
    save_steps: int = Field(default=500, gt=0)
    save_total_limit: int = Field(default=1, gt=0)
    bf16: bool = True
    tf32: bool = True
    save_safetensors: bool = False
    disable_tqdm: bool = False
    remove_unused_columns: bool = False
    prediction_loss_only: bool = True
    label_names: str = "labels"
    report_to: str = "none"

    enable_lora: bool = False
    lora_dim: int = Field(default=8, gt=0)
    lora_alpha: int = Field(default=16, gt=0)
    lora_dropout: float = Field(default=0.0, ge=0)
    target_modules: Optional[List[str]] = None


class TrainingJobSubmitted(BaseModel):
    job_id: str
    status: JobStatus


class TrainingJobResponse(BaseModel):
    job_id: str
    status: JobStatus
    created_at: str
    updated_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    pid: Optional[int] = None
    exit_code: Optional[int] = None
    output_dir: str
    log_path: str
    final_model_path: Optional[str] = None
    error: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    request: Dict[str, Any]


class TrainingJobList(BaseModel):
    jobs: List[TrainingJobResponse]


class TrainingJobLogs(BaseModel):
    job_id: str
    log_path: str
    lines: List[str]


def model_to_dict(model: BaseModel) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()

