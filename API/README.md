# DP Training API

This API lets a frontend trigger one local GPU training run at a time using the existing training script at `research/synthetic-text-generation-with-DP/fine-tune-dp.py`.

## Install

Install the API dependencies:

```bash
pip install -r API/requirements.txt
```

Install the training dependencies from the research project as needed:

```bash
pip install -r research/synthetic-text-generation-with-DP/requirements.txt
```

## Configure

The API is configured through environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `API_DATA_ROOT` | repo root | Directory that training `.jsonl` files must live under. |
| `API_OUTPUT_ROOT` | `API/runtime/outputs` | Directory where model outputs are written. |
| `API_RUNTIME_ROOT` | `API/runtime` | Directory for logs and the SQLite job database. |
| `API_DATABASE_PATH` | `API/runtime/training_jobs.sqlite3` | SQLite job metadata database. |
| `API_TRAINING_SCRIPT` | `research/synthetic-text-generation-with-DP/fine-tune-dp.py` | Training script path. |
| `API_TRAINING_PYTHON` | current Python executable | Python executable used for the training subprocess. |
| `API_DEFAULT_GPU_DEVICE` | `0` | Default `CUDA_VISIBLE_DEVICES` value. |
| `API_DEFAULT_MODEL_NAME` | `meta-llama/Llama-3.2-1B` | Default Hugging Face model. |
| `API_DEFAULT_SEQUENCE_LEN` | `3072` | Default sequence length. |

Example:

```bash
export API_DATA_ROOT=/mnt/nvme1/yidan/MIA/data
export API_OUTPUT_ROOT=/mnt/nvme1/srini/training-api-outputs
export API_DEFAULT_GPU_DEVICE=0
uvicorn app.main:app --app-dir API --host 0.0.0.0 --port 8000
```

## Submit a Job

```bash
curl -X POST http://localhost:8000/training-jobs \
  -H 'content-type: application/json' \
  -d '{
    "train_file": "/mnt/nvme1/yidan/MIA/data/cls/example.jsonl",
    "target_epsilon": 2,
    "gpu_device": "0",
    "output_name": "example-eps-2"
  }'
```

The response includes a `job_id`:

```json
{
  "job_id": "f4f7b8423dc849d293b64c3d9d7b52c8",
  "status": "queued"
}
```

## Frontend Updates

Use server-sent events for live status and log updates:

```ts
const events = new EventSource(`/training-jobs/${jobId}/events`);

events.addEventListener("status", (event) => {
  const status = JSON.parse(event.data);
  console.log(status.status);
});

events.addEventListener("log", (event) => {
  const line = JSON.parse(event.data);
  console.log(line.message);
});

events.addEventListener("done", (event) => {
  const finalJob = JSON.parse(event.data);
  console.log(finalJob.final_model_path);
  events.close();
});
```

Polling endpoints are also available:

```bash
GET /training-jobs/{job_id}
GET /training-jobs/{job_id}/logs?tail=200
POST /training-jobs/{job_id}/cancel
```

## Notes

- The worker runs one subprocess at a time per API process.
- Input files must be `.jsonl` files under `API_DATA_ROOT`.
- Output directories must be under `API_OUTPUT_ROOT` and cannot already contain files.
- Training stdout/stderr are captured to `API/runtime/logs/{job_id}.log`.
- Successful jobs expose the final model path as `<output_dir>/final`.

