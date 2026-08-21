#!/usr/bin/env bash

set -euo pipefail

dataset_name="psytar"
user_name="srini"
gpu_id="${1:-5}"
generation_batch_size="${GENERATION_BATCH_SIZE:-16}"
max_sequence_len="${MAX_SEQUENCE_LEN:-2048}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"
generate_script="$repo_root/generate.py"
original_train_file="/home/$user_name/dp-transformers/$dataset_name/train-original.jsonl"

export VLLM_WORKER_MULTIPROC_METHOD=spawn

for lr in 2e-5 1e-5 1e-4 2e-4; do
    for batch_size in 8 4 2; do
        for temperature in 0.4 0.6 0.8; do
            total_batch_size=$((batch_size * 4))
            model_dir="/data/$user_name/$dataset_name/lr_sweep/${lr}/epsilon_2_total_batch_size_${total_batch_size}/final"
            output_dir="/data/$user_name/$dataset_name/lr_sweep/${lr}/epsilon_2_total_batch_size_${total_batch_size}/inference"
            output_file="$output_dir/output_temperature_${temperature}.jsonl"

            if [ ! -d "$model_dir" ]; then
                echo "Skipping missing model dir: $model_dir"
                continue
            fi

            mkdir -p "$output_dir"

            echo "Running inference for lr=$lr total_batch_size=$total_batch_size temperature=$temperature on GPU $gpu_id"
            echo "  model_dir: $model_dir"
            echo "  output_file: $output_file"

            CUDA_VISIBLE_DEVICES="$gpu_id" python generate.py \
                --checkpoint_file "$model_dir" \
                --original_train_file "$original_train_file" \
                --output_file "$output_file" \
                --batch_size "$generation_batch_size" \
                --max_sequence_len "$max_sequence_len" \
                --temperature "$temperature"
        done
    done
done
