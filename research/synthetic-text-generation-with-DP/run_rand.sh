#!/usr/bin/env bash

#n2c2 batch size  2 , gradient accumulation 2
#eurlex batch size 4 , gradient accumulation 4


set -euo pipefail

dataset_name=${1:-}
user_name="srini"
gpu_device=${2:-}
start=${3:-}
end=${4:-}
rand=${5:-}
sequence_len=${6:-}
number_epochs=5
model_name=meta-llama/Llama-3.2-1B

if [ "$#" -ne 6 ]; then
    echo "Usage: $0 <dataset_name> <gpu_device> <start> <end> <rand_type> <sequence_len>" >&2
    echo "Error: All 6 arguments are required." >&2
    exit 1
fi

case "$dataset_name" in
    psytar) expected_sequence_len=128 ;;
    hoc) expected_sequence_len=128 ;;
    n2c2_2008) expected_sequence_len=3072 ;;
    Daniel-ML) expected_sequence_len=160 ;;
    asylax) expected_sequence_len=15000 ;;
    luckycat37) expected_sequence_len=2048 ;;
    tfns) expected_sequence_len=80 ;;
    eurlex) expected_sequence_len=2048 ;;
    Mimic) expected_sequence_len=4500 ;;
    *)
        echo "Error: Unknown dataset '$dataset_name'; no expected sequence length is configured." >&2
        exit 1
        ;;
esac

if ! [[ "$sequence_len" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: sequence_len must be a positive integer; got '$sequence_len'." >&2
    exit 1
fi

if [ "$sequence_len" -ne "$expected_sequence_len" ]; then
    echo "Error: sequence_len for '$dataset_name' must be $expected_sequence_len; got $sequence_len." >&2
    exit 1
fi

echo "Validated sequence_len=$sequence_len for dataset '$dataset_name'."

data_dir=/mnt/nvme1/yidan/MIA/data/cls/$dataset_name/D_sample/$rand
echo "data dir $data_dir"

for idx in $(seq $start $end); do
    dataset_file="$data_dir/dataset_${idx}.jsonl"
    for epsilon in 4; do
        echo "Processing $dataset_file with epsilon $epsilon"
        dataset_temp=$(basename $dataset_file)
        echo "dataset_temp: $dataset_temp"
        i=${dataset_temp%.jsonl}
        echo "i: $i"
        output_dir="/mnt/nvme1/srini/rerun_strategy1/$dataset_name/control_group/$rand/$i/$epsilon"
        echo "output dir: $output_dir"
        echo "epsilon: $epsilon"
        final_dir="$output_dir/final"
        if [ -d "$final_dir" ]; then
            echo "Final model already exists: $final_dir. Skipping training."
            continue
        fi
        mkdir -p "$output_dir"
        if [ "$epsilon" = "0" ]; then
            echo "no DP"
            CUDA_VISIBLE_DEVICES=$gpu_device python3 fine-tune-nodp.py \
                        --output_dir "$output_dir" \
                        --model_name $model_name \
                        --train_file "$dataset_file" \
                        --sequence_len $sequence_len \
                        --per_device_train_batch_size 4 \
                        --gradient_accumulation_steps 4 \
                        --log_level info \
                        --per_device_eval_batch_size 2 \
                        --eval_accumulation_steps 1 \
                        --seed 42 \
                        --prediction_loss_only \
                        --weight_decay 0.01 \
                        --remove_unused_columns False \
                        --num_train_epochs $number_epochs \
                        --logging_steps 5 \
                        --max_grad_norm 0 \
                        --lr_scheduler_type cosine \
                        --learning_rate 1e-4 \
                        --disable_tqdm False \
                        --dataloader_num_workers 2 \
                        --label_names labels \
                        --save_safetensors false \
                        --save_strategy steps \
                        --save_total_limit 1 \
                        --save_steps 500 \
                        --report_to none \
                        --tf32 True \
                        --bf16 True
        else
            echo "DP"
            echo "epsilon: $epsilon"
            CUDA_VISIBLE_DEVICES=$gpu_device python3 fine-tune-dp.py \
                        --output_dir "$output_dir" \
                        --model_name $model_name \
                        --train_file "$dataset_file" \
                        --sequence_len $sequence_len \
                        --per_device_train_batch_size 4 \
                        --gradient_accumulation_steps 4 \
                        --log_level info \
                        --per_device_eval_batch_size 2 \
                        --eval_accumulation_steps 1 \
                        --seed 42 \
                        --prediction_loss_only \
                        --target_epsilon "$epsilon" \
                        --per_sample_max_grad_norm 1.0 \
                        --weight_decay 0.01 \
                        --remove_unused_columns False \
                        --num_train_epochs $number_epochs \
                        --logging_steps 5 \
                        --max_grad_norm 0 \
                        --lr_scheduler_type cosine \
                        --learning_rate 1e-4 \
                        --disable_tqdm False \
                        --dataloader_num_workers 2 \
                        --label_names labels \
                        --save_safetensors false \
                        --save_strategy steps \
                        --save_total_limit 1 \
                        --save_steps 500 \
                        --report_to none \
                        --tf32 True \
                        --bf16 True
        fi
        find "$output_dir" -maxdepth 1 -type d -name 'checkpoint-*' -exec rm -rf {} +
    done
done
