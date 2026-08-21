#!/usr/bin/env bash


dataset_name=$1

sequence_len=256
user_name="srini"
gpu_device=$2
start=$3
end=$4
model_name=allenai/OLMo-2-0425-1B
number_epochs=10
# Check that all required input variables are provided correctly
if [ -z "$dataset_name" ]; then
    echo "Error: dataset_name argument is required."
    echo "Usage: $0 <dataset_name> <gpu_device> <start> <end>"
    exit 1
fi

if [ -z "$gpu_device" ]; then
    echo "Error: gpu_device argument is required."
    echo "Usage: $0 <dataset_name> <gpu_device> <start> <end>"
    exit 1
fi

if [ -z "$start" ]; then
    echo "Error: start argument is required."
    echo "Usage: $0 <dataset_name> <gpu_device> <start> <end>"
    exit 1
fi

if [ -z "$end" ]; then
    echo "Error: end argument is required."
    echo "Usage: $0 <dataset_name> <gpu_device> <start> <end>"
    exit 1
fi

set -euo pipefail 
data_dir=/mnt/nvme1/yidan/MIA/data/cls/$dataset_name/original
for idx in $(seq $start $end); do
    dataset_file="$data_dir/train-original.jsonl"
    for epsilon in 0 0.5 1 2 4; do
        echo "Processing $dataset_file with epsilon $epsilon"
        dataset_temp=$(basename $dataset_file)
        echo "dataset_temp: $dataset_temp"
        i=${dataset_temp%.jsonl}
        echo "i: $i"
        output_dir="/mnt/nvme1/srini/rerun_strategy1/$dataset_name/$model_name/full_split/${i}/${epsilon}"
        echo "output dir: $output_dir"
        echo "epsilon: $epsilon"
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

