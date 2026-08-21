#!/bin/bash


user_name="srini"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

    # if [ $i -eq 63 ] || [ $i -eq 64 ] || [ $i -eq 65 ] || [ $i -eq 66 ] || [ $i -eq 67 ] || [ $i -eq 68 ]; then
    #     continue
    # fii=0
dataset_name=$1
GPU=$2
start=$3
end=$4

# Ensure all 4 arguments are provided by the user
if [ -z "$dataset_name" ] || [ -z "$GPU" ] || [ -z "$start" ] || [ -z "$end" ]; then
    echo "Usage: $0 <dataset_name> <GPU> <start> <end>"
    echo "Error: All 4 arguments are required."
    exit 1
fi
set -euo pipefail 


sequence_len=256
model_name=allenai/OLMo-2-0425-1B
data_dir=/mnt/nvme1/yidan/MIA/data/cls/$dataset_name/original

for idx in $(seq $start $end); do
    dataset_file="$data_dir/train-original.jsonl"
    echo "dataset_file: $dataset_file"
    dataset_temp=$(basename $dataset_file)
    i=${dataset_temp%.jsonl}
    echo "i: $i"
    for epsilon in 0 0.5 1 2 4; do
        echo "Model checkpoint ${i} with noise ${epsilon}"
        output_dir="/mnt/nvme1/srini/dp_outputs/$dataset_name/$model_name/full_split/10_epochs/${i}/${epsilon}"
        output_file="$output_dir/output.jsonl"
        if [ -f "$output_file" ]; then
            echo "Output file already exists: $output_file"
            continue
        fi
        mkdir -p "$output_dir"
        model_dir="/mnt/nvme1/srini/rerun_strategy1/$dataset_name/$model_name/full_split/${i}/${epsilon}/final/"
        echo " Loading Model from $model_dir"
        CUDA_VISIBLE_DEVICES=$GPU python generate.py \
            --checkpoint_file "$model_dir" \
            --original_train_file  "$dataset_file" \
            --dataset $dataset_name \
            --output_file "$output_file" \
            --dataset_description "dptransformer" \
            --batch_size 16 \
            --max_sequence_len $sequence_len \
            --temperature 0.9
    done
done
