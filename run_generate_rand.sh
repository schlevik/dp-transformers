#!/bin/bash

set -euo pipefail


user_name="srini"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

    # if [ $i -eq 63 ] || [ $i -eq 64 ] || [ $i -eq 65 ] || [ $i -eq 66 ] || [ $i -eq 67 ] || [ $i -eq 68 ]; then
    #     continue
    # fii=0
dataset_name=${1:-}
GPU=${2:-}
start=${3:-}
end=${4:-}
rand_type=${5:-}
sequence_len=${6:-}



# Ensure all 6 arguments are provided by the user
if [ "$#" -ne 6 ]; then
    echo "Usage: $0 <dataset_name> <GPU> <start> <end> <rand_type> <sequence_len>" >&2
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


data_dir=/mnt/nvme1/yidan/MIA/data/cls/$dataset_name/D_sample/$rand_type

for idx in $(seq $start $end); do
    dataset_file="$data_dir/dataset_${idx}.jsonl"
    echo "dataset_file: $dataset_file"
    dataset_temp=$(basename $dataset_file)
    i=${dataset_temp%.jsonl}
    echo "i: $i"
    for epsilon in 0 1 4; do
        echo "Model checkpoint ${i} with noise ${epsilon}"
        output_dir="/mnt/nvme1/srini/dp_outputs/$dataset_name/control_group/$rand_type/${i}/${epsilon}"
        output_file="$output_dir/output.jsonl"
        if [ -f "$output_file" ]; then
            echo "Output file already exists: $output_file"
            continue
        fi
        mkdir -p "$output_dir"
        model_dir="/mnt/nvme1/srini/rerun_strategy1/$dataset_name/control_group/$rand_type/$i/${epsilon}/final/"
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
