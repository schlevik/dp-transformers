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
rand_type=$5

data_dir=/mnt/nvme1/yidan/MIA/data/cls/$dataset_name/D_sample/sampled_${rand_type}_datasets
echo "Rand type: $rand_type"
for idx in $(seq $start $end); do
    dataset_file="$data_dir/dataset_$idx.jsonl"
    echo "dataset_file: $dataset_file"
    dataset_temp=$(basename $dataset_file)
    i=${dataset_temp%.jsonl}
    echo "i: $i"
    for epsilon in 0; do
        echo "Model checkpoint ${i} with noise ${epsilon}"
        output_dir="/mnt/nvme1/srini/dp_outputs/$dataset_name/control_group/sampled_${rand_type}_datasets/${i}/${epsilon}"
        output_file="$output_dir/output.jsonl"
        if [ -f "$output_file" ]; then
            echo "Output file already exists: $output_file"
            continue
        fi
        
        mkdir -p "$output_dir"
        model_dir="/mnt/nvme1/srini/rerun_strategy1/$dataset_name/control_group/sampled_${rand_type}_datasets/${i}/${epsilon}/final/"
        echo " Loading Model from $model_dir"
        CUDA_VISIBLE_DEVICES=$GPU python generate.py \
            --checkpoint_file "$model_dir" \
            --original_train_file  "$dataset_file" \
            --dataset $dataset_name \
            --output_file "$output_file" \
            --dataset_description "dptransformer" \
            --batch_size 16 \
            --max_sequence_len 160 \
            --temperature 0.9
    done
done
