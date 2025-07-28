#!/bin/bash

# Inference script for psytar with Llama-3.1-8B-Instruct
dataset_name="psytar"
model_short_name="Llama-3.1-8B-Instruct"

export VLLM_WORKER_MULTIPROC_METHOD=spawn

echo "Starting inference for psytar with Llama-3.1-8B-Instruct"

for epsilon in 0.5 1 2 4 nodp; do
    echo "Processing epsilon $epsilon"
    
    # Set checkpoint path based on epsilon value
    checkpoint_dir="research/synthetic-text-generation-with-DP/result/$dataset_name/$model_short_name/$epsilon/final"
    
    # Check if checkpoint exists
    if [ ! -d "$checkpoint_dir" ]; then
        echo "Warning: Checkpoint not found at $checkpoint_dir, skipping..."
        continue
    fi
    
    # Create output directory
    output_dir="./generated-data/$dataset_name/$model_short_name/$epsilon"
    mkdir -p "$output_dir"
    
    echo "Generating synthetic data from checkpoint: $checkpoint_dir"
    echo "Output will be saved to: $output_dir/output.jsonl"
    
    CUDA_VISIBLE_DEVICES=0 python generate.py \
        --checkpoint_file "$checkpoint_dir" \
        --original_train_file "research/synthetic-text-generation-with-DP/data/cls/$dataset_name/original/train-original.jsonl" \
        --dataset "$dataset_name" \
        --output_file "$output_dir/output.jsonl" \
        --dataset_description "dptransformer" \
        --batch_size 4 \
        --max_sequence_len 195 \
        --temperature 0.9
    
    if [ $? -eq 0 ]; then
        echo "✓ Successfully generated data for epsilon=$epsilon"
    else
        echo "✗ Failed to generate data for epsilon=$epsilon"
    fi
done

echo "Completed inference for psytar with Llama-3.1-8B-Instruct"
