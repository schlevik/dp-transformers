#!/bin/bash

# Debug Training script for n2c2 with Llama-3.2-3B-Instruct
dataset_name="n2c2"
sequence_len=3072
user_name="srini"
model_name="meta-llama/Llama-3.2-3B-Instruct"

# Dataset file path
dataset_file="data/cls/$dataset_name/original/train-original.jsonl"
echo "dataset_file: $dataset_file"

if [ -f "$dataset_file" ]; then
    echo "Processing $dataset_file with model $model_name (DEBUG MODE)"
    
    for epsilon in 4; do
        echo "Processing $dataset_file with epsilon $epsilon"
        output_dir="debug_result/$dataset_name/${model_name##*/}/${epsilon}"
        mkdir -p "$output_dir"
        
        CUDA_VISIBLE_DEVICES=0 python fine-tune-dp.py \
            --output_dir "$output_dir" \
            --model_name "$model_name" \
            --train_file "$dataset_file" \
            --sequence_len $sequence_len \
            --per_device_train_batch_size 1 \
            --gradient_accumulation_steps 2 \
            --log_level info \
            --per_device_eval_batch_size 2 \
            --eval_accumulation_steps 1 \
            --seed 42 \
            --prediction_loss_only \
            --target_epsilon $epsilon \
            --per_sample_max_grad_norm 1.0 \
            --weight_decay 0.01 \
            --remove_unused_columns False \
            --num_train_epochs 0.01 \
            --logging_steps 2 \
            --max_grad_norm 0 \
            --lr_scheduler_type cosine \
            --learning_rate 1e-4 \
            --disable_tqdm False \
            --dataloader_num_workers 2 \
            --label_names labels \
            --save_safetensors false \
            --save_strategy steps \
            --save_total_limit 1 \
            --save_steps 5 \
            --tf32 True \
            --enable_lora \
            --target_modules "['q_proj', 'v_proj', 'gate_proj', 'down_proj', 'up_proj', 'k_proj', 'o_proj']"
        
        # Delete checkpoint files to save space
        find "$output_dir" -maxdepth 1 -type d -name 'checkpoint-*' -exec rm -rf {} +
        
        # Debug mode: delete model after successful training
        if [ $? -eq 0 ]; then
            echo "✓ Training completed successfully for epsilon $epsilon - cleaning up debug files"
            rm -rf "$output_dir"
            echo "✓ Debug files cleaned up for epsilon $epsilon"
        else
            echo "✗ Training failed for epsilon $epsilon - keeping files for debugging"
        fi
    done
    
    # No-DP Training
    echo "Starting no-DP training for $dataset_name with $model_name (DEBUG MODE)"
    for epsilon in "nodp"; do
        echo "Processing $dataset_file with epsilon $epsilon"
        output_dir="debug_result/$dataset_name/${model_name##*/}/${epsilon}"
        mkdir -p "$output_dir"
        
        CUDA_VISIBLE_DEVICES=0 python fine-tune-nodp.py \
            --output_dir "$output_dir" \
            --model_name "$model_name" \
            --train_file "$dataset_file" \
            --sequence_len $sequence_len \
            --per_device_train_batch_size 1 \
            --gradient_accumulation_steps 2 \
            --log_level info \
            --per_device_eval_batch_size 2 \
            --eval_accumulation_steps 1 \
            --seed 42 \
            --prediction_loss_only \
            --weight_decay 0.01 \
            --remove_unused_columns False \
            --num_train_epochs 0.01 \
            --logging_steps 2 \
            --max_grad_norm 0 \
            --lr_scheduler_type cosine \
            --learning_rate 1e-4 \
            --disable_tqdm False \
            --dataloader_num_workers 2 \
            --label_names labels \
            --save_safetensors false \
            --save_strategy steps \
            --save_total_limit 1 \
            --save_steps 5 \
            --tf32 True \
            --enable_lora \
            --target_modules "['q_proj', 'v_proj', 'gate_proj', 'down_proj', 'up_proj', 'k_proj', 'o_proj']"
        
        # Delete checkpoint files to save space
        find "$output_dir" -maxdepth 1 -type d -name 'checkpoint-*' -exec rm -rf {} +
        
        # Debug mode: delete model after successful training
        if [ $? -eq 0 ]; then
            echo "✓ No-DP training completed successfully - cleaning up debug files"
            rm -rf "$output_dir"
            echo "✓ Debug files cleaned up for no-DP"
        else
            echo "✗ No-DP training failed - keeping files for debugging"
        fi
    done
    
    echo "Completeddebug training for $dataset_name with $model_name"
else
    echo "Dataset file not found: $dataset_file"
    exit 1
fi
