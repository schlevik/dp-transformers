#!/bin/bash

# Master script to run all dataset-model combinations
# Usage: ./run_all_training.sh [dataset] [model] [debug]
# If no arguments provided, runs all combinations
# Use 'debug' as third argument to run debug versions

datasets=("psytar" "Daniel-ML" "asylax" "n2c2_2008")
models=("Llama-3.2-1B-Instruct" "Llama-3.2-3B-Instruct" "Llama-3.1-8B-Instruct" "Llama-3.3-70B-Instruct")

target_dataset="$1"
target_model="$2"
debug_mode="$3"

if [[ "$debug_mode" == "debug" ]]; then
    echo "Running in DEBUG MODE - short training with cleanup"
    script_prefix="debug_"
else
    echo "Running in PRODUCTION MODE - full training"
    script_prefix=""
fi

echo "Starting training runs..."
echo "Target dataset: ${target_dataset:-all}"
echo "Target model: ${target_model:-all}"

for dataset in "${datasets[@]}"; do
    # Skip if specific dataset requested and this isn't it
    if [[ -n "$target_dataset" && "$dataset" != "$target_dataset" ]]; then
        continue
    fi
    
    for model in "${models[@]}"; do
        # Skip if specific model requested and this isn't it
        if [[ -n "$target_model" && "$model" != "$target_model" ]]; then
            continue
        fi
        
        script_name="${script_prefix}run_${dataset}_${model}.sh"
        echo "Running: $script_name"
        
        if [[ -f "$script_name" ]]; then
            chmod +x "$script_name"
            ./"$script_name"
            
            if [[ $? -eq 0 ]]; then
                echo "✓ Completed: $script_name"
            else
                echo "✗ Failed: $script_name"
            fi
        else
            echo "✗ Script not found: $script_name"
        fi
        
        echo "---"
    done
done

echo "All training runs completed!"
