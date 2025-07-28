#!/bin/bash

# Master script to run all dataset-model inference combinations
# Usage: ./run_all_inference.sh [dataset] [model]
# If no arguments provided, runs all combinations

datasets=("psytar" "Daniel-ML" "asylax" "n2c2_2008")
models=("Llama-3.2-1B-Instruct" "Llama-3.2-3B-Instruct" "Llama-3.1-8B-Instruct" "Llama-3.3-70B-Instruct")

target_dataset="$1"
target_model="$2"

echo "Starting inference runs..."
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
        
        script_name="generate_${dataset}_${model}.sh"
        echo "Running: $script_name"
        
        if [[ -f "./generated_inference_scripts/${script_name}" ]]; then
            chmod +x "./generated_inference_scripts/${script_name}"
            ./generated_inference_scripts/"$script_name"
            
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

echo "All inference runs completed!"
