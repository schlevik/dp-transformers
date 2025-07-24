#!/bin/bash

# Debug master script - runs quick validation tests
# Usage: ./run_debug_training.sh [dataset] [model]

datasets=("psytar" "Daniel-ml" "asylax" "n2c2") 
models=("Llama-3.2-1B-Instruct" "Llama-3.2-3B-Instruct" "Llama-3.1-8B-Instruct" "Llama-3.3-70B-Instruct")

target_dataset="$1"
target_model="$2"

echo "=== DEBUG MODE: Quick Training Validation ==="
echo "This will run very short training sessions to validate setup"
echo "Models will be deleted after successful completion"
echo ""
echo "Target dataset: ${target_dataset:-all}"
echo "Target model: ${target_model:-all}"
echo ""

failed_runs=()
successful_runs=()

for dataset in "${datasets[@]}"; do
    if [[ -n "$target_dataset" && "$dataset" != "$target_dataset" ]]; then
        continue
    fi
    
    for model in "${models[@]}"; do
        if [[ -n "$target_model" && "$model" != "$target_model" ]]; then
            continue
        fi
        
        script_name="debug_run_${dataset}_${model}.sh"
        echo "🧪 Testing: $dataset + $model"
        
        if [[ -f "$script_name" ]]; then
            chmod +x "$script_name"
            start_time=$(date +%s)
            ./"$script_name"
            end_time=$(date +%s)
            duration=$((end_time - start_time))
            
            if [[ $? -eq 0 ]]; then
                echo "✅ PASSED: $dataset + $model (${duration}s)"
                successful_runs+=("$dataset + $model")
            else
                echo "❌ FAILED: $dataset + $model (${duration}s)"
                failed_runs+=("$dataset + $model")
            fi
        else
            echo "❌ SCRIPT NOT FOUND: $script_name"
            failed_runs+=("$dataset + $model (script missing)")
        fi
        
        echo ""
    done
done

echo "=== DEBUG SUMMARY ==="
echo "Successful runs: ${#successful_runs[@]}"
for run in "${successful_runs[@]}"; do
    echo "  ✅ $run"
done

echo ""
echo "Failed runs: ${#failed_runs[@]}"
for run in "${failed_runs[@]}"; do
    echo "  ❌ $run"
done

if [[ ${#failed_runs[@]} -eq 0 ]]; then
    echo ""
    echo "🎉 All tests passed! Your setup is ready for production training."
else
    echo ""
    echo "⚠️  Some tests failed. Please check the logs above."
    exit 1
fi
