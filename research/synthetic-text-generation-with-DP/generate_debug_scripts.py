#!/usr/bin/env python3

import os
from pathlib import Path

# Configuration
datasets = {
    "psytar": {"max_length": 195, "batch_size": 16},
    "Daniel-ML": {"max_length": 135, "batch_size": 16}, 
    "asylax": {"max_length": 4141, "batch_size": 2},
    "n2c2_2008": {"max_length": 3072, "batch_size": 2}
}

models = {
    "Llama-3.2-1B-Instruct": {
        "name": "meta-llama/Llama-3.2-1B-Instruct",
        "batch_size_multiplier": 1.0,
        "grad_accum": 1
    },
    "Llama-3.2-3B-Instruct": {
        "name": "meta-llama/Llama-3.2-3B-Instruct", 
        "batch_size_multiplier": 0.5,
        "grad_accum": 2
    },
    "Llama-3.1-8B-Instruct": {
        "name": "meta-llama/Llama-3.1-8B-Instruct",
        "batch_size_multiplier": 0.25,
        "grad_accum": 4
    },
    "Llama-3.3-70B-Instruct": {
        "name": "meta-llama/Llama-3.3-70B-Instruct",
        "batch_size_multiplier": 0.125,
        "grad_accum": 8
    }
}

def generate_script(dataset_name, dataset_config, model_name, model_config, debug=False):
    """Generate a bash script for a specific dataset-model combination."""
    
    # Calculate effective batch size
    base_batch_size = dataset_config["batch_size"]
    effective_batch_size = max(1, int(base_batch_size * model_config["batch_size_multiplier"]))
    
    # Debug mode parameters
    debug_suffix = "_debug" if debug else ""
    dp_epochs = "0.01" if debug else "10"  # Very short for debug
    nodp_epochs = "0.01" if debug else "0.2"  # Very short for debug, slightly longer for production nodp
    save_steps = "5" if debug else "100"
    epsilon_list = "4" if debug else "0.5 1 2 4"  # Only one epsilon for debug
    result_dir = "debug_result" if debug else "result"
    
    script_content = f'''#!/bin/bash

# {"Debug " if debug else ""}Training script for {dataset_name} with {model_name}
dataset_name="{dataset_name}"
sequence_len={dataset_config["max_length"]}
user_name="srini"
model_name="{model_config["name"]}"

# Dataset file path
dataset_file="data/cls/$dataset_name/original/train-original.jsonl"
echo "dataset_file: $dataset_file"

if [ -f "$dataset_file" ]; then
    echo "Processing $dataset_file with model $model_name{" (DEBUG MODE)" if debug else ""}"
    
    for epsilon in {epsilon_list}; do
        echo "Processing $dataset_file with epsilon $epsilon"
        output_dir="{result_dir}/$dataset_name/${{model_name##*/}}/${{epsilon}}"
        mkdir -p "$output_dir"
        
        CUDA_VISIBLE_DEVICES=0 python fine-tune-dp.py \\
            --output_dir "$output_dir" \\
            --model_name "$model_name" \\
            --train_file "$dataset_file" \\
            --sequence_len $sequence_len \\
            --per_device_train_batch_size {effective_batch_size} \\
            --gradient_accumulation_steps {model_config["grad_accum"]} \\
            --log_level info \\
            --per_device_eval_batch_size 2 \\
            --eval_accumulation_steps 1 \\
            --seed 42 \\
            --prediction_loss_only \\
            --target_epsilon $epsilon \\
            --per_sample_max_grad_norm 1.0 \\
            --weight_decay 0.01 \\
            --remove_unused_columns False \\
            --num_train_epochs {nodp_epochs} \\
            --logging_steps 2 \\
            --max_grad_norm 0 \\
            --lr_scheduler_type cosine \\
            --learning_rate 1e-4 \\
            --disable_tqdm False \\
            --dataloader_num_workers 2 \\
            --label_names labels \\
            --save_safetensors false \\
            --save_strategy steps \\
            --save_total_limit 1 \\
            --save_steps {save_steps} \\
            --tf32 True \\
            --enable_lora \\
            --target_modules "['q_proj', 'v_proj', 'gate_proj', 'down_proj', 'up_proj', 'k_proj', 'o_proj']"
        
        # Delete checkpoint files to save space
        find "$output_dir" -maxdepth 1 -type d -name 'checkpoint-*' -exec rm -rf {{}} +
        
        {"# Debug mode: delete model after successful training" if debug else ""}
        {"if [ $? -eq 0 ]; then" if debug else ""}
            {"echo \"✓ Training completed successfully for epsilon $epsilon - cleaning up debug files\"" if debug else ""}
            {"rm -rf \"$output_dir\"" if debug else ""}
            {"echo \"✓ Debug files cleaned up for epsilon $epsilon\"" if debug else ""}
        {"else" if debug else ""}
            {"echo \"✗ Training failed for epsilon $epsilon - keeping files for debugging\"" if debug else ""}
        {"fi" if debug else ""}
    done
    
    # No-DP Training
    echo "Starting no-DP training for $dataset_name with $model_name{" (DEBUG MODE)" if debug else ""}"
    for epsilon in "nodp"; do
        echo "Processing $dataset_file with epsilon $epsilon"
        output_dir="{result_dir}/$dataset_name/${{model_name##*/}}/${{epsilon}}"
        mkdir -p "$output_dir"
        
        CUDA_VISIBLE_DEVICES=0 python fine-tune-nodp.py \\
            --output_dir "$output_dir" \\
            --model_name "$model_name" \\
            --train_file "$dataset_file" \\
            --sequence_len $sequence_len \\
            --per_device_train_batch_size {effective_batch_size} \\
            --gradient_accumulation_steps {model_config["grad_accum"]} \\
            --log_level info \\
            --per_device_eval_batch_size 2 \\
            --eval_accumulation_steps 1 \\
            --seed 42 \\
            --prediction_loss_only \\
            --weight_decay 0.01 \\
            --remove_unused_columns False \\
            --num_train_epochs {dp_epochs} \\
            --logging_steps 2 \\
            --max_grad_norm 0 \\
            --lr_scheduler_type cosine \\
            --learning_rate 1e-4 \\
            --disable_tqdm False \\
            --dataloader_num_workers 2 \\
            --label_names labels \\
            --save_safetensors false \\
            --save_strategy steps \\
            --save_total_limit 1 \\
            --save_steps {save_steps} \\
            --tf32 True \\
            --enable_lora \\
            --target_modules "['q_proj', 'v_proj', 'gate_proj', 'down_proj', 'up_proj', 'k_proj', 'o_proj']"
        
        # Delete checkpoint files to save space
        find "$output_dir" -maxdepth 1 -type d -name 'checkpoint-*' -exec rm -rf {{}} +
        
        {"# Debug mode: delete model after successful training" if debug else ""}
        {"if [ $? -eq 0 ]; then" if debug else ""}
            {"echo \"✓ No-DP training completed successfully - cleaning up debug files\"" if debug else ""}
            {"rm -rf \"$output_dir\"" if debug else ""}
            {"echo \"✓ Debug files cleaned up for no-DP\"" if debug else ""}
        {"else" if debug else ""}
            {"echo \"✗ No-DP training failed - keeping files for debugging\"" if debug else ""}
        {"fi" if debug else ""}
    done
    
    echo "Completed{"debug " if debug else " "}training for $dataset_name with $model_name"
else
    echo "Dataset file not found: $dataset_file"
    exit 1
fi
'''
    
    return script_content

def generate_master_script():
    """Generate a master script that runs all combinations."""
    
    master_content = '''#!/bin/bash

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
'''
    
    return master_content

def generate_debug_master_script():
    """Generate a debug-specific master script."""
    
    debug_master_content = '''#!/bin/bash

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
'''
    
    return debug_master_content

def main():
    """Generate all training scripts."""
    
    # Create output directory
    output_dir = Path("generated_scripts")
    output_dir.mkdir(exist_ok=True)
    
    print("Generating training scripts...")
    
    # Generate individual scripts for each combination (both regular and debug)
    for dataset_name, dataset_config in datasets.items():
        for model_key, model_config in models.items():
            # # Regular script
            # script_content = generate_script(dataset_name, dataset_config, model_key, model_config, debug=False)
            # script_filename = f"run_{dataset_name}_{model_key}.sh"
            # script_path = output_dir / script_filename
            
            # with open(script_path, 'w') as f:
            #     f.write(script_content)
            # os.chmod(script_path, 0o755)
            # print(f"Generated: {script_filename}")
            
            # Debug script
            debug_script_content = generate_script(dataset_name, dataset_config, model_key, model_config, debug=True)
            debug_script_filename = f"debug_run_{dataset_name}_{model_key}.sh"
            debug_script_path = output_dir / debug_script_filename
            
            with open(debug_script_path, 'w') as f:
                f.write(debug_script_content)
            os.chmod(debug_script_path, 0o755)
            print(f"Generated: {debug_script_filename}")
    
    # Generate master script
    master_script = generate_master_script()
    master_path = output_dir / "run_all_training.sh"
    
    with open(master_path, 'w') as f:
        f.write(master_script)
    os.chmod(master_path, 0o755)
    print(f"Generated master script: run_all_training.sh")
    
    # Generate debug master script
    debug_master_script = generate_debug_master_script()
    debug_master_path = output_dir / "run_debug_training.sh"
    
    with open(debug_master_path, 'w') as f:
        f.write(debug_master_script)
    os.chmod(debug_master_path, 0o755)
    print(f"Generated debug master script: run_debug_training.sh")
    
    # Generate configuration summary
    summary_content = "# Training Configuration Summary\n\n"
    summary_content += "## Datasets\n"
    for name, config in datasets.items():
        summary_content += f"- {name}: max_length={config['max_length']}, base_batch_size={config['batch_size']}\n"
    
    summary_content += "\n## Models\n"
    for name, config in models.items():
        summary_content += f"- {name}: batch_multiplier={config['batch_size_multiplier']}, grad_accum={config['grad_accum']}\n"
    
    summary_content += "\n## Generated Scripts\n"
    summary_content += f"Total production scripts: {len(datasets) * len(models)}\n"
    summary_content += f"Total debug scripts: {len(datasets) * len(models)}\n"
    summary_content += f"Total scripts: {len(datasets) * len(models) * 2}\n"
    
    summary_content += "\n## Debug vs Production\n"
    summary_content += "**Debug scripts:**\n"
    summary_content += "- Train for only 0.01 epochs (~few steps)\n"
    summary_content += "- Test only epsilon=4 and nodp\n"
    summary_content += "- Save to `debug_result/` folder\n"
    summary_content += "- Auto-delete models after successful completion\n"
    summary_content += "- Keep failed models for debugging\n"
    summary_content += "- Perfect for validating setup\n\n"
    summary_content += "**Production scripts:**\n"
    summary_content += "- Train for full 10 epochs\n"
    summary_content += "- Test all epsilon values: 0.5, 1, 2, 4, nodp\n"
    summary_content += "- Save to `result/` folder\n"
    summary_content += "- Keep all models\n"
    summary_content += "- For actual experiments\n"
    
    summary_content += "\n## Usage\n"
    summary_content += "```bash\n"
    summary_content += "# Test everything quickly (recommended first step)\n"
    summary_content += "./run_debug_training.sh\n\n"
    summary_content += "# Test specific dataset\n"
    summary_content += "./run_debug_training.sh psytar\n\n"
    summary_content += "# Test specific combination\n"
    summary_content += "./run_debug_training.sh psytar Llama-3.2-1B-Instruct\n\n"
    summary_content += "# Run production training (after debug passes)\n"
    summary_content += "./run_all_training.sh\n\n"
    summary_content += "# Run production with debug flag\n"
    summary_content += "./run_all_training.sh '' '' debug\n\n"
    summary_content += "# Run individual debug script\n"
    summary_content += "./debug_run_psytar_Llama-3.2-1B-Instruct.sh\n"
    summary_content += "```\n"
    
    with open(output_dir / "README.md", 'w') as f:
        f.write(summary_content)
    
    print("\nGeneration complete!")
    print(f"Scripts saved to: {output_dir}")
    print(f"Total individual scripts: {len(datasets) * len(models) * 2} (production + debug)")
    print("\n🧪 RECOMMENDED: Run './generated_scripts/run_debug_training.sh' first to validate setup")
    print("✅ Then run './generated_scripts/run_all_training.sh' for production training")

if __name__ == "__main__":
    main()