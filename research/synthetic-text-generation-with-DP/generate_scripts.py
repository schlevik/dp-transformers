#!/usr/bin/env python3

import os
from pathlib import Path

# Configuration
datasets = {
    "psytar": {"max_length": 195, "batch_size": 32, 'num_epochs': 10},
    "Daniel-ml": {"max_length": 135, "batch_size": 32, 'num_epochs': 10}, 
    "asylax": {"max_length": 4141, "batch_size": 2, 'num_epochs': 10},
    "n2c2": {"max_length": 3072, "batch_size": 2, 'num_epochs': 10}
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

def generate_script(dataset_name, dataset_config, model_name, model_config):
    """Generate a bash script for a specific dataset-model combination."""
    
    # Calculate effective batch size
    base_batch_size = dataset_config["batch_size"]
    effective_batch_size = max(1, int(base_batch_size * model_config["batch_size_multiplier"]))
    
    script_content = f'''#!/bin/bash

# Training script for {dataset_name} with {model_name}
dataset_name="{dataset_name}"
sequence_len={dataset_config["max_length"]}
user_name="srini"
model_name="{model_config["name"]}"

# Dataset file path
dataset_file="data/cls/$dataset_name/original/train-original.jsonl"
echo "dataset_file: $dataset_file"

if [ -f "$dataset_file" ]; then
    echo "Processing $dataset_file with model $model_name"
    
    for epsilon in 0.5 1 2 4; do
        echo "Processing $dataset_file with epsilon $epsilon"
        output_dir="result/$dataset_name/${{model_name##*/}}/${{epsilon}}"
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
            --num_train_epochs 10 \\
            --logging_steps 5 \\
            --max_grad_norm 0 \\
            --lr_scheduler_type cosine \\
            --learning_rate 1e-4 \\
            --disable_tqdm False \\
            --dataloader_num_workers 2 \\
            --label_names labels \\
            --save_safetensors false \\
            --save_strategy steps \\
            --save_total_limit 2 \\
            --save_steps 100 \\
            --tf32 True \\
            --enable_lora \\
            --target_modules "['q_proj', 'v_proj', 'gate_proj', 'down_proj', 'up_proj', 'k_proj', 'o_proj']"
        
        # Delete checkpoint files to save space
        find "$output_dir" -maxdepth 1 -type d -name 'checkpoint-*' -exec rm -rf {{}} +
    done
    
    # No-DP Training
    echo "Starting no-DP training for $dataset_name with $model_name"
    for epsilon in "nodp"; do
        echo "Processing $dataset_file with epsilon $epsilon"
        output_dir="result/$dataset_name/${{model_name##*/}}/${{epsilon}}"
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
            --num_train_epochs {dataset_config['num_epochs']} \\
            --logging_steps 5 \\
            --max_grad_norm 0 \\
            --lr_scheduler_type cosine \\
            --learning_rate 1e-4 \\
            --disable_tqdm False \\
            --dataloader_num_workers 2 \\
            --label_names labels \\
            --save_safetensors false \\
            --save_strategy steps \\
            --save_total_limit 2 \\
            --save_steps 100 \\
            --tf32 True \\
            --enable_lora \\
            --target_modules "['q_proj', 'v_proj', 'gate_proj', 'down_proj', 'up_proj', 'k_proj', 'o_proj']"
        
        # Delete checkpoint files to save space
        find "$output_dir" -maxdepth 1 -type d -name 'checkpoint-*' -exec rm -rf {{}} +
    done
    
    echo "Completed training for $dataset_name with $model_name"
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
# Usage: ./run_all_training.sh [dataset] [model]
# If no arguments provided, runs all combinations

datasets=("psytar" "Daniel-ml" "asylax" "n2c2")
models=("Llama-3.2-1B-Instruct" "Llama-3.2-3B-Instruct" "Llama-3.1-8B-Instruct" "Llama-3.3-70B-Instruct")

target_dataset="$1"
target_model="$2"

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
        
        script_name="run_${dataset}_${model}.sh"
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

def main():
    """Generate all training scripts."""
    
    # Create output directory
    output_dir = Path("generated_scripts")
    output_dir.mkdir(exist_ok=True)
    
    print("Generating training scripts...")
    
    # Generate individual scripts for each combination
    for dataset_name, dataset_config in datasets.items():
        for model_key, model_config in models.items():
            script_content = generate_script(dataset_name, dataset_config, model_key, model_config)
            
            # Create filename
            script_filename = f"run_{dataset_name}_{model_key}.sh"
            script_path = output_dir / script_filename
            
            # Write script
            with open(script_path, 'w') as f:
                f.write(script_content)
            
            # Make executable
            os.chmod(script_path, 0o755)
            
            print(f"Generated: {script_filename}")
    
    # Generate master script
    master_script = generate_master_script()
    master_path = output_dir / "run_all_training.sh"
    
    with open(master_path, 'w') as f:
        f.write(master_script)
    
    os.chmod(master_path, 0o755)
    print(f"Generated master script: run_all_training.sh")
    
    # Generate configuration summary
    summary_content = "# Training Configuration Summary\n\n"
    summary_content += "## Datasets\n"
    for name, config in datasets.items():
        summary_content += f"- {name}: max_length={config['max_length']}, base_batch_size={config['batch_size']}\n"
    
    summary_content += "\n## Models\n"
    for name, config in models.items():
        summary_content += f"- {name}: batch_multiplier={config['batch_size_multiplier']}, grad_accum={config['grad_accum']}\n"
    
    summary_content += "\n## Generated Scripts\n"
    summary_content += f"Total scripts generated: {len(datasets) * len(models)}\n"
    summary_content += "\n## Usage\n"
    summary_content += "```bash\n"
    summary_content += "# Run all combinations\n"
    summary_content += "./run_all_training.sh\n\n"
    summary_content += "# Run specific dataset\n"
    summary_content += "./run_all_training.sh psytar\n\n"
    summary_content += "# Run specific dataset-model combination\n"
    summary_content += "./run_all_training.sh psytar Llama-3.2-1B-Instruct\n\n"
    summary_content += "# Run individual script\n"
    summary_content += "./run_psytar_Llama-3.2-1B-Instruct.sh\n"
    summary_content += "```\n"
    
    with open(output_dir / "README.md", 'w') as f:
        f.write(summary_content)
    
    print("\nGeneration complete!")
    print(f"Scripts saved to: {output_dir}")
    print(f"Total individual scripts: {len(datasets) * len(models)}")
    print("Run './generated_scripts/run_all_training.sh' to execute all combinations")

if __name__ == "__main__":
    main()