#!/usr/bin/env python3

import os
from pathlib import Path

# Configuration
datasets = {
    "psytar": {"max_length": 195, "batch_size": 16, "temperature": 0.9},
    "Daniel-ML": {"max_length": 135, "batch_size": 16, "temperature": 0.9}, 
    "asylax": {"max_length": 4141, "batch_size": 4, "temperature": 0.9},
    "n2c2_2008": {"max_length": 3072, "batch_size": 4, "temperature": 0.9}
}

models = {
    "Llama-3.2-1B-Instruct": {
        "name": "meta-llama/Llama-3.2-1B-Instruct",
        "batch_size_multiplier": 1.0
    },
    "Llama-3.2-3B-Instruct": {
        "name": "meta-llama/Llama-3.2-3B-Instruct", 
        "batch_size_multiplier": 0.5
    },
    "Llama-3.1-8B-Instruct": {
        "name": "meta-llama/Llama-3.1-8B-Instruct",
        "batch_size_multiplier": 0.25
    },
    "Llama-3.3-70B-Instruct": {
        "name": "meta-llama/Llama-3.3-70B-Instruct",
        "batch_size_multiplier": 0.125
    }
}

# Epsilon values to generate for
epsilon_values = ["0.5", "1", "2", "4", "nodp"]

def generate_inference_script(dataset_name, dataset_config, model_name, model_config):
    """Generate a bash script for inference for a specific dataset-model combination."""
    
    # Calculate effective batch size
    base_batch_size = dataset_config["batch_size"]
    effective_batch_size = max(1, int(base_batch_size * model_config["batch_size_multiplier"]))
    
    script_content = f'''#!/bin/bash

# Inference script for {dataset_name} with {model_name}
dataset_name="{dataset_name}"
model_short_name="{model_name}"

export VLLM_WORKER_MULTIPROC_METHOD=spawn

echo "Starting inference for {dataset_name} with {model_name}"

for epsilon in {' '.join(epsilon_values)}; do
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
    
    CUDA_VISIBLE_DEVICES=0 python generate.py \\
        --checkpoint_file "$checkpoint_dir" \\
        --original_train_file "research/synthetic-text-generation-with-DP/data/cls/$dataset_name/original/train-original.jsonl" \\
        --dataset "$dataset_name" \\
        --output_file "$output_dir/output.jsonl" \\
        --dataset_description "dptransformer" \\
        --batch_size {effective_batch_size} \\
        --max_sequence_len {dataset_config["max_length"]} \\
        --temperature {dataset_config["temperature"]}
    
    if [ $? -eq 0 ]; then
        echo "✓ Successfully generated data for epsilon=$epsilon"
    else
        echo "✗ Failed to generate data for epsilon=$epsilon"
    fi
done

echo "Completed inference for {dataset_name} with {model_name}"
'''
    
    return script_content

def generate_master_inference_script():
    """Generate a master script that runs all inference combinations."""
    
    master_content = '''#!/bin/bash

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

echo "All inference runs completed!"
'''
    
    return master_content

def main():
    """Generate all inference scripts."""
    
    # Create output directory
    output_dir = Path("generated_inference_scripts")
    output_dir.mkdir(exist_ok=True)
    
    print("Generating inference scripts...")
    
    # Generate individual scripts for each combination
    for dataset_name, dataset_config in datasets.items():
        for model_key, model_config in models.items():
            script_content = generate_inference_script(dataset_name, dataset_config, model_key, model_config)
            
            # Create filename
            script_filename = f"generate_{dataset_name}_{model_key}.sh"
            script_path = output_dir / script_filename
            
            # Write script
            with open(script_path, 'w') as f:
                f.write(script_content)
            
            # Make executable
            os.chmod(script_path, 0o755)
            
            print(f"Generated: {script_filename}")
    
    # Generate master script
    master_script = generate_master_inference_script()
    master_path = output_dir / "run_all_inference.sh"
    
    with open(master_path, 'w') as f:
        f.write(master_script)
    
    os.chmod(master_path, 0o755)
    print(f"Generated master script: run_all_inference.sh")
    
    # Generate configuration summary
    summary_content = "# Inference Configuration Summary\n\n"
    summary_content += "## Datasets\n"
    for name, config in datasets.items():
        summary_content += f"- {name}: max_length={config['max_length']}, batch_size={config['batch_size']}, temperature={config['temperature']}\n"
    
    summary_content += "\n## Models\n"
    for name, config in models.items():
        summary_content += f"- {name}: batch_multiplier={config['batch_size_multiplier']}\n"
    
    summary_content += "\n## Epsilon Values\n"
    summary_content += f"- {', '.join(epsilon_values)}\n"
    
    summary_content += "\n## Expected Model Paths\n"
    summary_content += "Models should be located at:\n"
    summary_content += "`research/synthetic-text-generation-with-DP/result/{dataset}/{model}/{epsilon}/final/`\n"
    
    summary_content += "\n## Generated Scripts\n"
    summary_content += f"Total scripts generated: {len(datasets) * len(models)}\n"
    summary_content += "\n## Usage\n"
    summary_content += "```bash\n"
    summary_content += "# Run all combinations\n"
    summary_content += "./run_all_inference.sh\n\n"
    summary_content += "# Run specific dataset\n"
    summary_content += "./run_all_inference.sh psytar\n\n"
    summary_content += "# Run specific dataset-model combination\n"
    summary_content += "./run_all_inference.sh psytar Llama-3.2-1B-Instruct\n\n"
    summary_content += "# Run individual script\n"
    summary_content += "./generate_psytar_Llama-3.2-1B-Instruct.sh\n"
    summary_content += "```\n"
    
    with open(output_dir / "README.md", 'w') as f:
        f.write(summary_content)
    
    print("\nGeneration complete!")
    print(f"Scripts saved to: {output_dir}")
    print(f"Total individual scripts: {len(datasets) * len(models)}")
    print("Run './generated_inference_scripts/run_all_inference.sh' to execute all combinations")

if __name__ == "__main__":
    main()