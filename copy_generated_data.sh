#!/bin/bash

# Script to copy generated data to research folder with proper structure
# Usage: ./copy_generated_data.sh

SOURCE_DIR="generated-data"
DEST_BASE="research/synthetic-text-generation-with-DP/data/cls"

# Check if source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: Source directory '$SOURCE_DIR' does not exist"
    exit 1
fi

# Process each dataset folder
for dataset_dir in "$SOURCE_DIR"/*; do
    if [ -d "$dataset_dir" ]; then
        dataset_name=$(basename "$dataset_dir")
        echo "Processing dataset: $dataset_name"
        
        # Process each model folder within the dataset
        for model_dir in "$dataset_dir"/*; do
            if [ -d "$model_dir" ]; then
                model_name=$(basename "$model_dir")
                echo "  Processing model: $model_name"
                
                # Create destination folder with dp-transformers-{model-name} format
                dest_approach_dir="$DEST_BASE/$dataset_name/dp-transformers-$model_name"
                mkdir -p "$dest_approach_dir"
                
                # Process each epsilon folder (including nodp)
                for epsilon_dir in "$model_dir"/*; do
                    if [ -d "$epsilon_dir" ]; then
                        epsilon_value=$(basename "$epsilon_dir")
                        
                        # Look for output.jsonl file
                        if [ -f "$epsilon_dir/output.jsonl" ]; then
                            # Determine output filename based on epsilon value
                            if [ "$epsilon_value" = "nodp" ]; then
                                output_file="trainnodp-dp-transformers-$model_name.jsonl"
                            else
                                output_file="train${epsilon_value}-dp-transformers-$model_name.jsonl"
                            fi
                            
                            # Copy the file with the new name
                            cp "$epsilon_dir/output.jsonl" "$dest_approach_dir/$output_file"
                            echo "    Copied: $epsilon_dir/output.jsonl -> $dest_approach_dir/$output_file"
                        else
                            echo "    Warning: No output.jsonl found in $epsilon_dir"
                        fi
                    fi
                done
            fi
        done
    fi
done

echo "Copy operation completed!"