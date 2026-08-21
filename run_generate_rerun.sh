dataset_name="sampled_psytar"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Read each line from rerun.txt and parse the filename
while IFS= read -r filename; do
    # Skip empty lines
    if [ -z "$filename" ]; then
        continue
    fi
    
    # Parse the filename to extract index and epsilon
    # Format: dataset_{idx}_e{eps}_samples.jsonl
    if [[ $filename =~ dataset_([0-9]+)_e([0-9.]+)_samples\.jsonl ]]; then
        idx="${BASH_REMATCH[1]}"
        epsilon="${BASH_REMATCH[2]}"
        
        echo "Processing: $filename (index: $idx, epsilon: $epsilon)"
        
        # Create output directory
        output_dir="/data/srini/$dataset_name/output_v3/${idx}/${epsilon}"
        mkdir -p "$output_dir"
        
        # only run if the index is 76 and epsilon is 0
        if [ $idx -eq 76 ] && [ $epsilon -eq 0 ]; then

            echo "Model checkpoint ${idx} with noise ${epsilon}"
            # Run inference
            CUDA_VISIBLE_DEVICES=5 python generate.py \
                --checkpoint_file "/data/srini/$dataset_name/${idx}/${epsilon}/final/" \
                --original_train_file "/home/srini/dp-transformers/$dataset_name/dataset_${idx}.jsonl" \
                --dataset psytar \
                --output_file "/data/srini/$dataset_name/output_v3/${idx}/${epsilon}/output.jsonl" \
                --dataset_description "dptransformer" \
                --batch_size 16 \
                --max_sequence_len 2048 \
                --temperature 0.9
                
            echo "Completed processing: $filename"
            echo "----------------------------------------"
        fi
    else
        echo "Warning: Could not parse filename: $filename"
    fi
done < rerun.txt

echo "All processing completed!"