dataset_name="psytar"
user_name="srini"

export VLLM_WORKER_MULTIPROC_METHOD=spawn

    # if [ $i -eq 63 ] || [ $i -eq 64 ] || [ $i -eq 65 ] || [ $i -eq 66 ] || [ $i -eq 67 ] || [ $i -eq 68 ]; then
    #     continue
    # fi
for leakage in 25 50 75; do
    for i in {1..10}; do
        dataset_file="/data/yidan/synth-data/leakage/psytar/${leakage}/train-original-partialcleaned-${leakage}-${i}.jsonl"
        output_dir="/data/$user_name/$dataset_name/leakage/${leakage}/${i}"
        echo "dataset_file: $dataset_file"
        if [ -f "$dataset_file" ]; then
            echo "Processing $dataset_file"
            CUDA_VISIBLE_DEVICES=1 python generate.py \
                --checkpoint_file "/data/$user_name/$dataset_name/leakage/${leakage}/${i}/final/" \
                --original_train_file  "$dataset_file" \
                --dataset psytar \
                --output_file "/data/$user_name/$dataset_name/leakage/${leakage}/${i}/output.jsonl"
        fi
    done
done

