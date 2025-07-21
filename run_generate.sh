dataset_name="tfns"


dataset_name="tfns"
user_name="srini"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
for i in 0; do
    # if [ $i -eq 63 ] || [ $i -eq 64 ] || [ $i -eq 65 ] || [ $i -eq 66 ] || [ $i -eq 67 ] || [ $i -eq 68 ]; then
    #     continue
    # fi
    for epsilon in 0 0.5 1 2 4; do
        echo "Model checkpoint ${i} with noise ${epsilon}"
        output_dir="/data/$user_name/$dataset_name/output_v2/${i}/${epsilon}"
        mkdir -p "$output_dir"
        CUDA_VISIBLE_DEVICES=7 python generate.py \
            --checkpoint_file "/data/$user_name/$dataset_name/${i}/${epsilon}/final/" \
            --original_train_file  "/home/$user_name/dp-transformers/$dataset_name/train-original.jsonl" \
            --dataset nlg-reddit \
            --output_file "/data/$user_name/$dataset_name/output_v2/${i}/${epsilon}/output.jsonl" \
            --dataset_description "dptransformer" \
            --batch_size 16 \
            --max_sequence_len 2048 \
            --temperature 0.9
    done
done
# CUDA_VISIBLE_DEVICES=4 python generate.py \
#     --checkpoint_file /home/srini/dp-transformers/research/synthetic-text-generation-with-DP/result/$dataset_name/2/final \
#     --original_train_file /home/srini/dp-transformers/$dataset_name/final.csv \
#     --dataset nlg-reddit \
#     --output_file /home/srini/dp-transformers/$dataset_name/output_2.jsonl \
#     --dataset_description "dptransformer" \
#     --batch_size 16 \
#     --max_sequence_len 4300 &

# CUDA_VISIBLE_DEVICES=5 python generate.py \
#     --checkpoint_file /home/srini/dp-transformers/research/synthetic-text-generation-with-DP/result/$dataset_name/4/final \
#     --original_train_file /home/srini/dp-transformers/$dataset_name/final.csv \
#     --dataset nlg-reddit \
#     --output_file /home/srini/dp-transformers/$dataset_name/output_4.jsonl \
#     --dataset_description "dptransformer" \
#     --batch_size 16 \
#     --max_sequence_len 4300

# Wait for all background jobs to complete

