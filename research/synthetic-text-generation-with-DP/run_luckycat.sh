dataset_name="luckycat"
echo "dataset_name: $dataset_name"
# Iterate through all dataset files in the folder
for i in 0; do
    dataset_file="/home/srini/dp-transformers/$dataset_name/train_filtered_2048.jsonl"
    if [ -f "$dataset_file" ]; then
        echo "Processing $dataset_file"
        for epsilon in 3.36 6.19 11.56 20.0; do
            echo "Processing $dataset_file with epsilon $epsilon"
            output_dir="/data/srini/$dataset_name/${i}/${epsilon}"
            mkdir -p "$output_dir"
            CUDA_VISIBLE_DEVICES=3,5,7 python3 -m torch.distributed.run --master_port 29501 --nproc_per_node 3 fine-tune-dp.py \
                --output_dir "$output_dir" \
                --model_name meta-llama/Llama-3.2-1B \
                --train_file "$dataset_file" \
                --sequence_len 2048 \
                --per_device_train_batch_size 1 \
                --gradient_accumulation_steps 2 \
                --log_level info \
                --per_device_eval_batch_size 4 \
                --seed 42 \
                --target_epsilon $epsilon \
                --per_sample_max_grad_norm 1.0 \
                --weight_decay 0.01 \
                --remove_unused_columns False \
                --num_train_epochs 10 \
                --logging_steps 5 \
                --max_grad_norm 0 \
                --lr_scheduler_type constant \
                --learning_rate 0.0001 \
                --save_strategy epoch \
                --save_total_limit 1 \
                --save_safetensors False \
                --dataloader_num_workers 2 \
                --disable_tqdm True
            # delete checkpoint files
            find "$output_dir" -maxdepth 1 -type d -name 'checkpoint-*' -exec rm -rf {} +
        done
    fi
done

# for epsilon in "2.96 5.34 9.77 18.01"; do
#     CUDA_VISIBLE_DEVICES=0,1,2,4 python3 -m torch.distributed.run --nproc_per_node 4 fine-tune-dp.py \
#         --output_dir result/$dataset_name/$epsilon \
#         --model_name meta-llama/Llama-3.2-1B  \
#         --train_file /home/srini/dp-transformers/$dataset_name/dataset_0.jsonl \
#         --sequence_len 1024 \
#         --per_device_train_batch_size 2 \
#         --gradient_accumulation_steps 4 \
#         --log_level info \
#         --per_device_eval_batch_size 1 \
#         --eval_accumulation_steps 1 \
#         --seed 42 \
#         --target_epsilon $epsilon \
#         --per_sample_max_grad_norm 1.0 \
#         --weight_decay 0.01 \
#         --remove_unused_columns False \
#         --num_train_epochs 10 \
#         --logging_steps 5 \
#         --max_grad_norm 0 \
#         --lr_scheduler_type constant \
#         --learning_rate 0.0001 \
#         --save_strategy epoch \
#         --save_total_limit 1 \
#         --save_safetensors False \
#         --dataloader_num_workers 2 \
#         --load_best_model_at_end True \
#         --disable_tqdm True
# done



# epsilon=0
# dataset_name="asylex"
# CUDA_VISIBLE_DEVICES=0,1,3,4,5,6 python3 -m torch.distributed.run --nproc_per_node 6 fine-tune-nodp.py \
#     --output_dir result/$dataset_name/$epsilon \
#     --model_name meta-llama/Llama-3.2-1B  \
#     --train_file /home/srini/dp-transformers/$dataset_name/final.csv \
#     --sequence_len 4300 \
#     --per_device_train_batch_size 1 \
#     --gradient_accumulation_steps 4 \
#     --log_level info \
#     --seed 42 \
#     --weight_decay 0.01 \
#     --remove_unused_columns False \
#     --num_train_epochs 5 \
#     --logging_steps 5 \
#     --max_grad_norm 0 \
#     --lr_scheduler_type constant \
#     --learning_rate 0.0001 \
#     --save_strategy epoch \
#     --save_safetensors False \
#     --dataloader_num_workers 2 \
#     --disable_tqdm True 

