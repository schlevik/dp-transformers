dataset_name="psytar"
sequence_len=256
user_name="srini"
# Iterate through all dataset files in the folder
    dataset_file="data/cls/$dataset_name/original/train-original.jsonl"
    echo "dataset_file: $dataset_file"
    if [ -f "$dataset_file" ]; then
        echo "Processing $dataset_file"
        for epsilon in 0.5 1 2 4; do #0.5 1 2 4; do
            echo "Processing $dataset_file with epsilon $epsilon"
            output_dir="result/$dataset_name/${epsilon}"
            mkdir -p "$output_dir"
            CUDA_VISIBLE_DEVICES=0 python fine-tune-dp.py \
                --output_dir "$output_dir"  \
                --model_name meta-llama/Llama-3.2-1B \
                --train_file "$dataset_file" \
                --sequence_len $sequence_len \
                --per_device_train_batch_size 16 \
                --gradient_accumulation_steps 1 \
                --log_level info \
                --per_device_eval_batch_size 2 \
                --eval_accumulation_steps 1 \
                --seed 42 \
                --prediction_loss_only \
                --target_epsilon $epsilon \
                --per_sample_max_grad_norm 1.0 \
                --weight_decay 0.01 \
                --remove_unused_columns False \
                --num_train_epochs 10 \
                --logging_steps 5 \
                --max_grad_norm 0 \
                --lr_scheduler_type cosine \
                --learning_rate 1e-4 \
                --disable_tqdm False \
                --dataloader_num_workers 2 \
                --label_names labels \
                --save_safetensors false \
                --save_strategy steps \
                --save_total_limit 2 \
                --save_steps 100 \
                --tf32 True \
                --enable_lora \
                --target_modules "['q_proj', 'v_proj', 'gate_proj', 'down_proj', 'up_proj', 'k_proj', 'o_proj']"
            # delete checkpoint files
            
            find "$output_dir" -maxdepth 1 -type d -name 'checkpoint-*' -exec rm -rf {} +
        done

#        for epsilon in 0; do
#            echo "Processing $dataset_file with epsilon $epsilon"
#            output_dir="result/$dataset_name/nodp"
#            mkdir -p "$output_dir"
#            CUDA_VISIBLE_DEVICES=0,1,2,3,4 python3 -m torch.distributed.run --nproc_per_node 5 fine-tune-nodp.py \
#                --output_dir "$output_dir" \
#                --model_name meta-llama/Llama-3.2-1B \
#                --train_file "$dataset_file" \
#                --sequence_len $sequence_len \
#                --per_device_train_batch_size 4 \
#                --gradient_accumulation_steps 4 \
#                --log_level info \
#                --per_device_eval_batch_size 2 \
#                --eval_accumulation_steps 1 \
#                --seed 42 \
#                --prediction_loss_only \
#                --weight_decay 0.01 \
#                --remove_unused_columns False \
#                --num_train_epochs 10 \
#                --logging_steps 5 \
#                --max_grad_norm 0 \
#                --lr_scheduler_type cosine \
#                --learning_rate 1e-4 \
#                --disable_tqdm False \
#                --dataloader_num_workers 2 \
#                --label_names labels \
#                --save_safetensors false \
#                --tf32 True\
#                --bf16 True 
#            # delete checkpoint files
#            
#            find "$output_dir" -maxdepth 1 -type d -name 'checkpoint-*' -exec rm -rf {} +
#        done
    fi
# ## finetune without DP
# python -m torch.distributed.run --nproc_per_node 8 fine-tune-nodp.py \
#     --data_dir $DATA \
#     --output_dir $OUTPUT_DIR \
#     --model_name gpt2 \
#     --per_device_train_batch_size 4 \
#     --gradient_accumulation_steps 1 \
#     --evaluation_strategy epoch \
#     --save_strategy epoch \
#     --log_level info \
#     --per_device_eval_batch_size 64 \
#     --eval_accumulation_steps 1 \
#     --seed 42 \
#     --weight_decay 0.01 \
#     --remove_unused_columns False \
#     --num_train_epochs 5 \
#     --logging_steps 2400 \
#     --max_grad_norm 0 \
#     --sequence_len 128 \
#     --learning_rate 0.00005 \
#     --lr_scheduler_type constant \
#     --dataloader_num_workers 2 \
#     --disable_tqdm True \
#     --load_best_model_at_end True \

# dataset_name="sampled_n2c2"
# # Iterate through all dataset files in the folder
# for i in {1..99}; do
#     # skip 39 41 61 79
#     if [ $i -eq 39 ] || [ $i -eq 41 ] || [ $i -eq 61 ] || [ $i -eq 79 ]; then
#         continue
#     fi
#     dataset_file="/home/srini/dp-transformers/$dataset_name/dataset_${i}.jsonl"
#     if [ -f "$dataset_file" ]; then
#         echo "Processing $dataset_file"
#         for epsilon in 2.61 4.59 8.19 14.68; do
#             echo "Processing $dataset_file with epsilon $epsilon"
#             output_dir="/data/srini/$dataset_name/${i}/${epsilon}"
#             mkdir -p "$output_dir"
#             CUDA_VISIBLE_DEVICES=0,1,3,4,5,6 python3 -m torch.distributed.run --nproc_per_node 6 fine-tune-dp.py \
#                 --output_dir "$output_dir" \
#                 --model_name meta-llama/Llama-3.2-1B \
#                 --train_file "$dataset_file" \
#                 --sequence_len 3072 \
#                 --per_device_train_batch_size 2 \
#                 --gradient_accumulation_steps 4 \
#                 --log_level info \
#                 --per_device_eval_batch_size 2 \
#                 --eval_accumulation_steps 1 \
#                 --seed 42 \
#                 --target_epsilon $epsilon \
#                 --per_sample_max_grad_norm 1.0 \
#                 --prediction_loss_only \
#                 --weight_decay 0.01 \
#                 --remove_unused_columns False \
#                 --num_train_epochs 10 \
#                 --logging_steps 5 \
#                 --max_grad_norm 0 \
#                 --lr_scheduler_type cosine \
#                 --learning_rate 1e-4 \
#                 --disable_tqdm False \
#                 --dataloader_num_workers 2 \
#                 --label_names labels \
#                 --save_safetensors false \
#                 --tf32 True\
#                 --bf16 True 
#             # delete checkpoint files
            
#             find "$output_dir" -maxdepth 1 -type d -name 'checkpoint-*' -exec rm -rf {} +
#         done
#     fi
# done

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

