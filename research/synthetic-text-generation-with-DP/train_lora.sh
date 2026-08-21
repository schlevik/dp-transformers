dataset_name="n2c2"
sequence_len=2048
epsilon=4
output_dir="/home/srini/dp-transformers/research/synthetic-text-generation-with-DP/sample_lora_${epsilon}"
dataset_file="/home/srini/dp-transformers/$dataset_name/train-original_2048.jsonl"
CUDA_VISIBLE_DEVICES=5,6 python3 -m torch.distributed.run --nproc_per_node 2 fine-tune-dp-lora.py \
                --output_dir "$output_dir" \
                --model_name meta-llama/Llama-3.2-1B \
                --train_file "$dataset_file" \
                --sequence_len $sequence_len \
                --per_device_train_batch_size 4 \
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
                --num_train_epochs 20 \
                --logging_steps 5 \
                --max_grad_norm 0 \
                --lr_scheduler_type cosine \
                --learning_rate 3e-4 \
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



# CUDA_VISIBLE_DEVICES=0,1,2,7 python3 -m torch.distributed.run --nproc_per_node 4 fine-tune-nodp.py \
#                 --output_dir "$output_dir" \
#                 --model_name meta-llama/Llama-3.2-1B \
#                 --train_file "$dataset_file" \
#                 --sequence_len $sequence_len \
#                 --per_device_train_batch_size 2 \
#                 --gradient_accumulation_steps 4 \
#                 --log_level info \
#                 --per_device_eval_batch_size 2 \
#                 --eval_accumulation_steps 1 \
#                 --seed 42 \
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
            # delete checkpoint files
            