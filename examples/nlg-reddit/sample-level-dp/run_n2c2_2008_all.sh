PROJECT_ROOT=../../../../..
mkdir -p result/n2c2_2008
for epsilon in 4; do #0.5 1 2 4; do
    # -m torch.distributed.run --nproc_per_node 8 fine-tune-dp.py \
    python fine-tune-dp.py \
    --output_dir result/n2c2_2008/$epsilon \
    --model_name meta-llama/Llama-3.2-1B \
    --train_file $PROJECT_ROOT/data/cls/n2c2_2008/dp-transformers/input-train-dp-transformers.jsonl \
    --sequence_len 3072 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy steps \
    --eval_steps 45 \
    --log_level info \
    --per_device_eval_batch_size 4 \
    --eval_accumulation_steps 1 \
    --seed 42 \
    --target_epsilon $epsilon \
    --per_sample_max_grad_norm 1.0 \
    --prediction_loss_only \
    --weight_decay 0.01 \
    --remove_unused_columns False \
    --num_train_epochs 3 \
    --logging_steps 5 \
    --max_grad_norm 0 \
    --lr_scheduler_type constant \
    --learning_rate 1e-5 \
    --disable_tqdm True \
    --dataloader_num_workers 2 \
    --label_names labels \
    --tf32 True\
    --save_safetensors false \
    # --bf16 True\
    # --fsdp full_shard
    # --enable_lora \
    # --lora_dim 256 \
    # --lora_alpha 2 \
done

# python -m torch.distributed.run --nproc_per_node 4 fine-tune-nodp.py \
#     --output_dir result/n2c2_2008/nodp_lora \
#     --model_name meta-llama/Llama-3.2-1B \
#     --train_file $PROJECT_ROOT/data/cls/n2c2_2008/dp-transformers/input-train-dp-transformers.jsonl \
#     --sequence_len 100 \
#     --per_device_train_batch_size 2 \
#     --gradient_accumulation_steps 2 \
#     --evaluation_strategy steps \
#     --eval_steps 45 \
#     --log_level info \
#     --per_device_eval_batch_size 4 \
#     --eval_accumulation_steps 1 \
#     --seed 42 \
#     --prediction_loss_only \
#     --weight_decay 0.01 \
#     --remove_unused_columns False \
#     --num_train_epochs 3 \
#     --logging_steps 5 \
#     --max_grad_norm 0 \
#     --lr_scheduler_type constant \
#     --learning_rate 1e-4 \
#     --disable_tqdm True \
#     --dataloader_num_workers 2 \
#     --label_names labels \
#     --save_safetensors false \
#     --enable_lora \
#     --lora_dim 256 \
#     --lora_alpha 2