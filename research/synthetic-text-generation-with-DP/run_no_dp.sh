
epsilon=0
dataset_name="daniel-ml"
CUDA_VISIBLE_DEVICES=5,6 python3 -m torch.distributed.run --nproc_per_node 2 fine-tune-nodp.py \
    --output_dir result/$dataset_name/$epsilon \
    --model_name meta-llama/Llama-3.2-1B  \
    --train_file /home/srini/dp-transformers/$dataset_name/train_original.jsonl \
    --sequence_len 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --eval_strategy epoch \
    --log_level info \
    --per_device_eval_batch_size 4 \
    --eval_accumulation_steps 1 \
    --seed 42 \
    --weight_decay 0.01 \
    --remove_unused_columns False \
    --num_train_epochs 3 \
    --logging_steps 5 \
    --max_grad_norm 0 \
    --lr_scheduler_type constant \
    --learning_rate 0.0001 \
    --save_strategy epoch \
    --save_safetensors False \
    --dataloader_num_workers 2 \
    --disable_tqdm True 
