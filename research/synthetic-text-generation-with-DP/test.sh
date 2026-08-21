dataset_name="psytar"
sequence_len=4300
user_name="srini"
dataset_file="/home/$user_name/dp-transformers/$dataset_name/train-original.jsonl"
output_dir="/data/$user_name/$dataset_name/test"

CUDA_VISIBLE_DEVICES=5,6 python3 -m torch.distributed.run --nproc_per_node 2 fine-tune-nodp.py \
                    --output_dir "$output_dir" \
                    --model_name meta-llama/Llama-3.2-1B \
                    --train_file "$dataset_file" \
                    --sequence_len $sequence_len \
                    --per_device_train_batch_size 2 \
                    --gradient_accumulation_steps 4 \
                    --log_level info \
                    --per_device_eval_batch_size 2 \
                    --eval_accumulation_steps 1 \
                    --seed 42 \
                    --prediction_loss_only \
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
                    --tf32 True\
                    --bf16 True 