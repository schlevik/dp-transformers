dataset_name="sampled_private_reduced_datasets"
sequence_len=128
# epsilon=16
# output_dir="/home/srini/dp-transformers/research/synthetic-text-generation-with-DP/sample_lora"
# dataset_file="/home/srini/dp-transformers/$dataset_name/train-original.jsonl"

data_path=/mnt/nvme1/yidan/MIA/data/cls/psytar/D_sample/sampled_4ref_datasets_outliers

for dataset_file in $data_path/*.jsonl; do
    for epsilon in 0.5 1 2 4; do
        dataset_name=$(basename $dataset_file)
        i=${dataset_name%.jsonl}
        echo "i: $i"
        echo "Processing $dataset_file with epsilon $epsilon"
        output_dir="/data/srini/$dataset_name/${i}/${epsilon}"
        mkdir -p "$output_dir"
        CUDA_VISIBLE_DEVICES=7 python3 -m torch.distributed.run --nproc_per_node 1 fine-tune-dp.py \
                        --output_dir "$output_dir" \
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
                        --learning_rate 3e-4 \
                        --disable_tqdm False \
                        --dataloader_num_workers 2 \
                        --label_names labels \
                        --save_safetensors false \
                        --save_strategy steps \
                        --save_total_limit 2 \
                        --save_steps 100 \
                        --tf32 True \
                        --bf16 True
        find "$output_dir" -maxdepth 1 -type d -name 'checkpoint-*' -exec rm -rf {} +
    done
done

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
            