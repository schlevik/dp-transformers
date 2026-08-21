dataset_name="psytar"
sequence_len=128
user_name="srini"
# Iterate through all dataset files in the folder
for lr in 2e-5 1e-5 1e-4 2e-4; do
    dataset_file="/home/$user_name/dp-transformers/$dataset_name/train-original.jsonl"

    echo "dataset_file: $dataset_file" and learning rate: $lr
    if [ -f "$dataset_file" ]; then
        echo "Processing $dataset_file"
        for batch_size in 8 4 2; do
            echo "Processing $dataset_file with epsilon 2 and batch size $batch_size"
            output_dir="/data/$user_name/$dataset_name/lr_sweep/${lr}/epsilon_2_total_batch_size_$((batch_size * 4))"
            mkdir -p "$output_dir"
            CUDA_VISIBLE_DEVICES=2 python3 fine-tune-dp.py \
                --output_dir "$output_dir" \
                --model_name meta-llama/Llama-3.2-1B \
                --train_file "$dataset_file" \
                --sequence_len $sequence_len \
                --per_device_train_batch_size $batch_size \
                --gradient_accumulation_steps 4 \
                --log_level info \
                --per_device_eval_batch_size 2 \
                --eval_accumulation_steps 1 \
                --seed 42 \
                --prediction_loss_only \
                --target_epsilon 2 \
                --per_sample_max_grad_norm 1.0 \
                --weight_decay 0.01 \
                --remove_unused_columns False \
                --num_train_epochs 5 \
                --logging_steps 5 \
                --max_grad_norm 0 \
                --lr_scheduler_type cosine \
                --learning_rate $lr \
                --disable_tqdm False \
                --dataloader_num_workers 2 \
                --label_names labels \
                --save_safetensors false \
                --save_strategy steps \
                --save_total_limit 2 \
                --save_steps 500 \
                --report_to wandb \
                --run_name "lr_sweep_${lr}_epsilon_2_total_batch_size_$((batch_size * 4))" \
                --tf32 True \
                --bf16 True
        done
    fi
done