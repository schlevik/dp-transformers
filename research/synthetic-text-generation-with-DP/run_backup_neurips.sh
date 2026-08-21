dataset_name=$1

sequence_len=2048
user_name="srini"
gpu_device=$2
start=$3
end=$4
rand_type=$5

data_dir=/mnt/nvme1/yidan/MIA/data/cls/$dataset_name/D_sample/sampled_${rand_type}_datasets
for idx in $(seq $start $end); do
    dataset_file="$data_dir/dataset_$idx.jsonl"
    for epsilon in 4; do
        echo "Processing $dataset_file with epsilon $epsilon"
        dataset_temp=$(basename $dataset_file)
        echo "dataset_temp: $dataset_temp"
        i=${dataset_temp%.jsonl}
        echo "i: $i"
        output_dir="/mnt/nvme1/srini/rerun_strategy1/$dataset_name/control_group/sampled_${rand_type}_datasets/${i}/${epsilon}"
        echo "output dir: $output_dir"
        echo "epsilon: $epsilon"
        mkdir -p "$output_dir"
        if [ "$epsilon" = "0" ]; then
            echo "no DP"
            CUDA_VISIBLE_DEVICES=$gpu_device python3 fine-tune-nodp.py \
                        --output_dir "$output_dir" \
                        --model_name meta-llama/Llama-3.2-1B \
                        --train_file "$dataset_file" \
                        --sequence_len $sequence_len \
                        --per_device_train_batch_size 2 \
                        --gradient_accumulation_steps 2 \
                        --log_level info \
                        --per_device_eval_batch_size 2 \
                        --eval_accumulation_steps 1 \
                        --seed 42 \
                        --prediction_loss_only \
                        --weight_decay 0.01 \
                        --remove_unused_columns False \
                        --num_train_epochs 5 \
                        --logging_steps 5 \
                        --max_grad_norm 0 \
                        --lr_scheduler_type cosine \
                        --learning_rate 1e-4 \
                        --disable_tqdm False \
                        --dataloader_num_workers 2 \
                        --label_names labels \
                        --save_safetensors false \
                        --save_strategy steps \
                        --save_total_limit 1 \
                        --save_steps 500 \
                        --report_to none \
                        --tf32 True \
                        --bf16 True
        else
            echo "DP"
            echo "epsilon: $epsilon"
            CUDA_VISIBLE_DEVICES=$gpu_device python3 fine-tune-dp.py \
                        --output_dir "$output_dir" \
                        --model_name meta-llama/Llama-3.2-1B \
                        --train_file "$dataset_file" \
                        --sequence_len $sequence_len \
                        --per_device_train_batch_size 2 \
                        --gradient_accumulation_steps 2 \
                        --log_level info \
                        --per_device_eval_batch_size 2 \
                        --eval_accumulation_steps 1 \
                        --seed 42 \
                        --prediction_loss_only \
                        --target_epsilon "$epsilon" \
                        --per_sample_max_grad_norm 1.0 \
                        --weight_decay 0.01 \
                        --remove_unused_columns False \
                        --num_train_epochs 5 \
                        --logging_steps 5 \
                        --max_grad_norm 0 \
                        --lr_scheduler_type cosine \
                        --learning_rate 1e-4 \
                        --disable_tqdm False \
                        --dataloader_num_workers 2 \
                        --label_names labels \
                        --save_safetensors false \
                        --save_strategy steps \
                        --save_total_limit 1 \
                        --save_steps 500 \
                        --report_to none \
                        --tf32 True \
                        --bf16 True
        fi
        find "$output_dir" -maxdepth 1 -type d -name 'checkpoint-*' -exec rm -rf {} +
    done
done

