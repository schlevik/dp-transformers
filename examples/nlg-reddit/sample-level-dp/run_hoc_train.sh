PROJECT_ROOT=../../../../..
python -m torch.distributed.run --nproc_per_node 8 fine-tune-dp.py \
--output_dir scratch \
--model_name meta-llama/Llama-3.2-1B \
--train_file $PROJECT_ROOT/data/cls/hallmarks_of_cancer/dp-transformers/input-train-augpe.jsonl \
--sequence_len 100 \
--per_device_train_batch_size 2 \
--gradient_accumulation_steps 2 \
--evaluation_strategy steps \
--eval_steps 45 \
--log_level info \
--per_device_eval_batch_size 4 \
--eval_accumulation_steps 1 \
--seed 42 \
--target_epsilon 8 \
--per_sample_max_grad_norm 1.0 \
--prediction_loss_only \
--weight_decay 0.01 \
--remove_unused_columns False \
--num_train_epochs 3 \
--logging_steps 5 \
--max_grad_norm 0 \
--lr_scheduler_type constant \
--learning_rate 1e-4 \
--disable_tqdm True \
--dataloader_num_workers 2 \
--label_names labels \
--save_safetensors false