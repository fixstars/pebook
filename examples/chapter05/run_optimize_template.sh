weight_path=meta-llama/Llama-3.2-1B
export WANDB_MODE=disabled
num_gpus=4
epoch=1

deepspeed --num_gpus $num_gpus  \
    --master_port 51336  train.py  \
    --model_name_or_path  $weight_path \
    --data_path ./alpaca_data.json \
    --bf16 True \
    --output_dir out_load_test/$MODE \
    --num_train_epochs $epoch \
    --gradient_checkpointing false \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size 1 \
    --eval_strategy no \
    --save_strategy steps  \
    --save_steps 10000 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --learning_rate 0 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --tf32 True \
    --deepspeed ${deepspeed_config_path}
