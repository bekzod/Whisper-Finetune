#!/bin/bash
export NVIDIA_TF32_OVERRIDE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_VISIBLE_DEVICES=0,1
export WANDB_API_KEY='2dfc22d8af7805df156e7f31ea3bc090ec99d52e'

accelerate launch --multi_gpu --num_processes=2 --config_fikle ./accelerate_config.yaml finetune_whisper_lora.py \
  --train_data ../datasets/train.json \
  --test_data ../datasets/test.json \
  --base_model ../models/whisper-large-v3 \
  --output_dir ./output/whisper-large-v3-uzbek-2xH100-adalora \
  --num_train_epochs 6 \
  --per_device_train_batch_size 12 \
  --per_device_eval_batch_size 16 \
  --gradient_accumulation_steps 6 \
  --learning_rate 3e-4 \
  --warmup_ratio 0.1 \
  --lr_scheduler_type cosine \
  --use_adalora True \
  --lora_r 32 \
  --lora_alpha 64 \
  --lora_dropout 0.05 \
  --num_workers 16 \
  --group_by_length True \
  --eval_steps 3000 \
  --save_steps 6000 \
  --save_total_limit 5 \
  --use_compile True \
  --wandb_project whisper-uzbek \
  --wandb_run_name whisper-v3-uzbek-2xH100-adalora \
  --wandb_tags uzbek,whisper,adalora,H100
