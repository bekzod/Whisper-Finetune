#!/bin/bash

export NVIDIA_TF32_OVERRIDE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_VISIBLE_DEVICES=0,1
export WANDB_API_KEY='2dfc22d8af7805df156e7f31ea3bc090ec99d52e'
export RAYON_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

accelerate launch --multi_gpu --num_processes=2 --config_file ./configs/accelerate.yaml finetune.py \
  --base_model ../models/whisper-large-v3 \
  --output_dir ../models/output \
  --num_train_epochs 6 \
  --per_device_train_batch_size 12 \
  --per_device_eval_batch_size 12 \
  --gradient_accumulation_steps 4 \
  --learning_rate 2.5e-4 \
  --warmup_ratio 0.15 \
  --lr_scheduler_type cosine \
  --use_adalora True \
  --lora_r 32 \
  --lora_alpha 64 \
  --lora_dropout 0.02 \
  --logging_steps 200 \
  --eval_steps 200 \
  --save_steps 600 \
  --save_total_limit 5 \
  --wandb_project whisper-uzbek \
  --wandb_run_name whisper-v3-uzbek-2xH100-adalora \
  --wandb_tags uzbek,whisper,adalora,H100 \
  --output_dir ../models/output \
  --train_data ../datasets/uzbek_voice/data/train/metadata.csv \
  # --test_data ../datasets/uzbek_voice/test.json \
