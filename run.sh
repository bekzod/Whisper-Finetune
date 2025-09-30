#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1 WANDB_API_KEY='2dfc22d8af7805df156e7f31ea3bc090ec99d52e' accelerate launch --multi_gpu --num_processes=2 finetune_whisper_lora.py \
  --train_data ../datasets/train.json \
  --test_data ../datasets/test.json \
  --base_model ../models/whisper-large-v3 \
  --output_dir ./output/whisper-large-v3-uzbek-2gpu-adalora \
  --num_train_epochs 4 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps 8 \
  --learning_rate 0.0005 \
  --use_adalora True \
  --lora_r 32 \
  --lora_alpha 64 \
  --lora_dropout 0.05 \
  --num_workers 8 \
  --eval_steps 500 \
  --save_steps 1000 \
  --wandb_project whisper-uzbek \
  --wandb_run_name whisper-v3-uzbek-2gpu-adalora-aggressive \
  --wandb_tags uzbek,whisper,adalora,2gpu
