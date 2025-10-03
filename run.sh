#!/bin/bash
export NVIDIA_TF32_OVERRIDE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1
export WANDB_API_KEY='2dfc22d8af7805df156e7f31ea3bc090ec99d52e'
export TOKENIZERS_PARALLELISM=false
export RAYON_NUM_THREADS=1

# Diagnostics / stability
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=600
export NCCL_IB_DISABLE=1             # single-node hygiene

# EITHER: disable W&B to test
# export WANDB_DISABLED=true

# OR: pre-login headlessly (preferred long-term)
# export WANDB_API_KEY="<NEW_KEY>"
# python -c "import os,wandb; wandb.login(key=os.environ['WANDB_API_KEY'])"

accelerate launch --multi_gpu --num_processes=2 --config_file ./configs/accelerate.yaml finetune.py \
  --base_model ../models/whisper-large-v3 \
  --output_dir ../models/output \
  --num_train_epochs 8 \
  --per_device_train_batch_size 64 \
  --per_device_eval_batch_size 128 \
  --gradient_accumulation_steps 1 \
  --learning_rate 4e-4 \
  --warmup_ratio 0.15 \
  --lr_scheduler_type cosine \
  --use_adalora True \
  --lora_r 256 \
  --lora_alpha 128 \
  --lora_dropout 0.025 \
  --logging_steps 100 \
  --eval_steps 100 \
  --save_steps 200 \
  --save_total_limit 5 \
  --wandb_project whisper-uzbek \
  --wandb_run_name whisper-v3-uzbek-2xH100-adalora \
  --wandb_tags uzbek,whisper,adalora,H100 \
  --train_data ../datasets/uzbek_voice/data/train/metadata.csv
