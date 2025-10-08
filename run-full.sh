#!/bin/bash
export HF_HOME=/workspace/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets

export TRANSFORMERS_VERBOSITY=debug
export DATASETS_VERBOSITY=debug

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_VISIBLE_DEVICES=0,1
export WANDB_API_KEY='2dfc22d8af7805df156e7f31ea3bc090ec99d52e'
export RAYON_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

# Diagnostics / stability
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=600
export NCCL_IB_DISABLE=1             # single-node hygiene

accelerate launch --multi_gpu --num_processes=2 --config_file ./configs/accelerate.yaml finetune.py \
  --base_model ../models/whisper-large-v3 \
  --output_dir ../models/output-full-finetune \
  --num_train_epochs 6 \
  --per_device_train_batch_size 40 \
  --per_device_eval_batch_size 64 \
  --gradient_accumulation_steps 2 \
  --freeze_encoder_epochs 1 \
  --learning_rate 1e-5 \
  --logging_steps 100 \
  --eval_steps 200 \
  --save_steps 200 \
  --warmup_ratio 0.1 \
  --lr_scheduler_type cosine \
  --use_lora False \
  --weight_decay 0.01 \
  --save_total_limit 4 \
  --wandb_project whisper-uzbek \
  --wandb_run_name whisper-v3-uzbek \
  --wandb_tags uzbek,whisper,full-finetune,H100 \
  --train_data ./configs/datasets.json
