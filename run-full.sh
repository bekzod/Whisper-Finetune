#!/bin/bash

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
  --num_train_epochs 8 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 64 \
  --gradient_accumulation_steps 2 \
  --learning_rate 1e-5 \
  --warmup_ratio 0.1 \
  --lr_scheduler_type cosine \
  --use_lora False \
  --weight_decay 0.005 \
  --save_total_limit 5 \
  --wandb_project whisper-uzbek \
  --wandb_run_name whisper-v3-uzbek \
  --wandb_tags uzbek,whisper,full-finetune,H100 \
  --train_data ../datasets/uzbek_voice
  # --train_data ../datasets/uzbek_voice+../datasets/FeruzaSpeech:train+../datasets/FeruzaSpeech:dev+../datasets/dataset_for_stt_ttsmodels+../datasets/fleurs_uz_uz:train+../datasets/fleurs_uz_uz:dev+../datasets/uzbek_voice_2:train+../datasets/uzbek_voice_2:dev+../datasets/uzbek_voice_2:validated+../datasets/uzbek_voice_3+../datasets/uzbek_voice_4+../datasets/uzbekvoice_filtered:train \
  # --test_data ../datasets/FeruzaSpeech:test+../datasets/fleurs_uz_uz:test+../datasets/uzbek_voice_2:test+ ../datasets/uzbekvoice_filtered:validate
