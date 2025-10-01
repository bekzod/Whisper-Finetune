#!/bin/bash

export NVIDIA_TF32_OVERRIDE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_VISIBLE_DEVICES=0

python3.11 ./evaluation.py \
  --test_data ../datasets/dataset_for_stt_ttsmodels/metadata.csv \
  --model_path ../models/output/whisper-large-v3-0/checkpoint-final-merged \
  --batch_size 16 \
  --num_workers 8 \
  --language Uzbek \
  --remove_pun True \
  --timestamps False \
  --min_audio_len 0.5 \
  --max_audio_len 30 \
  --local_files_only True \
  --task transcribe \
  --metric wer
