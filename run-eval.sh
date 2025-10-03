#!/bin/bash

export NVIDIA_TF32_OVERRIDE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_VISIBLE_DEVICES=0

python ./evaluation.py \
  --test_data ../datasets/dataset_for_stt_ttsmodels/metadata.csv \
  --model_path ../models/whisper-large-v3 \
  --adapter_path ../models/output/whisper-large-v3-1/checkpoint-best \
  --batch_size 16 \
  --num_workers 8 \
  --language Uzbek \
  --remove_pun True \
  --timestamps False \
  --min_audio_len 0.5 \
  --max_audio_len 30 \
  --local_files_only True \
  --task transcribe \
  --metric wer \
  --kenlm_path /workspace/models/uzbek.o5.arpa \
  --kenlm_alpha 0.6 \
  --kenlm_top_k 200
