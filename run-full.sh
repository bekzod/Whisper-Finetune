#!/bin/bash
export HF_HOME=/workspace/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets

export TRANSFORMERS_VERBOSITY=debug
export DATASETS_VERBOSITY=debug

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_MAX_CONNECTIONS=1
unset CUDA_VISIBLE_DEVICES
export RAYON_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

export ATEN_ALLOW_TF32=1
export TORCH_ALLOW_TF32_CUBLAS=1
export TORCH_ALLOW_TF32_CUDNN=1
export CUFFT_ALLOW_TF32=1
# export NCCL_NVLS_ENABLE=0
# export NCCL_P2P_DISABLE=1           # disable peer-to-peer GPU direct (forces PCIe instead of NVLink)
# export NCCL_NVLINK_DISABLE=1        # guard against NVLink path getting re-enabled
# export NCCL_P2P_LEVEL=PXB           # ensure NCCL routes over PCIe

# Diagnostics / stability
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=600
export NCCL_IB_DISABLE=1             # single-node hygiene

accelerate launch --config_file ./configs/accelerate-single.yaml finetune.py \
  --base_model ../models/whisper-large-v3 \
  --output_dir ../models/output-full-finetune \
  --num_train_epochs 5 \
  --per_device_train_batch_size 22 \
  --per_device_eval_batch_size 44 \
  --gradient_accumulation_steps 4 \
  --freeze_encoder_epochs 1 \
  --unfreeze_finish_ratio 0.28 \
  --learning_rate 1.6e-5 \
  --logging_steps 200 \
  --eval_steps 1000 \
  --save_steps 2000 \
  --warmup_ratio 0.06 \
  --lr_scheduler_type cosine \
  --use_lora False \
  --weight_decay 0.008 \
  --save_total_limit 4 \
  --wandb_project whisper-uzbek \
  --wandb_run_name whisper-v3-uzbek \
  --wandb_tags uzbek,whisper,full-finetune,H100 \
  --train_data ./configs/datasets.json
