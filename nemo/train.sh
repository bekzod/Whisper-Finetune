#!/usr/bin/env bash
set -euo pipefail

python speech_to_text_rnnt_bpe.py \
  --config-path=. \
  --config-name=train \
  exp_manager.create_wandb_logger=True \
  exp_manager.wandb_logger_kwargs.name="rnnt-bpe-uzbek" \
  exp_manager.wandb_logger_kwargs.project="whisper-finetune-nemo"
