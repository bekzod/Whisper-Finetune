#!/usr/bin/env bash
set -euo pipefail

python prepare_tokenizer.py \
  --data_file "../../uzbek.cleaned.txt" \
  --manifest "output/train_manifest.jsonl,output/eval_manifest.jsonl" \
  --data_root "." \
  --vocab_size 2048 \
  --tokenizer "spe" \
  --spe_type "bpe" \
  --spe_character_coverage 1.0 \
  --spe_remove_extra_whitespaces \
  --no_lower_case \
  --log
