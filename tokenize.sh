#!/usr/bin/env bash
set -euo pipefail

python nemo/prepare_tokenizer.py \
  --manifest "output/train_manifest.jsonl,output/eval_manifest.jsonl" \
  --data_root "tokenizers/uzbek" \
  --vocab_size 2048 \
  --tokenizer "spe" \
  --spe_type "bpe" \
  --spe_character_coverage 1.0 \
  --spe_split_digits \
  --spe_remove_extra_whitespaces \
  --log
