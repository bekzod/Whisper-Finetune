#!/usr/bin/env bash
set -euo pipefail

python prepare_tokenizer.py \
  --data_file "./output/merged_corpus.txt" \
  --data_root "." \
  --vocab_size 1024 \
  --tokenizer "spe" \
  --spe_type "unigram" \
  --spe_character_coverage 1.0 \
  --spe_remove_extra_whitespaces \
  --no_lower_case \
  --log
