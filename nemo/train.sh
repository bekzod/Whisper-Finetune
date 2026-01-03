#!/usr/bin/env bash
set -euo pipefail

python speech_to_text_rnnt_bpe.py \
  --config-path=. \
  --config-name=train
