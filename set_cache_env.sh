#!/usr/bin/env bash
set -euo pipefail

# Source this script to redirect HF/cache/temp files to /workspace.
export HF_HOME=/workspace/.hf
export HF_HUB_CACHE=/workspace/.hf/hub
export HF_DATASETS_CACHE=/workspace/.hf/datasets
export HF_MODULES_CACHE=/workspace/.hf/modules

export XDG_CACHE_HOME=/workspace/.cache

export TMPDIR=/workspace/tmp
export TEMP=/workspace/tmp
export TMP=/workspace/tmp
