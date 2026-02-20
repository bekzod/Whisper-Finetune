# NeMo Training Guide (Step by Step)

This guide explains how to run end-to-end ASR training in this `nemo/` folder using:
- `prepare_dataset.py` for NeMo manifests
- `prepare_tokenizer.py` for SentencePiece tokenizer
- `speech_to_text_rnnt_bpe.py` + `train.yaml` for training

## 1) Environment Setup

From repository root:

```bash
cd nemo
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

Optional cache setup (recommended in Docker/workspace environments):

```bash
source ./set_cache_env.sh
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$XDG_CACHE_HOME" "$TMPDIR"
```

Optional authentication:

```bash
# Needed only for private/gated HF datasets
export HF_TOKEN=your_hf_token

# Optional if using Weights & Biases logging
export WANDB_API_KEY=your_wandb_key
```

## 2) Prepare Train/Eval Manifests

Edit dataset sources in `nemo/datasets.json` first, then run:

```bash
python3 prepare_dataset.py \
  --config ./datasets.json \
  --groups train eval \
  --output-dir ./output \
  --num-workers 8 \
  --cache-mode per-dataset \
  --hf-token "${HF_TOKEN:-}"
```

Expected outputs:
- `nemo/output/train_manifest.jsonl`
- `nemo/output/eval_manifest.jsonl`
- `nemo/output/audio/...`

Quick sanity check:

```bash
wc -l ./output/train_manifest.jsonl ./output/eval_manifest.jsonl
```

## 3) Train Tokenizer

`train.yaml` expects tokenizer dir `./output/tokenizer_spe_unigram_v1024`, so create it from the train manifest:

```bash
python3 prepare_tokenizer.py \
  --manifest ./output/train_manifest.jsonl \
  --data_root ./output \
  --vocab_size 1024 \
  --tokenizer spe \
  --spe_type unigram \
  --spe_character_coverage 1.0 \
  --spe_remove_extra_whitespaces \
  --no_lower_case \
  --log
```

Check that files exist:

```bash
ls -lah ./output/tokenizer_spe_unigram_v1024
```

## 4) Review `train.yaml`

Before training, confirm these paths match your environment:
- `model.train_ds.manifest_filepath`
- `model.validation_ds.manifest_filepath`
- `model.tokenizer.dir`
- `exp_manager.exp_dir`

Current defaults already point to `./output/...` under `nemo/`.

## 5) Start Training

Single-GPU:

```bash
python3 speech_to_text_rnnt_bpe.py --config-path=. --config-name=train
```

Use all visible GPUs:

```bash
python3 speech_to_text_rnnt_bpe.py \
  --config-path=. \
  --config-name=train \
  trainer.devices=-1
```

Example with common runtime overrides:

```bash
python3 speech_to_text_rnnt_bpe.py \
  --config-path=. \
  --config-name=train \
  exp_manager.exp_dir=./output/nemo-training
```

## 6) Resume and Checkpoints

With current `train.yaml`, auto-resume is enabled if checkpoint exists in the same experiment directory.

Check training artifacts:

```bash
find ./output/nemo-training -maxdepth 3 -type f | head
```

## 7) Optional: Prepare Noise Augmentation Data

If you want to enable `train_ds.augmentor.noise` in `train.yaml`, generate a noise manifest first:

```bash
python3 prepare_noise.py \
  --out_dir ./output/noise_hf \
  --manifest_path ./output/noise_hf/noise_manifest.json
```

Then uncomment the `augmentor` block in `train.yaml`.

## 8) Troubleshooting

- OOM during validation:
  - Reduce `model.validation_ds.batch_size`
  - Reduce `model.train_ds.max_duration`
  - Reduce `model.train_ds.batch_size`
- Slow startup/downloading:
  - Use `source ./set_cache_env.sh`
  - Check HF token access and network stability
- Training not finding config:
  - Ensure command is run inside `nemo/` and includes `--config-path=. --config-name=train`
