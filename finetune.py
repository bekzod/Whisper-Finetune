#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import functools
import math
import os
import sys

import warnings
from typing import Any

import torch
from torch.utils.data import DataLoader, random_split
import evaluate
from datasets import load_dataset
from pathlib import Path
import json
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    TrainerCallback,
)
from transformers.trainer_callback import EarlyStoppingCallback
from transformers.trainer_utils import set_seed

# ---- Project utilities ----
from utils.callback import SavePeftModelCallback
from utils.data_utils import DataCollatorSpeechSeq2SeqWithPadding
from utils.model_utils import load_from_checkpoint
from utils.reader import CustomDataset, rate_limited_request
from utils.utils import print_arguments, make_inputs_require_grad, add_arguments

# ---- PEFT (LoRA / AdaLoRA) ----
from peft import (
    LoraConfig,
    AdaLoraConfig,
    get_peft_model,
    PeftModel,
)

# -------------------- Argparse --------------------
parser = argparse.ArgumentParser("Whisper-large-v3 LoRA finetune with W&B (bf16 only)")
add_arg = functools.partial(add_arguments, argparser=parser)

# Data & model
add_arg(
    "train_data",
    type=str,
    default="../datasets/train.json",
    help="Path to training data file or dataset",
)
add_arg(
    "test_data",
    type=str,
    default=None,
    help="Path to test/validation data file or dataset",
)
add_arg(
    "base_model",
    type=str,
    default="../models/whisper-large-v3",
    help="Path to base Whisper model",
)
add_arg(
    "output_dir", type=str, default="output/", help="Directory to save model outputs"
)

# Task / language
add_arg(
    "timestamps",
    type=bool,
    default=False,
    help="Whether to include timestamps in transcription",
)
add_arg("language", type=str, default="Uzbek", help="Language for transcription")
add_arg(
    "task",
    type=str,
    default="transcribe",
    help="Task type: transcribe or translate",
    choices=["transcribe", "translate"],
)

# Loader / augmentation
add_arg(
    "min_audio_len", type=float, default=0.5, help="Minimum audio length in seconds"
)
add_arg(
    "max_audio_len", type=float, default=30.0, help="Maximum audio length in seconds"
)
add_arg(
    "augment_config_path",
    type=str,
    default="./configs/augmentation.json",
    help="Path to augmentation config JSON file",
)
add_arg(
    "num_workers",
    type=int,
    default=min(8, os.cpu_count() or 1),
    help="Number of data loader workers",
)

# LoRA / AdaLoRA
add_arg("use_lora", type=bool, default=True, help="Whether to use LoRA for fine-tuning")
add_arg(
    "use_adalora",
    type=bool,
    default=True,
    help="Whether to use AdaLoRA instead of standard LoRA",
)
add_arg("lora_r", type=int, default=16, help="LoRA rank parameter")
add_arg("lora_alpha", type=int, default=32, help="LoRA alpha scaling parameter")
add_arg("lora_dropout", type=float, default=0.05, help="LoRA dropout rate")

# Training schedule
add_arg("num_train_epochs", type=int, default=3, help="Number of training epochs")
add_arg(
    "per_device_train_batch_size",
    type=int,
    default=8,
    help="Training batch size per device",
)
add_arg(
    "per_device_eval_batch_size",
    type=int,
    default=8,
    help="Evaluation batch size per device",
)
add_arg(
    "gradient_accumulation_steps",
    type=int,
    default=4,
    help="Number of gradient accumulation steps",
)
add_arg("learning_rate", type=float, default=2e-4, help="Learning rate for optimizer")
add_arg(
    "weight_decay",
    type=float,
    default=0.01,
    help="Weight decay regularization parameter",
)
add_arg(
    "max_grad_norm", type=float, default=1.0, help="Maximum gradient norm for clipping"
)
add_arg(
    "warmup_ratio",
    type=float,
    default=0.05,
    help="Warmup ratio for learning rate scheduler",
)
add_arg(
    "lr_scheduler_type",
    type=str,
    default="cosine",
    help="Type of learning rate scheduler",
)

# NEW: staged training
add_arg(
    "freeze_encoder_epochs",
    type=int,
    default=0,
    help="Freeze the encoder for the first N epochs, then unfreeze.",
)

# Logging / eval / saving
add_arg("logging_steps", type=int, default=None, help="Number of steps between logging")
add_arg(
    "eval_steps", type=int, default=None, help="Number of steps between evaluations"
)
add_arg(
    "save_steps", type=int, default=None, help="Number of steps between model saves"
)
add_arg(
    "save_total_limit",
    type=int,
    default=4,
    help="Maximum number of checkpoints to keep",
)
add_arg(
    "predict_with_generate",
    type=bool,
    default=True,
    help="Whether to use generation for evaluation",
)
add_arg(
    "early_stopping_patience",
    type=int,
    default=3,
    help="Number of evaluations with no improvement before stopping",
)
add_arg(
    "group_by_length",
    type=bool,
    default=False,
    help="Whether to group samples by length for efficiency",
)
add_arg(
    "length_column_name",
    type=str,
    default=None,
    help="Name of column containing sample lengths",
)
add_arg(
    "generation_max_length",
    type=int,
    default=225,
    help="Maximum length for generation during evaluation",
)

# Infra / misc
add_arg("seed", type=int, default=42, help="Random seed for reproducibility")
add_arg(
    "use_compile",
    type=bool,
    default=False,
    help="Whether to use torch.compile for optimization",
)
add_arg(
    "local_files_only",
    type=bool,
    default=True,
    help="Whether to only use local files (no downloading)",
)
add_arg(
    "resume_from_checkpoint",
    type=str,
    default=None,
    help="Path to checkpoint to resume training from",
)
add_arg(
    "push_to_hub",
    type=bool,
    default=False,
    help="Whether to push model to Hugging Face Hub",
)
add_arg("hub_model_id", type=str, default=None, help="Model ID for Hugging Face Hub")

# W&B
add_arg("wandb_project", type=str, default=None, help="Weights & Biases project name")
add_arg(
    "wandb_entity", type=str, default=None, help="Weights & Biases entity/team name"
)
add_arg("wandb_run_name", type=str, default=None, help="Weights & Biases run name")
add_arg("wandb_tags", type=str, default=None, help="Comma-separated tags for W&B run")
add_arg(
    "wandb_table_rows",
    type=int,
    default=6,
    help="Number of samples to log to W&B table",
)

# Debugging
add_arg(
    "check_label_keep_ratio",
    type=bool,
    default=True,
    help="Whether to check label retention ratio during data loading",
)

args = parser.parse_args()

# enforce precision policy
args.bf16 = True
args.fp16 = False
print_arguments(args)

# -------------------- W&B Setup --------------------
USE_WANDB = args.wandb_project is not None
from datetime import datetime

dt_suffix = datetime.now().strftime("%Y%m%d-%H%M")
base_name = args.base_model[:-1] if args.base_model.endswith("/") else args.base_model
base_model_name = os.path.basename(base_name)
user_name = args.wandb_run_name or base_model_name
unique_name = f"{user_name}-{dt_suffix}"

output_dir = os.path.join(args.output_dir, f"{base_model_name}-{dt_suffix}")
os.makedirs(output_dir, exist_ok=True)

if USE_WANDB:
    os.environ["WANDB_RUN_NAME"] = unique_name
    os.environ["WANDB_PROJECT"] = args.wandb_project
    if args.wandb_entity:
        os.environ["WANDB_ENTITY"] = args.wandb_entity
    if args.wandb_tags:
        os.environ["WANDB_TAGS"] = args.wandb_tags
    os.environ["WANDB_DIR"] = os.path.join(output_dir, "wandb")
    os.environ["WANDB_RESUME"] = "never"


# -------------------- Rate Limiting Helper --------------------


# -------------------- Freeze / unfreeze helpers --------------------
def _unwrap_to_base(model: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying WhisperForConditionalGeneration even if wrapped by PEFT."""
    try:
        if isinstance(model, PeftModel):
            return model.get_base_model()
    except Exception:
        pass
    return model


def _get_encoder_module(model: torch.nn.Module) -> torch.nn.Module:
    base = _unwrap_to_base(model)
    # WhisperForConditionalGeneration has .model.encoder
    if hasattr(base, "model") and hasattr(base.model, "encoder"):
        return base.model.encoder
    raise ValueError("Could not locate the Whisper encoder module.")


def freeze_encoder(model: torch.nn.Module) -> None:
    enc = _get_encoder_module(model)
    for p in enc.parameters():
        p.requires_grad = False


def unfreeze_encoder(model: torch.nn.Module) -> None:
    enc = _get_encoder_module(model)
    for p in enc.parameters():
        p.requires_grad = True


def count_trainable(model: torch.nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


class UnfreezeEncoderCallback(TrainerCallback):
    """
    Unfreezes the encoder at the end of epoch `unfreeze_after_epochs`, and
    adds those params to the optimizer so they start training next epoch.
    """

    def __init__(
        self,
        unfreeze_after_epochs: int,
        lr: float | None = None,
        wd: float | None = None,
    ):
        self.unfreeze_after_epochs = int(unfreeze_after_epochs)
        self.lr = lr
        self.wd = wd
        self.did_unfreeze = False

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.did_unfreeze or self.unfreeze_after_epochs <= 0:
            return

        # At end of epoch, HF sets state.epoch to a 1-based float (e.g., 1.0)
        if state.epoch is not None and int(state.epoch) >= self.unfreeze_after_epochs:
            model = kwargs["model"]
            optimizer = kwargs.get("optimizer", None)

            unfreeze_encoder(model)

            if optimizer is not None:
                enc = _get_encoder_module(model)
                # Only add params that are now trainable (and not already in any group)
                new_params = [p for p in enc.parameters() if p.requires_grad]
                optimizer.add_param_group(
                    {
                        "params": new_params,
                        "lr": self.lr if self.lr is not None else args.learning_rate,
                        "weight_decay": self.wd
                        if self.wd is not None
                        else args.weight_decay,
                    }
                )

            self.did_unfreeze = True
            tr, tot = count_trainable(model)
            print(
                f"🔓 Unfroze encoder at end of epoch {int(state.epoch)} "
                f"(trainable now {tr / 1e6:.2f}M / {tot / 1e6:.2f}M params)."
            )


def main():
    # PyTorch perf knobs
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
    except Exception:
        pass
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    set_seed(args.seed)

    # ----- Processor (tokenizer + feature extractor) -----
    processor = WhisperProcessor.from_pretrained(
        args.base_model,
        language=args.language,
        task=args.task,
        no_timestamps=not args.timestamps,
        local_files_only=args.local_files_only,
    )

    # Ensure pad token exists & align ids
    tok = processor.tokenizer
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token  # standard Whisper choice

    # ----- Datasets -----
    datasets_info = [
        {
            "name": "Beehzod/dataset_for_STT_TTSmodels",
            "filter_fn": None,  # No filter needed
        },
        {
            "name": "google/fleurs",
            "filter_fn": None,  # No filter needed
        },
        {
            "name": "bekzod123/uzbek_voice",
            "filter_fn": lambda ex: (ex.get("is_correct") is True),
        },
        {
            "name": "mozilla-foundation/common_voice_17_0",
            "filter_fn": None,  # No filter needed
        },
        {
            "name": "DavronSherbaev/uzbekvoice-filtered",
            "filter_fn": lambda ex: (
                ex.get("reported_reasons") is None
                and ex.get("downvotes_count", 0) == 0
                and ex.get("reported_count", 0) == 0
                and ex.get("client_id")
                not in [
                    "56ac8e86-b8c9-4879-a342-0eeb94f686fc",
                    "3d3fca02-6a07-41e2-9af4-60886ea60300",
                    "231d3776-2dbe-4a42-a535-c67943427e3f",
                    "e2716f95-70b5-4832-b903-eef2343591a4",
                    "2a815774-e953-4031-931a-8a28052e5cf9",
                    "d6fd3dc4-a55d-4a80-9bbf-b713325d05be",
                    "10b29e87-bf01-4b16-bead-a044076f849b",
                    "e3412d51-f079-4167-b3f9-311a976443ce",
                ]
            ),
        },
        {
            "name": "bekzod123/uzbek_voice_2",
            "filter_fn": None,  # No filter needed
        },
        {
            "name": "bekzod123/uzbek_voice_3",
            "filter_fn": None,  # No filter needed
        },
        {
            "name": "bekzod123/uzbek_voice_4",
            "filter_fn": None,  # No filter needed
        },
        {
            "name": "k2speech/FeruzaSpeech",
            "filter_fn": None,  # No filter needed
        },
    ]

    # ----- Build dataset specs (supports JSON manifest and HF Hub prefetch) -----
    # If train_data points to a JSON file, treat it as a manifest:
    # - Dict form: {"train": [...], "eval": [...]} where entries can be:
    #     - local file/dir paths (optionally "path:subset" for HF saved datasets)
    #     - HF hub refs as "hf://org/name:split" or "org/name:split" or {"hf": "org/name", "split": "train"}
    # - List form: [...], treated as training entries only
    #
    # HF hub entries are materialized to disk under "<output_dir>/prefetched/<org__name>/<split>"
    # and then referenced as "<path>:<split>" for our CustomDataset loader.
    def _is_hf_repo_spec(s: str) -> bool:
        base = s.split(":", 1)[0]
        return ("/" in base) and (not os.path.exists(base))

    def _prefetch_hf(
        repo: str,
        splits: list[str],
        save_root: str,
        revision: str | None = None,
        subset: str | None = None,
    ) -> list[str]:
        materialized = []

        # Special handling for google/fleurs dataset
        if repo == "google/fleurs" and subset:
            from huggingface_hub import snapshot_download

            for sp in splits:
                try:
                    print(f"Prefetching google/fleurs subset {subset}, split {sp}")

                    # Download the specific files for this subset
                    cache_dir = os.getenv(
                        "HF_DATASETS_CACHE", None
                    ) or os.path.expanduser("~/.cache/huggingface/datasets")

                    # Download the tar.gz and tsv files for the subset
                    snapshot_download(
                        repo_id="google/fleurs",
                        repo_type="dataset",
                        allow_patterns=[
                            f"data/{subset}/audio/*.tar.gz",
                            f"data/{subset}/*.tsv",
                        ],
                        local_dir_use_symlinks=False,
                    )

                    subset_part = f"#{subset}" if subset else ""
                    materialized.append(f"hf://{repo}{subset_part}:{sp}")

                except Exception as e:
                    print(f"Failed to prefetch google/fleurs {subset}:{sp}: {e}")
                    continue
        else:
            # Original logic for other datasets
            for sp in splits:
                try:
                    # Warm HF cache (no saving to disk) with explicit subset(config) and revision
                    adj_subset = subset
                    adj_revision = revision
                    # Load dataset normally
                    _ = rate_limited_request(
                        load_dataset,
                        repo,
                        name=adj_subset,
                        revision=adj_revision,
                        split=sp,
                        download_mode="reuse_dataset_if_exists",
                    )
                except Exception as e:
                    print(
                        f"Failed to load dataset {repo}:{sp} (subset={adj_subset}, revision={adj_revision}) from HF Hub: {e}"
                    )
                    continue
                rev_part = f"@{adj_revision}" if adj_revision else ""
                subset_part = f"#{adj_subset}" if adj_subset else ""
                materialized.append(f"hf://{repo}{rev_part}{subset_part}:{sp}")
        return materialized

    def _collect_entries(
        entries, save_root: str, default_splits: list[str]
    ) -> list[str]:
        if not entries:
            return []
        paths = []
        for ent in entries:
            if isinstance(ent, dict):
                # Support:
                # - {"hf": "org/name", "split": "train"}
                # - {"name": "org/name", "subset": "uz_uz", "revision": "refs/convert/parquet", "splits": ["train","validation"]}
                repo = ent.get("hf") or ent.get("huggingface") or ent.get("name")
                revision = ent.get("revision")
                subset = ent.get("subset")
                splits = ent.get("splits") or (
                    [ent.get("split")] if ent.get("split") else default_splits
                )
                if repo:
                    paths.extend(
                        _prefetch_hf(
                            repo, splits, save_root, revision=revision, subset=subset
                        )
                    )
                    continue
                print(f"Unrecognized manifest dict entry (missing 'name'/'hf'): {ent}")
            elif isinstance(ent, str):
                val = ent.strip()
                # Accept forms:
                # - hf://org/name:split
                # - hf://org/name@rev#subset:split
                # - org/name:split
                # - org/name@rev#subset:split
                val_no_proto = val[5:] if val.startswith("hf://") else val
                if ":" in val_no_proto:
                    base, split = val_no_proto.split(":", 1)
                else:
                    base, split = val_no_proto, None
                # Parse repo@revision#subset
                repo = base
                revision = None
                subset = None
                if "@" in base and "#" in base:
                    repo_part, rest = base.split("@", 1)
                    rev_part, subset = rest.split("#", 1)
                    repo, revision = repo_part, rev_part
                elif "@" in base:
                    repo, revision = base.split("@", 1)
                elif "#" in base:
                    repo, subset = base.split("#", 1)
                if _is_hf_repo_spec(repo):
                    fallback_splits = [
                        s
                        for s in default_splits
                        if s in ("train", "validation", "test")
                    ]
                    fallback_splits = fallback_splits or ["train", "validation", "test"]
                    splits = [split] if split else fallback_splits
                    paths.extend(
                        _prefetch_hf(
                            repo, splits, save_root, revision=revision, subset=subset
                        )
                    )
                else:
                    # local file/dir (optionally with :subset handled by CustomDataset)
                    paths.append(val)
            else:
                print(f"Unrecognized manifest entry: {ent}")
        return paths

    # Defaults
    train_data_arg = None
    eval_data_arg = None

    # If a manifest is provided via train_data path, parse it; otherwise keep CLI paths.
    manifest_prefetch_root = os.path.join(output_dir, "prefetched")
    if (
        args.train_data
        and args.train_data.endswith(".json")
        and os.path.isfile(args.train_data)
    ):
        try:
            with open(args.train_data, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            if isinstance(manifest, dict):
                train_entries = manifest.get("train") or manifest.get("training") or []
                eval_entries = (
                    manifest.get("eval")
                    or manifest.get("validation")
                    or manifest.get("test")
                    or []
                )
            elif isinstance(manifest, list):
                train_entries = manifest
                eval_entries = []
            else:
                raise ValueError(
                    "Manifest must be a dict with 'train'/'eval' keys or a list"
                )

            train_paths = _collect_entries(
                train_entries,
                save_root=manifest_prefetch_root,
                default_splits=["train", "validation", "test", "dev", "validated"],
            )
            eval_paths = _collect_entries(
                eval_entries,
                save_root=manifest_prefetch_root,
                default_splits=["train", "validation", "test", "dev", "validated"],
            )

            train_data_arg = "+".join(train_paths) if train_paths else None
            eval_data_arg = "+".join(eval_paths) if eval_paths else None

            if train_data_arg:
                print(
                    f"Using {len(train_paths)} dataset entries from manifest for training"
                )
            if eval_data_arg:
                print(
                    f"Using {len(eval_paths)} dataset entries from manifest for evaluation"
                )
        except Exception as e:
            print(f"Failed to parse or use dataset manifest at {args.train_data}: {e}")

    if not train_data_arg:
        train_data_arg = args.train_data
    if not eval_data_arg:
        eval_data_arg = args.test_data

    if eval_data_arg is None:
        print("No test data provided. Splitting train data: 92% train, 8% test")
        full_dataset = CustomDataset(
            data_list_path=train_data_arg,
            processor=processor,
            language=args.language,
            timestamps=args.timestamps,
            min_duration=args.min_audio_len,
            max_duration=args.max_audio_len,
            augment_config_path=args.augment_config_path,
            dataset_filters=datasets_info,
        )
        total_size = len(full_dataset)
        eval_size = max(1, int(0.08 * total_size))
        train_size = total_size - eval_size
        train_dataset, eval_dataset = random_split(
            full_dataset,
            [train_size, eval_size],
            generator=torch.Generator().manual_seed(args.seed),
        )
    else:
        train_dataset = CustomDataset(
            data_list_path=train_data_arg,
            processor=processor,
            language=args.language,
            timestamps=args.timestamps,
            min_duration=args.min_audio_len,
            max_duration=args.max_audio_len,
            augment_config_path=args.augment_config_path,
            dataset_filters=datasets_info,
        )
        eval_dataset = CustomDataset(
            data_list_path=eval_data_arg,
            processor=processor,
            language=args.language,
            timestamps=args.timestamps,
            min_duration=args.min_audio_len,
            max_duration=args.max_audio_len,
            dataset_filters=datasets_info,
        )

    print(f"Training data: {len(train_dataset)}, Eval data: {len(eval_dataset)}")

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # Optional sanity check: keep ratio of non-masked labels
    if args.check_label_keep_ratio:
        dl = DataLoader(train_dataset, batch_size=2, collate_fn=data_collator)
        batch = next(iter(dl))
        labels = batch["labels"]
        keep_ratio = (labels != -100).float().mean().item()
        print(f"🔎 Label keep ratio: {keep_ratio:.3f} (want high, e.g., >0.7)")

    # ----- Device map / DDP -----
    device_map: Any = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}

    # ----- Load model -----
    model = WhisperForConditionalGeneration.from_pretrained(
        args.base_model,
        device_map=device_map,
        local_files_only=args.local_files_only,
    )

    # Align special tokens
    model.config.pad_token_id = tok.pad_token_id
    model.config.eos_token_id = tok.eos_token_id
    model.config.bos_token_id = tok.bos_token_id
    start_id = tok.convert_tokens_to_ids("<|startoftranscript|>")
    if start_id is not None:
        model.config.decoder_start_token_id = start_id

    # Set generation ids (shared by generate)
    model.generation_config.pad_token_id = tok.pad_token_id
    model.generation_config.eos_token_id = tok.eos_token_id

    # Save processor (with PAD) early
    processor.save_pretrained(output_dir)
    print(f"✅ Saved processor files (incl. tokenizer with PAD) to: {output_dir}")

    # TRAIN: keep decoder free; EVAL/GEN: force language/task prompt
    eval_forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=args.language, task=args.task
    )
    model.config.forced_decoder_ids = None  # training
    model.generation_config.forced_decoder_ids = eval_forced_decoder_ids  # eval
    # Optional clarity (Whisper respects forced ids; these help downstream tools)
    try:
        model.generation_config.language = "uz"
        model.generation_config.task = "transcribe"
        model.generation_config.no_timestamps = not args.timestamps
    except Exception:
        pass

    # Make Whisper’s conv1 track gradients (safer AMP/bf16)
    model.model.encoder.conv1.register_forward_hook(make_inputs_require_grad)

    # Gradient checkpointing
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    # ----- Calculate steps for logging/eval and AdaLoRA -----
    eff_batch = (
        args.per_device_train_batch_size
        * args.gradient_accumulation_steps
        * max(1, world_size)
    )
    steps_per_epoch = max(1, math.ceil(len(train_dataset) / eff_batch))
    if args.logging_steps is None:
        args.logging_steps = max(1, steps_per_epoch // 2)
        print(f"Setting logging_steps to {args.logging_steps} (half epoch)")
    if args.save_steps is None:
        args.save_steps = max(1, steps_per_epoch // 2)
        print(f"Setting save_steps to {args.save_steps} (half epoch)")

    if args.eval_steps is None:
        eval_strategy = "epoch"
        save_strategy = "epoch"
        eval_steps = None
        print("Setting evaluation strategy to 'epoch'")
        print("Setting save strategy to 'epoch'")
    else:
        eval_strategy = "steps"
        save_strategy = "steps"
        eval_steps = args.eval_steps
        print(f"Setting evaluation strategy to 'steps' (eval_steps={eval_steps})")
        print("Setting save strategy to 'steps'")

    # ----- LoRA / AdaLoRA -----
    target_modules = ["k_proj", "q_proj", "v_proj", "out_proj", "fc1", "fc2"]
    total_update_steps = args.num_train_epochs * steps_per_epoch

    if args.use_lora:
        if args.resume_from_checkpoint:
            print("Loading adapters from checkpoint (resume).")
            model = PeftModel.from_pretrained(
                model, args.resume_from_checkpoint, is_trainable=True
            )
        else:
            print("Adding LoRA/AdaLoRA adapters...")
            if args.use_adalora:
                config = AdaLoraConfig(
                    init_r=64,
                    target_r=8,
                    beta1=0.6,
                    beta2=0.6,
                    tinit=50,
                    tfinal=400,
                    deltaT=5,
                    lora_alpha=args.lora_alpha,
                    lora_dropout=args.lora_dropout,
                    orth_reg_weight=0.75,
                    target_modules=target_modules,
                    total_step=total_update_steps,  # optimizer steps estimate
                )
            else:
                config = LoraConfig(
                    r=args.lora_r,
                    lora_alpha=args.lora_alpha,
                    target_modules=target_modules,
                    lora_dropout=args.lora_dropout,
                    bias="none",
                )
            model = get_peft_model(model, config)
    else:
        print("Using full fine-tuning (no LoRA)...")

    # --- Optional staged un/freezing ---
    if args.freeze_encoder_epochs > 0:
        print(f"🧊 Freezing encoder for {args.freeze_encoder_epochs} epoch(s) ...")
        freeze_encoder(model)
        tr, tot = count_trainable(model)
        print(
            f"   -> Trainable params while frozen: {tr / 1e6:.2f}M / {tot / 1e6:.2f}M"
        )

    # ----- Training args -----
    training_args_dict = {
        "output_dir": output_dir,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "warmup_ratio": args.warmup_ratio,
        "lr_scheduler_type": args.lr_scheduler_type,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps if save_strategy == "steps" else None,
        "save_total_limit": args.save_total_limit,
        "eval_strategy": eval_strategy,
        "save_strategy": save_strategy,
        "load_best_model_at_end": True,
        "metric_for_best_model": "wer",
        "greater_is_better": False,
        "bf16": True,
        "fp16": False,
        "optim": "adamw_torch_fused",
        "torch_compile": args.use_compile,
        "ddp_find_unused_parameters": False if ddp else None,
        "dataloader_num_workers": args.num_workers,
        "dataloader_pin_memory": True,
        "dataloader_persistent_workers": True if args.num_workers > 0 else False,
        "remove_unused_columns": False,
        "label_names": ["labels"],
        "report_to": (["tensorboard", "wandb"] if USE_WANDB else ["tensorboard"]),
        "push_to_hub": args.push_to_hub,
        "group_by_length": args.group_by_length,
        "length_column_name": args.length_column_name,
        "predict_with_generate": args.predict_with_generate,
        "generation_max_length": args.generation_max_length,
    }
    if eval_strategy == "steps":
        training_args_dict["eval_steps"] = eval_steps

    training_args = Seq2SeqTrainingArguments(**training_args_dict)

    # ----- Metrics: WER -----
    wer_metric = evaluate.load("wer")

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        labels = labels.copy()
        labels[labels == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.tokenizer.batch_decode(preds, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    # ----- Trainer -----
    callbacks = [
        SavePeftModelCallback(),
        EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience),
    ]
    if args.freeze_encoder_epochs > 0:
        callbacks.append(
            UnfreezeEncoderCallback(
                unfreeze_after_epochs=args.freeze_encoder_epochs,
                lr=args.learning_rate,
                wd=args.weight_decay,
            )
        )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=processor,  # keep to match your environment
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )
    model.config.use_cache = False
    trainer._load_from_checkpoint = load_from_checkpoint

    # ---- Training ----
    model.config.forced_decoder_ids = None  # training stays prompt-free
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # ---- Save & Export ----
    trainer.save_state()
    model.config.use_cache = True
    model.config.forced_decoder_ids = (
        eval_forced_decoder_ids  # set for downstream eval/infer
    )

    if training_args.local_rank in (0, -1):
        final_dir = os.path.join(output_dir, "checkpoint-final")
        os.makedirs(final_dir, exist_ok=True)
        model.save_pretrained(final_dir, safe_serialization=True)
        processor.save_pretrained(final_dir)
        print(f"✅ Saved processor files to: {final_dir}")

        # (Optional) merge LoRA into base weights
        try:
            if isinstance(model, PeftModel):
                skip_merge = getattr(args, "skip_merge", False)
                merge_on_cpu = getattr(args, "merge_on_cpu", True)
                merge_bf16 = getattr(args, "merge_bf16", True)
                save_sharded = getattr(args, "save_sharded", True)
                merge_max_shard_size = getattr(args, "merge_max_shard_size", "2000MB")

                if skip_merge:
                    print("⏭️ Skipping merge_and_unload because skip_merge=True")
                else:
                    print("🔄 Merging LoRA adapters into base model...")
                    if merge_on_cpu:
                        try:
                            print("↪️ Moving model to CPU for merge...")
                            model = model.to("cpu")
                        except Exception as move_e:
                            warnings.warn(
                                f"Failed to move model to CPU for merge: {move_e}"
                            )

                    merged = model.merge_and_unload()

                    if merge_bf16:
                        try:
                            print("🧪 Casting merged model to bfloat16 before save...")
                            merged = merged.to(torch.bfloat16)
                        except Exception as cast_e:
                            warnings.warn(
                                f"bf16 cast failed, saving in original dtype: {cast_e}"
                            )

                    merged_dir = os.path.join(output_dir, "checkpoint-final-merged")
                    os.makedirs(merged_dir, exist_ok=True)
                    print(
                        f"💾 Saving merged model to: {merged_dir} (sharded={save_sharded}, max_shard_size={merge_max_shard_size})"
                    )

                    save_kwargs = {"safe_serialization": True}
                    if save_sharded:
                        save_kwargs["max_shard_size"] = merge_max_shard_size
                    merged.save_pretrained(merged_dir, **save_kwargs)
                    processor.save_pretrained(merged_dir)
                    print(f"✅ Saved processor files to: {merged_dir}")
            else:
                print("ℹ️ Model is not a PeftModel; skipping merge.")
        except Exception as e:
            warnings.warn(f"Merge-and-unload skipped: {e}")

    if training_args.push_to_hub and (training_args.local_rank in (0, -1)):
        hub_model_id = args.hub_model_id if args.hub_model_id else output_dir
        model.push_to_hub(hub_model_id)
        try:
            processor.push_to_hub(hub_model_id)
        except Exception as e:
            warnings.warn(f"Pushing processor to Hub skipped: {e}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        sys.exit(1)
