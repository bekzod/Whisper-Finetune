#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
import argparse
import functools
import os
import sys
import warnings
from typing import Any

import numpy as np
import torch
from torch.utils.data import random_split
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    GenerationConfig,
)
from transformers.trainer_callback import EarlyStoppingCallback, TrainerCallback
from transformers.trainer_utils import set_seed

# ---- Your project utilities (kept as-is) ----
from utils.callback import SavePeftModelCallback
from utils.data_utils import DataCollatorSpeechSeq2SeqWithPadding
from utils.model_utils import load_from_checkpoint
from utils.reader import CustomDataset
from utils.utils import print_arguments, make_inputs_require_grad, add_arguments

# ---- Metrics ----
import evaluate

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
    help="Path to the training dataset. Supports JSON, JSONL, CSV formats, and Hugging Face dataset folders. Multiple datasets can be combined using '+' separator. For HF datasets, use 'path:subset' format (e.g., '../dataset:train'). CSV format supports 'filename,text' or 'filename|text' (LJSpeech) formats.",
)
add_arg(
    "test_data",
    type=str,
    default=None,
    help="Path to the test dataset. Supports JSON, JSONL, CSV formats, and Hugging Face dataset folders. If not provided, 8% of train data will be used for testing. For HF datasets, use 'path:subset' format (e.g., '../dataset:train').",
)
add_arg(
    "base_model",
    type=str,
    default="../models/whisper-large-v3",
    help="Base Whisper model",
)
add_arg("output_dir", type=str, default="output/", help="Path to save checkpoints")

# Task / language
add_arg(
    "timestamps", type=bool, default=False, help="Use timestamp tokens during training"
)
add_arg(
    "language",
    type=str,
    default="Uzbek",
    help="Language name or code. If None, train multilingual",
)
add_arg(
    "task",
    type=str,
    default="transcribe",
    choices=["transcribe", "translate"],
    help="Task type",
)

# Loader / augmentation
add_arg("min_audio_len", type=float, default=0.5, help="Min audio length (s)")
add_arg(
    "max_audio_len",
    type=float,
    default=30.0,
    help="Max audio length (s, Whisper cap ~30s)",
)
add_arg(
    "augment_config_path",
    type=str,
    default="./configs/augmentation.json",
    help="Path to augmentation config (optional)",
)
add_arg(
    "num_workers",
    type=int,
    default=min(10, os.cpu_count() or 1),
    help="Dataloader workers",
)

# LoRA / AdaLoRA
add_arg(
    "use_lora",
    type=bool,
    default=True,
    help="Enable LoRA/AdaLoRA (False for full fine-tuning)",
)
add_arg(
    "use_adalora", type=bool, default=True, help="Use AdaLoRA instead of standard LoRA"
)
add_arg("lora_r", type=int, default=16, help="LoRA rank (typical 8-16 for Whisper v3)")
add_arg("lora_alpha", type=int, default=32, help="LoRA alpha")
add_arg("lora_dropout", type=float, default=0.05, help="LoRA dropout")

# AdaLoRA schedule is inlined in code (no extra CLI controls)

# Training schedule
add_arg("num_train_epochs", type=int, default=3, help="Epochs")
add_arg("per_device_train_batch_size", type=int, default=8, help="Per-GPU train batch")
add_arg("per_device_eval_batch_size", type=int, default=8, help="Per-GPU eval batch")
add_arg("gradient_accumulation_steps", type=int, default=4, help="Grad accumulation")
add_arg("learning_rate", type=float, default=2e-4, help="LR (LoRA typical 5e-5 ~ 5e-4)")
add_arg("weight_decay", type=float, default=0.01, help="Weight decay")
add_arg("max_grad_norm", type=float, default=1.0, help="Grad clip")
add_arg("warmup_ratio", type=float, default=0.05, help="Warmup ratio")
add_arg("lr_scheduler_type", type=str, default="cosine", help="Scheduler")

# Logging / eval / saving
add_arg(
    "logging_steps",
    type=int,
    default=None,
    help="Logging steps (if None, logs every half epoch)",
)
add_arg(
    "eval_steps",
    type=int,
    default=None,
    help="Eval steps (if None, evaluates every epoch)",
)
add_arg(
    "save_steps",
    type=int,
    default=None,
    help="Save steps (if None, saves every half epoch)",
)
add_arg("save_total_limit", type=int, default=10, help="Max checkpoints to keep")
add_arg(
    "predict_with_generate", type=bool, default=True, help="Use generate() during eval"
)
add_arg(
    "early_stopping_patience",
    type=int,
    default=4,
    help="Early stopping patience (eval calls)",
)
add_arg(
    "group_by_length",
    type=bool,
    default=False,
    help="Bucket by length to reduce padding",
)
add_arg(
    "length_column_name",
    type=str,
    default=None,
    help="Name of length column for bucketing (optional)",
)
add_arg(
    "generation_max_length",
    type=int,
    default=225,
    help="Max tokens for generation (eval)",
)

# Infra / misc
add_arg("seed", type=int, default=42, help="Random seed")
add_arg("use_compile", type=bool, default=False, help="torch.compile (PyTorch 2.x)")
add_arg("local_files_only", type=bool, default=True, help="Load only from local cache")
add_arg(
    "resume_from_checkpoint", type=str, default=None, help="Path to resume checkpoint"
)
add_arg("push_to_hub", type=bool, default=False, help="Push to HF Hub at end")
add_arg("hub_model_id", type=str, default=None, help="HF Hub repo id")

# Weights & Biases
add_arg(
    "wandb_project",
    type=str,
    default=None,
    help="W&B project name (enables W&B if set)",
)
add_arg("wandb_entity", type=str, default=None, help="W&B entity/org (optional)")
add_arg("wandb_run_name", type=str, default=None, help="W&B run name (optional)")
add_arg("wandb_tags", type=str, default=None, help="Comma-separated tags (optional)")
add_arg(
    "wandb_table_rows",
    type=int,
    default=6,
    help="How many eval samples to log as table each eval",
)

args = parser.parse_args()

# Enforce your precision policy:
args.bf16 = True
args.fp16 = False

print_arguments(args)

# -------------------- W&B Setup (robust for parallel runs) --------------------
USE_WANDB = args.wandb_project is not None

# compute base output_dir early (you already do this later; keep consistent)
base_name = args.base_model[:-1] if args.base_model.endswith("/") else args.base_model
output_dir = os.path.join(args.output_dir, os.path.basename(base_name))
os.makedirs(output_dir, exist_ok=True)

if USE_WANDB:
    import time, uuid

    # 1) unique run name: prefer user-provided name but append timestamp/pid/uuid
    timestamp = int(time.time())
    short_uuid = uuid.uuid4().hex[:6]
    pid = os.getpid()
    user_name = args.wandb_run_name or "run"
    unique_run_name = f"{user_name}-{timestamp}-{pid}-{short_uuid}"
    os.environ["WANDB_RUN_NAME"] = unique_run_name

    # 2) project/entity (keep your values)
    os.environ["WANDB_PROJECT"] = args.wandb_project
    if args.wandb_entity:
        os.environ["WANDB_ENTITY"] = args.wandb_entity
    if args.wandb_tags:
        os.environ["WANDB_TAGS"] = args.wandb_tags

    # 3) direct W&B to write inside the run's output dir (prevents ./wandb collisions)
    #    this will create output_dir/wandb instead of ./wandb
    os.environ["WANDB_DIR"] = os.path.join(output_dir, "wandb")

    # 4) ensure wandb doesn't try to resume an earlier run by accident
    #    If you intentionally want to resume set WANDB_RESUME or WANDB_RUN_ID explicitly.
    os.environ["WANDB_RESUME"] = "never"


# -------------------- Main --------------------
def main():
    # perf niceties
    # torch.backends.cuda.matmul.allow_tf32 = True
    # torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)

    # Optional: nudge PyTorch’s matmul heuristics
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

    # ---------- Save processor configs ----------
    # Ensure output dir exists and save processor there for training artifacts
    base_name = (
        args.base_model[:-1] if args.base_model.endswith("/") else args.base_model
    )
    output_dir = os.path.join(args.output_dir, os.path.basename(base_name))
    os.makedirs(output_dir, exist_ok=True)

    # 1) Save into the training output root (for convenience / reuse)
    processor.save_pretrained(output_dir)
    print(f"✅ Saved processor files (incl. preprocessor_config.json) to: {output_dir}")

    # (Optional) also keep a copy next to the base model (comment out if undesired)
    # processor.save_pretrained(args.base_model)
    # print(f"✅ Saved processor files next to base model: {args.base_model}")

    # ----- Dataset Filters -----
    datasets_info = [
        {
            "name": "uzbekvoice-filtered",
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
        {"name": "uzbek_voice", "filter_fn": lambda ex: (ex.get("is_correct") == True)},
    ]

    # ----- Datasets -----
    if args.test_data is None:
        # If no test data provided, load train data and split it
        print("No test data provided. Splitting train data: 92% train, 8% test")
        full_dataset = CustomDataset(
            data_list_path=args.train_data,
            processor=processor,
            language=args.language,
            timestamps=args.timestamps,
            min_duration=args.min_audio_len,
            max_duration=args.max_audio_len,
            augment_config_path=args.augment_config_path,
            dataset_filters=datasets_info,
        )

        # Calculate split sizes
        total_size = len(full_dataset)
        eval_size = int(0.08 * total_size)
        train_size = total_size - eval_size

        # Perform random split
        train_dataset, eval_dataset = random_split(
            full_dataset,
            [train_size, eval_size],
            generator=torch.Generator().manual_seed(args.seed),
        )
    else:
        # Load separate train and test datasets
        train_dataset = CustomDataset(
            data_list_path=args.train_data,
            processor=processor,
            language=args.language,
            timestamps=args.timestamps,
            min_duration=args.min_audio_len,
            max_duration=args.max_audio_len,
            augment_config_path=args.augment_config_path,
            dataset_filters=datasets_info,
        )
        eval_dataset = CustomDataset(
            data_list_path=args.test_data,
            processor=processor,
            language=args.language,
            timestamps=args.timestamps,
            min_duration=args.min_audio_len,
            max_duration=args.max_audio_len,
            dataset_filters=datasets_info,
        )

    print(f"Training data: {len(train_dataset)}, Eval data: {len(eval_dataset)}")

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # ----- Device map / DDP -----
    device_map: Any = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}

    # ----- Load model (no quantization) -----
    model = WhisperForConditionalGeneration.from_pretrained(
        args.base_model,
        device_map=device_map,
        local_files_only=args.local_files_only,
    )
    # Whisper generation defaults: set a fresh GenerationConfig to avoid migration warnings
    gen_cfg_path = os.path.join(args.base_model, "generation_config.json")
    if os.path.exists(gen_cfg_path):
        gen_cfg = GenerationConfig.from_pretrained(args.base_model)
    else:
        gen_cfg = GenerationConfig()
        # carry over existing begin_suppress_tokens/suppress_tokens/max_length if present
        if hasattr(model.config, "begin_suppress_tokens"):
            gen_cfg.begin_suppress_tokens = model.config.begin_suppress_tokens
        if hasattr(model.config, "suppress_tokens"):
            gen_cfg.suppress_tokens = model.config.suppress_tokens
        if hasattr(model.config, "max_length"):
            gen_cfg.max_length = model.config.max_length
    # Override with CLI/runtime prefs
    gen_cfg.task = args.task
    gen_cfg.language = args.language
    gen_cfg.no_timestamps = not args.timestamps
    gen_cfg.forced_decoder_ids = None
    if getattr(gen_cfg, "suppress_tokens", None) is None:
        gen_cfg.suppress_tokens = []
    model.generation_config = gen_cfg

    # Safer multi-GPU w/ Whisper’s conv1 front-end
    model.model.encoder.conv1.register_forward_hook(make_inputs_require_grad)

    # Gradient checkpointing (big memory saver)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    # ----- Calculate steps for logging and eval -----
    # Calculate steps per epoch based on batch size and gradient accumulation
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    effective_batch_size = (
        args.per_device_train_batch_size * args.gradient_accumulation_steps * world_size
    )
    steps_per_epoch = len(train_dataset) // effective_batch_size

    # Set logging_steps to half an epoch if not specified
    if args.logging_steps is None:
        args.logging_steps = max(1, steps_per_epoch // 2)
        print(f"Setting logging_steps to {args.logging_steps} (half epoch)")

    # Set save_steps to half an epoch if not specified
    if args.save_steps is None:
        args.save_steps = max(1, steps_per_epoch // 2)
        print(f"Setting save_steps to {args.save_steps} (half epoch)")

    # Set eval strategy based on whether eval_steps is provided
    # Decide eval strategy and align save strategy for compatibility with load_best_model_at_end
    if args.eval_steps is None:
        eval_strategy = "epoch"
        save_strategy = "epoch"
        eval_steps = None
        print("Setting evaluation strategy to 'epoch' (eval every epoch)")
        print("Setting save strategy to 'epoch' to match evaluation strategy")
    else:
        eval_strategy = "steps"
        save_strategy = "steps"
        eval_steps = args.eval_steps
        print(f"Setting evaluation strategy to 'steps' with eval_steps={eval_steps}")
        print("Setting save strategy to 'steps' to match evaluation strategy")

    # ----- LoRA / AdaLoRA -----
    total_step = args.num_train_epochs * max(1, len(train_dataset))
    target_modules = ["k_proj", "q_proj", "v_proj", "out_proj", "fc1", "fc2"]

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
                    total_step=total_step,
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
        # No PEFT adapters - model remains as is for full fine-tuning

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
        "bf16": True,  # enforced
        "fp16": False,  # enforced
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

    # Add eval_steps only if using steps strategy
    if eval_strategy == "steps":
        training_args_dict["eval_steps"] = eval_steps

    training_args = Seq2SeqTrainingArguments(**training_args_dict)

    # ----- Metrics: WER -----
    wer_metric = evaluate.load("wer")

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        # Replace -100 with pad id for decoding
        labels = labels.copy()
        labels[labels == -100] = processor.tokenizer.pad_token_id

        # Using generation_config.task/language; no need to toggle forced_decoder_ids

        # Decode strings
        pred_str = processor.tokenizer.batch_decode(preds, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)

        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    # ----- Trainer -----
    callbacks = [
        SavePeftModelCallback,
        EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience),
    ]

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=processor,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )
    model.config.use_cache = False
    trainer._load_from_checkpoint = load_from_checkpoint

    # ---- Training ----
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # ---- Save & Export ----
    trainer.save_state()
    model.config.use_cache = True

    # Save PEFT adapter
    if training_args.local_rank in (0, -1):
        final_dir = os.path.join(output_dir, "checkpoint-final")
        os.makedirs(final_dir, exist_ok=True)

        model.save_pretrained(final_dir, safe_serialization=True)
        # 2) Save processor next to the final adapter checkpoint
        processor.save_pretrained(final_dir)
        print(f"✅ Saved processor files to: {final_dir}")

        # (Optional) merge LoRA into base weights for a single merged model
        try:
            if isinstance(model, PeftModel):
                # Read optional knobs (fallback to sensible defaults if not provided via CLI)
                skip_merge = getattr(args, "skip_merge", False)
                merge_on_cpu = getattr(args, "merge_on_cpu", True)
                merge_bf16 = getattr(args, "merge_bf16", True)
                save_sharded = getattr(args, "save_sharded", True)
                merge_max_shard_size = getattr(args, "merge_max_shard_size", "2000MB")

                if skip_merge:
                    print("⏭️ Skipping merge_and_unload because skip_merge=True")
                else:
                    print(
                        "🔄 Starting merge_and_unload of LoRA adapters into base model..."
                    )
                    if merge_on_cpu:
                        try:
                            print(
                                "↪️ Moving model to CPU for merging to reduce GPU memory usage..."
                            )
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

                    # 3) Save processor next to the merged weights
                    processor.save_pretrained(merged_dir)
                    print(f"✅ Saved processor files to: {merged_dir}")
            else:
                print("ℹ️ Model is not a PeftModel; skipping merge.")
        except Exception as e:
            warnings.warn(f"Merge-and-unload skipped: {e}")

    # Push to Hub (optional)
    if training_args.push_to_hub and (training_args.local_rank in (0, -1)):
        hub_model_id = args.hub_model_id if args.hub_model_id else output_dir
        model.push_to_hub(hub_model_id)
        # When pushing to hub, processors are auto-handled if you call processor.push_to_hub
        try:
            processor.push_to_hub(hub_model_id)
        except Exception as e:
            warnings.warn(f"Pushing processor to Hub skipped: {e}")

    # W&B will be automatically finished by the trainer


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        sys.exit(1)
