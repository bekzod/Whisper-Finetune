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
    help="Path to the training dataset. Supports JSON, JSONL, CSV, and Parquet formats. Multiple datasets can be combined using '+' separator. Glob patterns are supported for parquet files, e.g., '../datasets/train.json+../datasets/cleaned.json' or '/data/train-*.parquet'. CSV format supports 'filename,text' or 'filename|text' (LJSpeech) formats.",
)
add_arg(
    "test_data",
    type=str,
    default=None,
    help="Path to the test dataset. Supports JSON, JSONL, CSV, and Parquet formats. If not provided, 8% of train data will be used for testing.",
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
add_arg("num_workers", type=int, default=os.cpu_count(), help="Dataloader workers")

# LoRA / AdaLoRA
add_arg(
    "use_adalora", type=bool, default=True, help="Use AdaLoRA instead of standard LoRA"
)
add_arg("lora_r", type=int, default=16, help="LoRA rank (typical 8-16 for Whisper v3)")
add_arg("lora_alpha", type=int, default=32, help="LoRA alpha")
add_arg("lora_dropout", type=float, default=0.05, help="LoRA dropout")

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
add_arg("logging_steps", type=int, default=250, help="Logging steps")
add_arg("eval_steps", type=int, default=1000, help="Eval steps")
add_arg("save_steps", type=int, default=1000, help="Save steps")
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

# -------------------- W&B Setup --------------------
USE_WANDB = args.wandb_project is not None
if USE_WANDB:
    os.environ["WANDB_PROJECT"] = args.wandb_project
    if args.wandb_entity:
        os.environ["WANDB_ENTITY"] = args.wandb_entity
    if args.wandb_run_name:
        os.environ["WANDB_RUN_NAME"] = args.wandb_run_name
    if args.wandb_tags:
        os.environ["WANDB_TAGS"] = args.wandb_tags


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
        )
        eval_dataset = CustomDataset(
            data_list_path=args.test_data,
            processor=processor,
            language=args.language,
            timestamps=args.timestamps,
            min_duration=args.min_audio_len,
            max_duration=args.max_audio_len,
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
    # Whisper generation defaults
    model.config.suppress_tokens = []

    # TRAIN: keep decoder prompt free; EVAL/GEN: use processor prompt ids
    train_forced_decoder_ids = None
    eval_forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=args.language, task=args.task
    )

    # Safer multi-GPU w/ Whisper’s conv1 front-end
    model.model.encoder.conv1.register_forward_hook(make_inputs_require_grad)

    # Gradient checkpointing (big memory saver)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    # ----- LoRA / AdaLoRA -----
    total_step = args.num_train_epochs * max(1, len(train_dataset))
    target_modules = ["k_proj", "q_proj", "v_proj", "out_proj", "fc1", "fc2"]

    if args.resume_from_checkpoint:
        print("Loading adapters from checkpoint (resume).")
        model = PeftModel.from_pretrained(
            model, args.resume_from_checkpoint, is_trainable=True
        )
    else:
        print("Adding LoRA/AdaLoRA adapters...")
        if args.use_adalora:
            config = AdaLoraConfig(
                init_r=max(8, args.lora_r),
                target_r=max(4, args.lora_r // 2),
                beta1=0.85,
                beta2=0.85,
                tinit=200,
                tfinal=1000,
                deltaT=10,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                orth_reg_weight=0.5,
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

    # ----- Output dir -----
    base_name = (
        args.base_model[:-1] if args.base_model.endswith("/") else args.base_model
    )
    output_dir = os.path.join(args.output_dir, os.path.basename(base_name))

    # ----- Training args -----
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        bf16=True,  # enforced
        fp16=False,  # enforced
        optim="adamw_torch_fused",
        torch_compile=args.use_compile,
        ddp_find_unused_parameters=False if ddp else None,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True if args.num_workers > 0 else False,
        remove_unused_columns=False,
        label_names=["labels"],
        report_to=(["tensorboard", "wandb"] if USE_WANDB else ["tensorboard"]),
        push_to_hub=args.push_to_hub,
        group_by_length=args.group_by_length,
        length_column_name=args.length_column_name,
        predict_with_generate=args.predict_with_generate,
        generation_max_length=args.generation_max_length,
    )

    # ----- Metrics: WER -----
    wer_metric = evaluate.load("wer")

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        # Replace -100 with pad id for decoding
        labels = labels.copy()
        labels[labels == -100] = processor.tokenizer.pad_token_id

        # Ensure eval uses language/task prompt
        prev_forced = model.config.forced_decoder_ids
        model.config.forced_decoder_ids = eval_forced_decoder_ids

        # Decode strings
        pred_str = processor.tokenizer.batch_decode(preds, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Restore previous forced ids (avoid side-effects)
        model.config.forced_decoder_ids = prev_forced

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
        tokenizer=processor.tokenizer,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )
    model.config.use_cache = False
    trainer._load_from_checkpoint = load_from_checkpoint

    # ---- Training ----
    # Keep training free of forced prompt so adapters can learn freely
    model.config.forced_decoder_ids = train_forced_decoder_ids
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # ---- Save & Export ----
    trainer.save_state()
    model.config.use_cache = True
    # Set eval prompt by default for downstream generation
    model.config.forced_decoder_ids = eval_forced_decoder_ids

    # Save PEFT adapter
    if training_args.local_rank in (0, -1):
        model.save_pretrained(
            os.path.join(output_dir, "checkpoint-final"), safe_serialization=True
        )

        # (Optional) merge LoRA into base weights for a single merged model
        try:
            if isinstance(model, PeftModel):
                merged = model.merge_and_unload()
                merged.save_pretrained(
                    os.path.join(output_dir, "checkpoint-final-merged"),
                    safe_serialization=True,
                )
        except Exception as e:
            warnings.warn(f"Merge-and-unload skipped: {e}")

    # Push to Hub (optional)
    if training_args.push_to_hub and (training_args.local_rank in (0, -1)):
        hub_model_id = args.hub_model_id if args.hub_model_id else output_dir
        model.push_to_hub(hub_model_id)

    # W&B will be automatically finished by the trainer


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        sys.exit(1)
