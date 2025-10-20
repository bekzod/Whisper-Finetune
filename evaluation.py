#!/usr/bin/env python3
"""
Evaluation script with optional KenLM rescoring and vocabulary-first-token biasing.

Usage examples:
    python eval_with_vocab_bias.py --test_data dataset/test.json --model_path models/whisper-tiny-finetune \
        --vocab_bias_path bias_words.txt --vocab_alpha 5.0

bias_words.txt example (one word/phrase per line):
    Alice
    OpenAI
    Tashkent
"""

import argparse
import functools
import gc
import os
import sys

import evaluate
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    LogitsProcessorList,
)

from utils.data_utils import DataCollatorSpeechSeq2SeqWithPadding, remove_punctuation
from utils.reader import CustomDataset
from utils.utils import print_arguments, add_arguments
from utils.logits_processors import (
    KenLMBiasLogitsProcessor,
    VocabularyFirstTokenBiasLogitsProcessor,
)

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg(
    "test_data", type=str, default="dataset/test.json", help="Path to the test dataset"
)
add_arg(
    "model_path",
    type=str,
    default="models/whisper-tiny-finetune",
    help="Path to the base or merged model, or the model name on huggingface",
)
add_arg(
    "adapter_path",
    type=str,
    default=None,
    help="Optional path or Hub ID of a PEFT adapter to merge into the base model before evaluation",
)
add_arg("batch_size", type=int, default=16, help="Batch size for evaluation")
add_arg("num_workers", type=int, default=8, help="Number of threads for data loading")
add_arg(
    "language",
    type=str,
    default="Uzbek",
    help="Set language, can be full name or abbreviation, if None then evaluates multilingual",
)
add_arg("remove_pun", type=bool, default=True, help="Whether to remove punctuation")
add_arg(
    "timestamps",
    type=bool,
    default=False,
    help="Whether to use timestamp data during evaluation",
)
add_arg(
    "min_audio_len", type=float, default=0.5, help="Minimum audio length in seconds"
)
add_arg("max_audio_len", type=float, default=30, help="Maximum audio length in seconds")
add_arg(
    "local_files_only",
    type=bool,
    default=True,
    help="Whether to only load models locally, without attempting to download",
)
add_arg(
    "task",
    type=str,
    default="transcribe",
    choices=["transcribe", "translate"],
    help="Task for the model",
)
add_arg(
    "metric", type=str, default="wer", choices=["cer", "wer"], help="Evaluation method"
)
add_arg(
    "kenlm_path",
    type=str,
    default=None,
    help="Optional path to a KenLM n-gram model (ARPA or binary) to bias decoding",
)
add_arg(
    "kenlm_alpha",
    type=float,
    default=0.5,
    help="KenLM weight to add to model logits (applied to LM log-prob deltas)",
)
add_arg(
    "kenlm_top_k",
    type=int,
    default=50,
    help="Rescore only the top-k tokens at each step with KenLM to reduce overhead",
)

# New args for vocabulary-first-token biasing
add_arg(
    "vocab_bias_path",
    type=str,
    default=None,
    help="Optional path to newline-separated file with words/phrases to bias (one per line).",
)
add_arg(
    "vocab_alpha",
    type=float,
    default=5.0,
    help="Additive logit boost (scalar) applied to the first token of each vocabulary bias entry.",
)

args = parser.parse_args()
print_arguments(args)


# -----------------------
# Sanity checks for model/adapters
# -----------------------
assert "openai" == os.path.dirname(args.model_path) or os.path.exists(
    args.model_path
), (
    f"Model file {args.model_path} does not exist, please check if the model has been successfully merged, or if it's an existing model on huggingface"
)
if args.adapter_path and args.local_files_only:
    assert os.path.exists(args.adapter_path), (
        f"Adapter path {args.adapter_path} does not exist; set local_files_only=False to allow downloading from the Hub"
    )


# -----------------------
# Main
# -----------------------
def main():
    processor = WhisperProcessor.from_pretrained(
        args.model_path,
        language=args.language,
        task=args.task,
        no_timestamps=not args.timestamps,
        local_files_only=args.local_files_only,
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model_path, device_map="auto", local_files_only=args.local_files_only
    )

    if args.adapter_path:
        try:
            from peft import PeftModel
        except Exception as e:
            raise RuntimeError(
                "You provided adapter_path, but the 'peft' package is not installed. Please install peft to use adapters."
            ) from e
        model = PeftModel.from_pretrained(
            model,
            args.adapter_path,
            local_files_only=args.local_files_only,
        )
        if hasattr(model, "merge_and_unload"):
            model = model.merge_and_unload()

    # build the proper prompt once (requires your processor)
    prompt_ids = processor.get_decoder_prompt_ids(language="uz", task="transcribe")

    model.generation_config.forced_decoder_ids = prompt_ids
    model.generation_config.no_timestamps = True  # or False, depending on your need
    model.eval()
    # Build logits processors list
    processors = []

    # KenLM (optional)
    if args.kenlm_path:
        try:
            import kenlm  # pip install kenlm
        except Exception as e:
            raise RuntimeError(
                "You provided kenlm_path, but the 'kenlm' package is not installed. Please install kenlm to enable KenLM biasing."
            ) from e
        if not os.path.exists(args.kenlm_path):
            raise FileNotFoundError(f"KenLM model file {args.kenlm_path} not found")
        lm = kenlm.Model(args.kenlm_path)
        processors.append(
            KenLMBiasLogitsProcessor(
                kenlm_model=lm,
                tokenizer=processor.tokenizer,
                alpha=args.kenlm_alpha,
                top_k=args.kenlm_top_k,
            )
        )

    # Vocab-first-token biasing (optional)
    if args.vocab_bias_path:
        if not os.path.exists(args.vocab_bias_path):
            raise FileNotFoundError(f"Vocab bias file {args.vocab_bias_path} not found")
        with open(args.vocab_bias_path, "r", encoding="utf-8") as fh:
            bias_words = [line.strip() for line in fh if line.strip()]
        if bias_words:
            processors.append(
                VocabularyFirstTokenBiasLogitsProcessor(
                    tokenizer=processor.tokenizer,
                    bias_words=bias_words,
                    alpha=args.vocab_alpha,
                )
            )

    logits_processor = LogitsProcessorList(processors) if processors else None

    # Prepare dataset and dataloader
    test_dataset = CustomDataset(
        data_list_path=args.test_data,
        processor=processor,
        timestamps=args.timestamps,
        min_duration=args.min_audio_len,
        max_duration=args.max_audio_len,
    )
    print(f"Test data: {len(test_dataset)}")

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    eval_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=data_collator,
    )

    metric = evaluate.load(f"metrics/{args.metric}.py")

    # Evaluation loop
    first_batch_logged = False
    for step, batch in enumerate(tqdm(eval_dataloader)):
        with torch.autocast(device_type="cuda"):
            with torch.no_grad():
                # Ensure inputs are on GPU if available
                input_feats = (
                    batch["input_features"].cuda()
                    if torch.cuda.is_available()
                    else batch["input_features"]
                )
                decoder_start = (
                    batch["labels"][:, :4].cuda()
                    if torch.cuda.is_available()
                    else batch["labels"][:, :4]
                )
                # Only pass logits_processor if there are processors
                generate_kwargs = {
                    "input_features": input_feats,
                    "decoder_input_ids": decoder_start,
                    "max_new_tokens": 255,
                }
                if logits_processor is not None:
                    generate_kwargs["logits_processor"] = logits_processor

                generated_tokens = model.generate(**generate_kwargs).cpu().numpy()

                labels = batch["labels"].cpu().numpy()
                labels = np.where(
                    labels != -100, labels, processor.tokenizer.pad_token_id
                )

                decoded_preds = processor.tokenizer.batch_decode(
                    generated_tokens, skip_special_tokens=True
                )
                decoded_labels = processor.tokenizer.batch_decode(
                    labels, skip_special_tokens=True
                )

                if args.remove_pun:
                    decoded_preds = remove_punctuation(decoded_preds)
                    decoded_labels = remove_punctuation(decoded_labels)

                # Log first batch for debugging
                if not first_batch_logged:
                    first_batch_logged = True
                    print("\n" + "=" * 80)
                    print("FIRST BATCH TRANSCRIPTIONS (for debugging)")
                    print("=" * 80)
                    for i in range(
                        min(3, len(decoded_preds))
                    ):  # Log up to first 3 examples
                        print(f"\nExample {i + 1}:")
                        print(f"  Ground Truth: {decoded_labels[i]}")
                        print(f"  Prediction:   {decoded_preds[i]}")
                        print(f"  Match: {decoded_preds[i] == decoded_labels[i]}")
                    print("=" * 80 + "\n")

                metric.add_batch(predictions=decoded_preds, references=decoded_labels)

        del generated_tokens, labels, batch
        gc.collect()

    result = metric.compute()
    print(f"Evaluation result: {args.metric}={round(result, 5)}")


if __name__ == "__main__":
    main()
