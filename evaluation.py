import argparse
import functools
import gc
import os

import evaluate
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from utils.data_utils import DataCollatorSpeechSeq2SeqWithPadding, remove_punctuation
from utils.reader import CustomDataset
from utils.utils import print_arguments, add_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg(
    "test_data", type=str, default="dataset/test.json", help="Path to the test dataset"
)
add_arg(
    "model_path",
    type=str,
    default="models/whisper-tiny-finetune",
    help="Path to the merged model, or the model name on huggingface",
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
    "metric", type=str, default="cer", choices=["cer", "wer"], help="Evaluation method"
)
args = parser.parse_args()
print_arguments(args)

# Check if the model path is valid
assert "openai" == os.path.dirname(args.model_path) or os.path.exists(
    args.model_path
), (
    f"Model file {args.model_path} does not exist, please check if the model has been successfully merged, or if it's an existing model on huggingface"
)


def main():
    # Get Whisper's data processor, which includes feature extractor and tokenizer
    processor = WhisperProcessor.from_pretrained(
        args.model_path,
        language=args.language,
        task=args.task,
        no_timestamps=not args.timestamps,
        local_files_only=args.local_files_only,
    )
    # Get the model
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model_path, device_map="auto", local_files_only=args.local_files_only
    )
    model.generation_config.language = args.language.lower()
    model.generation_config.forced_decoder_ids = None
    model.eval()

    # Get test data
    test_dataset = CustomDataset(
        data_list_path=args.test_data,
        processor=processor,
        timestamps=args.timestamps,
        min_duration=args.min_audio_len,
        max_duration=args.max_audio_len,
    )
    print(f"Test data: {len(test_dataset)}")

    # Data padding collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    eval_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=data_collator,
    )

    # Get evaluation metric
    metric = evaluate.load(f"metrics/{args.metric}.py")

    # Start evaluation
    for step, batch in enumerate(tqdm(eval_dataloader)):
        with torch.autocast(device_type="cuda"):
            with torch.no_grad():
                generated_tokens = (
                    model.generate(
                        input_features=batch["input_features"].cuda(),
                        decoder_input_ids=batch["labels"][:, :4].cuda(),
                        max_new_tokens=255,
                    )
                    .cpu()
                    .numpy()
                )
                labels = batch["labels"].cpu().numpy()
                labels = np.where(
                    labels != -100, labels, processor.tokenizer.pad_token_id
                )
                # Convert predicted and actual tokens to text
                decoded_preds = processor.tokenizer.batch_decode(
                    generated_tokens, skip_special_tokens=True
                )
                decoded_labels = processor.tokenizer.batch_decode(
                    labels, skip_special_tokens=True
                )
                # Remove punctuation
                if args.remove_pun:
                    decoded_preds = remove_punctuation(decoded_preds)
                    decoded_labels = remove_punctuation(decoded_labels)

                metric.add_batch(predictions=decoded_preds, references=decoded_labels)
        # Delete computation records
        del generated_tokens, labels, batch
        gc.collect()
    # Calculate evaluation results
    m = metric.compute()
    print(f"Evaluation result: {args.metric}={round(m, 5)}")


if __name__ == "__main__":
    main()
