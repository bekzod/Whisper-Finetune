import argparse
import functools
import os

from transformers import (
    WhisperForConditionalGeneration,
    WhisperFeatureExtractor,
    WhisperTokenizerFast,
    WhisperProcessor,
)
from peft import PeftModel, PeftConfig
from utils.utils import print_arguments, add_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg(
    "lora_model",
    type=str,
    default="output/whisper-tiny/checkpoint-best/",
    help="Path to the fine-tuned model",
)
add_arg(
    "output_dir", type=str, default="models/", help="Directory to save the merged model"
)
add_arg(
    "local_files_only",
    type=bool,
    default=False,
    help="Whether to load model locally only without attempting to download",
)
args = parser.parse_args()
print_arguments(args)

# Check if model file exists
assert os.path.exists(args.lora_model), f"Model file {args.lora_model} does not exist"
# Get LoRA configuration parameters
peft_config = PeftConfig.from_pretrained(args.lora_model)
# Get Whisper base model
base_model = WhisperForConditionalGeneration.from_pretrained(
    peft_config.base_model_name_or_path,
    device_map={"": "cpu"},
    local_files_only=args.local_files_only,
)
# Merge with LoRA model
model = PeftModel.from_pretrained(
    base_model, args.lora_model, local_files_only=args.local_files_only
)
feature_extractor = WhisperFeatureExtractor.from_pretrained(
    peft_config.base_model_name_or_path, local_files_only=args.local_files_only
)
tokenizer = WhisperTokenizerFast.from_pretrained(
    peft_config.base_model_name_or_path, local_files_only=args.local_files_only
)
processor = WhisperProcessor.from_pretrained(
    peft_config.base_model_name_or_path, local_files_only=args.local_files_only
)

# Merge parameters
model = model.merge_and_unload()
model.train(False)

# Save directory path
if peft_config.base_model_name_or_path.endswith("/"):
    peft_config.base_model_name_or_path = peft_config.base_model_name_or_path[:-1]
save_directory = os.path.join(
    args.output_dir, f"{os.path.basename(peft_config.base_model_name_or_path)}-finetune"
)
os.makedirs(save_directory, exist_ok=True)

# Save model to specified directory
model.save_pretrained(save_directory, max_shard_size="4GB")
feature_extractor.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)
processor.save_pretrained(save_directory)
print(f"Merged model saved at: {save_directory}")
