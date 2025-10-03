import argparse
import functools
import gc
import os

import evaluate
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    LogitsProcessor,
    LogitsProcessorList,
)

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
    "metric", type=str, default="cer", choices=["cer", "wer"], help="Evaluation method"
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
args = parser.parse_args()
print_arguments(args)

# Check if the model path is valid
assert "openai" == os.path.dirname(args.model_path) or os.path.exists(
    args.model_path
), (
    f"Model file {args.model_path} does not exist, please check if the model has been successfully merged, or if it's an existing model on huggingface"
)
# If adapter is provided and we're restricted to local files, ensure it exists
if args.adapter_path and args.local_files_only:
    assert os.path.exists(args.adapter_path), (
        f"Adapter path {args.adapter_path} does not exist; set local_files_only=False to allow downloading from the Hub"
    )


class KenLMBiasLogitsProcessor(LogitsProcessor):
    """
    A simple KenLM-based logits processor that rescales the logits for the top-k
    candidate tokens at each step using an n-gram language model.

    For each hypothesis in the batch, we:
      - Decode the current prefix to text with the tokenizer.
      - For the top-k candidate next tokens, decode each token piece to text,
        compute LM score for the new text and subtract the LM score of the prefix
        to obtain an incremental LM log-probability.
      - Add alpha * delta to the token's logit.

    Notes:
      - This operates on decoded text pieces; with subword/byte-level tokens the LM
        will see partial words. Despite being approximate, it often provides useful bias.
      - Use a moderate top_k to balance speed/quality.
    """

    def __init__(self, kenlm_model, tokenizer, alpha: float = 0.5, top_k: int = 50):
        self.lm = kenlm_model
        self.tokenizer = tokenizer
        self.alpha = float(alpha)
        self.top_k = int(top_k)
        self._cache = {}  # cache for prefix and extended text LM scores

    def _lm_score(self, text: str) -> float:
        if text in self._cache:
            return self._cache[text]
        # KenLM scores are log probabilities; exact base is not critical since alpha scales it.
        score = self.lm.score(text.strip(), bos=False, eos=False)
        self._cache[text] = score
        return score

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        # Decode current prefixes
        prefixes = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        vocab_size = scores.shape[-1]
        for i, prefix in enumerate(prefixes):
            base = self._lm_score(prefix)
            k = min(self.top_k, vocab_size)
            topk = torch.topk(scores[i], k)
            idxs = topk.indices
            # Compute deltas for top-k candidates
            deltas = []
            for tid in idxs.tolist():
                token_piece = self.tokenizer.decode([tid], skip_special_tokens=True)
                if not token_piece:
                    deltas.append(0.0)
                    continue
                new_text = prefix + token_piece
                delta = self._lm_score(new_text) - base
                deltas.append(self.alpha * float(delta))
            if deltas:
                scores[i, idxs] = scores[i, idxs] + torch.tensor(
                    deltas, device=scores.device, dtype=scores.dtype
                )
        return scores


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
    # If a PEFT adapter is provided, merge it into the base model
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
        # Merge LoRA/PEFT weights into the base model for efficient inference
        if hasattr(model, "merge_and_unload"):
            model = model.merge_and_unload()
    model.generation_config.language = args.language.lower()
    model.generation_config.forced_decoder_ids = None
    model.eval()

    # Optional: build KenLM logits processor
    logits_processor = None
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
        logits_processor = LogitsProcessorList(
            [
                KenLMBiasLogitsProcessor(
                    kenlm_model=lm,
                    tokenizer=processor.tokenizer,
                    alpha=args.kenlm_alpha,
                    top_k=args.kenlm_top_k,
                )
            ]
        )

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
                        logits_processor=logits_processor,
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
