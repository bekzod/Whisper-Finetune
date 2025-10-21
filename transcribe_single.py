#!/usr/bin/env python3
"""
Single-file transcription script with optional KenLM, vocabulary biasing, and basic voice enhancement.

Example usage:
    python transcribe_single.py --audio_path path/to/file.wav --model_path models/whisper-tiny-finetune
"""

from __future__ import annotations

import argparse
import functools
import os
from contextlib import nullcontext
from typing import Any, Iterable, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
from transformers import (
    LogitsProcessorList,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from utils.data_utils import remove_punctuation
from utils.logits_processors import (
    KenLMBiasLogitsProcessor,
    VocabularyFirstTokenBiasLogitsProcessor,
)
from utils.utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg(
    "audio_path",
    type=str,
    default="dataset/test.wav",
    help="Path to the audio file to transcribe",
)
add_arg(
    "model_path",
    type=str,
    default="models/whisper-tiny-finetune",
    help="Path to the base model, merged model, or Hugging Face repo ID",
)
add_arg(
    "adapter_path",
    type=str,
    default=None,
    help="Optional PEFT adapter path/Hub ID to merge before transcription",
)
add_arg(
    "language",
    type=str,
    default="Uzbek",
    help="Language passed to WhisperProcessor; set None for multilingual",
)
add_arg(
    "prompt_language",
    type=str,
    default=None,
    help="Optional language code for decoder prompt (defaults to `language` if unset)",
)
add_arg(
    "task",
    type=str,
    default="transcribe",
    choices=["transcribe", "translate"],
    help="Generation task for the model",
)
add_arg(
    "remove_pun",
    type=bool,
    default=True,
    help="Remove punctuation from the final transcript",
)
add_arg(
    "timestamps",
    type=bool,
    default=False,
    help="Allow timestamp tokens during decoding",
)
add_arg(
    "max_new_tokens",
    type=int,
    default=255,
    help="Maximum number of new tokens to generate",
)
add_arg(
    "num_beams",
    type=int,
    default=4,
    help="Beam search width used during generation",
)
add_arg(
    "do_sample",
    type=bool,
    default=True,
    help="Enable sampling for generation (set True to sample instead of deterministic decoding)",
)
add_arg(
    "temperature",
    type=float,
    default=0,
    help="Sampling temperature applied when do_sample is enabled",
)
add_arg(
    "top_p",
    type=float,
    default=1.0,
    help="Nucleus sampling probability mass (effective when do_sample is enabled)",
)
add_arg(
    "top_k",
    type=int,
    default=50,
    help="Top-k sampling limit (set 0 to disable; effective when do_sample is enabled)",
)
add_arg(
    "repetition_penalty",
    type=float,
    default=1.0,
    help="Penalty factor for repeated tokens during generation",
)
add_arg(
    "length_penalty",
    type=float,
    default=1.0,
    help="Length penalty applied during beam search decoding",
)
add_arg(
    "no_repeat_ngram_size",
    type=int,
    default=0,
    help="Prevent repeating n-grams of this size (0 disables constraint)",
)
add_arg(
    "device",
    type=str,
    default="auto",
    help="Device for inference (e.g. auto, cpu, cuda, cuda:0, mps)",
)
add_arg(
    "local_files_only",
    type=bool,
    default=True,
    help="Load models/adapters only from local cache without downloading",
)
add_arg(
    "kenlm_path",
    type=str,
    default=None,
    help="Optional path to a KenLM n-gram model (ARPA/binary) for biasing",
)
add_arg(
    "kenlm_alpha",
    type=float,
    default=0.3,
    help="KenLM weight applied to logit deltas",
)
add_arg(
    "kenlm_top_k",
    type=int,
    default=30,
    help="Rescore only the top-k tokens with KenLM",
)
add_arg(
    "vocab_bias_path",
    type=str,
    default=None,
    help="Optional newline-delimited file with bias phrases",
)
add_arg(
    "vocab_alpha",
    type=float,
    default=5.0,
    help="Logit boost applied to the first token of each bias phrase",
)
add_arg(
    "enhance_audio",
    type=bool,
    default=True,
    help="Apply simple gain boost to quiet audio before transcription",
)
add_arg(
    "enhance_target_rms",
    type=float,
    default=0.1,
    help="Target RMS amplitude for enhancement when gain is applied",
)
add_arg(
    "enhance_max_gain",
    type=float,
    default=10.0,
    help="Maximum gain multiplier applied during enhancement",
)

args = parser.parse_args()
print_arguments(args)


def resolve_device(device_str: Optional[str]) -> torch.device:
    """
    Resolve a device string to a torch.device, falling back to auto-detection.
    """
    if device_str in (None, "auto"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def load_audio(audio_path: str) -> Tuple[np.ndarray, int]:
    """
    Load an audio file and return mono float32 samples with its sample rate.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file {audio_path} not found")
    audio, sample_rate = sf.read(audio_path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return audio.astype(np.float32), int(sample_rate)


def resample_audio(
    audio: np.ndarray,
    original_sr: int,
    target_sr: int,
) -> np.ndarray:
    """
    Resample audio to the target sampling rate.
    """
    if original_sr == target_sr:
        return audio

    try:
        from scipy.signal import resample_poly

        # Use polyphase filtering for better quality resampling
        return resample_poly(audio, target_sr, original_sr).astype(np.float32)
    except Exception:
        # Fallback to linear interpolation if scipy is unavailable
        duration = audio.shape[0] / float(original_sr)
        target_length = int(round(duration * target_sr))
        if target_length <= 0:
            raise ValueError(
                f"Cannot resample audio with duration {duration:.6f}s "
                f"from {original_sr}Hz to {target_sr}Hz"
            )
        original_indices = np.linspace(0, duration, num=audio.shape[0], endpoint=False)
        target_indices = np.linspace(0, duration, num=target_length, endpoint=False)
        resampled_audio = np.interp(target_indices, original_indices, audio)
        return resampled_audio.astype(np.float32)


def enhance_audio(
    audio: np.ndarray,
    target_rms: float,
    max_gain: float,
) -> np.ndarray:
    """
    Boost the audio level if the RMS amplitude is below the target threshold.
    """
    if target_rms <= 0 or max_gain <= 1.0:
        return audio

    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    if rms <= 0 or rms >= target_rms:
        return audio

    gain = min(max_gain, target_rms / max(rms, 1e-8))
    if gain <= 1.0:
        return audio

    enhanced = audio * gain
    peak = float(np.max(np.abs(enhanced)))
    if peak > 1.0:
        enhanced /= peak
    return enhanced.astype(np.float32)


def build_logits_processors(
    tokenizer: Any,
    bias_words: Iterable[str],
    args: argparse.Namespace,
) -> Optional[LogitsProcessorList]:
    """
    Construct the logits processor list for KenLM and vocabulary biasing.
    """
    processors = []
    if args.kenlm_path:
        try:
            import kenlm  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "KenLM path provided but `kenlm` package is not installed."
            ) from exc
        if not os.path.exists(args.kenlm_path):
            raise FileNotFoundError(f"KenLM model file {args.kenlm_path} not found")
        lm_model = kenlm.Model(args.kenlm_path)
        processors.append(
            KenLMBiasLogitsProcessor(
                kenlm_model=lm_model,
                tokenizer=tokenizer,
                alpha=args.kenlm_alpha,
                top_k=args.kenlm_top_k,
            )
        )
    if bias_words:
        processors.append(
            VocabularyFirstTokenBiasLogitsProcessor(
                tokenizer=tokenizer,
                bias_words=bias_words,
                alpha=args.vocab_alpha,
            )
        )
    return LogitsProcessorList(processors) if processors else None


def main() -> None:
    device = resolve_device(args.device)
    torch_dtype = torch.float16 if device.type == "cuda" else torch.float32

    processor = WhisperProcessor.from_pretrained(
        args.model_path,
        language=args.language,
        task=args.task,
        no_timestamps=not args.timestamps,
        local_files_only=args.local_files_only,
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
        local_files_only=args.local_files_only,
    )

    if args.adapter_path:
        try:
            from peft import PeftModel
        except Exception as exc:
            raise RuntimeError(
                "PEFT adapter requested but `peft` package is not installed."
            ) from exc
        if args.local_files_only and not os.path.exists(args.adapter_path):
            raise FileNotFoundError(
                f"Adapter path {args.adapter_path} not found. Set local_files_only=False to download."
            )
        model = PeftModel.from_pretrained(
            model,
            args.adapter_path,
            local_files_only=args.local_files_only,
        )
        if hasattr(model, "merge_and_unload"):
            model = model.merge_and_unload()

    model.to(device)
    model.eval()

    prompt_language = args.prompt_language or args.language
    try:
        prompt_ids = (
            processor.get_decoder_prompt_ids(language=prompt_language, task=args.task)
            if prompt_language
            else None
        )
    except Exception:
        prompt_ids = None
    model.generation_config.forced_decoder_ids = prompt_ids
    model.generation_config.no_timestamps = not args.timestamps

    bias_words = []
    if args.vocab_bias_path:
        if not os.path.exists(args.vocab_bias_path):
            raise FileNotFoundError(f"Vocab bias file {args.vocab_bias_path} not found")
        with open(args.vocab_bias_path, "r", encoding="utf-8") as bias_file:
            bias_words = [line.strip() for line in bias_file if line.strip()]

    logits_processor = build_logits_processors(
        tokenizer=processor.tokenizer, bias_words=bias_words, args=args
    )

    audio_samples, sample_rate = load_audio(args.audio_path)
    target_sample_rate = processor.feature_extractor.sampling_rate
    if sample_rate != target_sample_rate:
        audio_samples = resample_audio(
            audio_samples, original_sr=sample_rate, target_sr=target_sample_rate
        )
        sample_rate = target_sample_rate
    if args.enhance_audio:
        audio_samples = enhance_audio(
            audio=audio_samples,
            target_rms=args.enhance_target_rms,
            max_gain=args.enhance_max_gain,
        )

    inputs = processor(
        audio=audio_samples,
        sampling_rate=sample_rate,
        return_tensors="pt",
    )
    input_features = inputs.input_features.to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    generate_kwargs = {
        "input_features": input_features,
        "max_new_tokens": args.max_new_tokens,
        "num_beams": args.num_beams,
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
        "length_penalty": args.length_penalty,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
    }
    if attention_mask is not None:
        generate_kwargs["attention_mask"] = attention_mask
    if logits_processor is not None:
        generate_kwargs["logits_processor"] = logits_processor

    with torch.inference_mode():
        autocast_enabled = device.type == "cuda"
        autocast_context = (
            torch.autocast(device_type="cuda", enabled=True)
            if autocast_enabled
            else nullcontext()
        )
        with autocast_context:
            generated = model.generate(**generate_kwargs)

    sequences = generated.sequences if hasattr(generated, "sequences") else generated
    transcript = processor.tokenizer.decode(sequences[0], skip_special_tokens=True)
    if args.remove_pun:
        transcript = remove_punctuation(transcript)

    print("\n" + "=" * 80)
    print("TRANSCRIPTION")
    print("=" * 80)
    print(transcript.strip())
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
