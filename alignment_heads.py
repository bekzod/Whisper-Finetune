#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute Whisper cross-attention heads that correlate with word-level timing
and emit them as base85-encoded (n_layers, n_heads) boolean arrays.

Example:
    # Using Hugging Face model ID:
    python alignment_heads.py \
        --model-id openai/whisper-small \
        --dataset-name mozilla-foundation/common_voice_11_0 \
        --dataset-config en \
        --dataset-split validation \
        --threshold 0.90 \
        --device cuda

    # Using local finetuned model path:
    python alignment_heads.py \
        --model-path ./models/whisper-small-finetuned \
        --dataset-name mozilla-foundation/common_voice_11_0 \
        --dataset-config en \
        --dataset-split validation \
        --threshold 0.90 \
        --device cuda

The script prints a Python dict like:
_ALIGNMENT_HEADS = {
    "small": b"ABzY8..."}
and also verifies roundtrip decoding.
"""

import argparse
import base64
import io
import json
import math
import os
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torchaudio
from datasets import load_dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# ---------------------------
# Data loading
# ---------------------------


@dataclass
class Word:
    text: str
    start: float
    end: float


@dataclass
class Sample:
    audio_path: str
    words: List[Word]
    transcript: str


def parse_inline_timestamps(text: str) -> Tuple[List[Word], str]:
    """
    Parse inline timestamp format like: word[start-end] word2[start-end]

    Example input: "Hello[0.00-0.50] world[0.55-1.00]"
    Example output: ([Word("Hello", 0.00, 0.50), Word("world", 0.55, 1.00)], "Hello world")

    Also supports format: "Allo,[0.00-0.80] eshitayapsanmi[0.85-1.60] Kechqurun[1.76-2.40]"

    Returns:
        words: List of Word objects with timing info
        transcript: Plain text without timestamps
    """
    import re

    # Pattern to match word[start-end]
    pattern = r"(\S+?)\[(\d+\.?\d*)-(\d+\.?\d*)\]"
    matches = re.findall(pattern, text)

    words = []
    transcript_parts = []

    for word_text, start, end in matches:
        # Remove punctuation from word_text for the Word object if needed
        # but keep it in transcript
        words.append(Word(word_text, float(start), float(end)))
        transcript_parts.append(word_text)

    transcript = " ".join(transcript_parts)
    return words, transcript


def load_samples_from_hf_dataset(
    dataset,
    audio_column: str = "audio",
    words_column: str = "words",
    transcript_column: Optional[str] = "transcript",
    max_samples: Optional[int] = None,
) -> List[Sample]:
    """
    Load samples from a Hugging Face dataset object.

    Args:
        dataset: A Hugging Face Dataset object
        audio_column: Name of the audio column
        words_column: Name of the words column with timing info
        transcript_column: Name of the transcript column (optional)
        max_samples: Maximum number of samples to process

    Expected dataset format:
    - audio_column: dict with 'path' or 'array' and 'sampling_rate'
    - words_column: list of dicts with 'text', 'start', 'end' keys
                    OR string with inline timestamps like "word[start-end]"
    - transcript_column: optional text string (if not provided, joins words)
    """
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    samples = []
    for item in dataset:
        # Handle audio - HF datasets provide audio as dict with 'path' or 'array'
        audio_data = item[audio_column]
        if isinstance(audio_data, dict):
            # Audio is already loaded with 'array' and 'sampling_rate'
            # or has 'path'
            audio_path = audio_data.get("path")
            if audio_path is None:
                # If no path, we need to handle the array directly
                # For now, skip samples without paths (can be extended)
                print(f"Warning: Skipping sample without audio path")
                continue
        else:
            audio_path = audio_data

        # Parse words - handle both list format and inline timestamp string format
        words_data = item[words_column]

        if isinstance(words_data, str):
            # Inline timestamp format: "word[start-end] word2[start-end]"
            words, transcript = parse_inline_timestamps(words_data)
        elif isinstance(words_data, list):
            # Standard list format: [{"text": "word", "start": 0.0, "end": 1.0}, ...]
            words = [
                Word(w["text"], float(w["start"]), float(w["end"])) for w in words_data
            ]
            # Get transcript
            if transcript_column and transcript_column in item:
                transcript = item[transcript_column]
            else:
                transcript = " ".join([w.text for w in words])
        else:
            print(f"Warning: Unsupported words_column format, skipping sample")
            continue

        samples.append(Sample(audio_path, words, transcript))

    return samples


def load_samples_from_dataset(
    dataset_name: str,
    dataset_config: Optional[str] = None,
    dataset_split: str = "train",
    audio_column: str = "audio",
    words_column: str = "words",
    transcript_column: Optional[str] = "transcript",
    max_samples: Optional[int] = None,
) -> List[Sample]:
    """
    Load samples from a Hugging Face dataset by name.

    Args:
        dataset_name: HF dataset identifier (e.g., 'mozilla-foundation/common_voice_11_0')
        dataset_config: Dataset configuration/subset
        dataset_split: Split to use (train/validation/test)
        audio_column: Name of the audio column
        words_column: Name of the words column with timing info
        transcript_column: Name of the transcript column (optional)
        max_samples: Maximum number of samples to process

    Expected dataset format:
    - audio_column: dict with 'path' or 'array' and 'sampling_rate'
    - words_column: list of dicts with 'text', 'start', 'end' keys
                    OR string with inline timestamps like "word[start-end]"
    - transcript_column: optional text string (if not provided, joins words)
    """
    dataset = load_dataset(
        dataset_name,
        dataset_config,
        split=dataset_split,
        trust_remote_code=True,
    )

    return load_samples_from_hf_dataset(
        dataset=dataset,
        audio_column=audio_column,
        words_column=words_column,
        transcript_column=transcript_column,
        max_samples=max_samples,
    )


# ---------------------------
# Token ↔ word-time mapping
# ---------------------------


def build_char_spans(
    words: List[Word], transcript: str
) -> List[Tuple[int, int, float]]:
    """
    Map each word to a (char_start, char_end, midpoint_time).
    Simple greedy match of words (joined by single spaces) inside transcript.
    Assumes transcript is exactly the join of given `words` (common in aligned sets).
    """
    spans = []
    cursor = 0
    for i, w in enumerate(words):
        # Find next occurrence starting from cursor (robust to multiple spaces)
        # We allow minimal whitespace flexibility by skipping whitespace in transcript.
        # But for aligned corpora, a simple exact match generally suffices.
        token = w.text
        # skip any leading spaces in transcript
        while cursor < len(transcript) and transcript[cursor].isspace():
            cursor += 1
        start = transcript.find(token, cursor)
        if start == -1:
            # Fallback: treat the next non-space block as the word
            # (keeps us going even if punctuation or casing differs)
            start = cursor
            end = min(len(transcript), start + len(token))
        else:
            end = start + len(token)
        midpoint = (w.start + w.end) / 2.0
        spans.append((start, end, midpoint))
        cursor = end
        # After word, try to consume a single space if present
        if cursor < len(transcript) and transcript[cursor] == " ":
            cursor += 1
    return spans


def token_mid_times_from_offsets(
    offsets: List[Tuple[int, int]], word_spans: List[Tuple[int, int, float]]
) -> List[Optional[float]]:
    """
    For each token char offset [s,e), assign the midpoint time of
    the word whose char span overlaps it the most. If no overlap, None.
    """
    mids = []
    for s, e in offsets:
        if s == e:  # special token in fast tokenizer often returns (0,0)
            mids.append(None)
            continue
        best_overlap = 0
        best_mid = None
        for ws, we, mid in word_spans:
            ov = max(0, min(e, we) - max(s, ws))
            if ov > best_overlap:
                best_overlap = ov
                best_mid = mid
        mids.append(best_mid)
    return mids


# ---------------------------
# Attention → predicted time
# ---------------------------


def attention_times(attn: torch.Tensor, enc_times: torch.Tensor) -> torch.Tensor:
    """
    Compute expected time per target token from attention distribution.
    attn: (num_heads, tgt_len, src_len) probabilities (sum over src_len = 1).
    enc_times: (src_len,) times (seconds) per encoder position.
    Returns: (num_heads, tgt_len) predicted times.
    """
    # (num_heads, tgt_len, src_len) @ (src_len,) -> (num_heads, tgt_len)
    return torch.einsum("hts,s->ht", attn, enc_times)


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3:
        return np.nan
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    xm = x - x.mean()
    ym = y - y.mean()
    denom = np.sqrt((xm * xm).sum() * (ym * ym).sum())
    if denom == 0.0:
        return np.nan
    return float((xm * ym).sum() / denom)


# ---------------------------
# Bit-pack ↔ base85 helpers
# ---------------------------


def bool_array_to_base85(mask: np.ndarray) -> bytes:
    """
    Pack a (n_layers, n_heads) boolean array into bytes (row-major, little-endian bit order),
    then Ascii85-encode to match the style of Whisper.
    """
    flat = mask.astype(np.uint8).reshape(-1)
    # Pack bits, little-endian so the first element is the least-significant bit of the first byte
    packed = np.packbits(flat, bitorder="little").tobytes()
    return base64.a85encode(packed)


def base85_to_bool_array(b: bytes, shape: Tuple[int, int]) -> np.ndarray:
    raw = base64.a85decode(b)
    bits = np.frombuffer(raw, dtype=np.uint8)
    unpacked = np.unpackbits(bits, bitorder="little")
    need = shape[0] * shape[1]
    if unpacked.size < need:
        # pad with zeros if needed (shouldn't happen if you stick to the same packer)
        pad = np.zeros(need - unpacked.size, dtype=np.uint8)
        unpacked = np.concatenate([unpacked, pad], axis=0)
    return unpacked[:need].astype(bool).reshape(shape)


# ---------------------------
# Main computation
# ---------------------------


def compute_alignment_heads(
    samples: List[Sample],
    device: str = "cuda",
    batch_size: int = 1,
    threshold: float = 0.90,
    language: Optional[str] = None,
    task: str = "transcribe",
    model_id: Optional[str] = None,
    model_path: Optional[str] = None,
) -> Tuple[np.ndarray, Dict[Tuple[int, int], float]]:
    """
    Returns:
        mask: (n_layers, n_heads) boolean array of alignment heads
        scores: dict[(layer, head)] -> Pearson r
    """
    if model_id is None and model_path is None:
        raise ValueError("Either model_id or model_path must be provided")
    if model_id is not None and model_path is not None:
        raise ValueError("Cannot specify both model_id and model_path")

    model_source = model_path if model_path is not None else model_id
    processor = WhisperProcessor.from_pretrained(
        model_source, language=language, task=task
    )
    model = WhisperForConditionalGeneration.from_pretrained(model_source)
    model.to(device)
    model.eval()

    n_layers = model.config.decoder_layers
    n_heads = model.model.decoder.layers[
        0
    ].self_attn.num_heads  # cross_attn uses same n_heads

    # Accumulate per (layer, head) all (pred_time, true_time) pairs
    per_head_true: Dict[Tuple[int, int], list] = {
        (L, H): [] for L in range(n_layers) for H in range(n_heads)
    }
    per_head_pred: Dict[Tuple[int, int], list] = {
        (L, H): [] for L in range(n_layers) for H in range(n_heads)
    }

    for sample in samples:
        # Load audio
        wav, sr = torchaudio.load(sample.audio_path)
        wav = wav.mean(dim=0, keepdim=False)  # mono
        wav = torchaudio.functional.resample(
            wav, orig_freq=sr, new_freq=processor.feature_extractor.sampling_rate
        )
        sr = processor.feature_extractor.sampling_rate
        duration = wav.shape[-1] / float(sr)

        # Inputs
        inputs = processor(wav.numpy(), sampling_rate=sr, return_tensors="pt")
        input_features = inputs.input_features.to(device)

        # Labels via tokenizer w/ offsets so we can map tokens → words
        # Use fast tokenizer (Transformers provides WhisperTokenizerFast)
        tok = processor.tokenizer
        enc = tok(
            sample.transcript,
            add_special_tokens=True,
            return_offsets_mapping=True,
            return_attention_mask=False,
        )
        input_ids = torch.tensor(
            enc["input_ids"], dtype=torch.long, device=device
        ).unsqueeze(0)  # (1, T)
        offsets = enc["offset_mapping"]  # list of (s,e) per token

        # Build per-token target mid-times (None for specials/non-overlapping)
        word_spans = build_char_spans(sample.words, sample.transcript)
        token_mids = token_mid_times_from_offsets(offsets, word_spans)
        # Model forward with teacher forcing to get cross-attentions
        with torch.no_grad():
            out = model(
                input_features=input_features,
                decoder_input_ids=input_ids[:, :-1],  # teacher-forcing
                labels=input_ids,  # enables loss but we don't use it
                output_attentions=True,
                use_cache=False,  # ensure full attentions are returned
            )

        # cross_attentions: tuple(len = n_layers) of tensors (batch, num_heads, tgt_len, src_len)
        cross_atts = out.cross_attentions
        assert len(cross_atts) == n_layers

        # Map encoder positions to real seconds (uniform over encoder length)
        # Encoder seq_len is cross_atts[0].shape[-1]
        enc_len = cross_atts[0].shape[-1]
        # Uniform mapping: center of each encoder position
        enc_times = torch.linspace(0.0, duration, steps=enc_len, device=device)

        # For each layer, compute per-head predicted times per target token
        # Skip BOS at position 0 if present in decoder_input_ids (we used [:, :-1] above)
        tgt_len = cross_atts[0].shape[2]
        # Build mask of which target token indices have a valid true time
        # Align positions: decoder_input_ids positions correspond to token_mids[:-1]
        true_times = []
        valid_t_idx = []
        for t_idx in range(tgt_len):
            mid = (
                token_mids[t_idx + 1] if (t_idx + 1) < len(token_mids) else None
            )  # +1 because labels include BOS
            if mid is not None and not math.isnan(mid):
                true_times.append(mid)
                valid_t_idx.append(t_idx)
        if not valid_t_idx:
            continue
        true_times_np = np.array(true_times, dtype=np.float32)

        for L in range(n_layers):
            # (1, H, T, S) -> (H, T, S)
            att = cross_atts[L][0].detach()  # (num_heads, tgt_len, src_len)
            # Normalize along src (should already be softmaxed, but safe)
            att = att / (att.sum(dim=-1, keepdim=True) + 1e-9)

            pred_ts = attention_times(att, enc_times)  # (H, T)
            # Gather only valid token indices
            pred_sel = pred_ts[:, valid_t_idx].cpu().numpy()  # (H, N_valid)

            for H in range(n_heads):
                per_head_pred[(L, H)].extend(pred_sel[H].tolist())
                per_head_true[(L, H)].extend(true_times_np.tolist())

    # Compute Pearson r per head
    scores: Dict[Tuple[int, int], float] = {}
    mask = np.zeros((n_layers, n_heads), dtype=bool)
    for (L, H), y_list in per_head_true.items():
        y = np.array(y_list, dtype=np.float32)
        x = np.array(per_head_pred[(L, H)], dtype=np.float32)
        if x.size == 0 or y.size != x.size:
            r = float("nan")
        else:
            r = pearson_r(x, y)
        scores[(L, H)] = r
        if np.isfinite(r) and r >= threshold and r > 0:
            mask[L, H] = True

    return mask, scores


# ---------------------------
# CLI
# ---------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model-id",
        default=None,
        help="Hugging Face model ID (e.g., openai/whisper-small or openai/whisper-large-v3)",
    )
    ap.add_argument(
        "--model-path",
        default=None,
        help="Local path to a finetuned Whisper model directory",
    )
    ap.add_argument(
        "--model-alias",
        default=None,
        help='Key to use in the output dict (e.g., "small", "large-v3"). Default: derived from --model-id.',
    )
    ap.add_argument(
        "--dataset-name",
        required=True,
        help="Hugging Face dataset name (e.g., 'mozilla-foundation/common_voice_11_0')",
    )
    ap.add_argument(
        "--dataset-config",
        default=None,
        help="Dataset configuration/subset (e.g., 'en' for language)",
    )
    ap.add_argument(
        "--dataset-split",
        default="train",
        help="Dataset split to use (default: 'train')",
    )
    ap.add_argument(
        "--audio-column",
        default="audio",
        help="Name of the audio column in the dataset (default: 'audio')",
    )
    ap.add_argument(
        "--words-column",
        default="words",
        help="Name of the words column with timing info (default: 'words')",
    )
    ap.add_argument(
        "--transcript-column",
        default="transcript",
        help="Name of the transcript column (default: 'transcript')",
    )
    ap.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (default: all)",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.90,
        help="Pearson r threshold to flag a head",
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--language", default=None, help="Force language for processor (e.g., 'en')"
    )
    ap.add_argument("--task", default="transcribe", choices=["transcribe", "translate"])
    args = ap.parse_args()

    # Validate that either model_id or model_path is provided
    if args.model_id is None and args.model_path is None:
        ap.error("Either --model-id or --model-path must be provided")
    if args.model_id is not None and args.model_path is not None:
        ap.error("Cannot specify both --model-id and --model-path")

    samples = load_samples_from_dataset(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        dataset_split=args.dataset_split,
        audio_column=args.audio_column,
        words_column=args.words_column,
        transcript_column=args.transcript_column,
        max_samples=args.max_samples,
    )
    mask, scores = compute_alignment_heads(
        samples=samples,
        device=args.device,
        threshold=args.threshold,
        language=args.language,
        task=args.task,
        model_id=args.model_id,
        model_path=args.model_path,
    )

    # Encode
    encoded = bool_array_to_base85(mask)
    alias = args.model_alias
    if alias is None:
        # Derive a compact alias (similar to Whisper keys)
        # Examples: "openai/whisper-small.en" -> "small.en", "openai/whisper-large-v3" -> "large-v3"
        model_source = args.model_path if args.model_path is not None else args.model_id
        name = model_source.split("/")[-1]
        alias = name.replace("whisper-", "")

    # Print in the requested format
    print(
        "# base85-encoded (n_layers, n_heads) boolean arrays indicating the cross-attention heads"
    )
    print(
        "# that are highly correlated to the word-level timing, i.e. the alignment between audio and text tokens."
    )
    print("_ALIGNMENT_HEADS = {")
    print(f'    "{alias}": {encoded!r},')
    print("}")

    # Optional: verify roundtrip decode
    roundtrip = base85_to_bool_array(encoded, mask.shape)
    ok = np.array_equal(mask, roundtrip)
    print(f"# Roundtrip decode OK: {ok}")
    print(f"# Shape: {mask.shape}, Heads selected: {int(mask.sum())}")
    # Also print top-10 heads by r for inspection
    sorted_heads = sorted(
        scores.items(), key=lambda kv: (-(kv[1] if np.isfinite(kv[1]) else -999))
    )
    print("# Top heads by Pearson r:")
    for (L, H), r in sorted_heads[:10]:
        print(f"# layer {L:02d}, head {H:02d}: r={r:.4f}")


if __name__ == "__main__":
    main()
