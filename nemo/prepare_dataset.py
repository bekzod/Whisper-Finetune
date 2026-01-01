#!/usr/bin/env python3
"""Prepare HuggingFace datasets into NeMo-style JSONL manifests with cleaned text."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import re
import shutil
import sys
import tarfile
import time
import unicodedata
from collections.abc import Iterable as IterableCollection
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import chain
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import librosa
import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset
from huggingface_hub import constants as hf_constants
from huggingface_hub import snapshot_download
from tqdm import tqdm

# Allow running the script from the ``nemo`` directory without installing the repo as a package.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.tsv_parser import iter_tsv_dict_rows


def _load_cleanup_utils() -> Any:
    utils_path = Path(__file__).with_name("utils.py")
    spec = importlib.util.spec_from_file_location("nemo_cleanup_utils", utils_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load cleanup utils from {utils_path}")
    module = importlib.util.module_from_spec(spec)
    # Register module in sys.modules before exec_module so that dataclass
    # decorator can find it when processing classes during module execution
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_cleanup_utils = _load_cleanup_utils()
normalize_text = _cleanup_utils.normalize_text
contains_standalone_c = _cleanup_utils.contains_standalone_c
get_misspelling_stats = _cleanup_utils.get_misspelling_stats
reset_misspelling_stats = _cleanup_utils.reset_misspelling_stats
# Frequency-based typo detection
WordFrequencyCollector = _cleanup_utils.WordFrequencyCollector
FrequencyBasedTypoDetector = _cleanup_utils.FrequencyBasedTypoDetector
get_frequency_collector = _cleanup_utils.get_frequency_collector
reset_frequency_collector = _cleanup_utils.reset_frequency_collector
get_typo_detector = _cleanup_utils.get_typo_detector
reset_typo_detector = _cleanup_utils.reset_typo_detector

MAX_TRANSCRIPT_CHAR_LIMIT = 680
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_MIN_DURATION = 0.5
DEFAULT_MAX_DURATION = 30.0
DEFAULT_MIN_CHARS = 1
DEFAULT_MAX_CHARS = 680
DEFAULT_SAMPLING_SEED = 3407
DEFAULT_CACHE_ROOT = Path("/workspace")
MAX_TAR_PATH_COMPONENT = 960
DEFAULT_HF_LOAD_RETRIES = 8
DEFAULT_HF_RETRY_WAIT = (
    240  # 4 minutes base backoff; exponential: 240s, 480s, 960s, ...
)
DEFAULT_HF_RATE_LIMIT_WAIT = (
    300  # Base wait for 429 rate limits (5 min per HF quota window)
)

_HF_AUDIO_CANDIDATE_KEYS = (
    "audio",
    "audio_path",
    "audio_filepath",
    "filepath",
    "file",
    "filename",
    "path",
    "audio_file",
    "wav",
)
_HF_TEXT_CANDIDATE_KEYS = (
    "sentence",
    "text",
    "transcription",
    "transcript",
    "label",
    "normalized_text",
    "norm_text",
    "normalized_text_with_punct",
    "raw_text",
    "target_text",
    "translation",
)

_FILTERED_CLIENT_IDS = {
    "56ac8e86-b8c9-4879-a342-0eeb94f686fc",
    "3d3fca02-6a07-41e2-9af4-60886ea60300",
    "231d3776-2dbe-4a42-a535-c67943427e3f",
    "e2716f95-70b5-4832-b903-eef2343591a4",
    "2a815774-e953-4031-931a-8a28052e5cf9",
    "d6fd3dc4-a55d-4a80-9bbf-b713325d05be",
    "10b29e87-bf01-4b16-bead-a044076f849b",
    "e3412d51-f079-4167-b3f9-311a976443ce",
}


def _filter_bekzod123_uzbek_voice(ex: Dict[str, Any]) -> bool:
    return ex.get("is_correct") is True


def _filter_davron_sherbaev_uzbekvoice(ex: Dict[str, Any]) -> bool:
    return (
        ex.get("reported_reasons") is None
        and ex.get("downvotes_count", 0) == 0
        and ex.get("reported_count", 0) == 0
        and ex.get("client_id") not in _FILTERED_CLIENT_IDS
    )


@dataclass(frozen=True)
class DatasetSpec:
    group: str
    repo: str
    subset: Optional[str]
    revision: Optional[str]
    splits: List[str]
    percentage: Optional[float]
    seed: Optional[int]
    data_dir: Optional[str]
    data_files: Optional[Any]
    trust_remote_code: Optional[bool]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert HuggingFace datasets into NeMo-style manifests with text cleanup."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("datasets.json"),
        help="Path to dataset config JSON (train/eval entries).",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        default=["train", "eval"],
        help="Config groups to process (default: train eval).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Output directory for audio and manifest files.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional cache root (used in default mode or as temp parent).",
    )
    parser.add_argument(
        "--cache-mode",
        choices=("default", "per-dataset"),
        default="per-dataset",
        help=(
            "Cache strategy: 'per-dataset' uses a temporary cache per split and "
            "deletes it after processing; 'default' uses the shared HF cache."
        ),
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Optional HuggingFace token for gated datasets.",
    )
    parser.add_argument(
        "--hf-read-timeout",
        type=int,
        default=None,
        help="Override HF_HUB_READ_TIMEOUT in seconds for Hub requests.",
    )
    parser.add_argument(
        "--hf-connect-timeout",
        type=int,
        default=None,
        help="Override HF_HUB_CONNECT_TIMEOUT in seconds for Hub requests.",
    )
    parser.add_argument(
        "--hf-etag-timeout",
        type=int,
        default=None,
        help="Override HF_HUB_ETAG_TIMEOUT in seconds for Hub metadata calls.",
    )
    parser.add_argument(
        "--hf-max-retries",
        type=int,
        default=None,
        help="Override HF_HUB_MAX_RETRIES for Hub HTTP retries.",
    )
    parser.add_argument(
        "--hf-load-retries",
        type=int,
        default=DEFAULT_HF_LOAD_RETRIES,
        help=(
            "Retry load_dataset on transient network errors "
            f"(default: {DEFAULT_HF_LOAD_RETRIES})."
        ),
    )
    parser.add_argument(
        "--hf-retry-wait",
        type=float,
        default=DEFAULT_HF_RETRY_WAIT,
        help=(
            "Base backoff in seconds between load_dataset retries "
            f"(default: {DEFAULT_HF_RETRY_WAIT:g})."
        ),
    )
    parser.add_argument(
        "--hf-rate-limit-wait",
        type=float,
        default=DEFAULT_HF_RATE_LIMIT_WAIT,
        help=(
            "Base backoff in seconds for 429 rate limit errors "
            f"(default: {DEFAULT_HF_RATE_LIMIT_WAIT:g}). HF quota resets every 5 minutes."
        ),
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help="Target sample rate for audio (default: 16000).",
    )
    parser.add_argument(
        "--no-resample",
        action="store_true",
        help="Keep original sample rate instead of resampling.",
    )
    parser.add_argument(
        "--mono",
        dest="mono",
        action="store_true",
        default=True,
        help="Convert audio to mono (default: enabled).",
    )
    parser.add_argument(
        "--no-mono",
        dest="mono",
        action="store_false",
        help="Keep audio channels (disable mono conversion).",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=DEFAULT_MIN_DURATION,
        help="Minimum audio duration in seconds.",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=DEFAULT_MAX_DURATION,
        help="Maximum audio duration in seconds (-1 for no limit).",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=DEFAULT_MIN_CHARS,
        help="Minimum cleaned transcript length in chars.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=DEFAULT_MAX_CHARS,
        help="Maximum cleaned transcript length in chars.",
    )
    parser.add_argument(
        "--default-splits",
        nargs="+",
        default=["train"],
        help="Fallback splits when none provided in config (default: train).",
    )
    parser.add_argument(
        "--absolute-paths",
        action="store_true",
        help="Write absolute audio paths into the manifest.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max rows per split for quick smoke tests.",
    )
    # Frequency-based typo detection arguments
    parser.add_argument(
        "--enable-typo-detection",
        action="store_true",
        help="Enable frequency-based typo detection (two-pass processing).",
    )
    parser.add_argument(
        "--typo-min-frequency-ratio",
        type=float,
        default=10.0,
        help="Minimum ratio of correction_freq / typo_freq to consider a typo (default: 10.0).",
    )
    parser.add_argument(
        "--typo-max-edit-distance",
        type=int,
        default=2,
        help="Maximum edit distance to consider for corrections (default: 2).",
    )
    parser.add_argument(
        "--typo-min-correction-frequency",
        type=int,
        default=100,
        help="Minimum frequency of the correction word (default: 100).",
    )
    parser.add_argument(
        "--typo-min-word-length",
        type=int,
        default=3,
        help="Minimum length of word to consider as potential typo (default: 3).",
    )
    parser.add_argument(
        "--typo-confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum confidence score to include a typo (default: 0.5).",
    )
    parser.add_argument(
        "--typo-report-only",
        action="store_true",
        help="Only report detected typos without applying corrections.",
    )
    return parser.parse_args()


def _configure_hf_http_settings(
    read_timeout: Optional[int],
    connect_timeout: Optional[int],
    etag_timeout: Optional[int],
    max_retries: Optional[int],
) -> None:
    try:
        from huggingface_hub import constants as hf_constants
    except Exception:
        hf_constants = None

    settings = {
        "HF_HUB_READ_TIMEOUT": read_timeout,
        "HF_HUB_CONNECT_TIMEOUT": connect_timeout,
        "HF_HUB_ETAG_TIMEOUT": etag_timeout,
        "HF_HUB_MAX_RETRIES": max_retries,
    }
    for key, value in settings.items():
        if value is None:
            continue
        os.environ[key] = str(value)
        if hf_constants is not None and hasattr(hf_constants, key):
            setattr(hf_constants, key, value)

    if hf_constants is None or not hasattr(hf_constants, "DEFAULT_REQUEST_TIMEOUT"):
        return
    current_timeout = hf_constants.DEFAULT_REQUEST_TIMEOUT
    if isinstance(current_timeout, tuple) and len(current_timeout) == 2:
        current_connect, current_read = current_timeout
        new_connect = (
            connect_timeout if connect_timeout is not None else current_connect
        )
        new_read = read_timeout if read_timeout is not None else current_read
        hf_constants.DEFAULT_REQUEST_TIMEOUT = (new_connect, new_read)
    elif read_timeout is not None:
        hf_constants.DEFAULT_REQUEST_TIMEOUT = read_timeout


def _is_retryable_hf_error(exc: BaseException) -> bool:
    if isinstance(exc, TimeoutError):
        return True
    message = str(exc).lower()
    retry_markers = (
        "read timed out",
        "read timeout",
        "read operation timed out",
        "timed out",
        "timeout",
        "connection reset",
        "connection aborted",
        "connection broken",
        "temporarily unavailable",
        "service unavailable",
        "bad gateway",
        "gateway timeout",
        "too many requests",
        "rate limit",
        "429",
        "502",
        "503",
        "504",
    )
    return any(marker in message for marker in retry_markers)


def _is_rate_limit_error(exc: BaseException) -> bool:
    """Check if the exception is a rate limit (429) error."""
    message = str(exc).lower()
    return "429" in message or "too many requests" in message or "rate limit" in message


def _load_dataset_with_retries(
    repo: str,
    load_kwargs: Dict[str, Any],
    max_retries: int,
    retry_wait: float,
    rate_limit_wait: float = DEFAULT_HF_RATE_LIMIT_WAIT,
) -> Any:
    retries = max(0, max_retries)
    delay = max(0.0, retry_wait)
    attempt = 0
    # Rate limit errors require longer waits (HF quota is per 5-minute window)
    rate_limit_base_wait = max(0.0, rate_limit_wait)
    while True:
        try:
            return load_dataset(repo, **load_kwargs)
        except Exception as exc:
            if attempt >= retries or not _is_retryable_hf_error(exc):
                raise
            # Use longer backoff for rate limit errors
            if _is_rate_limit_error(exc):
                # For rate limits, wait exactly 5 minutes (no exponential backoff)
                sleep_for = rate_limit_base_wait
                print(
                    f"  Rate limited while loading {repo}; waiting {sleep_for:.0f}s "
                    f"before retry ({attempt + 1}/{retries}): {exc}"
                )
            else:
                sleep_for = delay * (2**attempt)
                print(
                    f"  Network error while loading {repo}; retrying in {sleep_for:.1f}s "
                    f"({attempt + 1}/{retries}): {exc}"
                )
            time.sleep(sleep_for)
            attempt += 1


def load_dataset_config(path: Path) -> Dict[str, Sequence[Dict[str, Any]]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Dataset config must be a dict with train/eval keys.")
    return raw


def parse_repo_spec(name: str) -> Tuple[str, Optional[str], Optional[str]]:
    normalized = name.strip()
    if normalized.startswith("hf://"):
        normalized = normalized[5:]
    subset = None
    revision = None
    before_hash = normalized
    if "#" in normalized:
        before_hash, subset = normalized.split("#", 1)
    if "@" in before_hash:
        repo, revision = before_hash.split("@", 1)
    else:
        repo = before_hash
    return repo, revision, subset


def coerce_splits(raw_value: Any, default_splits: Sequence[str]) -> List[str]:
    if raw_value is None:
        return list(default_splits)
    if isinstance(raw_value, str):
        return [raw_value]
    if isinstance(raw_value, IterableCollection):
        return [str(split) for split in raw_value]
    raise TypeError(f"Unsupported split definition: {raw_value!r}")


def iter_dataset_specs(
    config: Dict[str, Sequence[Dict[str, Any]]],
    groups: Sequence[str],
    default_splits: Sequence[str],
) -> Iterable[DatasetSpec]:
    for group in groups:
        for entry in config.get(group, []):
            if not isinstance(entry, dict):
                continue
            name = entry.get("name") or entry.get("hf") or entry.get("huggingface")
            if not name:
                continue
            repo, revision_inline, subset_inline = parse_repo_spec(str(name))
            subset = entry.get("subset") or subset_inline
            revision = entry.get("revision") or revision_inline
            splits = coerce_splits(
                entry.get("splits") or entry.get("split"), default_splits
            )
            percentage = entry.get("percentage")
            if percentage is not None:
                try:
                    percentage = float(percentage)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid percentage value '{percentage}' for entry {entry}"
                    ) from exc
                if not (0 < percentage <= 100):
                    raise ValueError(
                        f"Percentage must be within (0, 100], got {percentage} for entry {entry}"
                    )
            seed = entry.get("seed")
            if seed is not None:
                try:
                    seed = int(seed)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Seed must be an integer, got {seed!r} for entry {entry}"
                    ) from exc
            yield DatasetSpec(
                group=group,
                repo=repo,
                subset=subset,
                revision=revision,
                splits=splits,
                percentage=percentage,
                seed=seed,
                data_dir=entry.get("data_dir"),
                data_files=entry.get("data_files"),
                trust_remote_code=entry.get("trust_remote_code"),
            )


def _flatten_transcript_text(transcript: Any) -> str:
    if transcript is None:
        return ""
    if isinstance(transcript, str):
        return normalize_text(transcript)
    if isinstance(transcript, dict):
        return normalize_text(str(transcript.get("text", "")))
    if isinstance(transcript, Sequence) and not isinstance(
        transcript, (str, bytes, bytearray)
    ):
        parts: List[str] = []
        for segment in transcript:
            flattened = _flatten_transcript_text(segment)
            if flattened:
                parts.append(flattened)
        return " ".join(parts).strip()
    return normalize_text(str(transcript))


def sanitize_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-")
    return cleaned or "dataset"


def pick_text(item: Dict[str, Any]) -> str:
    if "sentences" in item and item["sentences"] is not None:
        text = _flatten_transcript_text(item["sentences"])
        return unicodedata.normalize("NFKC", text)
    for key in _HF_TEXT_CANDIDATE_KEYS:
        if key in item and item[key] is not None:
            text = _flatten_transcript_text(item[key])
            return unicodedata.normalize("NFKC", text)
    return ""


# Common audio file extensions for path validation
_AUDIO_EXTENSIONS = frozenset(
    {
        ".wav",
        ".mp3",
        ".flac",
        ".ogg",
        ".m4a",
        ".aac",
        ".wma",
        ".opus",
        ".webm",
        ".mp4",
        ".aiff",
        ".aif",
        ".au",
        ".raw",
        ".pcm",
        ".sph",
    }
)


def _looks_like_audio_path(s: str) -> bool:
    """Check if a string looks like an audio file path rather than plain text.

    Returns True if the string:
    - Has a recognized audio file extension, OR
    - Contains path separators (/ or \) suggesting it's a file path
    - AND does not contain multiple spaces (suggesting it's text/sentence)
    """
    if not s or len(s) > 1000:  # Too long to be a reasonable path
        return False
    # If it contains multiple consecutive spaces, it's likely text
    if "  " in s:
        return False
    # Check for audio file extension
    s_lower = s.lower()
    for ext in _AUDIO_EXTENSIONS:
        if s_lower.endswith(ext):
            return True
    # Check if it looks like a file path (contains path separators)
    if "/" in s or "\\" in s:  # noqa: W605
        # But filter out things that look like sentences with slashes
        # (e.g., "this/that" patterns in text)
        # Real paths typically have extensions or start with common patterns
        if any(s_lower.endswith(ext) for ext in _AUDIO_EXTENSIONS):
            return True
        # Paths often start with / or ./ or drive letters
        if s.startswith("/") or s.startswith("./") or s.startswith("../"):
            return True
        if len(s) > 2 and s[1] == ":" and s[2] == "\\":  # Windows drive path
            return True
    return False


def _read_audio_from_bytes(
    audio_bytes: bytes, mono: bool = True
) -> Tuple[Optional[np.ndarray], Optional[int]]:
    """Read audio data from raw bytes."""
    try:
        import io

        samples, sample_rate = sf.read(
            io.BytesIO(audio_bytes), dtype="float32", always_2d=True
        )
        if mono:
            samples = samples.mean(axis=1).astype(np.float32)
        else:
            samples = samples.astype(np.float32)
        return samples, int(sample_rate)
    except Exception as exc:
        print(f"  [DEBUG] Failed to read audio from bytes: {exc}")
        return None, None


def _decode_audio_decoder(
    decoder: Any, mono: bool = True
) -> Tuple[Optional[np.ndarray], Optional[int]]:
    """
    Decode an AudioDecoder object from newer HuggingFace datasets versions.

    AudioDecoder is a lazy decoder that needs to be called to get audio data.
    Supports torchcodec AudioDecoder which uses get_all_samples() method.
    """
    try:
        # Method 1: Try get_all_samples() for torchcodec AudioDecoder
        if hasattr(decoder, "get_all_samples"):
            samples = decoder.get_all_samples()
            if samples is not None:
                # torchcodec returns a tensor, convert to numpy
                if hasattr(samples, "numpy"):
                    arr = samples.numpy().astype(np.float32)
                elif hasattr(samples, "data"):
                    arr = np.asarray(samples.data, dtype=np.float32)
                else:
                    arr = np.asarray(samples, dtype=np.float32)

                # Get sample rate from metadata
                sr = None
                if hasattr(decoder, "metadata"):
                    meta = decoder.metadata
                    if hasattr(meta, "sample_rate"):
                        sr = int(meta.sample_rate)
                    elif isinstance(meta, dict):
                        sr = meta.get("sample_rate") or meta.get("sampling_rate")
                        if sr:
                            sr = int(sr)

                # Handle mono conversion - torchcodec returns (channels, samples)
                if mono and arr.ndim == 2:
                    arr = arr.mean(axis=0).astype(np.float32)
                elif arr.ndim == 2 and arr.shape[0] == 1:
                    arr = arr[0]  # Remove channel dimension if single channel

                return arr, sr

        # Method 2: Try calling the decoder (older HF datasets API)
        if callable(decoder):
            decoded = decoder()
            if isinstance(decoded, dict):
                arr = decoded.get("array")
                sr = decoded.get("sampling_rate")
                if arr is not None:
                    arr = np.asarray(arr, dtype=np.float32)
                    if mono and arr.ndim == 2:
                        arr = arr.mean(axis=0).astype(np.float32)
                    elif mono and arr.ndim == 1:
                        pass  # already mono
                    return arr, int(sr) if sr else None

        # Method 3: Try accessing as an object with array attribute
        if hasattr(decoder, "array") and hasattr(decoder, "sampling_rate"):
            arr = np.asarray(decoder.array, dtype=np.float32)
            sr = int(decoder.sampling_rate) if decoder.sampling_rate else None
            if mono and arr.ndim == 2:
                arr = arr.mean(axis=0).astype(np.float32)
            return arr, sr

        # Method 4: Try the __call__ method explicitly
        if hasattr(decoder, "__call__"):
            result = decoder.__call__()
            if isinstance(result, dict):
                arr = result.get("array")
                sr = result.get("sampling_rate")
                if arr is not None:
                    arr = np.asarray(arr, dtype=np.float32)
                    if mono and arr.ndim == 2:
                        arr = arr.mean(axis=0).astype(np.float32)
                    return arr, int(sr) if sr else None
            elif isinstance(result, tuple) and len(result) >= 2:
                arr, sr = result[0], result[1]
                arr = np.asarray(arr, dtype=np.float32)
                if mono and arr.ndim == 2:
                    arr = arr.mean(axis=0).astype(np.float32)
                return arr, int(sr) if sr else None

    except Exception as e:
        print(f"  [DEBUG] Failed to decode AudioDecoder: {e}")
        import traceback

        traceback.print_exc()

    return None, None


def resolve_audio_blob(
    blob: Any,
    mono: bool = True,
) -> Tuple[Optional[np.ndarray], Optional[int], Optional[str]]:
    """
    Resolve various audio representations to (array, sample_rate, path_reference).

    Handles:
    - AudioDecoder objects (lazy decoding from newer HuggingFace datasets)
    - dict with 'array' key (decoded HuggingFace Audio)
    - dict with 'bytes' key (raw audio bytes)
    - dict with 'path' key (file path reference)
    - raw numpy array or list
    - file path string
    - bytes/bytearray
    """
    if blob is None:
        return None, None, None

    # Handle AudioDecoder objects from newer HuggingFace datasets
    blob_type_name = type(blob).__name__
    if "AudioDecoder" in blob_type_name or "Decoder" in blob_type_name:
        arr, sr = _decode_audio_decoder(blob, mono=mono)
        if arr is not None:
            return arr, sr, None
        # If decoding failed, continue to try other methods

    if isinstance(blob, dict):
        # Try to get decoded array first
        array_candidate = blob.get("array")
        # Also check alternative keys used by some datasets
        if array_candidate is None:
            for alt_key in ("waveform", "samples", "values"):
                if alt_key in blob and blob[alt_key] is not None:
                    array_candidate = blob[alt_key]
                    break

        if array_candidate is not None:
            try:
                arr = np.asarray(array_candidate, dtype=np.float32)
            except Exception:
                arr = None
            if arr is not None and arr.size > 0:
                return (
                    arr,
                    int(blob.get("sampling_rate") or 0) or None,
                    blob.get("path"),
                )

        # Check for raw bytes (some datasets provide audio as bytes)
        bytes_value = blob.get("bytes")
        if isinstance(bytes_value, (bytes, bytearray, memoryview)):
            arr, sr = _read_audio_from_bytes(bytes(bytes_value), mono=mono)
            if arr is not None:
                return arr, sr, None

        # Fall back to path reference
        for key in ("path", "audio_path", "audio_filepath", "filename", "file"):
            ref = blob.get(key)
            if isinstance(ref, str) and ref and _looks_like_audio_path(ref):
                return None, None, ref
        return None, None, None

    if isinstance(blob, (list, tuple, np.ndarray)):
        try:
            arr = np.asarray(blob, dtype=np.float32)
        except Exception:
            return None, None, None
        if arr.size == 0:
            return None, None, None
        return arr, None, None

    # Handle raw bytes
    if isinstance(blob, (bytes, bytearray, memoryview)):
        arr, sr = _read_audio_from_bytes(bytes(blob), mono=mono)
        if arr is not None:
            return arr, sr, None
        return None, None, None

    if isinstance(blob, str):
        # Only treat as audio path if it looks like a file path, not arbitrary text
        if _looks_like_audio_path(blob):
            return None, None, blob
        return None, None, None

    if hasattr(os, "PathLike") and isinstance(blob, os.PathLike):
        return None, None, os.fspath(blob)
    return None, None, None


def read_audio_from_item(
    item: Dict[str, Any],
    debug: bool = False,
) -> Tuple[Optional[np.ndarray], Optional[int], Optional[str]]:
    # First try known audio column names
    for key in _HF_AUDIO_CANDIDATE_KEYS:
        if key not in item:
            continue
        blob = item.get(key)
        if debug:
            blob_type = type(blob).__name__
            if blob is None:
                blob_preview = "None"
            elif isinstance(blob, dict):
                # Show dict keys and types for audio dicts
                dict_info = {k: type(v).__name__ for k, v in blob.items()}
                has_array = "array" in blob and blob["array"] is not None
                array_shape = None
                if has_array:
                    try:
                        import numpy as np

                        arr = np.asarray(blob["array"])
                        array_shape = arr.shape
                    except:
                        pass
                blob_preview = f"dict keys={dict_info}, has_array={has_array}, array_shape={array_shape}, sr={blob.get('sampling_rate')}"
            elif (
                "AudioDecoder" in type(blob).__name__
                or "Decoder" in type(blob).__name__
            ):
                blob_preview = f"AudioDecoder object (lazy decoding)"
            else:
                blob_preview = str(blob)[:100]
            print(
                f"    [DEBUG] read_audio_from_item: key='{key}', type={blob_type}, {blob_preview}"
            )
        arr, sr, ref = resolve_audio_blob(blob)
        if arr is not None or ref:
            if debug:
                print(
                    f"    [DEBUG] read_audio_from_item: found audio via key='{key}', arr={arr is not None}, sr={sr}, ref={ref}"
                )
            return arr, sr, ref
    # Fallback: try all values
    for key, value in item.items():
        if key in _HF_AUDIO_CANDIDATE_KEYS:
            continue  # Already checked
        arr, sr, ref = resolve_audio_blob(value)
        if arr is not None or ref:
            if debug:
                print(
                    f"    [DEBUG] read_audio_from_item: found audio via fallback key='{key}'"
                )
            return arr, sr, ref
    if debug:
        print(f"    [DEBUG] read_audio_from_item: no audio found in item")
    return None, None, None


def download_common_voice_subset(
    repo_id: str, subset: str, cache_root: Optional[Path]
) -> Path:
    """Download Common Voice dataset files for a specific language subset.

    Args:
        repo_id: HuggingFace repository ID (e.g., 'fsicoli/common_voice_17_0').
        subset: Language code (e.g., 'uz' for Uzbek).
        cache_root: Root directory for caching downloaded files.

    Returns:
        Path to the local directory containing downloaded files.
    """
    if cache_root is None:
        cache_root = DEFAULT_CACHE_ROOT
    # Use sanitized repo name for cache directory
    repo_name = sanitize_name(repo_id)
    local_dir = cache_root / repo_name / subset
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {repo_id} files to {local_dir}")
    print(f"  Patterns: audio/{subset}/*, transcript/{subset}/*")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=[
            f"audio/{subset}/**",
            f"transcript/{subset}/*",
        ],
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
    )
    return local_dir


def download_fleurs_subset(
    subset: str, cache_root: Optional[Path], revision: Optional[str]
) -> Path:
    if cache_root is None:
        cache_root = DEFAULT_CACHE_ROOT
    local_dir = cache_root / "google_fleurs" / subset
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="google/fleurs",
        repo_type="dataset",
        allow_patterns=[
            f"data/{subset}/audio/*.tar.gz",
            f"data/{subset}/*.tsv",
        ],
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        revision=revision,
    )
    return local_dir


def _hash_path_component(component: str, *, force: bool = False) -> str:
    if not force and len(component) <= MAX_TAR_PATH_COMPONENT:
        return component
    base, ext = os.path.splitext(component)
    digest = hashlib.md5(component.encode("utf-8")).hexdigest()
    if ext and len(ext) <= 10:
        return f"{digest}{ext}"
    return digest


def _shorten_rel_path(rel_path: str, *, force_filename_hash: bool = False) -> str:
    parts = PurePosixPath(rel_path).parts
    if not parts:
        return rel_path
    if force_filename_hash:
        *dirs, last = parts
        hashed_dirs = [_hash_path_component(part) for part in dirs]
        hashed_last = _hash_path_component(last, force=True)
        return "/".join([*hashed_dirs, hashed_last])
    return "/".join(_hash_path_component(part) for part in parts)


def _safe_extract_tarball(
    root_path: Path, tar_name: str, tar_root_to_subdirs: Dict[str, set]
) -> None:
    root_abs = root_path.resolve()
    if tar_name.endswith(".tar.gz"):
        stem = tar_name[: -len(".tar.gz")]
    elif tar_name.endswith(".tar"):
        stem = tar_name[: -len(".tar")]
    else:
        stem = os.path.splitext(tar_name)[0]
    tar_root_to_subdirs.setdefault(str(root_abs), set()).add(stem)

    marker_path = root_abs / f".{stem}_extracted"
    if marker_path.exists():
        return
    tar_path = root_abs / tar_name
    mode = "r:gz" if tar_name.endswith(".tar.gz") else "r:"

    print(f"  Extracting {tar_name}...")

    def _extract_with_target(member: tarfile.TarInfo, rel_name: str) -> None:
        dest_path = root_abs / rel_name
        if member.isdir():
            dest_path.mkdir(parents=True, exist_ok=True)
            return
        if not member.isfile():
            print(
                f"  Warning: skipping unsupported tar entry in {tar_name}: {member.name}"
            )
            return
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        file_obj = tar.extractfile(member)
        if file_obj is None:
            return
        with file_obj:
            with open(dest_path, "wb") as out_f:
                shutil.copyfileobj(file_obj, out_f)

    with tarfile.open(tar_path, mode) as tar:
        members = tar.getmembers()
        for member in tqdm(members, desc=f"    {tar_name}", unit="files"):
            member_path = PurePosixPath(member.name)
            parts = member_path.parts
            if member_path.is_absolute() or ".." in parts:
                raise RuntimeError(
                    f"Blocked path traversal while extracting {tar_name}"
                )
            shortened_name = _shorten_rel_path(member.name)
            if shortened_name != member.name:
                print(
                    f"  Warning: shortening tar entry in {tar_name}: {member.name} -> {shortened_name}"
                )
                _extract_with_target(member, shortened_name)
                continue
            try:
                target_path = (root_abs / member.name).resolve()
            except OSError as exc:
                print(f"  Warning: path error in {tar_name} for {member.name}: {exc}")
                forced_name = _shorten_rel_path(member.name, force_filename_hash=True)
                if forced_name != member.name:
                    _extract_with_target(member, forced_name)
                continue
            if os.path.commonpath([str(root_abs), str(target_path)]) != str(root_abs):
                raise RuntimeError(
                    f"Blocked path traversal while extracting {tar_name}"
                )
            try:
                tar.extract(member, root_abs)
            except OSError as exc:
                print(
                    f"  Warning: failed to extract {member.name} from {tar_name}: {exc}"
                )
                forced_name = _shorten_rel_path(member.name, force_filename_hash=True)
                if forced_name != member.name:
                    _extract_with_target(member, forced_name)
                continue

    marker_path.write_text("extracted", encoding="utf-8")


def iter_common_voice_items(
    *,
    local_dir: Path,
    subset: str,
    split: Optional[str],
    percentage: Optional[float],
    seed: int,
    limit: Optional[int],
) -> Iterable[Dict[str, Any]]:
    audio_dir = local_dir / "audio" / subset
    transcript_dir = local_dir / "transcript" / subset
    audio_dir_abs = audio_dir.resolve()

    tar_root_to_subdirs: Dict[str, set] = {}
    # Search for tarballs in audio_dir and its parent (Common Voice structure can vary)
    search_dirs = [audio_dir_abs, audio_dir_abs.parent, local_dir / "audio"]
    for search_dir in search_dirs:
        if search_dir.is_dir():
            for current_dir, _, files in os.walk(search_dir):
                current_path = Path(current_dir)
                for tar_file in files:
                    if tar_file.endswith((".tar", ".tar.gz")):
                        _safe_extract_tarball(
                            current_path, tar_file, tar_root_to_subdirs
                        )

    subset_aliases = {
        "validation": ["validated"],
        "validated": ["validation"],
        "dev": ["development"],
        "development": ["dev"],
    }

    subset_variants: List[str] = []
    if split:
        base = str(split).strip().lower()
        if base:
            subset_variants.append(base)
    else:
        subset_variants.extend(
            ["train", "validation", "test", "dev", "validated", "invalidated", "other"]
        )

    for variant in list(subset_variants):
        for alias in subset_aliases.get(variant, []):
            if alias not in subset_variants:
                subset_variants.append(alias)

    candidate_dirs: List[str] = []
    seen_dirs: set = set()

    def _add_candidate(dir_path: Path) -> None:
        abs_path = dir_path.resolve()
        if abs_path.is_dir():
            abs_str = str(abs_path)
            if abs_str not in seen_dirs:
                seen_dirs.add(abs_str)
                candidate_dirs.append(abs_str)

    _add_candidate(audio_dir_abs)
    for variant in subset_variants:
        _add_candidate(audio_dir_abs / variant)

    # Also check parent directories - Common Voice structure can vary
    # e.g., audio might be directly under audio/{subset} without split subdirs
    _add_candidate(audio_dir_abs.parent)
    _add_candidate(local_dir / "audio")
    _add_candidate(local_dir)

    # Check for common audio subdirectory patterns
    for variant in subset_variants:
        _add_candidate(audio_dir_abs.parent / variant)
        _add_candidate(local_dir / "audio" / variant)

    for base_dir in list(candidate_dirs):
        for suffix in tar_root_to_subdirs.get(base_dir, set()):
            _add_candidate(Path(base_dir) / suffix)
        _add_candidate(Path(base_dir) / "clips")
        # Also check for nested subset directories inside extracted tarballs
        _add_candidate(Path(base_dir) / subset)

    # Recursively scan audio_dir parent for any subdirectories containing audio files
    audio_parent = audio_dir_abs.parent
    if audio_parent.is_dir():
        for subdir in audio_parent.iterdir():
            if subdir.is_dir():
                _add_candidate(subdir)
                for variant in subset_variants:
                    _add_candidate(subdir / variant)

    if not candidate_dirs:
        candidate_dirs.append(str(audio_dir_abs))

    # Print diagnostic info about search directories
    print(f"  Audio search directories ({len(candidate_dirs)} candidates):")
    for i, d in enumerate(candidate_dirs[:5]):  # Show first 5
        print(f"    [{i + 1}] {d}")
    if len(candidate_dirs) > 5:
        print(f"    ... and {len(candidate_dirs) - 5} more")

    path_cache: Dict[str, Optional[str]] = {}
    missing_logged: set = set()

    def _resolve_audio_path(filename: str) -> Optional[str]:
        if not filename:
            return None
        normalized = filename.strip().replace("\\", os.sep)
        has_ext = bool(os.path.splitext(normalized)[1])
        exts = [".mp3", ".wav", ".flac", ".ogg", ".m4a"]

        cached = path_cache.get(normalized)
        if cached is not None:
            return cached

        candidates = []
        names = [normalized]
        base_name = os.path.basename(normalized)
        if base_name != normalized:
            names.append(base_name)
        if has_ext:
            candidates.extend(names)
        else:
            for name in names:
                for ext in exts:
                    candidates.append(f"{name}{ext}")

        resolved = None
        for name in candidates:
            if os.path.isabs(name):
                if os.path.exists(name):
                    resolved = name
                    break
                continue
            for base_dir in candidate_dirs:
                direct_path = os.path.join(base_dir, name)
                if os.path.exists(direct_path):
                    resolved = direct_path
                    break
                for suffix in tar_root_to_subdirs.get(base_dir, set()):
                    nested_path = os.path.join(base_dir, suffix, os.path.basename(name))
                    if os.path.exists(nested_path):
                        resolved = nested_path
                        break
                if resolved:
                    break
            if resolved:
                break

        if resolved is None and os.sep in normalized:
            for base_dir in candidate_dirs:
                fallback_path = os.path.join(base_dir, base_name)
                if os.path.exists(fallback_path):
                    resolved = fallback_path
                    break

        # Last resort: recursive search in audio directory tree (caches results)
        if resolved is None:
            search_roots = [audio_dir_abs, audio_dir_abs.parent, local_dir / "audio"]
            for search_root in search_roots:
                if not search_root.is_dir():
                    continue
                for root, dirs, files in os.walk(search_root):
                    if base_name in files:
                        resolved = os.path.join(root, base_name)
                        # Cache all files found in this directory for faster future lookups
                        for f in files:
                            if f not in path_cache:
                                path_cache[f] = os.path.join(root, f)
                        break
                if resolved:
                    break

        if resolved:
            resolved = os.path.abspath(resolved)
        path_cache[normalized] = resolved
        return resolved

    keep_prob = 1.0
    if percentage is not None and percentage < 100:
        keep_prob = max(0.0, min(1.0, float(percentage) / 100.0))
    rng = random.Random(seed)
    kept = 0

    seen_tsv: set = set()
    for variant in subset_variants or ["train", "validation", "test"]:
        tsv_file = transcript_dir / f"{variant}.tsv"
        if not tsv_file.exists() or str(tsv_file) in seen_tsv:
            continue
        seen_tsv.add(str(tsv_file))
        print(f"  Processing TSV file: {tsv_file}")
        with tsv_file.open("r", encoding="utf-8", newline="") as f:
            header_keys, header_lookup, dict_rows = iter_tsv_dict_rows(f)
            if not header_keys:
                continue

            def _select_column(
                candidates: Sequence[str], default_idx: int
            ) -> Optional[str]:
                for candidate in candidates:
                    cand_key = candidate.strip().lower()
                    if cand_key in header_lookup:
                        return cand_key
                if 0 <= default_idx < len(header_keys):
                    return header_keys[default_idx]
                if header_keys:
                    return header_keys[0]
                return None

            path_column = _select_column(
                ["path", "audio", "audio_filename", "filename", "file"],
                default_idx=0,
            )
            text_column = _select_column(
                [
                    "sentence",
                    "text",
                    "transcription",
                    "transcript",
                    "text_latin",
                    "normalized_text",
                    "raw_text",
                ],
                default_idx=2 if len(header_keys) > 2 else len(header_keys) - 1,
            )

            if path_column is None or text_column is None:
                print(
                    f"  Warning: could not detect required columns in {tsv_file}. "
                    f"Header: {list(header_lookup.values())}"
                )
                continue

            for row_number, row_dict in dict_rows:
                audio_filename = (row_dict.get(path_column) or "").strip()
                if not audio_filename:
                    continue

                transcription_raw = row_dict.get(text_column, "")
                transcription = (
                    str(transcription_raw).strip()
                    if transcription_raw is not None
                    else ""
                )
                if not transcription:
                    continue

                if keep_prob < 1.0 and rng.random() > keep_prob:
                    continue

                resolved_path = _resolve_audio_path(audio_filename)
                if resolved_path:
                    yield {"audio": {"path": resolved_path}, "text": transcription}
                    kept += 1
                    if limit and kept >= limit:
                        print(
                            f"  Summary: {kept} audio files found, {len(missing_logged)} unique files missing"
                        )
                        return
                elif audio_filename not in missing_logged:
                    print(
                        f"  Warning (line {row_number}): audio file '{audio_filename}' not found under {audio_dir_abs}"
                    )
                    missing_logged.add(audio_filename)

    # Print summary after processing all TSV files
    print(
        f"  Summary: {kept} audio files found, {len(missing_logged)} unique files missing"
    )
    if missing_logged:
        print(
            f"  Tip: Check that all tar archives were extracted and audio files exist in the cache directory"
        )


def iter_fleurs_items(
    *,
    local_dir: Path,
    subset: str,
    split: Optional[str],
    percentage: Optional[float],
    seed: int,
    limit: Optional[int],
) -> Iterable[Dict[str, Any]]:
    data_dir = local_dir / "data" / subset
    audio_dir = data_dir / "audio"
    audio_dir_abs = audio_dir.resolve()

    tar_root_to_subdirs: Dict[str, set] = {}
    if audio_dir_abs.is_dir():
        for current_dir, _, files in os.walk(audio_dir_abs):
            current_path = Path(current_dir)
            for tar_file in files:
                if tar_file.endswith((".tar", ".tar.gz")):
                    _safe_extract_tarball(current_path, tar_file, tar_root_to_subdirs)

    audio_exts = (".wav", ".mp3", ".flac", ".m4a", ".ogg")
    path_index: Dict[str, str] = {}
    noext_index: Dict[str, str] = {}
    if audio_dir_abs.is_dir():
        for root, _, files in os.walk(audio_dir_abs):
            for fn in files:
                if fn.lower().endswith(audio_exts):
                    full_path = os.path.join(root, fn)
                    path_index[fn] = full_path
                    base_no_ext, _ = os.path.splitext(fn)
                    noext_index.setdefault(base_no_ext, full_path)

    def resolve_audio(filename: str) -> Optional[str]:
        if not filename:
            return None
        normalized = str(filename).strip().replace("\\", os.sep)

        direct = audio_dir_abs / normalized
        try:
            if direct.exists():
                return str(direct)
        except OSError:
            pass

        shortened = _shorten_rel_path(normalized)
        if shortened != normalized:
            direct_short = audio_dir_abs / shortened
            try:
                if direct_short.exists():
                    return str(direct_short)
            except OSError:
                pass

        forced = _shorten_rel_path(normalized, force_filename_hash=True)
        if forced != normalized:
            direct_forced = audio_dir_abs / forced
            try:
                if direct_forced.exists():
                    return str(direct_forced)
            except OSError:
                pass

        base = os.path.basename(normalized)
        if base in path_index:
            return path_index[base]

        base_no_ext, ext = os.path.splitext(base)
        if not ext:
            for suf in audio_exts:
                cand = base_no_ext + suf
                if cand in path_index:
                    return path_index[cand]
                shortened_cand = _hash_path_component(cand)
                if shortened_cand != cand and shortened_cand in path_index:
                    return path_index[shortened_cand]
                forced_cand = _hash_path_component(cand, force=True)
                if forced_cand != cand and forced_cand in path_index:
                    return path_index[forced_cand]
        if base_no_ext in noext_index:
            return noext_index[base_no_ext]
        shortened_base = _hash_path_component(base)
        if shortened_base != base and shortened_base in path_index:
            return path_index[shortened_base]
        forced_base = _hash_path_component(base, force=True)
        if forced_base != base and forced_base in path_index:
            return path_index[forced_base]

        return None

    split_candidates: List[str] = []
    if split:
        base = str(split).strip().lower()
        if base:
            split_candidates.append(base)
            if base == "dev" and "validation" not in split_candidates:
                split_candidates.append("validation")
            if base == "validation" and "dev" not in split_candidates:
                split_candidates.append("dev")
    else:
        split_candidates.extend(["train", "validation", "test"])

    keep_prob = 1.0
    if percentage is not None and percentage < 100:
        keep_prob = max(0.0, min(1.0, float(percentage) / 100.0))
    rng = random.Random(seed)
    kept = 0

    missing_logged: set = set()
    seen_tsv: set = set()
    for variant in split_candidates or ["train", "validation", "test"]:
        tsv_file = data_dir / f"{variant}.tsv"
        if not tsv_file.exists() or str(tsv_file) in seen_tsv:
            continue
        seen_tsv.add(str(tsv_file))
        print(f"  Processing TSV file: {tsv_file}")
        with tsv_file.open("r", encoding="utf-8", newline="") as f:
            header_keys, header_lookup, dict_rows = iter_tsv_dict_rows(f)
            if not header_keys:
                continue

            dict_rows = iter(dict_rows)
            try:
                first_item = next(dict_rows)
            except StopIteration:
                continue
            dict_rows = chain([first_item], dict_rows)
            _, first_row = first_item

            def _peek_resolved_audio_column() -> Optional[str]:
                candidates: List[str] = []
                for key in header_keys:
                    candidate_value = first_row.get(key, "")
                    if not candidate_value:
                        continue
                    resolved = resolve_audio(str(candidate_value))
                    if resolved:
                        candidates.append(key)
                return candidates[0] if candidates else None

            def _pick_text_column(existing_audio_col: Optional[str]) -> Optional[str]:
                best_key: Optional[str] = None
                best_score = -1
                for key in header_keys:
                    if key == existing_audio_col:
                        continue
                    value = str(first_row.get(key, "")).strip()
                    if not value:
                        continue
                    score = 0
                    if " " in value:
                        score += 2
                    score += min(len(value), 120) / 40.0
                    if score > best_score:
                        best_score = score
                        best_key = key
                return best_key

            def _resolve_column(
                candidates: Sequence[str], default_idx: int
            ) -> Optional[str]:
                for candidate in candidates:
                    cand_key = candidate.strip().lower()
                    if cand_key in header_lookup:
                        return cand_key
                if 0 <= default_idx < len(header_keys):
                    return header_keys[default_idx]
                return None

            audio_column = _resolve_column(
                ["path", "filename", "file", "audio", "audio_filename"],
                default_idx=0,
            )
            text_column = _resolve_column(
                [
                    "text",
                    "transcription",
                    "sentence",
                    "transcript",
                    "raw_transcription",
                    "normalized_text",
                    "raw_text",
                ],
                default_idx=2 if len(header_keys) > 2 else len(header_keys) - 1,
            )

            def _looks_like_audio_path(value: str) -> bool:
                if not value:
                    return False
                lowered = str(value).lower()
                return lowered.endswith(audio_exts)

            resolved_audio_col = _peek_resolved_audio_column()
            if resolved_audio_col and resolved_audio_col in header_keys:
                audio_column = resolved_audio_col

            header_is_synthetic = all(
                key.startswith("column_") and header_lookup.get(key) == key
                for key in header_keys
            )
            if header_is_synthetic:
                first_col_value = (
                    first_row.get(header_keys[0], "") if header_keys else ""
                )
                second_col_value = (
                    first_row.get(header_keys[1], "") if len(header_keys) > 1 else ""
                )
                third_col_value = (
                    first_row.get(header_keys[2], "") if len(header_keys) > 2 else ""
                )
                if (
                    len(header_keys) > 1
                    and not _looks_like_audio_path(first_col_value)
                    and _looks_like_audio_path(second_col_value)
                ):
                    audio_column = header_keys[1]
                if (
                    len(header_keys) > 2
                    and third_col_value
                    and not _looks_like_audio_path(third_col_value)
                ):
                    text_column = header_keys[2]
                if text_column is None:
                    text_column = _pick_text_column(audio_column)

            if text_column is None:
                text_column = _pick_text_column(audio_column)

            if header_is_synthetic:
                if len(header_keys) > 1 and audio_column == header_keys[0]:
                    audio_column = header_keys[1]
                if len(header_keys) > 2:
                    if text_column is None or text_column == audio_column:
                        text_column = header_keys[2]
                elif text_column == audio_column and len(header_keys) > 1:
                    text_column = header_keys[1]

            if text_column == audio_column:
                text_column = _pick_text_column(audio_column)

            if audio_column is None or text_column is None:
                print(
                    f"  Warning: could not detect audio/text columns in {tsv_file}. "
                    f"Header: {list(header_lookup.values())}"
                )
                continue

            for line_number, row_dict in dict_rows:
                audio_filename = (row_dict.get(audio_column) or "").strip()
                if not audio_filename:
                    continue

                transcription_raw = row_dict.get(text_column, "")
                transcription = (
                    str(transcription_raw).strip()
                    if transcription_raw is not None
                    else ""
                )
                if not transcription:
                    continue

                if keep_prob < 1.0 and rng.random() > keep_prob:
                    continue

                audio_path = resolve_audio(audio_filename)
                if audio_path:
                    yield {"audio": {"path": audio_path}, "text": transcription}
                    kept += 1
                    if limit and kept >= limit:
                        return
                elif audio_filename not in missing_logged:
                    print(
                        f"  Warning (line {line_number}): audio file '{audio_filename}' not found under {audio_dir_abs}"
                    )
                    missing_logged.add(audio_filename)


def load_audio(
    arr: Optional[np.ndarray],
    sr: Optional[int],
    ref: Optional[str],
    *,
    target_sr: Optional[int],
    mono: bool,
    no_resample: bool,
) -> Tuple[Optional[np.ndarray], Optional[int]]:
    if arr is None:
        if not ref:
            return None, None
        # Regular file path
        try:
            samples, sample_rate = sf.read(ref, dtype="float32", always_2d=True)
        except Exception as exc:
            print(f"  [DEBUG] Failed to read audio from '{ref}': {exc}")
            return None, None
        if mono:
            samples = samples.mean(axis=1).astype(np.float32)
        else:
            samples = samples.astype(np.float32)
        arr = samples
        sr = int(sample_rate)
    else:
        arr = arr.astype(np.float32, copy=False)
        if arr.ndim == 2:
            if mono:
                arr = arr.mean(axis=1).astype(np.float32)
            else:
                arr = arr.astype(np.float32)

    if arr is None or arr.size == 0:
        return None, None

    if sr is None:
        return None, None

    if not no_resample and target_sr and sr != target_sr:
        if arr.ndim == 1:
            arr = librosa.resample(arr, orig_sr=sr, target_sr=target_sr).astype(
                np.float32
            )
        else:
            channels = []
            for ch in range(arr.shape[1]):
                channels.append(
                    librosa.resample(
                        arr[:, ch], orig_sr=sr, target_sr=target_sr
                    ).astype(np.float32)
                )
            arr = np.stack(channels, axis=1)
        sr = target_sr
    return arr, sr


def write_audio(
    samples: np.ndarray, sample_rate: int, path: Path, *, subtype: str = "PCM_16"
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, samples, sample_rate, subtype=subtype)


def apply_sampling(ds, percentage: Optional[float], seed: int, limit: Optional[int]):
    if percentage is None:
        percentage = 100.0
    if percentage >= 100 and not limit:
        return ds
    try:
        length = len(ds)
    except TypeError:
        return ds
    if length == 0:
        return ds
    if percentage < 100:
        target = max(1, int(length * (percentage / 100.0)))
    else:
        target = length
    if limit:
        target = min(target, limit)
    if target >= length:
        return ds
    ds = ds.shuffle(seed=seed)
    return ds.select(range(target))


def get_filter_fn(repo: str):
    if repo == "bekzod123/uzbek_voice":
        return _filter_bekzod123_uzbek_voice
    if repo == "DavronSherbaev/uzbekvoice-filtered":
        return _filter_davron_sherbaev_uzbekvoice
    return None


def apply_dataset_filter(ds, filter_fn, label: str):
    if filter_fn is None:
        return ds
    try:
        before = len(ds)
    except TypeError:
        before = None
    try:
        ds = ds.filter(filter_fn, batched=False, desc=f"Filtering {label}")
        if before is not None:
            print(f"  Kept {len(ds)}/{before} after filter for {label}")
        return ds
    except Exception as exc:
        print(f"  Filter failed for {label} ({exc}); falling back to index scan.")
        try:
            kept = []
            for idx, item in enumerate(tqdm(ds, desc=f"Scanning {label}")):
                try:
                    if filter_fn(item):
                        kept.append(idx)
                except Exception:
                    continue
            print(f"  Kept {len(kept)}/{len(ds)} after index scan for {label}")
            return ds.select(kept)
        except Exception as exc2:
            print(f"  Failed to apply filter for {label}: {exc2}")
            return ds


@contextmanager
def dataset_cache_context(
    cache_mode: str, cache_root: Optional[Path]
) -> Optional[Path]:
    if cache_root is None:
        cache_root = DEFAULT_CACHE_ROOT
    cache_root.mkdir(parents=True, exist_ok=True)
    if cache_mode == "default":
        yield cache_root
        return

    # Use a persistent cache directory instead of a temporary one.
    # This allows reusing downloaded datasets across runs.
    persistent_cache = cache_root / "hf_cache"
    persistent_cache.mkdir(parents=True, exist_ok=True)
    prev_env = {
        "HF_HOME": os.environ.get("HF_HOME"),
        "HF_DATASETS_CACHE": os.environ.get("HF_DATASETS_CACHE"),
        "HUGGINGFACE_HUB_CACHE": os.environ.get("HUGGINGFACE_HUB_CACHE"),
    }
    os.environ["HF_HOME"] = str(persistent_cache)
    os.environ["HF_DATASETS_CACHE"] = str(persistent_cache)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(persistent_cache / "hub")
    try:
        yield persistent_cache
    finally:
        for key, value in prev_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        # Do not cleanup - reuse the cache for subsequent runs


def to_manifest_path(output_dir: Path, audio_path: Path, absolute: bool) -> str:
    if absolute:
        return str(audio_path.resolve())
    return str(audio_path.relative_to(output_dir))


# =============================================================================
# Frequency Collection (Single-Pass) and Manifest Post-Processing
# =============================================================================


def post_process_manifest_with_typo_corrections(
    manifest_path: Path,
    typo_detector: "FrequencyBasedTypoDetector",
) -> int:
    """Apply typo corrections to an existing manifest file.

    Reads the manifest, applies corrections to text fields, and writes back.

    Args:
        manifest_path: Path to the manifest file
        typo_detector: Configured typo detector with analyzed frequencies

    Returns:
        Number of entries corrected
    """
    if not manifest_path.exists():
        print(f"  Warning: Manifest not found: {manifest_path}")
        return 0

    # Read all entries
    entries = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    if not entries:
        return 0

    # Apply corrections
    corrected_count = 0
    for entry in entries:
        if "text" in entry and entry["text"]:
            original = entry["text"]
            corrected = typo_detector.correct_text(original, record_stats=True)
            if corrected != original:
                entry["text"] = corrected
                corrected_count += 1

    # Write back
    with manifest_path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False))
            f.write("\n")

    return corrected_count


def process_items(
    items: Iterable[Dict[str, Any]],
    *,
    output_dir: Path,
    audio_root: Path,
    manifest_file,
    desc: str,
    sample_rate: int,
    no_resample: bool,
    mono: bool,
    min_duration: float,
    max_duration: float,
    min_chars: int,
    max_chars: int,
    absolute_paths: bool,
    frequency_collector: Optional["WordFrequencyCollector"] = None,
    debug: bool = False,
) -> Dict[str, int]:
    counts = {
        "total": 0,
        "kept": 0,
        "no_text": 0,
        "no_audio": 0,
        "dur_filtered": 0,
        "text_filtered": 0,
        "failed": 0,
    }

    for idx, item in enumerate(tqdm(items, desc=desc)):
        counts["total"] += 1
        if not isinstance(item, dict):
            counts["failed"] += 1
            if debug and idx < 3:
                print(f"  [DEBUG] Item {idx}: not a dict, type={type(item).__name__}")
            continue

        # Debug: print first few items' structure
        if debug and idx < 3:
            print(f"  [DEBUG] Item {idx} keys: {list(item.keys())}")
            for k, v in item.items():
                v_type = type(v).__name__
                v_preview = repr(v)[:100] if v is not None else "None"
                print(f"    {k}: {v_type} = {v_preview}")

        transcript = pick_text(item)
        if not transcript:
            counts["no_text"] += 1
            if debug and idx < 5:
                print(f"  [DEBUG] Item {idx}: no text found")
            continue
        if len(transcript) < min_chars:
            counts["text_filtered"] += 1
            continue
        effective_max = MAX_TRANSCRIPT_CHAR_LIMIT
        if max_chars != -1:
            effective_max = min(effective_max, max_chars)
        if len(transcript) > effective_max:
            counts["text_filtered"] += 1
            continue
        # Filter out items containing C/c not followed by h (standalone C/c)
        if contains_standalone_c(transcript):
            counts["text_filtered"] += 1
            continue

        arr, sr, ref = read_audio_from_item(item, debug=(debug and idx < 5))
        if debug and idx < 5:
            print(
                f"  [DEBUG] Item {idx}: audio read result: arr={arr is not None}, sr={sr}, ref={ref}"
            )
        arr, sr = load_audio(
            arr,
            sr,
            ref,
            target_sr=None if no_resample else sample_rate,
            mono=mono,
            no_resample=no_resample,
        )
        if arr is None or sr is None:
            counts["no_audio"] += 1
            if debug and idx < 5:
                print(f"  [DEBUG] Item {idx}: no audio after load_audio")
            continue

        duration = float(arr.shape[0] / float(sr))
        if duration < min_duration:
            counts["dur_filtered"] += 1
            continue
        if max_duration != -1 and duration > max_duration:
            counts["dur_filtered"] += 1
            continue

        audio_path = audio_root / f"{idx:09d}.wav"
        try:
            write_audio(arr, sr, audio_path)
        except Exception:
            counts["failed"] += 1
            continue

        # Collect word frequencies if collector is provided (for typo detection)
        if frequency_collector is not None:
            frequency_collector.add_text(transcript)

        manifest_entry = {
            "audio_filepath": to_manifest_path(output_dir, audio_path, absolute_paths),
            "duration": round(duration, 6),
            "text": transcript,
        }
        manifest_file.write(json.dumps(manifest_entry, ensure_ascii=False))
        manifest_file.write("\n")
        counts["kept"] += 1

    return counts


def prepare_group(
    group: str,
    specs: Sequence[DatasetSpec],
    *,
    output_dir: Path,
    cache_dir: Optional[Path],
    cache_mode: str,
    hf_token: Optional[str],
    sample_rate: int,
    no_resample: bool,
    mono: bool,
    min_duration: float,
    max_duration: float,
    min_chars: int,
    max_chars: int,
    absolute_paths: bool,
    limit: Optional[int],
    hf_load_retries: int = DEFAULT_HF_LOAD_RETRIES,
    hf_retry_wait: float = DEFAULT_HF_RETRY_WAIT,
    hf_rate_limit_wait: float = DEFAULT_HF_RATE_LIMIT_WAIT,
    frequency_collector: Optional["WordFrequencyCollector"] = None,
) -> None:
    manifest_path = output_dir / f"{group}_manifest.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        for spec in specs:
            if spec.group != group:
                continue
            dataset_id = sanitize_name(spec.repo)
            try:
                _process_single_spec(
                    spec=spec,
                    group=group,
                    dataset_id=dataset_id,
                    manifest_file=manifest_file,
                    output_dir=output_dir,
                    cache_dir=cache_dir,
                    cache_mode=cache_mode,
                    hf_token=hf_token,
                    sample_rate=sample_rate,
                    no_resample=no_resample,
                    mono=mono,
                    min_duration=min_duration,
                    max_duration=max_duration,
                    min_chars=min_chars,
                    max_chars=max_chars,
                    absolute_paths=absolute_paths,
                    limit=limit,
                    hf_load_retries=hf_load_retries,
                    hf_retry_wait=hf_retry_wait,
                    hf_rate_limit_wait=hf_rate_limit_wait,
                    frequency_collector=frequency_collector,
                )
            except Exception as exc:
                print(f"[{group}] ERROR processing {spec.repo}: {exc}")
                import traceback

                traceback.print_exc()
                print(
                    f"[{group}] Skipping {spec.repo} and continuing with next dataset..."
                )
                continue


def _process_single_spec(
    *,
    spec: DatasetSpec,
    group: str,
    dataset_id: str,
    manifest_file,
    output_dir: Path,
    cache_dir: Optional[Path],
    cache_mode: str,
    hf_token: Optional[str],
    sample_rate: int,
    no_resample: bool,
    mono: bool,
    min_duration: float,
    max_duration: float,
    min_chars: int,
    max_chars: int,
    absolute_paths: bool,
    limit: Optional[int],
    hf_load_retries: int,
    hf_retry_wait: float,
    hf_rate_limit_wait: float,
    frequency_collector: Optional["WordFrequencyCollector"] = None,
) -> None:
    """Process a single dataset spec and write to manifest."""
    if spec.repo == "google/fleurs":
        if not spec.subset:
            print(f"[{group}] skipping {spec.repo} because no subset was provided.")
            return
        with dataset_cache_context(cache_mode, cache_dir) as cache_path:
            local_dir = download_fleurs_subset(spec.subset, cache_path, spec.revision)
            for split in spec.splits:
                print(
                    f"[{group}] loading {spec.repo} split={split} (subset={spec.subset})"
                )
                items = iter_fleurs_items(
                    local_dir=local_dir,
                    subset=spec.subset,
                    split=split,
                    percentage=spec.percentage,
                    seed=spec.seed if spec.seed is not None else DEFAULT_SAMPLING_SEED,
                    limit=limit,
                )
                audio_root = output_dir / "audio" / group / dataset_id / split
                counts = process_items(
                    items,
                    output_dir=output_dir,
                    audio_root=audio_root,
                    manifest_file=manifest_file,
                    desc=f"{dataset_id}:{split}",
                    sample_rate=sample_rate,
                    no_resample=no_resample,
                    mono=mono,
                    min_duration=min_duration,
                    max_duration=max_duration,
                    min_chars=min_chars,
                    max_chars=max_chars,
                    absolute_paths=absolute_paths,
                    frequency_collector=frequency_collector,
                    debug=True,
                )
                print(
                    f"[{group}] {spec.repo}:{split} "
                    f"total={counts['total']} kept={counts['kept']} "
                    f"no_text={counts['no_text']} no_audio={counts['no_audio']} "
                    f"dur_filtered={counts['dur_filtered']} text_filtered={counts['text_filtered']} "
                    f"failed={counts['failed']}"
                )
        return
    if spec.repo in (
        "mozilla-foundation/common_voice_17_0",
        "fsicoli/common_voice_17_0",
    ):
        if not spec.subset:
            print(f"[{group}] skipping {spec.repo} because no subset was provided.")
            return
        with dataset_cache_context(cache_mode, cache_dir) as cache_path:
            local_dir = download_common_voice_subset(spec.repo, spec.subset, cache_path)
            for split in spec.splits:
                print(
                    f"[{group}] loading {spec.repo} split={split} (subset={spec.subset})"
                )
                items = iter_common_voice_items(
                    local_dir=local_dir,
                    subset=spec.subset,
                    split=split,
                    percentage=spec.percentage,
                    seed=spec.seed if spec.seed is not None else DEFAULT_SAMPLING_SEED,
                    limit=limit,
                )
                audio_root = output_dir / "audio" / group / dataset_id / split
                counts = process_items(
                    items,
                    output_dir=output_dir,
                    audio_root=audio_root,
                    manifest_file=manifest_file,
                    desc=f"{dataset_id}:{split}",
                    sample_rate=sample_rate,
                    no_resample=no_resample,
                    mono=mono,
                    min_duration=min_duration,
                    max_duration=max_duration,
                    min_chars=min_chars,
                    max_chars=max_chars,
                    absolute_paths=absolute_paths,
                    frequency_collector=frequency_collector,
                    debug=True,
                )
                print(
                    f"[{group}] {spec.repo}:{split} "
                    f"total={counts['total']} kept={counts['kept']} "
                    f"no_text={counts['no_text']} no_audio={counts['no_audio']} "
                    f"dur_filtered={counts['dur_filtered']} text_filtered={counts['text_filtered']} "
                    f"failed={counts['failed']}"
                )
        return

    for split in spec.splits:
        print(
            f"[{group}] loading {spec.repo} split={split}"
            + (f" (subset={spec.subset})" if spec.subset else "")
        )
        with dataset_cache_context(cache_mode, cache_dir) as cache_path:
            load_kwargs: Dict[str, Any] = {"split": split}
            if spec.subset:
                load_kwargs["name"] = spec.subset
            if spec.revision:
                load_kwargs["revision"] = spec.revision
            if spec.data_dir:
                load_kwargs["data_dir"] = spec.data_dir
            if spec.data_files is not None:
                load_kwargs["data_files"] = spec.data_files
            if spec.trust_remote_code is not None:
                load_kwargs["trust_remote_code"] = spec.trust_remote_code
            if cache_path:
                load_kwargs["cache_dir"] = str(cache_path)
            if hf_token:
                load_kwargs["use_auth_token"] = hf_token

            ds = _load_dataset_with_retries(
                spec.repo,
                load_kwargs,
                hf_load_retries,
                hf_retry_wait,
                hf_rate_limit_wait,
            )
            # Debug: print dataset features and first item
            print(f"  [DEBUG] Dataset features: {ds.features}")
            print(f"  [DEBUG] Dataset columns: {ds.column_names}")
            print(f"  [DEBUG] Dataset length: {len(ds)}")

            # Print first 3 items raw structure before any processing
            print(f"  [DEBUG] First 3 raw items from dataset:")
            for i in range(min(3, len(ds))):
                item = ds[i]
                print(f"    [DEBUG] Item {i}:")
                for k, v in item.items():
                    v_type = type(v).__name__
                    if v is None:
                        print(f"      {k}: None")
                    elif isinstance(v, dict):
                        dict_keys = list(v.keys())
                        has_array = "array" in v and v["array"] is not None
                        sr = v.get("sampling_rate")
                        path = v.get("path")
                        print(
                            f"      {k}: dict with keys={dict_keys}, has_array={has_array}, sr={sr}, path={str(path)[:80] if path else None}"
                        )
                    elif isinstance(v, str):
                        print(
                            f"      {k}: str = '{v[:100]}{'...' if len(v) > 100 else ''}'"
                        )
                    else:
                        print(f"      {k}: {v_type} = {str(v)[:100]}")

            # Ensure Audio features are decoded - find and cast audio columns
            audio_columns_found = []
            for col_name in ds.column_names:
                feature = ds.features.get(col_name)
                if feature is None:
                    continue
                feature_str = str(feature)
                feature_type = type(feature).__name__
                print(
                    f"  [DEBUG] Column '{col_name}': feature_type={feature_type}, feature_str={feature_str[:100]}"
                )
                # Check if it's an Audio feature type
                is_audio_feature = (
                    isinstance(feature, Audio)
                    or feature_type == "Audio"
                    or feature_str.startswith("Audio")
                    or "audio" in feature_str.lower()
                    or (
                        hasattr(feature, "dtype")
                        and "audio" in str(getattr(feature, "dtype", "")).lower()
                    )
                )
                if is_audio_feature:
                    audio_columns_found.append(col_name)
                    print(
                        f"  [DEBUG] Casting audio column '{col_name}' to ensure decoding (feature type: {feature_type})"
                    )
                    ds = ds.cast_column(col_name, Audio(sampling_rate=16000))

            if not audio_columns_found:
                print(
                    f"  [WARNING] No audio columns detected! Available columns: {ds.column_names}"
                )

            # Print first item AFTER audio casting to verify decoding
            if len(ds) > 0:
                print(f"  [DEBUG] First item AFTER audio casting:")
                item = ds[0]
                for k, v in item.items():
                    if isinstance(v, dict):
                        has_array = "array" in v and v["array"] is not None
                        array_len = len(v["array"]) if has_array else 0
                        sr = v.get("sampling_rate")
                        print(
                            f"      {k}: dict has_array={has_array}, array_len={array_len}, sr={sr}"
                        )
                    elif isinstance(v, str):
                        print(f"      {k}: str = '{v[:60]}...'")
                    else:
                        print(f"      {k}: {type(v).__name__}")
            filter_fn = get_filter_fn(spec.repo)
            if filter_fn is not None:
                print(f"  Applying dataset filter for {spec.repo}")
                ds = apply_dataset_filter(ds, filter_fn, f"{spec.repo}:{split}")
            ds = apply_sampling(
                ds,
                spec.percentage,
                spec.seed if spec.seed is not None else DEFAULT_SAMPLING_SEED,
                limit,
            )

            audio_root = output_dir / "audio" / group / dataset_id / split
            counts = process_items(
                ds,
                output_dir=output_dir,
                audio_root=audio_root,
                manifest_file=manifest_file,
                desc=f"{dataset_id}:{split}",
                sample_rate=sample_rate,
                no_resample=no_resample,
                mono=mono,
                min_duration=min_duration,
                max_duration=max_duration,
                min_chars=min_chars,
                max_chars=max_chars,
                absolute_paths=absolute_paths,
                frequency_collector=frequency_collector,
                debug=True,
            )
            print(
                f"[{group}] {spec.repo}:{split} "
                f"total={counts['total']} kept={counts['kept']} "
                f"no_text={counts['no_text']} no_audio={counts['no_audio']} "
                f"dur_filtered={counts['dur_filtered']} text_filtered={counts['text_filtered']} "
                f"failed={counts['failed']}"
            )


def main() -> None:
    args = parse_args()
    reset_misspelling_stats()
    reset_frequency_collector()
    reset_typo_detector()

    _configure_hf_http_settings(
        args.hf_read_timeout,
        args.hf_connect_timeout,
        args.hf_etag_timeout,
        args.hf_max_retries,
    )
    config = load_dataset_config(args.config)
    specs = list(
        iter_dataset_specs(config, args.groups, default_splits=args.default_splits)
    )

    if not specs:
        raise SystemExit("No dataset entries found for the selected groups.")

    groups_present = {spec.group for spec in specs}

    # Initialize frequency collector if typo detection is enabled
    frequency_collector: Optional[WordFrequencyCollector] = None
    if args.enable_typo_detection:
        frequency_collector = get_frequency_collector()
        frequency_collector.reset()
        print("\n" + "=" * 50)
        print("Frequency-based typo detection ENABLED")
        print("Collecting word frequencies during processing...")
        print("=" * 50 + "\n")

    # Process datasets (single pass - collects frequencies if enabled)
    for group in args.groups:
        if group not in groups_present:
            print(f"Skipping group '{group}' (no entries in config).")
            continue
        prepare_group(
            group,
            specs,
            output_dir=args.output_dir,
            cache_dir=args.cache_dir,
            cache_mode=args.cache_mode,
            hf_token=args.hf_token,
            sample_rate=args.sample_rate,
            no_resample=args.no_resample,
            mono=args.mono,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
            min_chars=args.min_chars,
            max_chars=args.max_chars,
            absolute_paths=args.absolute_paths,
            limit=args.limit,
            hf_load_retries=args.hf_load_retries,
            hf_retry_wait=args.hf_retry_wait,
            hf_rate_limit_wait=args.hf_rate_limit_wait,
            frequency_collector=frequency_collector,
        )

    # Report misspelling fix statistics (static dictionary-based)
    stats = get_misspelling_stats()
    if stats.total_fixes > 0:
        print("\n" + "=" * 50)
        print("STATIC MISSPELLING CORRECTION REPORT")
        print("=" * 50)
        print(stats.report())
        print("=" * 50 + "\n")

    # Apply frequency-based typo corrections if enabled
    if frequency_collector is not None and args.enable_typo_detection:
        print("\n" + "=" * 50)
        print("Analyzing collected word frequencies...")
        print("=" * 50)
        print(f"  Total words collected: {frequency_collector.total_words:,}")
        print(f"  Unique vocabulary size: {frequency_collector.vocabulary_size:,}")
        print(f"  Top 20 most common words:")
        for word, count in frequency_collector.most_common(20):
            print(f"    '{word}': {count:,}")

        # Create and configure the typo detector
        typo_detector = FrequencyBasedTypoDetector(
            frequency_collector,
            min_frequency_ratio=args.typo_min_frequency_ratio,
            max_edit_distance=args.typo_max_edit_distance,
            min_correction_frequency=args.typo_min_correction_frequency,
            min_typo_length=args.typo_min_word_length,
            confidence_threshold=args.typo_confidence_threshold,
        )

        # Analyze and detect typos
        candidates = typo_detector.analyze()
        print(f"\nDetected {len(candidates)} potential typos")

        # Print typo report
        print("\n" + typo_detector.get_typo_report())

        if args.typo_report_only:
            print("\n[typo-report-only] Typos reported but not applied to manifests.")
        else:
            # Apply corrections to manifest files
            print("\n" + "=" * 50)
            print("Applying typo corrections to manifest files...")
            print("=" * 50)

            total_corrected = 0
            for group in args.groups:
                if group not in groups_present:
                    continue
                manifest_path = args.output_dir / f"{group}_manifest.jsonl"
                corrected = post_process_manifest_with_typo_corrections(
                    manifest_path, typo_detector
                )
                print(
                    f"  [{group}] Corrected {corrected} entries in {manifest_path.name}"
                )
                total_corrected += corrected

            print(f"\nTotal entries with typo corrections: {total_corrected}")

            # Report frequency-based typo correction statistics
            typo_stats = typo_detector.stats
            if typo_stats.total_corrections_applied > 0:
                print("\n" + "=" * 50)
                print("FREQUENCY-BASED TYPO CORRECTION REPORT")
                print("=" * 50)
                print(typo_stats.report())
                print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
