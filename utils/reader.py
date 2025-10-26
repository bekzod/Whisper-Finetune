import csv
import functools
import gc
import hashlib
import json
import os
import random
import re
import sys
import tarfile
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import chain
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import soundfile
from datasets import Audio, Dataset as HFDataset, DatasetDict as HFDatasetDict
from datasets import load_dataset, load_from_disk, DatasetDict
from huggingface_hub import snapshot_download  # needed for fleurs/common_voice branches
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.binary import DatasetReader
from utils.audio_augmentation import AudioAugmenter, resample
from utils.tsv_parser import iter_tsv_dict_rows

try:
    import pyarrow as pa
except ImportError:  # pragma: no cover - optional at runtime
    pa = None

try:
    csv.field_size_limit(sys.maxsize)
except (OverflowError, AttributeError):
    # Fall back to a large but safe limit if sys.maxsize is not accepted
    csv.field_size_limit(2**31 - 1)

MAX_TRANSCRIPT_CHAR_LIMIT = 600
DEFAULT_HF_SAMPLING_SEED = 3407
ORIGINAL_INDEX_COLUMN = "__orig_idx__"

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
_HF_OPTIONAL_KEYS = (
    "duration",
    "language",
    "sentences",
    "speaker",
    "speaker_id",
    "accent",
    "id",
    "name",
    "start_time",
    "end_time",
    "segment",
    "offset",
)
_HF_ALLOWED_COLUMNS = (
    set(_HF_AUDIO_CANDIDATE_KEYS)
    | set(_HF_TEXT_CANDIDATE_KEYS)
    | set(_HF_OPTIONAL_KEYS)
)

_APOSTROPHE_TRANSLATION = str.maketrans(
    {
        "\u2019": "'",
        "\u02bc": "'",
        "\u02bb": "'",
        "`": "'",
        "´": "'",
        "ʻ": "'",
        "ʼ": "'",
        "‛": "'",
    }
)
_ALLOWED_TEXT_RE = re.compile(r"[^a-zA-ZА-Яа-яЎўҚқҒғҲҳ0-9\s,.'-]+")
_MULTISPACE_RE = re.compile(r"\s+")
_UZBEK_CYRILLIC_TO_LATIN = {
    "А": "A",
    "а": "a",
    "Б": "B",
    "б": "b",
    "В": "V",
    "в": "v",
    "Г": "G",
    "г": "g",
    "Д": "D",
    "д": "d",
    "Е": "E",
    "е": "e",
    "Ё": "Yo",
    "ё": "yo",
    "Ж": "J",
    "ж": "j",
    "З": "Z",
    "з": "z",
    "И": "I",
    "и": "i",
    "Й": "Y",
    "й": "y",
    "К": "K",
    "к": "k",
    "Л": "L",
    "л": "l",
    "М": "M",
    "м": "m",
    "Н": "N",
    "н": "n",
    "О": "O",
    "о": "o",
    "П": "P",
    "п": "p",
    "Р": "R",
    "р": "r",
    "С": "S",
    "с": "s",
    "Т": "T",
    "т": "t",
    "У": "U",
    "у": "u",
    "Ф": "F",
    "ф": "f",
    "Х": "X",
    "х": "x",
    "Ц": "Ts",
    "ц": "ts",
    "Ч": "Ch",
    "ч": "ch",
    "Ш": "Sh",
    "ш": "sh",
    "Щ": "Sh",
    "щ": "sh",
    "Ъ": "'",
    "ъ": "'",
    "Ы": "I",
    "ы": "i",
    "Ь": "",
    "ь": "",
    "Э": "E",
    "э": "e",
    "Ю": "Yu",
    "ю": "yu",
    "Я": "Ya",
    "я": "ya",
    "Ў": "O'",
    "ў": "o'",
    "Қ": "Q",
    "қ": "q",
    "Ғ": "G'",
    "ғ": "g'",
    "Ҳ": "H",
    "ҳ": "h",
}
_UZBEK_CYRILLIC_CHARS = set(_UZBEK_CYRILLIC_TO_LATIN.keys())


def _preview_text(text: str, limit: int = 120) -> str:
    """Return a single-line preview of the given text, capped at `limit` characters."""
    snippet = text[:limit].replace("\n", " ")
    return f"{snippet}..." if len(text) > limit else snippet


@dataclass(slots=True)
class ManifestEntry:
    audio_path: str
    sentence: Optional[str] = None
    sentences: Optional[Any] = None
    duration: Optional[float] = None
    language: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None


def rate_limited_request(func, *args, **kwargs):
    """
    Execute a function with exponential backoff retry for rate limiting.
    """
    max_retries = 5
    base_delay = 150

    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_msg = str(e).lower()
            if "rate limit" in error_msg or "quota" in error_msg or "2500" in error_msg:
                if attempt < max_retries - 1:
                    delay = base_delay * (2**attempt)  # Exponential backoff
                    print(
                        f"Rate limit hit. Waiting {delay} seconds before retry {attempt + 1}/{max_retries}..."
                    )
                    time.sleep(delay)
                    continue
                else:
                    print(
                        "Max retries reached for rate limiting. Please upgrade your HF organization or wait."
                    )
                    raise
            else:
                # Re-raise non-rate-limit errors immediately
                raise


def normalize_text(text):
    """
    Normalize text by replacing various apostrophe-like characters with a uniform apostrophe
    and keeping only alphanumeric characters plus commas, dots, spaces, apostrophes, and dashes.
    This handles Uzbek text like "ko'p" to ensure consistent character usage and also
    transliterates Uzbek Cyrillic characters to their Latin counterparts.
    """
    if text is None:
        return text

    # Coerce to string to handle non-string inputs (e.g., ints)
    try:
        normalized = str(text)
    except Exception:
        # As a last resort, return empty string if conversion fails
        return ""

    normalized = _transliterate_uzbek_cyrillic(normalized)

    # Replace apostrophe-like characters with standard apostrophe
    normalized = normalized.translate(_APOSTROPHE_TRANSLATION)

    # This regex keeps Latin & Cyrillic letters, digits, spaces, comma, dot, apostrophe, dash
    normalized = _ALLOWED_TEXT_RE.sub("", normalized)

    # Clean up multiple spaces
    normalized = _MULTISPACE_RE.sub(" ", normalized).strip()

    return normalized


def _transliterate_uzbek_cyrillic(text: str) -> str:
    """
    Transliterate Uzbek Cyrillic characters to Latin.
    """
    if not text:
        return text

    if not any(char in _UZBEK_CYRILLIC_CHARS for char in text):
        return text

    return "".join(_UZBEK_CYRILLIC_TO_LATIN.get(char, char) for char in text)


class CustomDataset(Dataset):
    def __init__(
        self,
        data_list_path,
        processor,
        mono=True,
        language=None,
        timestamps=False,
        sample_rate=16000,
        min_duration=0.5,
        max_duration=30,
        min_sentence=1,
        max_sentence=480,
        augment_config_path=None,
        dataset_filters=None,
        hf_sampling_config=None,
    ):
        """
        Args:
            data_list_path: Path(s) to dataset manifests or HF dataset specs. Supports '+' to concat.
            processor: Whisper preprocessing tool, obtained from WhisperProcessor.from_pretrained
            mono: Whether to convert audio to mono channel (recommended True)
            language: Language code for fine-tuning (optional)
            timestamps: Whether labels contain per-segment timestamps
            sample_rate: Target sample rate (default 16k)
            min_duration, max_duration: duration bounds in seconds (0.5 <= min <= 30)
            min_sentence, max_sentence: character-count bounds for transcripts
            augment_config_path: JSON file path with augmentation configs
            dataset_filters: List[{'name': str, 'filter_fn': callable(row)->bool}]
            hf_sampling_config: Optional mapping of split identifiers to sampling directives
                (e.g., {'hf://org/name:train': {'percentage': 25}}).
        """
        super(CustomDataset, self).__init__()
        assert min_duration >= 0.5, (
            f"min_duration cannot be less than 0.5, current value: {min_duration}"
        )
        assert max_duration <= 30, (
            f"max_duration cannot be greater than 30, current value: {max_duration}"
        )
        assert min_sentence >= 1, (
            f"min_sentence cannot be less than 1, current value: {min_sentence}"
        )
        assert max_sentence <= 500, (
            f"max_sentence cannot be greater than 500, current value: {max_sentence}"
        )
        self.data_list_path = data_list_path
        self.processor = processor
        self.sample_rate = sample_rate
        self.mono = mono
        self.language = language
        self.timestamps = timestamps
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.min_sentence = min_sentence
        self.max_sentence = max_sentence

        self.dataset_filters = dataset_filters or []

        self.hf_sampling_config: Dict[str, Dict[str, float]] = {}
        if hf_sampling_config:
            for key, cfg in hf_sampling_config.items():
                if not cfg:
                    continue
                normalized_key = str(key).strip()
                if not normalized_key:
                    continue
                cfg_copy = dict(cfg)
                percentage = cfg_copy.get("percentage")
                if percentage is None:
                    continue
                try:
                    pct_value = float(percentage)
                except (TypeError, ValueError):
                    raise ValueError(
                        f"Invalid percentage value '{percentage}' for HF sampling config key '{key}'."
                    ) from None
                if not (0 < pct_value <= 100):
                    raise ValueError(
                        f"Percentage for HF sampling config key '{key}' must be within (0, 100], got {pct_value}."
                    )
                cfg_copy["percentage"] = pct_value
                if "seed" in cfg_copy and cfg_copy["seed"] is not None:
                    try:
                        cfg_copy["seed"] = int(cfg_copy["seed"])
                    except (TypeError, ValueError):
                        raise ValueError(
                            f"Seed override for HF sampling config key '{key}' must be an integer, got {cfg_copy['seed']!r}."
                        ) from None
                self.hf_sampling_config[normalized_key] = cfg_copy
                if normalized_key.startswith("hf://"):
                    alt_key = normalized_key[5:]
                    self.hf_sampling_config.setdefault(alt_key, cfg_copy)

        # --- SAFER: choose num_proc for HF Datasets map/filter ---
        cpu_cnt = os.cpu_count() or 4
        self.num_proc = min(8, max(2, cpu_cnt // 4))

        self.vocab = self.processor.tokenizer.get_vocab()
        self.startoftranscript = self.vocab["<|startoftranscript|>"]
        self.endoftext = self.vocab["<|endoftext|>"]
        if "<|nospeech|>" in self.vocab.keys():
            self.nospeech = self.vocab["<|nospeech|>"]
            self.timestamp_begin = None
        else:
            # Compatible with old models
            self.nospeech = self.vocab["<|nocaptions|>"]
            self.timestamp_begin = self.vocab["<|notimestamps|>"] + 1

        # Python-side materialized entries (CSV/JSONL/etc.)
        self.data_list: List[Union[str, ManifestEntry]] = []
        # Lazy HF datasets: store splits to avoid 100s of thousands of Python dicts
        # Each entry: { 'dataset': HFDataset, 'indices': Optional[List[int]], 'name': str }
        self.hf_splits = []

        inline_cache_root = os.environ.get("WHISPER_INLINE_AUDIO_DIR")
        if inline_cache_root:
            inline_cache_root = os.path.abspath(inline_cache_root)
        else:
            inline_cache_root = os.path.join(
                tempfile.gettempdir(), "whisper_finetune_inline_audio"
            )
        os.makedirs(inline_cache_root, exist_ok=True)
        self.inline_audio_cache_dir = inline_cache_root

        # Label tracking for debugging TSV/dataset issues
        self._dataset_label_counts = {}
        self._label_examples = {}
        self._load_data_list()

        # Summary logging to make filtering effects obvious
        try:
            materialized = len(self.data_list)
            lazy_splits = len(self.hf_splits)
            total = len(self)
            print(
                f"📊 Dataset loaded: materialized={materialized}, hf_splits={lazy_splits}, total={total}"
            )

            # Print summary of label examples collected during loading
            if self._dataset_label_counts:
                print("📝 Label examples summary:")
                for dataset_key, count in self._dataset_label_counts.items():
                    examples = self._label_examples.get(dataset_key, [])
                    print(f"   Dataset '{dataset_key}': {count} total entries")
                    for i, example in enumerate(examples, 1):
                        print(
                            f"     Example {i}: chars={example['char_length']}, tokens={example['token_length']}"
                        )
                        if example["char_length"] > 1000 or (
                            isinstance(example["token_length"], int)
                            and example["token_length"] > 400
                        ):
                            print(
                                f"       ⚠️  This example is very long - check TSV parsing!"
                            )
        except Exception:
            pass

        self.augmenter = None
        if augment_config_path:
            with open(augment_config_path, "r", encoding="utf-8") as f:
                augment_configs = json.load(f)
                self.augmenter = AudioAugmenter(augment_configs)

    @staticmethod
    def _normalize_dataset_key(name: Optional[str]) -> str:
        """
        Normalize dataset identifiers so filters match exact repos/paths instead of substrings.
        Examples:
            hf://org/name@rev#subset:split -> org/name
            /foo/bar/ds.jsonl -> /foo/bar/ds.jsonl
            ds.jsonl -> ds.jsonl
        """
        if not name:
            return ""
        key = str(name).strip()
        # Strip scheme prefix
        if key.startswith("hf://"):
            key = key[5:]
        # Drop split/subset/revision suffixes
        for sep in (":", "#", "@"):
            if sep in key:
                key = key.split(sep, 1)[0]
        return key.strip("/").lower()

    def _get_filter_config_for_path(self, data_path: str) -> Optional[dict]:
        """Return the filter config that exactly matches the dataset path/repo."""
        path_norm = self._normalize_dataset_key(data_path)
        base_norm = self._normalize_dataset_key(os.path.basename(data_path))

        for filter_config in self.dataset_filters:
            cfg_name = filter_config.get("name")
            if not cfg_name:
                continue
            cfg_norm = self._normalize_dataset_key(cfg_name)
            cfg_base_norm = self._normalize_dataset_key(os.path.basename(cfg_name))
            if cfg_norm and (cfg_norm == path_norm or cfg_norm == base_norm):
                return filter_config
            if cfg_base_norm and (
                cfg_base_norm == path_norm or cfg_base_norm == base_norm
            ):
                return filter_config
        return None

    @staticmethod
    def _coerce_optional_float(value) -> Optional[float]:
        if value is None:
            return None
        try:
            # Convert bools/ints/strings safely
            coerced = float(value)
        except (TypeError, ValueError):
            return None
        if np.isnan(coerced):
            return None
        return coerced

    @staticmethod
    def _transcript_length(entry: ManifestEntry) -> int:
        if entry.sentence:
            return len(entry.sentence)
        sentences = entry.sentences
        if sentences is None:
            return 0
        if isinstance(sentences, list):
            total = 0
            for segment in sentences:
                if isinstance(segment, dict):
                    total += len(str(segment.get("text", "")))
                else:
                    total += len(str(segment))
            return total
        return len(str(sentences))

    @staticmethod
    def _ensure_2d_features(features: Any):
        """
        Whisper expects log-Mel features shaped (mel_bins, frames). Drop any
        leading singleton batch/channel dimensions that may sneak in.
        """
        if isinstance(features, (list, tuple)):
            if not features:
                raise ValueError("Received empty input_features sequence.")
            features = features[0]

        if hasattr(features, "ndim") and hasattr(features, "shape"):
            while getattr(features, "ndim", 0) >= 3 and features.shape[0] == 1:
                features = features[0]

        return features

    def _write_inline_bytes(self, payload: Union[bytes, bytearray, memoryview]) -> str:
        """
        Persist encoded audio bytes to disk and return the resulting file path.
        """
        if isinstance(payload, memoryview):
            payload = payload.tobytes()
        digest = hashlib.sha1(payload).hexdigest()
        target_path = os.path.join(self.inline_audio_cache_dir, f"{digest}.bin")
        if not os.path.exists(target_path):
            tmp_path = f"{target_path}.tmp-{os.getpid()}-{random.randint(0, 1_000_000)}"
            with open(tmp_path, "wb") as handle:
                handle.write(payload)
            os.replace(tmp_path, target_path)
        return target_path

    def _write_inline_array(
        self, array_data: Any, sampling_rate: Optional[int]
    ) -> Optional[str]:
        """
        Persist a numpy/list audio array to disk as a WAV file and return its path.
        """
        if array_data is None:
            return None
        if not isinstance(array_data, np.ndarray):
            array = np.array(array_data, dtype=np.float32)
        else:
            array = array_data.astype(np.float32, copy=False)
        if array.size == 0:
            return None

        if array.ndim == 2:
            if array.shape[1] == 1:
                array = array[:, 0]
            elif self.mono and array.shape[1] > 1:
                array = array.mean(axis=1, dtype=np.float32)
        elif array.ndim > 2:
            array = array.reshape(-1)
        array = np.ascontiguousarray(array, dtype=np.float32)

        try:
            sr_value = (
                int(sampling_rate)
                if sampling_rate is not None
                else int(self.sample_rate)
            )
        except (TypeError, ValueError):
            sr_value = int(self.sample_rate)
        if sr_value <= 0:
            sr_value = int(self.sample_rate)

        digest = hashlib.sha1(
            array.tobytes() + str(sr_value).encode("utf-8")
        ).hexdigest()
        target_path = os.path.join(
            self.inline_audio_cache_dir, f"{digest}_{sr_value}.wav"
        )
        if not os.path.exists(target_path):
            tmp_path = f"{target_path}.tmp-{os.getpid()}-{random.randint(0, 1_000_000)}"
            soundfile.write(tmp_path, array, sr_value)
            os.replace(tmp_path, target_path)
        return target_path

    def _materialize_inline_audio(self, blob: Any) -> Optional[str]:
        """
        Convert inline audio payloads (arrays/bytes) to cached files and return their path.
        """
        if blob is None:
            return None

        sampling_rate = self.sample_rate
        array_candidate: Optional[Any] = None

        if isinstance(blob, Mapping):
            sr_value = blob.get("sampling_rate")
            if sr_value is not None:
                sampling_rate = sr_value
            if "array" in blob and blob["array"] is not None:
                array_candidate = blob["array"]
            elif isinstance(blob.get("waveform"), (list, tuple, np.ndarray)):
                array_candidate = blob["waveform"]
            elif isinstance(blob.get("samples"), (list, tuple, np.ndarray)):
                array_candidate = blob["samples"]
            elif isinstance(blob.get("values"), (list, tuple, np.ndarray)):
                array_candidate = blob["values"]

            bytes_value = blob.get("bytes")
            if isinstance(bytes_value, (bytes, bytearray, memoryview)):
                return self._write_inline_bytes(bytes_value)
        elif isinstance(blob, np.ndarray):
            array_candidate = blob
        elif isinstance(blob, (list, tuple)):
            array_candidate = blob
        elif isinstance(blob, (bytes, bytearray, memoryview)):
            return self._write_inline_bytes(blob)

        if array_candidate is None:
            return None

        return self._write_inline_array(array_candidate, sampling_rate)

    def _flatten_transcript_text(self, transcript: Any) -> str:
        """
        Normalize and flatten transcript content into a single string for validation.
        Supports raw strings, dict segments, and sequences of segments.
        """
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
                flattened = self._flatten_transcript_text(segment)
                if flattened:
                    parts.append(flattened)
            return " ".join(parts).strip()

        return normalize_text(str(transcript))

    def _enforce_char_limit(self, text: str, *, message_prefix: str) -> bool:
        """
        Ensure `text` stays within the global transcript length cap; log a message when exceeded.
        """
        if len(text) <= MAX_TRANSCRIPT_CHAR_LIMIT:
            return True
        preview = _preview_text(text)
        print(
            f"{message_prefix} transcript has {len(text)} chars "
            f"(limit {MAX_TRANSCRIPT_CHAR_LIMIT}). Preview: '{preview}'"
        )
        return False

    def _check_transcript_limits(
        self,
        transcript: Any,
        *,
        audio_path: Optional[str] = None,
        dataset_source: Optional[str] = None,
        enforce_dataset_bounds: bool = False,
    ) -> bool:
        """
        Validate transcript character length, optionally enforcing dataset-level min/max sentence bounds.
        Returns True when the transcript passes all checks; otherwise logs a warning and returns False.
        """
        cleaned_text = self._flatten_transcript_text(transcript)
        if not cleaned_text:
            return False

        char_length = len(cleaned_text)

        def format_context() -> str:
            parts: List[str] = []
            if dataset_source:
                parts.append(str(dataset_source))
            if audio_path:
                basename = os.path.basename(str(audio_path))
                parts.append(basename or str(audio_path))
            return f" [{' | '.join(parts)}]" if parts else ""

        context = format_context()
        preview = _preview_text(cleaned_text)

        if enforce_dataset_bounds and char_length < self.min_sentence:
            return False
        if (
            enforce_dataset_bounds
            and self.max_sentence != -1
            and char_length > self.max_sentence
        ):
            print(
                f"⚠️  Dropping entry{context}: transcript has {char_length} characters "
                f"(dataset max_sentence limit {self.max_sentence}). Preview: '{preview}'"
            )
            return False

        if not self._enforce_char_limit(
            cleaned_text, message_prefix=f"⚠️  Dropping entry{context}:"
        ):
            return False

        return True

    def _extract_transcript_source(self, sample: Dict[str, Any]) -> Any:
        """
        Determine the most appropriate transcript-like field from a dataset sample.
        Handles both plain text and timestamped segmentation formats.
        """
        if not isinstance(sample, dict):
            return None

        if self.timestamps:
            return sample.get("sentences") or sample.get("text")

        for key in ("sentence", "text", "transcription", "transcript", "label"):
            value = sample.get(key)
            if value:
                return value

        return sample.get("sentences")

    @staticmethod
    def _resolve_audio_blob(
        blob: Any,
    ) -> Tuple[Optional[str], Optional[float], Optional[float]]:
        """
        Normalize various audio representations to a filesystem path and optional timing metadata.
        """
        start_time = None
        end_time = None

        if isinstance(blob, Mapping):
            start_time = blob.get("start_time")
            end_time = blob.get("end_time")
            path_candidates = (
                blob.get("path"),
                blob.get("audio_path"),
                blob.get("audio_filepath"),
                blob.get("filepath"),
                blob.get("filename"),
                blob.get("file"),
                blob.get("path_or_url"),
            )
            for candidate in path_candidates:
                if candidate is None:
                    continue
                if isinstance(candidate, (str, bytes)):
                    return str(candidate), start_time, end_time
                if hasattr(os, "PathLike") and isinstance(candidate, os.PathLike):
                    return os.fspath(candidate), start_time, end_time
                try:
                    return str(candidate), start_time, end_time
                except Exception:
                    continue
            return None, start_time, end_time

        if isinstance(blob, (str, bytes)):
            return str(blob), start_time, end_time

        if hasattr(os, "PathLike") and isinstance(blob, os.PathLike):
            return os.fspath(blob), start_time, end_time

        return None, start_time, end_time

    @staticmethod
    def _extract_audio_reference(sample: Dict[str, Any]) -> Optional[str]:
        """
        Extract a representative audio path reference from a dataset sample when available.
        """
        if not isinstance(sample, dict):
            return None

        for key in _HF_AUDIO_CANDIDATE_KEYS:
            candidate_blob = sample.get(key)
            if candidate_blob is None:
                continue
            path, _, _ = CustomDataset._resolve_audio_blob(candidate_blob)
            if path:
                return path

        return None

    def _hf_item_is_valid(
        self, sample: Dict[str, Any], dataset_source: Optional[str]
    ) -> bool:
        """
        Validate a Hugging Face dataset sample against duration and transcript limits.
        """
        if not isinstance(sample, dict):
            return False

        duration = self._coerce_optional_float(sample.get("duration"))
        if duration is not None:
            if duration < self.min_duration:
                return False
            if self.max_duration != -1 and duration > self.max_duration:
                return False

        transcript_source = self._extract_transcript_source(sample)
        if transcript_source is None:
            return False

        audio_reference = self._extract_audio_reference(sample)

        return self._check_transcript_limits(
            transcript_source,
            audio_path=audio_reference,
            dataset_source=dataset_source,
            enforce_dataset_bounds=True,
        )

    def _collect_valid_hf_indices(
        self,
        ds_obj: HFDataset,
        split_name_label: str,
        *,
        base_indices: Optional[List[int]] = None,
    ) -> Optional[List[int]]:
        """
        Evaluate HF dataset entries once during load time and collect indices that pass validation.
        Returns None when no items are dropped so the entire split can be used directly.
        """
        label = split_name_label or "<unnamed>"

        try:
            scan_dataset = ds_obj
            for audio_col in _HF_AUDIO_CANDIDATE_KEYS:
                if audio_col not in scan_dataset.column_names:
                    continue
                try:
                    scan_dataset = scan_dataset.cast_column(
                        audio_col, Audio(decode=False)
                    )
                except Exception:
                    continue
        except Exception:
            scan_dataset = ds_obj

        if base_indices is None:
            index_selection = list(range(len(ds_obj)))
        else:
            index_selection = list(base_indices)

        total_candidates = len(index_selection)
        if total_candidates == 0:
            raise ValueError(f"No samples available to validate in split '{label}'.")

        print(
            f"  Pre-filtering split '{label}': validating {total_candidates} samples prior to training."
        )

        candidate_ds = (
            scan_dataset
            if base_indices is None
            else scan_dataset.select(index_selection)
        )
        candidate_ds = candidate_ds.add_column(ORIGINAL_INDEX_COLUMN, index_selection)

        validator = functools.partial(self._hf_item_is_valid, dataset_source=label)
        num_proc = None
        if total_candidates > 500 and getattr(self, "num_proc", None):
            try:
                import cloudpickle

                cloudpickle.dumps(validator)
                num_proc = self.num_proc
            except Exception:
                num_proc = None

        try:
            filtered_ds = candidate_ds.filter(
                validator,
                batched=False,
                with_indices=False,
                num_proc=num_proc,
                desc=f"Validating {label}",
            )
        except Exception as filter_error:
            print(
                f"  HF fast filter failed ({filter_error}); falling back to Python validation for split '{label}'."
            )
            valid_indices: List[int] = []
            rejected = 0
            iterator = tqdm(index_selection, desc=f"Validating {label}", unit="samples")
            for original_idx in iterator:
                try:
                    sample = scan_dataset[original_idx]
                except Exception:
                    rejected += 1
                    continue

                if self._hf_item_is_valid(sample, dataset_source=label):
                    valid_indices.append(original_idx)
                else:
                    rejected += 1

            if rejected:
                kept = len(valid_indices)
                print(
                    f"  Validation for split '{label}' kept {kept}/{total_candidates} samples "
                    f"({rejected} removed)."
                )
            else:
                print(
                    f"  Validation for split '{label}' kept all {total_candidates} samples."
                )

            if not valid_indices:
                raise ValueError(
                    f"All samples were filtered out during validation for split '{label}'."
                )

            if base_indices is None and rejected == 0:
                return None
            if base_indices is not None and rejected == 0:
                return index_selection
            return valid_indices

        kept = len(filtered_ds)
        rejected = total_candidates - kept

        if kept == 0:
            raise ValueError(
                f"All samples were filtered out during validation for split '{label}'."
            )

        if rejected:
            print(
                f"  Validation for split '{label}' kept {kept}/{total_candidates} samples "
                f"({rejected} removed)."
            )
        else:
            print(
                f"  Validation for split '{label}' kept all {total_candidates} samples."
            )

        if rejected == 0:
            if base_indices is None:
                return None
            return index_selection

        valid_indices = list(filtered_ds[ORIGINAL_INDEX_COLUMN])
        return valid_indices

    def _apply_hf_sampling(
        self,
        ds_obj: HFDataset,
        split_label: str,
        candidate_indices: Optional[List[int]],
        sampling_cfg: Dict[str, Any],
    ) -> Optional[List[int]]:
        """
        Apply percentage-based random sampling to a HuggingFace dataset split.
        """
        percentage = float(sampling_cfg.get("percentage", 100))
        if percentage >= 100:
            return candidate_indices
        if percentage <= 0:
            raise ValueError(
                f"Percentage for HF sampling on split '{split_label}' must be > 0."
            )
        if not hasattr(ds_obj, "__len__"):
            raise ValueError(
                f"Cannot apply percentage sampling to non-indexable dataset split '{split_label}'."
            )

        pool = (
            list(range(len(ds_obj)))
            if candidate_indices is None
            else list(candidate_indices)
        )
        total = len(pool)
        if total == 0:
            return candidate_indices

        target = int(total * (percentage / 100.0))
        if target <= 0:
            target = 1
        if target >= total:
            return candidate_indices

        seed_override = sampling_cfg.get("seed")
        seed_value = (
            int(seed_override)
            if seed_override is not None
            else DEFAULT_HF_SAMPLING_SEED
        )

        rng = random.Random(seed_value)
        selected = rng.sample(pool, target)
        selected.sort()
        print(
            f"  Sampling {target}/{total} entries (~{percentage:.2f}%) from split '{split_label}' with seed {seed_value}."
        )
        return selected

    def _ensure_hf_arrow_dataset(self, split_entry: Dict[str, Any]) -> Any:
        """
        Ensure a Hugging Face dataset entry uses Arrow formatting when no explicit
        columns were previously selected, and remember which columns are exposed.
        """
        ds_obj = split_entry.get("dataset")
        if not isinstance(ds_obj, HFDataset):
            return ds_obj

        format_dict = getattr(ds_obj, "format", None) or {}
        existing_columns = format_dict.get("columns")
        if existing_columns:
            # Respect caller-defined formatting but remember the visible columns.
            if "hf_allowed_columns" not in split_entry:
                try:
                    split_entry["hf_allowed_columns"] = list(existing_columns)
                except TypeError:
                    split_entry["hf_allowed_columns"] = list(existing_columns)
            return ds_obj

        allowed_columns = [
            col for col in ds_obj.column_names if col in _HF_ALLOWED_COLUMNS
        ]
        if not allowed_columns:
            allowed_columns = list(ds_obj.column_names)

        needs_update = (
            format_dict.get("type") != "arrow"
            or format_dict.get("output_all_columns", True)
            or not existing_columns
        )

        if needs_update:
            try:
                ds_obj = ds_obj.with_format(
                    type="arrow",
                    columns=allowed_columns,
                    output_all_columns=False,
                )
            except Exception:
                # Leave dataset unchanged if Arrow formatting is unavailable.
                split_entry.setdefault("hf_allowed_columns", allowed_columns)
                return split_entry.get("dataset", ds_obj)
            split_entry["dataset"] = ds_obj

        split_entry["hf_allowed_columns"] = allowed_columns
        return ds_obj

    def _hf_item_to_python(
        self, item: Any, allowed_columns: Optional[Sequence[str]]
    ) -> Any:
        """
        Convert a Hugging Face dataset item that may be backed by Arrow objects
        into standard Python types and drop unformatted columns.
        """
        if pa is not None:
            if isinstance(item, pa.StructScalar):
                item = item.as_py()
            elif isinstance(item, pa.Table):
                rows = item.to_pylist()
                item = rows[0] if len(rows) == 1 else rows
            elif isinstance(item, pa.RecordBatch):
                rows = item.to_pylist()
                item = rows[0] if len(rows) == 1 else rows
            elif isinstance(item, pa.Array):
                item = item.to_pylist()

        if isinstance(item, Mapping):
            item = dict(item)

        if allowed_columns and isinstance(item, dict):
            item = {key: item[key] for key in allowed_columns if key in item}
        return item

    def _validate_entry(
        self, entry: ManifestEntry, dataset_source: Optional[str] = None
    ) -> bool:
        if not entry.audio_path:
            return False

        duration = entry.duration
        if duration is not None:
            if duration < self.min_duration:
                return False
            if self.max_duration != -1 and duration > self.max_duration:
                return False

        transcript_len = self._transcript_length(entry)
        if transcript_len < self.min_sentence:
            return False
        if self.max_sentence != -1 and transcript_len > self.max_sentence:
            return False
        transcript_source = entry.sentence if entry.sentence else entry.sentences
        if not self._check_transcript_limits(
            transcript_source,
            audio_path=entry.audio_path,
            dataset_source=dataset_source,
            enforce_dataset_bounds=False,
        ):
            return False

        # Require at least one textual source
        if not entry.sentence and entry.sentences is None:
            return False

        return True

    def _manifest_from_mapping(self, row: dict) -> Optional[ManifestEntry]:
        if not isinstance(row, dict):
            return None

        audio_path = None
        start_time = None
        end_time = None

        for key in _HF_AUDIO_CANDIDATE_KEYS:
            candidate_blob = row.get(key)
            if candidate_blob is None:
                continue
            candidate_path, candidate_start, candidate_end = self._resolve_audio_blob(
                candidate_blob
            )
            if candidate_path is None:
                candidate_path = self._materialize_inline_audio(candidate_blob)
            if candidate_path:
                audio_path = candidate_path
                start_time = candidate_start
                end_time = candidate_end
                break

        if not audio_path:
            for candidate_blob in row.values():
                candidate_path = self._materialize_inline_audio(candidate_blob)
                if candidate_path:
                    audio_path = candidate_path
                    break

        if not audio_path:
            return None

        sentences = row.get("sentences")
        sentence = row.get("sentence")
        if sentence is None:
            for text_key in ("text", "transcription", "transcript", "label"):
                if text_key in row and row[text_key]:
                    sentence = row[text_key]
                    break

        duration = self._coerce_optional_float(row.get("duration"))
        if duration is not None and duration < 0:
            duration = None

        start_time = self._coerce_optional_float(start_time)
        end_time = self._coerce_optional_float(end_time)

        language = row.get("language")

        normalized_sentence = normalize_text(sentence) if sentence is not None else None

        return ManifestEntry(
            audio_path=str(audio_path),
            sentence=normalized_sentence,
            sentences=sentences,
            duration=duration,
            language=language,
            start_time=start_time,
            end_time=end_time,
        )

    def _try_store_manifest(self, row: dict, dataset_path: str = None) -> None:
        entry = self._manifest_from_mapping(row)
        if entry and self._validate_entry(entry, dataset_source=dataset_path):
            # Track and log first 2 labels from each dataset for debugging
            if dataset_path:
                dataset_key = (
                    os.path.basename(dataset_path) if dataset_path else "unknown"
                )
                if dataset_key not in self._dataset_label_counts:
                    self._dataset_label_counts[dataset_key] = 0
                    self._label_examples[dataset_key] = []

                self._dataset_label_counts[dataset_key] += 1

                # Log first 2 labels from each dataset
                if len(self._label_examples[dataset_key]) < 2:
                    label_text = entry.sentence or ""
                    label_length = len(label_text)
                    # Also check token length if processor is available
                    token_length = "N/A"
                    if self.processor and label_text:
                        try:
                            tokens = self.processor.tokenizer.encode(
                                label_text, add_special_tokens=False
                            )
                            token_length = len(tokens)
                        except:
                            pass

                    self._label_examples[dataset_key].append(
                        {
                            "text": label_text[:100] + "..."
                            if len(label_text) > 100
                            else label_text,
                            "char_length": label_length,
                            "token_length": token_length,
                        }
                    )

                    print(
                        f"📝 Dataset '{dataset_key}' label example #{len(self._label_examples[dataset_key])}:"
                    )
                    print(
                        f"   Text: '{label_text[:100]}{'...' if len(label_text) > 100 else ''}'"
                    )
                    print(f"   Character length: {label_length}")
                    print(f"   Token length: {token_length}")
                    if label_length > 1000 or (
                        isinstance(token_length, int) and token_length > 400
                    ):
                        print(
                            f"   ⚠️  WARNING: Label seems very long! This might indicate TSV parsing issues."
                        )

            self.data_list.append(entry)

    # Load data list
    def _load_data_list(self):
        # Support multiple dataset paths separated by '+'
        data_paths = self.data_list_path.split("+")
        self.data_list = []

        for data_path in data_paths:
            data_path = data_path.strip()  # Remove any whitespace

            # Parse path:subset format for HuggingFace datasets
            dataset_subset = None
            is_hf_scheme = isinstance(data_path, str) and data_path.startswith("hf://")
            if ":" in data_path and not data_path.startswith("http"):
                # Check if this is a path:subset format (not a URL or Windows drive)
                parts = data_path.rsplit(":", 1)
                if len(parts) == 2 and not parts[1].startswith("//"):
                    potential_path, potential_subset = parts
                    # If local path exists, treat as path:subset on disk; otherwise allow HF repo specs
                    if os.path.exists(potential_path):
                        data_path = potential_path
                        dataset_subset = potential_subset
                    elif is_hf_scheme or (
                        "/" in potential_path and not os.path.exists(potential_path)
                    ):
                        # hf://org/name:split or org/name:split -> load from HF hub
                        data_path = potential_path
                        dataset_subset = potential_subset

            # Check if it's a directory or HF Hub spec - treat as HuggingFace datasets
            if (
                os.path.isdir(data_path)
                or is_hf_scheme
                or ("/" in data_path and not os.path.exists(data_path))
            ):
                try:
                    self._load_huggingface_dataset(data_path, dataset_subset)
                except Exception as e:
                    print(
                        f"Warning: Failed to load {data_path} as HuggingFace dataset: {e}"
                    )
                    print(f"Skipping directory: {data_path}")
            elif data_path.endswith(".header"):
                # Get binary data list
                dataset_reader = DatasetReader(
                    data_header_path=data_path,
                    min_duration=self.min_duration,
                    max_duration=self.max_duration,
                )
                current_data_list = dataset_reader.get_keys()
                self.data_list.extend(current_data_list)
            elif data_path.endswith(".tsv"):
                self._load_csv_data(data_path, delimiter="\t")
            elif data_path.endswith(".csv"):
                # Load CSV file
                self._load_csv_data(data_path)
            else:
                # Get data list from JSON/JSONL
                filter_config = self._get_filter_config_for_path(data_path)
                filter_fn = None
                dataset_name = os.path.basename(data_path)
                if filter_config:
                    filter_fn = filter_config["filter_fn"]
                    print(
                        f"  Found filter for dataset '{filter_config['name']}' (matched '{data_path}')"
                    )
                else:
                    print(f"  No dataset-specific filter configured for '{data_path}'")

                with open(data_path, "r", encoding="utf-8") as f:
                    for line_idx, raw_line in enumerate(
                        tqdm(
                            f, desc=f"Reading data list from {data_path}", unit="lines"
                        ),
                        start=1,
                    ):
                        line_str = raw_line.strip()
                        if not line_str:
                            continue
                        try:
                            line = json.loads(line_str)
                        except json.JSONDecodeError as exc:
                            raise ValueError(
                                f"Failed to parse JSON entry at line {line_idx} in {data_path}"
                            ) from exc
                        if not isinstance(line, dict):
                            continue

                        # Handle text/transcription columns - use 'text' as fallback
                        if "sentence" not in line and "sentences" not in line:
                            if "text" in line:
                                line["sentence"] = line["text"]
                            elif "transcription" in line:
                                line["sentence"] = line["transcription"]
                            elif "transcript" in line:
                                line["sentence"] = line["transcript"]

                        # Skip audio that exceeds duration limits
                        if line["duration"] < self.min_duration:
                            continue
                        if (
                            self.max_duration != -1
                            and line["duration"] > self.max_duration
                        ):
                            continue
                        # Skip audio that exceeds sentence character count limits
                        if "sentence" in line.keys():
                            if (
                                len(line["sentence"]) < self.min_sentence
                                or len(line["sentence"]) > self.max_sentence
                            ):
                                continue
                        elif "sentences" in line.keys():
                            sentence_len = 0
                            for s in line["sentences"]:
                                if isinstance(s, dict):
                                    sentence_len += len(s.get("text", ""))
                                else:
                                    sentence_len += len(str(s))
                            if (
                                sentence_len < self.min_sentence
                                or sentence_len > self.max_sentence
                            ):
                                continue

                        # Apply custom filter if available for JSON/JSONL data
                        if filter_fn and not filter_fn(line):
                            continue

                        self._try_store_manifest(line, data_path)

    def _load_huggingface_dataset(self, data_path, dataset_subset=None):
        """Load data from a Hugging Face dataset folder."""
        print(
            f"Loading Hugging Face dataset from {data_path}"
            + (f" (subset: {dataset_subset})" if dataset_subset else "")
        )

        filter_config = self._get_filter_config_for_path(data_path)
        filter_fn = None
        if filter_config:
            filter_fn = filter_config["filter_fn"]
            print(
                f"  Found filter for dataset '{filter_config['name']}' (matched '{data_path}')"
            )
        else:
            print(f"  No dataset-specific filter configured for '{data_path}'")

        try:
            # Load the dataset (supports HF Hub refs via cache or local saved datasets)
            if os.path.isdir(data_path):
                dataset = load_from_disk(data_path)
            else:
                # Treat as HF Hub repo; allow:
                # - hf://org/name:split
                # - hf://org/name@revision#subset:split
                # - org/name:split
                # - org/name@revision#subset:split
                repo_spec = (
                    data_path[5:]
                    if isinstance(data_path, str) and data_path.startswith("hf://")
                    else data_path
                )
                # Split off optional split (already provided via dataset_subset when parsed earlier)
                # Parse repo@revision#subset
                repo = repo_spec
                revision = None
                subset_name = None
                if "@" in repo and "#" in repo:
                    repo_part, rest = repo.split("@", 1)
                    rev_part, subset_name = rest.split("#", 1)
                    repo, revision = repo_part, rev_part
                elif "@" in repo:
                    repo, revision = repo.split("@", 1)
                elif "#" in repo:
                    repo, subset_name = repo.split("#", 1)

                adj_revision = revision

                # Special handling for google/fleurs dataset
                if repo == "google/fleurs" and subset_name:
                    print(
                        f"  Special handling for google/fleurs with subset {subset_name}"
                    )

                    # Download the specific files for this subset
                    cache_dir = os.getenv(
                        "HF_DATASETS_CACHE", None
                    ) or os.path.expanduser("~/.cache/huggingface/datasets")
                    local_dir = os.path.join(cache_dir, "google_fleurs", subset_name)

                    # Download the tar.gz and tsv files for the subset
                    print(f"  Downloading google/fleurs files to {local_dir}")
                    snapshot_download(
                        repo_id="google/fleurs",
                        repo_type="dataset",
                        allow_patterns=[
                            f"data/{subset_name}/audio/*.tar.gz",
                            f"data/{subset_name}/*.tsv",
                        ],
                        local_dir=local_dir,
                    )

                    # Process the downloaded files
                    data_dir = os.path.join(local_dir, "data", subset_name)
                    audio_dir = os.path.join(data_dir, "audio")

                    # Extract tar.gz files if they exist
                    for tar_file in (
                        os.listdir(audio_dir) if os.path.exists(audio_dir) else []
                    ):
                        if tar_file.endswith(".tar.gz"):
                            tar_path = os.path.join(audio_dir, tar_file)
                            extract_dir = os.path.join(
                                audio_dir, tar_file.replace(".tar.gz", "")
                            )

                            # Check if already extracted by looking for audio files
                            if (
                                not os.path.exists(extract_dir)
                                or not any(
                                    f.endswith((".wav", ".mp3", ".flac"))
                                    for f in os.listdir(extract_dir)
                                    if os.path.isfile(os.path.join(extract_dir, f))
                                )
                                if os.path.exists(extract_dir)
                                else True
                            ):
                                print(f"  Extracting {tar_file} to {extract_dir}")
                                os.makedirs(extract_dir, exist_ok=True)
                                # Extract with lower memory usage
                                with tarfile.open(tar_path, "r:gz") as tar:
                                    # Extract files one by one to avoid memory issues
                                    for member in tar:
                                        tar.extract(member, extract_dir)

                    # Create a generator for TSV files to avoid loading all in memory
                    def fleurs_generator():
                        # Determine which TSV splits to process
                        if dataset_subset:
                            base = str(dataset_subset).strip().lower()
                            split_candidates = [base]
                            # Allow common aliasing between "dev" and "validation"
                            if base == "dev" and "validation" not in split_candidates:
                                split_candidates.append("validation")
                            if base == "validation" and "dev" not in split_candidates:
                                split_candidates.append("dev")
                        else:
                            # No split specified: iterate common splits
                            split_candidates = ["train", "validation", "test"]

                        # Build a recursive index of audio files inside extracted tar folders and split subdirs
                        path_index: Dict[str, str] = {}
                        noext_index: Dict[str, str] = {}
                        for root, _, files in os.walk(audio_dir):
                            for fn in files:
                                if fn.lower().endswith(
                                    (".wav", ".mp3", ".flac", ".m4a", ".ogg")
                                ):
                                    full_path = os.path.join(root, fn)
                                    # index by exact filename
                                    path_index[fn] = full_path
                                    # also index by name without extension
                                    base_no_ext, _ = os.path.splitext(fn)
                                    noext_index.setdefault(base_no_ext, full_path)

                        def resolve_audio(filename: str) -> Optional[str]:
                            if not filename:
                                return None
                            normalized = str(filename).strip().replace("\\", os.sep)

                            # Try direct relative path under audio_dir
                            direct = os.path.join(audio_dir, normalized)
                            if os.path.exists(direct):
                                return direct

                            # Try basename match in recursive index
                            base = os.path.basename(normalized)
                            if base in path_index:
                                return path_index[base]

                            # Try without extension variants
                            base_no_ext, ext = os.path.splitext(base)
                            if not ext:
                                # try common suffixes
                                for suf in (".wav", ".mp3", ".flac", ".m4a", ".ogg"):
                                    cand = base_no_ext + suf
                                    if cand in path_index:
                                        return path_index[cand]
                            # fallback: noext index
                            if base_no_ext in noext_index:
                                return noext_index[base_no_ext]

                            return None

                        missing_logged: Set[str] = set()

                        for split in split_candidates:
                            tsv_file = os.path.join(data_dir, f"{split}.tsv")
                            if not os.path.exists(tsv_file):
                                continue
                            print(f"  Processing TSV file: {tsv_file}")
                            with open(tsv_file, "r", encoding="utf-8", newline="") as f:
                                header_keys, header_lookup, dict_rows = (
                                    iter_tsv_dict_rows(f)
                                )
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

                                def _pick_text_column(
                                    existing_audio_col: Optional[str],
                                ) -> Optional[str]:
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
                                    [
                                        "path",
                                        "filename",
                                        "file",
                                        "audio",
                                        "audio_filename",
                                    ],
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
                                    default_idx=2
                                    if len(header_keys) > 2
                                    else len(header_keys) - 1,
                                )

                                def _looks_like_audio_path(value: str) -> bool:
                                    if not value:
                                        return False
                                    lowered = str(value).lower()
                                    audio_exts = (
                                        ".wav",
                                        ".mp3",
                                        ".flac",
                                        ".m4a",
                                        ".ogg",
                                    )
                                    return lowered.endswith(audio_exts)

                                resolved_audio_col = _peek_resolved_audio_column()
                                if (
                                    resolved_audio_col
                                    and resolved_audio_col in header_keys
                                ):
                                    audio_column = resolved_audio_col

                                header_is_synthetic = all(
                                    key.startswith("column_")
                                    and header_lookup.get(key) == key
                                    for key in header_keys
                                )
                                if header_is_synthetic:
                                    first_col_value = (
                                        first_row.get(header_keys[0], "")
                                        if header_keys
                                        else ""
                                    )
                                    second_col_value = (
                                        first_row.get(header_keys[1], "")
                                        if len(header_keys) > 1
                                        else ""
                                    )
                                    third_col_value = (
                                        first_row.get(header_keys[2], "")
                                        if len(header_keys) > 2
                                        else ""
                                    )
                                    # google/fleurs TSV exports sometimes omit the header row.
                                    # The first column is an ID, the second is the audio path, and the third is text.
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
                                    if (
                                        len(header_keys) > 1
                                        and audio_column == header_keys[0]
                                    ):
                                        audio_column = header_keys[1]
                                    if len(header_keys) > 2:
                                        if (
                                            text_column is None
                                            or text_column == audio_column
                                        ):
                                            text_column = header_keys[2]
                                    elif (
                                        text_column == audio_column
                                        and len(header_keys) > 1
                                    ):
                                        text_column = header_keys[1]

                                if text_column == audio_column:
                                    text_column = _pick_text_column(audio_column)

                                if audio_column is None or text_column is None:
                                    print(
                                        f"  Warning: could not detect audio/text columns in {tsv_file}. Header: {list(header_lookup.values())}"
                                    )
                                    continue

                                skipped_long = 0

                                for line_number, row_dict in dict_rows:
                                    audio_filename = (
                                        row_dict.get(audio_column) or ""
                                    ).strip()
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
                                    if not self._enforce_char_limit(
                                        transcription,
                                        message_prefix=(
                                            f"  Dropping line {line_number} in "
                                            f"{os.path.basename(tsv_file)}:"
                                        ),
                                    ):
                                        skipped_long += 1
                                        continue

                                    audio_path = resolve_audio(audio_filename)
                                    if audio_path:
                                        yield {
                                            "audio": {"path": audio_path},
                                            "text": transcription,
                                        }
                                    elif audio_filename not in missing_logged:
                                        print(
                                            f"  Warning (line {line_number}): audio file '{audio_filename}' not found under {audio_dir}"
                                        )
                                        missing_logged.add(audio_filename)

                                if skipped_long:
                                    print(
                                        f"  Skipped {skipped_long} entries in {os.path.basename(tsv_file)} due to transcripts longer than {MAX_TRANSCRIPT_CHAR_LIMIT} characters."
                                    )

                    # Use the generator instead of a list
                    dataset = fleurs_generator()

                elif repo == "mozilla-foundation/common_voice_17_0" and subset_name:
                    print(
                        f"  Special handling for mozilla-foundation/common_voice_17_0 with subset {subset_name}"
                    )

                    # Download the specific files for this subset
                    cache_dir = os.getenv(
                        "HF_DATASETS_CACHE", None
                    ) or os.path.expanduser("~/.cache/huggingface/datasets")
                    local_dir = os.path.join(
                        cache_dir, "mozilla_common_voice", subset_name
                    )

                    # Download the audio and transcript files for the subset
                    print(
                        f"  Downloading mozilla-foundation/common_voice_17_0 files to {local_dir}"
                    )
                    snapshot_download(
                        repo_id="mozilla-foundation/common_voice_17_0",
                        repo_type="dataset",
                        allow_patterns=[
                            f"audio/{subset_name}/*",
                            f"transcript/{subset_name}/*",
                        ],
                        local_dir=local_dir,
                    )

                    # Process the downloaded files
                    audio_dir = os.path.join(local_dir, "audio", subset_name)
                    transcript_dir = os.path.join(local_dir, "transcript", subset_name)
                    audio_dir_abs = os.path.abspath(audio_dir)

                    tar_root_to_subdirs: Dict[str, Set[str]] = {}

                    def _record_tar_stem(
                        root_path: str, tar_name: str
                    ) -> tuple[str, str]:
                        root_abs = os.path.abspath(root_path)
                        if tar_name.endswith(".tar.gz"):
                            stem = tar_name[: -len(".tar.gz")]
                        elif tar_name.endswith(".tar"):
                            stem = tar_name[: -len(".tar")]
                        else:
                            stem = os.path.splitext(tar_name)[0]
                        tar_root_to_subdirs.setdefault(root_abs, set()).add(stem)
                        return root_abs, stem

                    def _safe_extract_tarball(root_path: str, tar_name: str) -> None:
                        root_abs, stem = _record_tar_stem(root_path, tar_name)
                        marker_path = os.path.join(root_abs, f".{stem}_extracted")
                        if os.path.exists(marker_path):
                            return
                        tar_path = os.path.join(root_abs, tar_name)
                        print(f"  Extracting {tar_name} to {root_abs}")

                        # Open with appropriate mode for better performance
                        mode = "r:gz" if tar_name.endswith(".tar.gz") else "r:"
                        with tarfile.open(tar_path, mode) as tar:
                            # Pre-validate all paths in batch for security
                            safe_members = []
                            for member in tar:
                                target_path = os.path.abspath(
                                    os.path.join(root_abs, member.name)
                                )
                                if (
                                    os.path.commonpath([root_abs, target_path])
                                    != root_abs
                                ):
                                    raise RuntimeError(
                                        f"Blocked path traversal while extracting {tar_name}"
                                    )
                                safe_members.append(member)

                            # Extract all validated members at once - much faster than individual extracts
                            tar.extractall(root_abs, members=safe_members)

                        with open(marker_path, "w", encoding="utf-8") as f:
                            f.write("extracted")

                    if os.path.isdir(audio_dir_abs):
                        for current_dir, _, files in os.walk(audio_dir_abs):
                            current_abs = os.path.abspath(current_dir)
                            for tar_file in files:
                                if tar_file.endswith((".tar", ".tar.gz")):
                                    _safe_extract_tarball(current_abs, tar_file)

                    # Create a generator for Common Voice TSV files to avoid loading all in memory
                    def common_voice_generator():
                        subset_aliases = {
                            "validation": ["validated"],
                            "validated": ["validation"],
                            "dev": ["development"],
                            "development": ["dev"],
                        }

                        subset_variants: List[str] = []
                        if dataset_subset:
                            base = str(dataset_subset).strip().lower()
                            if base:
                                subset_variants.append(base)
                        else:
                            subset_variants.extend(
                                [
                                    "train",
                                    "validation",
                                    "test",
                                    "dev",
                                    "validated",
                                    "invalidated",
                                    "other",
                                ]
                            )

                        for variant in list(subset_variants):
                            for alias in subset_aliases.get(variant, []):
                                if alias not in subset_variants:
                                    subset_variants.append(alias)

                        candidate_dirs: List[str] = []
                        seen_dirs: Set[str] = set()

                        def _add_candidate(dir_path: str) -> None:
                            abs_path = os.path.abspath(dir_path)
                            if os.path.isdir(abs_path) and abs_path not in seen_dirs:
                                seen_dirs.add(abs_path)
                                candidate_dirs.append(abs_path)

                        _add_candidate(audio_dir_abs)
                        for variant in subset_variants:
                            _add_candidate(os.path.join(audio_dir_abs, variant))

                        for base_dir in list(candidate_dirs):
                            for suffix in tar_root_to_subdirs.get(base_dir, set()):
                                _add_candidate(os.path.join(base_dir, suffix))
                            clips_dir = os.path.join(base_dir, "clips")
                            _add_candidate(clips_dir)

                        if not candidate_dirs:
                            candidate_dirs.append(audio_dir_abs)

                        path_cache: Dict[str, Optional[str]] = {}
                        missing_logged: Set[str] = set()

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
                                for nm in names:
                                    for e in exts:
                                        candidates.append(f"{nm}{e}")

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
                                    for suffix in tar_root_to_subdirs.get(
                                        base_dir, set()
                                    ):
                                        nested_path = os.path.join(
                                            base_dir, suffix, os.path.basename(name)
                                        )
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

                            if resolved:
                                resolved = os.path.abspath(resolved)

                            path_cache[normalized] = resolved
                            return resolved

                        seen_tsv: Set[str] = set()
                        for variant in subset_variants or [
                            "train",
                            "validation",
                            "test",
                        ]:
                            tsv_file = os.path.join(transcript_dir, f"{variant}.tsv")
                            if not os.path.exists(tsv_file) or tsv_file in seen_tsv:
                                continue
                            seen_tsv.add(tsv_file)
                            print(f"  Processing TSV file: {tsv_file}")
                            with open(tsv_file, "r", encoding="utf-8", newline="") as f:
                                header_keys, header_lookup, dict_rows = (
                                    iter_tsv_dict_rows(f)
                                )
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
                                    [
                                        "path",
                                        "audio",
                                        "audio_filename",
                                        "filename",
                                        "file",
                                    ],
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
                                    default_idx=2
                                    if len(header_keys) > 2
                                    else len(header_keys) - 1,
                                )

                                if path_column is None or text_column is None:
                                    print(
                                        f"  Warning: could not detect required columns in {tsv_file}. Header: {list(header_lookup.values())}"
                                    )
                                    continue

                                skipped_long = 0

                                for row_number, row_dict in dict_rows:
                                    audio_filename = (
                                        row_dict.get(path_column) or ""
                                    ).strip()
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
                                    if not self._enforce_char_limit(
                                        transcription,
                                        message_prefix=(
                                            f"  Dropping line {row_number} in "
                                            f"{os.path.basename(tsv_file)}:"
                                        ),
                                    ):
                                        skipped_long += 1
                                        continue

                                    resolved_path = _resolve_audio_path(audio_filename)
                                    if resolved_path:
                                        yield {
                                            "audio": {"path": resolved_path},
                                            "text": transcription,
                                        }
                                    elif audio_filename not in missing_logged:
                                        print(
                                            f"  Warning (line {row_number}): audio file '{audio_filename}' not found under {audio_dir_abs}"
                                        )
                                        missing_logged.add(audio_filename)

                                if skipped_long:
                                    print(
                                        f"  Skipped {skipped_long} rows in {os.path.basename(tsv_file)} due to transcripts longer than {MAX_TRANSCRIPT_CHAR_LIMIT} characters."
                                    )

                    # Use the generator instead of a list
                    dataset = common_voice_generator()

                else:
                    # Original loading logic for other datasets
                    if dataset_subset:
                        print(
                            f"  Loading HF dataset via hub: repo='{repo}' name='{subset_name}' revision='{adj_revision}' split='{dataset_subset}'"
                        )
                        # Use local caching instead of streaming to reduce API calls
                        dataset = rate_limited_request(
                            load_dataset,
                            repo,
                            name=subset_name,
                            revision=adj_revision,
                            split=dataset_subset,
                            streaming=False,  # ensure materialized dataset
                            download_mode="reuse_dataset_if_exists",
                            cache_dir=os.getenv("HF_DATASETS_CACHE", None),
                        )
                    else:
                        # May return a DatasetDict (multiple splits) or a single Dataset
                        print(
                            f"  Loading HF dataset via hub: repo='{repo}' name='{subset_name}' revision='{adj_revision}' (all splits)"
                        )
                        # Use local caching instead of streaming to reduce API calls
                        dataset = rate_limited_request(
                            load_dataset,
                            repo,
                            name=subset_name,
                            revision=adj_revision,
                            streaming=False,  # ensure materialized dataset
                            download_mode="reuse_dataset_if_exists",
                            cache_dir=os.getenv("HF_DATASETS_CACHE", None),
                        )

            # ---- IMPORTANT: Keep audio decoding ENABLED for proper loading ----
            # We want HuggingFace to handle audio loading and decoding for us
            # This ensures audio files are properly downloaded and decoded

            # Build iterator across splits; avoid concatenation/materialization
            if isinstance(dataset, DatasetDict):
                if dataset_subset:
                    if dataset_subset not in dataset:
                        available_subsets = list(dataset.keys())
                        raise ValueError(
                            f"Subset '{dataset_subset}' not found in dataset. "
                            f"Available subsets: {available_subsets}"
                        )
                    split_names = [dataset_subset]
                else:
                    split_names = list(dataset.keys())
                    print(f"No subset specified, iterating over subsets: {split_names}")
                dataset_iter = chain.from_iterable(dataset[sp] for sp in split_names)
            elif isinstance(dataset, list):
                # Handle list datasets (like from google/fleurs)
                dataset_iter = dataset
            elif callable(dataset):
                # Handle generator functions
                dataset_iter = dataset()
            else:
                dataset_iter = dataset

            # NEW: Avoid materializing massive Python dicts; keep HF splits lazily
            def _append_lazy_split(ds_obj, split_name_label: str):
                indices_override: Optional[List[int]] = None
                sampling_cfg = None
                if self.hf_sampling_config:
                    sampling_cfg = self.hf_sampling_config.get(split_name_label)
                    if sampling_cfg is None and split_name_label.startswith("hf://"):
                        sampling_cfg = self.hf_sampling_config.get(split_name_label[5:])

                if filter_fn:
                    print(f"  Applying filter to split: {split_name_label}")
                    # Try to apply filtering within HF Datasets to keep it on-disk
                    safe_num_proc = self.num_proc
                    try:
                        import cloudpickle

                        cloudpickle.dumps(filter_fn)
                    except Exception:
                        safe_num_proc = None

                    try:
                        before = len(ds_obj)
                        ds_obj = ds_obj.filter(
                            filter_fn,
                            batched=False,
                            num_proc=safe_num_proc,
                            desc=f"Filtering {split_name_label}",
                        )
                        after = len(ds_obj)
                        print(
                            f"  Kept {after}/{before} samples after filter in {split_name_label}"
                        )
                    except Exception as e:
                        # Fallback: build a compact list of indices in Python
                        print(
                            f"  Filter via HF failed ({e}); falling back to Python indices for {split_name_label}"
                        )
                        kept_idx = []
                        for i, it in enumerate(
                            tqdm(ds_obj, desc=f"Scanning {split_name_label}")
                        ):
                            try:
                                if filter_fn(it):
                                    kept_idx.append(i)
                            except Exception:
                                # If user filter fails on a row, skip it
                                pass
                        print(
                            f"  Kept {len(kept_idx)}/{len(ds_obj)} samples (indexed) in {split_name_label}"
                        )
                        indices_override = kept_idx

                validated_indices = self._collect_valid_hf_indices(
                    ds_obj, split_name_label, base_indices=indices_override
                )
                if sampling_cfg:
                    validated_indices = self._apply_hf_sampling(
                        ds_obj, split_name_label, validated_indices, sampling_cfg
                    )

                # If we reach here, we have an HF dataset object to use lazily
                self.hf_splits.append(
                    {
                        "dataset": ds_obj,
                        "indices": validated_indices,
                        "name": split_name_label,
                    }
                )

            if isinstance(dataset, DatasetDict):
                # Work per split lazily
                if dataset_subset:
                    if dataset_subset not in dataset:
                        available_subsets = list(dataset.keys())
                        raise ValueError(
                            f"Subset '{dataset_subset}' not found in dataset. Available subsets: {available_subsets}"
                        )
                    split_names = [dataset_subset]
                else:
                    split_names = list(dataset.keys())
                    print(f"No subset specified, using splits lazily: {split_names}")

                for sp in split_names:
                    _append_lazy_split(dataset[sp], f"{data_path}:{sp}")
            elif isinstance(dataset, HFDataset):
                _append_lazy_split(dataset, f"{data_path}")
            elif isinstance(dataset, list) or callable(dataset):
                # For generators or simple lists, fall back to incremental materialization
                dataset_iter = dataset() if callable(dataset) else dataset
                batch_size = 1000
                for idx, item in enumerate(
                    tqdm(dataset_iter, desc=f"Processing HF dataset {data_path}")
                ):
                    if filter_fn and not filter_fn(item):
                        continue
                    self._process_item(item, data_path)
                    if idx > 0 and idx % batch_size == 0:
                        gc.collect()
            else:
                # Unknown type; best effort iteration
                for item in tqdm(
                    dataset_iter, desc=f"Processing HF dataset {data_path}"
                ):
                    if filter_fn and not filter_fn(item):
                        continue
                    self._process_item(item, data_path)

        except Exception as e:
            print(f"Error loading Hugging Face dataset from {data_path}: {e}")
            raise

    def _load_csv_data(self, data_path, delimiter: Optional[str] = None):
        """Load data from CSV file. Supports multiple CSV formats with proper header mapping."""
        filter_config = self._get_filter_config_for_path(data_path)
        filter_fn = None
        if filter_config:
            filter_fn = filter_config["filter_fn"]
            print(
                f"  Found filter for dataset '{filter_config['name']}' (matched '{data_path}')"
            )
        else:
            print(f"  No dataset-specific filter configured for '{data_path}'")

        with open(data_path, "r", encoding="utf-8") as f:
            # Try to detect CSV format by reading first few lines
            sample_lines = []
            for i, line in enumerate(f):
                sample_lines.append(line.strip())
                if i >= 2:  # Read first 3 lines to detect format
                    break

        # Reset file pointer
        with open(data_path, "r", encoding="utf-8", newline="") as f:
            # Detect delimiter and format
            has_header = False
            detected_delimiter = delimiter or ","

            # Check if first line looks like a header
            first_line = sample_lines[0] if sample_lines else ""
            if any(
                keyword in first_line.lower()
                for keyword in [
                    "filename",
                    "file_name",
                    "path",
                    "audio",
                    "text",
                    "sentence",
                    "transcript",
                    "id",
                ]
            ):
                has_header = True

            # Check for pipe delimiter (LJSpeech format)
            if delimiter is None:
                if "|" in first_line and "," not in first_line:
                    detected_delimiter = "|"
                elif "\t" in first_line and "," not in first_line:
                    detected_delimiter = "\t"

            reader = csv.reader(f, delimiter=detected_delimiter)

            # Read and process header if present
            header_mapping = {}
            if has_header:
                header = next(reader)
                print(f"Detected CSV header: {header}")

                # Create mapping from header to column indices
                for idx, col_name in enumerate(header):
                    col_name = col_name.strip().lower()
                    header_mapping[col_name] = idx

                # Define column mappings for different field names
                audio_path_columns = [
                    "file_name",
                    "filename",
                    "path",
                    "audio_path",
                    "audio",
                    "file",
                ]
                text_columns = [
                    "text",
                    "sentence",
                    "transcript",
                    "transcription",
                    "text_latin",
                ]

                # Find the correct columns
                audio_col_idx = None
                text_col_idx = None

                for col in audio_path_columns:
                    if col in header_mapping:
                        audio_col_idx = header_mapping[col]
                        break

                for col in text_columns:
                    if col in header_mapping:
                        text_col_idx = header_mapping[col]
                        break

                if audio_col_idx is None or text_col_idx is None:
                    print(
                        f"Warning: Could not find required columns in header {header}"
                    )
                    print(f"Looking for audio path in: {audio_path_columns}")
                    print(f"Looking for text in: {text_columns}")

            for row in tqdm(reader, desc=f"Reading CSV data from {data_path}"):
                if len(row) < 2:
                    continue

                # Extract filename and text from row based on header mapping
                if (
                    has_header
                    and audio_col_idx is not None
                    and text_col_idx is not None
                ):
                    # Use header-based mapping
                    if len(row) > max(audio_col_idx, text_col_idx):
                        filename = row[audio_col_idx].strip()
                        text = row[text_col_idx].strip()
                    else:
                        print(f"Warning: Row has insufficient columns: {row}")
                        continue
                else:
                    # Fallback to old logic for headerless or unrecognized formats
                    if detected_delimiter == "|":
                        # LJSpeech format: filename|text
                        if "|" in row[0] and len(row) == 1:
                            filename, text = row[0].split("|", 1)
                        else:
                            filename, text = row[0], row[1] if len(row) > 1 else ""
                    else:
                        # Standard CSV format: filename,text or audio_path,transcription
                        filename, text = row[0], row[1]

                # Debug TSV parsing issues - check for suspiciously long text
                if len(text) > 2000:  # Character count threshold
                    print(f"🚨 POTENTIAL TSV PARSING ISSUE:")
                    print(f"   File: {data_path}")
                    print(f"   Row length: {len(row)}")
                    print(f"   Text length: {len(text)} characters")
                    print(f"   Text preview: '{text[:200]}...'")
                    print(
                        f"   This might indicate incorrect delimiter or column mapping"
                    )
                    # Count newlines and tabs in the text to detect multi-line issues
                    newlines = text.count("\n")
                    tabs = text.count("\t")
                    if newlines > 0 or tabs > 10:
                        print(
                            f"   Contains {newlines} newlines and {tabs} tabs - likely parsing error"
                        )

                # Skip empty entries
                if not filename or not text:
                    continue

                # Create line dict in expected format
                sentence = normalize_text(text.strip())
                if not sentence:
                    continue
                if len(sentence) > MAX_TRANSCRIPT_CHAR_LIMIT:
                    preview = _preview_text(sentence)
                    print(
                        f"Skipping row in {data_path}: transcript exceeds {MAX_TRANSCRIPT_CHAR_LIMIT} characters "
                        f"(got {len(sentence)}). Preview: '{preview}'"
                    )
                    continue

                # Try to get audio duration if file exists
                try:
                    duration_val = None
                    if os.path.isfile(filename):
                        audio_path = filename
                    else:
                        # Try relative to CSV file directory
                        csv_dir = os.path.dirname(data_path)
                        audio_path = os.path.join(csv_dir, filename)

                    if os.path.isfile(audio_path):
                        # FAST: use soundfile.info (no full decode)
                        info = soundfile.info(audio_path)
                        duration = round(info.frames / float(info.samplerate), 2)
                        duration_val = duration
                    else:
                        # Skip if audio file not found
                        print(f"Warning: Audio file not found: {filename}")
                        continue

                except Exception as e:
                    print(f"Warning: Could not read audio file {filename}: {e}")
                    continue

                candidate = {
                    "audio": {"path": audio_path},
                    "sentence": sentence,
                    "duration": duration_val,
                }

                # Apply custom filter if available for CSV data
                if filter_fn and not filter_fn(candidate):
                    continue

                self._try_store_manifest(candidate, data_path)

    # Get audio data, sample rate, and text from data list
    def _get_list_data(self, idx):
        if self.data_list_path.endswith(".header"):
            raw_entry = self.dataset_reader.get_data(self.data_list[idx])
            entry = self._manifest_from_mapping(raw_entry)
        else:
            raw_entry = self.data_list[idx]
            if isinstance(raw_entry, ManifestEntry):
                entry = raw_entry
            else:
                entry = self._manifest_from_mapping(raw_entry)

        if entry is None:
            raise ValueError(f"Unable to resolve manifest entry at index {idx}")

        audio_file = entry.audio_path

        # --------- FAST IO + CHEAP MONO ----------
        if entry.start_time is not None and entry.end_time is not None:
            sample, sample_rate = self.slice_from_file(
                audio_file, start=entry.start_time, end=entry.end_time
            )
        else:
            sample, sample_rate = soundfile.read(
                audio_file, dtype="float32", always_2d=True
            )

        if self.timestamps:
            transcript = (
                entry.sentences if entry.sentences is not None else entry.sentence or ""
            )
        else:
            transcript = entry.sentence or ""
            if not transcript and entry.sentences is not None:
                if isinstance(entry.sentences, list):
                    pieces = []
                    for seg in entry.sentences:
                        if isinstance(seg, dict):
                            pieces.append(normalize_text(seg.get("text", "")))
                        else:
                            pieces.append(normalize_text(str(seg)))
                    transcript = " ".join(filter(None, pieces)).strip()
                else:
                    transcript = normalize_text(str(entry.sentences))
            else:
                transcript = normalize_text(transcript)

        language = entry.language

        # Convert to mono channel cheaply
        if self.mono:
            if sample.ndim == 2:
                if sample.shape[1] > 1:
                    sample = sample.mean(axis=1).astype(np.float32)
                else:
                    sample = sample[:, 0].astype(np.float32)
            else:
                sample = sample.astype(np.float32)
        else:
            if sample.ndim == 2 and sample.shape[1] == 1:
                sample = sample[:, 0].astype(np.float32)

        if self.augmenter:
            sample, sample_rate = self.augment(sample, sample_rate)

        if sample_rate > 16000:
            sample = resample(sample, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000
        elif self.sample_rate != sample_rate:
            sample = resample(sample, orig_sr=sample_rate, target_sr=self.sample_rate)
            sample_rate = self.sample_rate

        return sample, sample_rate, transcript, language

    def _load_timestamps_transcript(self, transcript: List[dict]):
        assert isinstance(transcript, list), (
            f"transcript should be list, current type: {type(transcript)}"
        )
        data = dict()
        labels = self.processor.tokenizer.get_vocab()
        labels = self.processor.tokenizer.prefix_tokens[:3]
        for t in transcript:
            # Encode target text as label IDs
            start = (
                t["start"] if round(t["start"] * 100) % 2 == 0 else t["start"] + 0.01
            )
            if self.timestamp_begin is None:
                start = self.vocab[f"<|{start:.2f}|>"]
            else:
                start = self.timestamp_begin + round(start * 100) // 2
            end = t["end"] if round(t["end"] * 100) % 2 == 0 else t["end"] - 0.01
            if self.timestamp_begin is None:
                end = self.vocab[f"<|{end:.2f}|>"]
            else:
                end = self.timestamp_begin + round(end * 100) // 2
            label = self.processor(text=t["text"]).input_ids[4:-1]
            labels.extend([start])
            labels.extend(label)
            labels.extend([end])
        data["labels"] = labels + [self.endoftext]
        return data

    def _process_item(self, item, dataset_path=None):
        """Helper method to process a single dataset item."""
        try:
            # Handle both dict (HF datasets) and ManifestEntry objects
            if isinstance(item, ManifestEntry):
                # Convert ManifestEntry to dict format for consistency
                candidate = {
                    "audio": {"path": str(item.audio_path)},
                    "sentence": item.sentence,
                    "sentences": item.sentences,
                    "duration": item.duration,
                    "language": item.language,
                }
                if item.start_time is not None:
                    candidate["audio"]["start_time"] = item.start_time
                if item.end_time is not None:
                    candidate["audio"]["end_time"] = item.end_time

                self._try_store_manifest(candidate, dataset_path)
                return

            # Handle dict format (from HF datasets)
            audio_path = None
            start_time = None
            end_time = None

            audio_data = item.get("audio")
            if isinstance(audio_data, dict):
                audio_path = (
                    audio_data.get("path")
                    or audio_data.get("filename")
                    or audio_data.get("file")
                )
                start_time = audio_data.get("start_time")
                end_time = audio_data.get("end_time")
            elif audio_data:
                audio_path = audio_data

            if not audio_path:
                for alt_key in ("audio_path", "file", "filename", "path", "wav"):
                    alt_value = item.get(alt_key)
                    if alt_value:
                        audio_path = alt_value
                        break

            if not audio_path:
                return

            sentences = item.get("sentences")
            text_value = None
            for text_key in [
                "transcription",
                "text",
                "sentence",
                "transcript",
                "label",
            ]:
                if text_key in item:
                    candidate_value = item[text_key]
                    if isinstance(candidate_value, str):
                        text_value = candidate_value
                        break
                    if sentences is None and candidate_value is not None:
                        sentences = candidate_value

            audio_dict = {"path": str(audio_path)}
            if start_time is not None:
                audio_dict["start_time"] = start_time
            if end_time is not None:
                audio_dict["end_time"] = end_time

            candidate = {
                "audio": audio_dict,
                "sentence": text_value,
                "sentences": sentences,
                "duration": item.get("duration"),
                "language": item.get("language"),
            }

            self._try_store_manifest(candidate, dataset_path)

        except Exception as e:
            print(f"Error processing item: {e}")

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            indices = range(*idx.indices(len(self)))
            return [self.__getitem__(i) for i in indices]

        try:
            # Route: materialized list first, then lazy HF splits
            if idx < len(self.data_list):
                sample, sample_rate, transcript, language = self._get_list_data(idx=idx)
            else:
                # Map into HF splits
                rem = idx - len(self.data_list)
                ds = None
                row_idx = None
                language = None
                dataset_source = None
                # Find which split contains this index
                for entry in self.hf_splits:
                    n = (
                        len(entry["indices"])
                        if entry["indices"] is not None
                        else len(entry["dataset"])
                    )
                    if rem < n:
                        ds = entry["dataset"]
                        row_idx = (
                            entry["indices"][rem]
                            if entry["indices"] is not None
                            else rem
                        )
                        break
                    rem -= n

                if ds is None:
                    raise IndexError("Index out of range for dataset")
                dataset_source = entry.get("name")

                if isinstance(idx, (int, slice)):
                    ds = self._ensure_hf_arrow_dataset(entry)
                allowed_columns = entry.get("hf_allowed_columns")
                item = ds[row_idx]
                item = self._hf_item_to_python(item, allowed_columns)
                if isinstance(item, list):
                    raise TypeError(
                        "CustomDataset does not support slice retrieval from HF splits."
                    )
                if not isinstance(item, dict):
                    raise TypeError(
                        f"Expected dict-like item from dataset '{dataset_source}', "
                        f"received {type(item).__name__}."
                    )

                # Extract audio path or inline sample similar to _process_item
                audio_file = None
                sample = None
                sample_rate = None
                audio_reference = None

                def _assign_sample_from_array(
                    array_like, sampling_rate_hint=None, reference=None
                ):
                    nonlocal sample, sample_rate, audio_reference
                    if array_like is None:
                        return False
                    try:
                        arr = (
                            array_like
                            if isinstance(array_like, np.ndarray)
                            else np.array(array_like, dtype=np.float32)
                        )
                    except Exception:
                        return False
                    if arr.size == 0:
                        return False
                    if arr.dtype != np.float32:
                        arr = arr.astype(np.float32)
                    if arr.ndim == 2:
                        if arr.shape[1] > 1 and self.mono:
                            arr = arr.mean(axis=1).astype(np.float32)
                        elif arr.shape[1] == 1:
                            arr = arr[:, 0].astype(np.float32)
                    elif arr.ndim > 2:
                        arr = arr.reshape(-1).astype(np.float32)
                    sample = arr
                    try:
                        sr_val = (
                            int(sampling_rate_hint)
                            if sampling_rate_hint is not None
                            else int(self.sample_rate)
                        )
                    except (TypeError, ValueError):
                        sr_val = int(self.sample_rate)
                    sample_rate = sr_val
                    if reference and not audio_reference:
                        audio_reference = reference
                    return True

                def _handle_audio_blob(blob):
                    nonlocal audio_file, audio_reference
                    if blob is None:
                        return False
                    if isinstance(blob, dict):
                        array_candidate = blob.get("array")
                        if array_candidate is None:
                            for alt_key in ("waveform", "samples", "values"):
                                if isinstance(
                                    blob.get(alt_key), (list, tuple, np.ndarray)
                                ):
                                    array_candidate = blob[alt_key]
                                    break
                        sr_hint = blob.get("sampling_rate")
                        reference = (
                            blob.get("path")
                            or blob.get("audio_path")
                            or blob.get("audio_filepath")
                            or blob.get("filename")
                            or blob.get("file")
                        )
                        if array_candidate is not None and _assign_sample_from_array(
                            array_candidate, sr_hint, reference
                        ):
                            return True
                        bytes_value = blob.get("bytes")
                        if isinstance(bytes_value, (bytes, bytearray, memoryview)):
                            inline_path = self._write_inline_bytes(bytes_value)
                            audio_file = inline_path
                            audio_reference = inline_path
                            return True
                        if reference:
                            audio_file = str(reference)
                            audio_reference = audio_file
                            return True
                        inline_path = self._materialize_inline_audio(blob)
                        if inline_path:
                            audio_file = inline_path
                            audio_reference = inline_path
                            return True
                        return False
                    if isinstance(blob, (np.ndarray, list, tuple)):
                        return _assign_sample_from_array(blob)
                    if isinstance(blob, (bytes, bytearray, memoryview)):
                        inline_path = self._materialize_inline_audio(blob)
                        if inline_path:
                            audio_file = inline_path
                            audio_reference = inline_path
                            return True
                        return False
                    if hasattr(os, "PathLike") and isinstance(blob, os.PathLike):
                        audio_file = os.fspath(blob)
                        audio_reference = audio_file
                        return True
                    if isinstance(blob, str) and blob.strip():
                        audio_file = blob
                        audio_reference = blob
                        return True
                    return False

                candidate_keys: List[str] = []
                if "audio" in item:
                    candidate_keys.append("audio")
                for key in _HF_AUDIO_CANDIDATE_KEYS:
                    if key == "audio":
                        continue
                    if key in item:
                        candidate_keys.append(key)

                for key in candidate_keys:
                    if sample is not None or audio_file is not None:
                        break
                    _handle_audio_blob(item.get(key))

                # As a fallback, scan remaining values for inline blobs
                if sample is None and audio_file is None:
                    for value in item.values():
                        if _handle_audio_blob(value):
                            break

                # Transcript selection
                if self.timestamps:
                    transcript = item.get("sentences", item.get("text", ""))
                else:
                    txt = None
                    for text_key in _HF_TEXT_CANDIDATE_KEYS:
                        if text_key in item:
                            txt = item[text_key]
                            break
                    transcript = normalize_text(txt) if txt else ""

                language = item.get("language", None)

                if not self._check_transcript_limits(
                    transcript,
                    audio_path=audio_reference,
                    dataset_source=dataset_source,
                    enforce_dataset_bounds=True,
                ):
                    raise ValueError(
                        "Transcript constraints should be enforced before training; "
                        f"found invalid sample in dataset '{dataset_source}' "
                        f"with audio reference '{audio_reference}'."
                    )

                # Read audio from file only if no decoded array available
                if sample is None:
                    if not audio_file:
                        raise ValueError("Missing audio path in HF item")
                    # Ensure audio_file is a string path, not a dict
                    if isinstance(audio_file, dict):
                        raise ValueError(f"Invalid file: {audio_file}")
                    if not isinstance(audio_file, str):
                        audio_file = str(audio_file)

                    try:
                        sample, sample_rate = soundfile.read(
                            audio_file, dtype="float32", always_2d=True
                        )
                    except Exception as e:
                        # Log the actual error for debugging
                        raise ValueError(
                            f"Error reading audio file '{audio_file}': {e}"
                        )

            # Can set language for individual data
            self.processor.tokenizer.set_prefix_tokens(
                language=language if language is not None else self.language
            )

            if len(transcript) > 0:
                if self.timestamps:
                    # -------- timestamps training: keep your existing label building --------
                    data = self._load_timestamps_transcript(transcript=transcript)
                    # Calculate log-Mel input features from input audio array
                    feats = self.processor(
                        audio=sample, sampling_rate=self.sample_rate
                    ).input_features
                    data["input_features"] = self._ensure_2d_features(feats)
                else:
                    # -------- non-timestamps training: use processor to build labels --------
                    processed = self.processor(
                        audio=sample, sampling_rate=self.sample_rate, text=transcript
                    )
                    feats = processed.get("input_features")
                    if feats is None:
                        raise ValueError(
                            "Processor returned no input_features for non-timestamp sample."
                        )

                    raw_labels = processed.get("labels")
                    if raw_labels is None:
                        raw_labels = processed.get("label_ids")
                    if raw_labels is None:
                        raise ValueError(
                            "Processor returned neither 'labels' nor 'label_ids'."
                        )

                    # Flatten any nested structures from single-sample processor calls
                    if isinstance(raw_labels, (list, tuple)) and raw_labels:
                        first_elem = raw_labels[0]
                        if isinstance(first_elem, (list, tuple, np.ndarray)):
                            raw_labels = first_elem
                    if hasattr(raw_labels, "tolist"):
                        raw_labels = raw_labels.tolist()
                    labels = list(raw_labels)

                    data = {
                        "input_features": self._ensure_2d_features(feats),
                        "labels": labels,
                        "text": transcript,
                    }
            else:
                # If there's no text, use <|nospeech|> token (kept as IDs; collator pads)
                data = self.processor(audio=sample, sampling_rate=self.sample_rate)
                data["input_features"] = self._ensure_2d_features(
                    data["input_features"]
                )
                data["labels"] = [self.startoftranscript, self.nospeech, self.endoftext]

            return data

        except Exception as e:
            raise RuntimeError(f"Error reading data at index {idx}: {e}") from e

    def __len__(self):
        # Materialized entries
        n = len(self.data_list)
        # Add lazy HF splits sizes
        for entry in self.hf_splits:
            n += (
                len(entry["indices"])
                if entry["indices"] is not None
                else len(entry["dataset"])
            )
        return n

    # Split and read audio
    @staticmethod
    def slice_from_file(file, start, end):
        sndfile = soundfile.SoundFile(file)
        sample_rate = sndfile.samplerate
        duration = round(float(len(sndfile)) / sample_rate, 3)
        start = round(start, 3)
        end = round(end, 3)
        # Count from the end
        if start < 0.0:
            start += duration
        if end < 0.0:
            end += duration
        # Ensure data doesn't go out of bounds
        if start < 0.0:
            start = 0.0
        if end > duration:
            end = duration
        if end < 0.0:
            raise ValueError("Slice end position (%f s) out of bounds" % end)
        if start > end:
            raise ValueError(
                "Slice start position (%f s) is later than slice end position (%f s)"
                % (start, end)
            )
        start_frame = int(start * sample_rate)
        end_frame = int(end * sample_rate)
        sndfile.seek(start_frame)
        # ensure always_2d for consistent downstream handling
        sample = sndfile.read(
            frames=end_frame - start_frame, dtype="float32", always_2d=True
        )
        return sample, sample_rate

    # Data augmentation
    def augment(self, sample, sample_rate):
        if self.augmenter:
            return self.augmenter.augment(sample, sample_rate)
        return sample, sample_rate
