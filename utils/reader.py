import csv
import gc
import json
import os
import random
import sys
import tarfile
import time
import warnings
from dataclasses import dataclass
from itertools import chain, zip_longest
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, TextIO, Union
import re

import numpy as np
import soundfile
from datasets import Audio, Dataset as HFDataset, DatasetDict as HFDatasetDict
from datasets import load_dataset, load_from_disk, DatasetDict
from huggingface_hub import snapshot_download  # needed for fleurs/common_voice branches
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.binary import DatasetReader
from utils.audio_augmentation import AudioAugmenter, resample

try:
    csv.field_size_limit(sys.maxsize)
except (OverflowError, AttributeError):
    # Fall back to a large but safe limit if sys.maxsize is not accepted
    csv.field_size_limit(2**31 - 1)

MAX_TRANSCRIPT_CHAR_LIMIT = 800
MAX_TRANSCRIPT_TOKEN_LIMIT = 448


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
    and keeping only alphabetical characters, commas, dots, spaces and apostrophes.
    This handles Uzbek text like "ko'p" to ensure consistent character usage.
    """
    if text is None:
        return text

    # Coerce to string to handle non-string inputs (e.g., ints)
    try:
        normalized = str(text)
    except Exception:
        # As a last resort, return empty string if conversion fails
        return ""

    # Various apostrophe-like characters that might appear
    apostrophe_variants = [
        "\u2019",  # Right single quotation mark
        "\u0027",  # Apostrophe
        "\u02bc",  # Modifier letter apostrophe
        "\u02bb",  # Modifier letter turned comma
        "`",  # Grave accent
        "´",  # Acute accent
        "ʻ",  # Modifier letter turned comma (another variant)
        "ʼ",  # Modifier letter apostrophe (another variant)
        "'",  # Right single quotation mark (direct)
        "‛",  # Single high-reversed-9 quotation mark
    ]

    # Replace all variants with standard apostrophe (')
    for variant in apostrophe_variants:
        normalized = normalized.replace(variant, "'")

    # This regex keeps Latin & Cyrillic letters, spaces, comma, dot, apostrophe, dash
    normalized = re.sub(r"[^a-zA-ZА-Яа-яЎўҚқҒғҲҳ0-9\s,.'-]+", "", normalized)

    # Clean up multiple spaces
    normalized = re.sub(r"\s+", " ", normalized).strip()

    return normalized


def _detect_distributed() -> bool:
    """
    Prefer PyTorch distributed init if available, fall back to env heuristics.
    """
    # Prefer PyTorch's ground truth if initialized
    try:
        import torch.distributed as dist  # noqa

        if dist.is_available() and dist.is_initialized():
            return True
    except Exception:
        pass

    # Fallback to environment heuristics
    if os.environ.get("WORLD_SIZE", "1") not in ("", "1"):
        return True
    if any(
        os.environ.get(k) is not None
        for k in ("LOCAL_RANK", "RANK", "OMPI_COMM_WORLD_SIZE", "PMI_SIZE")
    ):
        return True
    return False


def _strip_bom(value: str) -> str:
    """Remove a UTF-8 BOM prefix if present."""
    if value and value[0] == "\ufeff":
        return value[1:]
    return value


def _iter_tsv_rows(
    file_obj: TextIO,
    *,
    delimiter: str = "\t",
    strip_cells: bool = True,
    skip_comments: bool = True,
) -> Iterator[List[str]]:
    """
    Yield TSV rows using the csv module to honor quoting and embedded delimiters.
    """
    reader = csv.reader(
        file_obj,
        delimiter=delimiter,
        quoting=csv.QUOTE_MINIMAL,
        skipinitialspace=False,
    )
    for raw_row in reader:
        if not raw_row:
            continue
        if strip_cells:
            row = []
            for cell in raw_row:
                cell = _strip_bom(cell.strip()) if isinstance(cell, str) else cell
                row.append(cell)
        else:
            row = [
                _strip_bom(cell) if isinstance(cell, str) else cell for cell in raw_row
            ]
        if skip_comments:
            for cell in row:
                if not cell:
                    continue
                if isinstance(cell, str) and cell.startswith("#"):
                    row = []
                break
            if not row:
                continue
        yield row


def _prepare_tsv_header(raw_header: Sequence[str]) -> List[str]:
    """
    Normalize TSV header names to lowercase unique keys for robust lookups.
    Empty headers are replaced with positional placeholders.
    """
    normalized: List[str] = []
    seen: Dict[str, int] = {}
    for idx, cell in enumerate(raw_header):
        if isinstance(cell, str):
            header_name = _strip_bom(cell).strip().lower()
        elif cell is None:
            header_name = ""
        else:
            header_name = str(cell).strip().lower()

        if not header_name:
            header_name = f"column_{idx}"

        base = header_name
        count = seen.get(base, 0)
        if count > 0:
            header_name = f"{base}_{count + 1}"
        seen[base] = count + 1
        normalized.append(header_name)

    return normalized


def _iter_tsv_dict_rows(
    file_obj: TextIO,
    *,
    delimiter: str = "\t",
    strip_cells: bool = True,
    skip_comments: bool = True,
) -> tuple[List[str], Dict[str, str], Iterator[tuple[int, Dict[str, str]]]]:
    """
    Yield TSV rows as dictionaries keyed by normalized lowercase header names.

    Returns:
        header_keys: normalized lowercase header names.
        header_lookup: mapping of normalized names to their original header string.
        row_iter: iterator yielding (line_number, row_dict) pairs.
    """
    row_iter = _iter_tsv_rows(
        file_obj,
        delimiter=delimiter,
        strip_cells=strip_cells,
        skip_comments=skip_comments,
    )
    header_row = next(row_iter, None)
    if not header_row:
        return [], {}, iter(())

    original_header: List[str] = []
    normalized_header_lower: List[str] = []
    for idx, cell in enumerate(header_row):
        if isinstance(cell, str):
            cleaned = _strip_bom(cell).strip()
        elif cell is None:
            cleaned = ""
        else:
            cleaned = str(cell).strip()
        original_header.append(cleaned)
        normalized_header_lower.append(cleaned.lower())

    header_hint_substrings = (
        "path",
        "audio",
        "filename",
        "file",
        "text",
        "sentence",
        "transcript",
        "transcription",
        "normalized",
        "raw",
        "duration",
        "language",
        "id",
    )
    header_present = any(
        any(hint in cell for hint in header_hint_substrings) and cell
        for cell in normalized_header_lower
    )

    if header_present:
        header_keys = _prepare_tsv_header(original_header)
        header_lookup = {
            header_keys[idx]: (original_header[idx] or header_keys[idx])
            for idx in range(len(header_keys))
        }
        data_iter: Iterator[List[str]] = row_iter
        first_data_line_number = 2
    else:
        header_keys = [f"column_{idx}" for idx in range(len(original_header))]
        header_lookup = {key: key for key in header_keys}
        data_iter = chain([original_header], row_iter)
        first_data_line_number = 1
        source_desc = getattr(file_obj, "name", "<tsv>")
        print(f"  Info: No header detected in {source_desc}; using positional columns.")

    def generator() -> Iterator[tuple[int, Dict[str, str]]]:
        extras_warning_emitted = False
        for line_number, row in enumerate(data_iter, start=first_data_line_number):
            if not row:
                continue
            if len(row) > len(header_keys) and not extras_warning_emitted:
                extras_warning_emitted = True
                ignored = len(row) - len(header_keys)
                print(
                    f"  Warning: detected {ignored} extra columns starting at line {line_number}; ignoring surplus values."
                )

            row_dict: Dict[str, str] = {}
            for key, value in zip_longest(header_keys, row, fillvalue=""):
                if key is None:
                    continue
                if value is None:
                    cell_value = ""
                elif isinstance(value, str):
                    cell_value = value
                else:
                    cell_value = str(value)
                row_dict[key] = cell_value
            yield line_number, row_dict

    return header_keys, header_lookup, generator()


def _detect_column_index(
    header: Sequence[str], candidates: Sequence[str], default_idx: Optional[int]
) -> Optional[int]:
    """
    Locate the index of the first matching candidate column in a normalized header.
    """
    normalized = {col.strip().lower(): idx for idx, col in enumerate(header) if col}
    for candidate in candidates:
        idx = normalized.get(candidate)
        if idx is not None:
            return idx
    return default_idx


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
        max_sentence=300,
        augment_config_path=None,
        dataset_filters=None,
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
        assert max_sentence <= 300, (
            f"max_sentence cannot be greater than 300, current value: {max_sentence}"
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

        # --- SAFER: choose num_proc for HF Datasets map/filter ---
        cpu_cnt = os.cpu_count() or 4
        self.num_proc = min(6, max(2, cpu_cnt // 4))

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

    def _check_transcript_limits(
        self,
        transcript: Any,
        *,
        audio_path: Optional[str] = None,
        dataset_source: Optional[str] = None,
        enforce_dataset_bounds: bool = False,
    ) -> bool:
        """
        Validate transcript length and token count, optionally enforcing dataset-level min/max sentence bounds.
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
        preview = cleaned_text[:120].replace("\n", " ")
        if len(cleaned_text) > 120:
            preview += "..."

        if enforce_dataset_bounds:
            if char_length < self.min_sentence:
                return False
            if self.max_sentence != -1 and char_length > self.max_sentence:
                print(
                    f"⚠️  Dropping entry{context}: transcript has {char_length} characters "
                    f"(dataset max_sentence limit {self.max_sentence}). Preview: '{preview}'"
                )
                return False

        if char_length > MAX_TRANSCRIPT_CHAR_LIMIT:
            print(
                f"⚠️  Dropping entry{context}: transcript has {char_length} characters "
                f"(limit {MAX_TRANSCRIPT_CHAR_LIMIT}). Preview: '{preview}'"
            )
            return False

        if not self.processor:
            return True

        try:
            token_count = len(
                self.processor.tokenizer.encode(cleaned_text, add_special_tokens=False)
            )
        except Exception as exc:
            print(f"Warning: Could not tokenize text for validation: {exc}")
            return False

        if token_count > MAX_TRANSCRIPT_TOKEN_LIMIT:
            print(
                f"⚠️  Dropping entry{context}: transcript has {token_count} tokens "
                f"(limit {MAX_TRANSCRIPT_TOKEN_LIMIT}). Preview: '{preview}'"
            )
            return False
        if token_count > MAX_TRANSCRIPT_TOKEN_LIMIT - 80:
            print(
                f"📏 Warning{context}: transcript has {token_count} tokens "
                f"(approaching limit {MAX_TRANSCRIPT_TOKEN_LIMIT}). Preview: '{preview}'"
            )

        return True

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

        audio_blob = row.get("audio")
        if isinstance(audio_blob, dict):
            audio_path = (
                audio_blob.get("path")
                or audio_blob.get("filename")
                or audio_blob.get("file")
            )
            start_time = audio_blob.get("start_time")
            end_time = audio_blob.get("end_time")
        elif audio_blob:
            audio_path = audio_blob

        if not audio_path:
            for key in ("wav", "audio_path", "filepath", "file", "filename", "path"):
                candidate = row.get(key)
                if candidate:
                    audio_path = candidate
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
                                    _iter_tsv_dict_rows(f)
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

                                    transcription_value = row_dict.get(text_column, "")
                                    if transcription_value is None:
                                        transcription = ""
                                    else:
                                        transcription = str(transcription_value).strip()

                                    if not transcription:
                                        continue
                                    if len(transcription) > MAX_TRANSCRIPT_CHAR_LIMIT:
                                        skipped_long += 1
                                        preview = transcription[:120].replace("\n", " ")
                                        msg = (
                                            f"  Dropping line {line_number} in {os.path.basename(tsv_file)}: "
                                            f"transcript has {len(transcription)} chars (limit {MAX_TRANSCRIPT_CHAR_LIMIT}). "
                                            f"Preview: '{preview}{'...' if len(transcription) > 120 else ''}'"
                                        )
                                        print(msg)
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
                                    _iter_tsv_dict_rows(f)
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

                                    transcription_value = row_dict.get(text_column, "")
                                    if transcription_value is None:
                                        transcription = ""
                                    else:
                                        transcription = str(transcription_value).strip()

                                    if not transcription:
                                        continue
                                    if len(transcription) > MAX_TRANSCRIPT_CHAR_LIMIT:
                                        skipped_long += 1
                                        preview = transcription[:120].replace("\n", " ")
                                        msg = (
                                            f"  Dropping line {row_number} in {os.path.basename(tsv_file)}: "
                                            f"transcript has {len(transcription)} chars (limit {MAX_TRANSCRIPT_CHAR_LIMIT}). "
                                            f"Preview: '{preview}{'...' if len(transcription) > 120 else ''}'"
                                        )
                                        print(msg)
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
                        self.hf_splits.append(
                            {
                                "dataset": ds_obj,
                                "indices": kept_idx,
                                "name": split_name_label,
                            }
                        )
                        return

                # If we reach here, we have an HF dataset object to use lazily
                self.hf_splits.append(
                    {"dataset": ds_obj, "indices": None, "name": split_name_label}
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
                    preview = sentence[:120]
                    print(
                        f"Skipping row in {data_path}: transcript exceeds {MAX_TRANSCRIPT_CHAR_LIMIT} characters "
                        f"(got {len(sentence)}). Preview: '{preview}{'...' if len(sentence) > 120 else ''}'"
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

                item = ds[row_idx]

                # Extract audio path and text similar to _process_item
                audio_file = None
                sample = None
                sample_rate = None
                audio_reference = None
                if "audio" in item:
                    audio_data = item["audio"]
                    if isinstance(audio_data, dict):
                        # Prefer already-decoded array from HF datasets to avoid path issues
                        if "array" in audio_data and audio_data["array"] is not None:
                            arr = audio_data["array"]
                            # Handle both numpy arrays and lists
                            if not isinstance(arr, np.ndarray):
                                arr = np.array(arr, dtype=np.float32)
                            # Ensure float32
                            if arr.dtype != np.float32:
                                arr = arr.astype(np.float32)
                            # If stereo/2D, average to mono if requested
                            if arr.ndim == 2 and arr.shape[1] > 1 and self.mono:
                                arr = arr.mean(axis=1).astype(np.float32)
                            # If 2D with a single channel, squeeze to 1D
                            if arr.ndim == 2 and arr.shape[1] == 1:
                                arr = arr[:, 0].astype(np.float32)
                            sample = arr
                            sample_rate = int(
                                audio_data.get("sampling_rate", self.sample_rate)
                            )
                            audio_reference = (
                                audio_data.get("path")
                                or audio_data.get("filename")
                                or audio_data.get("file")
                            )
                            # Don't try to extract audio_file if we already have the sample
                        else:
                            # No array present, extract path
                            audio_file = (
                                audio_data.get("path")
                                or audio_data.get("filename")
                                or audio_data.get("file")
                            )
                            audio_reference = audio_file
                    else:
                        audio_file = str(audio_data)
                        audio_reference = audio_file

                # Only check other fields if we don't have a sample yet
                if sample is None and audio_file is None:
                    if "audio_path" in item:
                        audio_file = item["audio_path"]
                        audio_reference = audio_file
                    elif "file" in item:
                        audio_file = item["file"]
                        audio_reference = audio_file
                    elif "filename" in item:
                        audio_file = item["filename"]
                        audio_reference = audio_file
                    elif "path" in item:
                        # Make sure we extract string path, not a dict
                        path_val = item["path"]
                        if isinstance(path_val, dict):
                            # Check if this is actually an audio dict with array
                            if "array" in path_val and isinstance(
                                path_val["array"], np.ndarray
                            ):
                                # This is a mislabeled audio dict, process it as such
                                arr = path_val["array"]
                                if arr.dtype != np.float32:
                                    arr = arr.astype(np.float32)
                                if arr.ndim == 2 and arr.shape[1] > 1 and self.mono:
                                    arr = arr.mean(axis=1).astype(np.float32)
                                if arr.ndim == 2 and arr.shape[1] == 1:
                                    arr = arr[:, 0].astype(np.float32)
                                sample = arr
                                sample_rate = int(
                                    path_val.get("sampling_rate", self.sample_rate)
                                )
                            else:
                                # If path is a dict without array, try to extract the actual path string
                                audio_file = (
                                    path_val.get("path")
                                    or path_val.get("filename")
                                    or path_val.get("file")
                                )
                                audio_reference = audio_file
                        else:
                            audio_file = path_val
                            audio_reference = audio_file

                # Transcript selection
                if self.timestamps:
                    transcript = item.get("sentences", item.get("text", ""))
                else:
                    txt = None
                    for text_key in [
                        "sentence",
                        "text",
                        "transcription",
                        "transcript",
                        "label",
                    ]:
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
                    return self.__getitem__(random.randint(0, self.__len__() - 1))

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
                    # -------- non-timestamps training: return features + RAW TEXT --------
                    feats = self.processor(
                        audio=sample, sampling_rate=self.sample_rate
                    ).input_features
                    data = {
                        "input_features": self._ensure_2d_features(
                            feats
                        ),  # (80, T) or torch tensor of same
                        "text": transcript,  # <-- collator will batch-tokenize
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
            print(
                f"Error reading data, index: {idx}, error message: {e}", file=sys.stderr
            )
            return self.__getitem__(random.randint(0, self.__len__() - 1))

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
