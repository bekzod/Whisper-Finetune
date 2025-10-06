import csv
import gc
import json
import os
import random
import sys
import tarfile
import time
import warnings
from typing import List
from itertools import chain

import librosa
import numpy as np
from datasets import load_from_disk, DatasetDict, load_dataset
from huggingface_hub import snapshot_download

import soundfile
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.binary import DatasetReader


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
    if not text:
        return text

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
    normalized = text
    for variant in apostrophe_variants:
        normalized = normalized.replace(variant, "'")

    # Keep only alphabetical characters, comma, dot, apostrophe, and spaces
    import re

    # This regex keeps:
    # - \w matches word characters (letters, digits, underscore) - we'll refine this
    # - \s matches whitespace
    # - , . ' are literally those characters
    # Better regex that keeps only letters (Latin and Cyrillic), spaces, comma, dot, apostrophe
    normalized = re.sub(r"[^a-zA-ZА-Яа-яЎўҚқҒғҲҳ\s,.']+", "", normalized)

    # Clean up multiple spaces
    normalized = re.sub(r"\s+", " ", normalized).strip()

    return normalized


def remove_silence_librosa(
    y: np.ndarray,
    sr: int,
    top_db: int = 40,
    min_silence_ms: float = 200,
    pad_ms: float = 50,
) -> np.ndarray:
    intervals = librosa.effects.split(y, top_db=top_db)

    if len(intervals) == 0:
        return np.array([], dtype=y.dtype)

    def merge_intervals(iv, min_gap):
        merged = []
        for start, end in iv:
            if not merged:
                merged.append([start, end])
                continue
            if start - merged[-1][1] <= min_gap:
                merged[-1][1] = end
            else:
                merged.append([start, end])
        return np.array(merged, dtype=int)

    min_gap = int((min_silence_ms / 1000.0) * sr)
    intervals = merge_intervals(intervals, min_gap=min_gap)

    pad = int((pad_ms / 1000.0) * sr)
    chunks = []
    for s, e in intervals:
        s = max(0, s - pad)
        e = min(len(y), e + pad)
        chunks.append(y[s:e])
    return np.concatenate(chunks) if chunks else np.array([], dtype=y.dtype)


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
        max_sentence=200,
        augment_config_path=None,
        # --- optional knobs for Example A ---
        silence_top_db: int = 40,
        silence_min_gap_ms: int = 200,
        silence_pad_ms: int = 50,
        dataset_filters=None,
    ):
        """
        Args:
            data_list_path: Path to the data list file, or path to the binary list header file, or Hugging Face dataset folder.
                           Supports JSON, JSONL, CSV formats, and Hugging Face dataset directories.
                           Multiple paths can be specified separated by '+' to combine datasets,
                           e.g., '../datasets/train.json+../datasets/cleaned.json'
                           For Hugging Face datasets, use 'path:subset' format to specify a subset,
                           e.g., '../dataset:train' or '../dataset:validation'
                           If no subset is specified for HF dataset, all available data will be used.
                           CSV format supports:
                           - Standard format: filename,text
                           - LJSpeech format: filename|text
                           - Header detection for column names like 'filename', 'text', 'audio_path', etc.
                           Hugging Face dataset folders should contain arrow/parquet files
            processor: Whisper preprocessing tool, obtained from WhisperProcessor.from_pretrained
            mono: Whether to convert audio to mono channel, this must be True
            language: Language of the fine-tuning data
            timestamps: Whether to use timestamps during fine-tuning
            sample_rate: Audio sample rate, default is 16000
            min_duration: Audio shorter than this duration will be truncated, in seconds, cannot be less than 0.5, default 0.5s
            max_duration: Audio longer than this duration will be truncated, in seconds, cannot be greater than 30, default 30s
            min_sentence: Minimum sentence character count for fine-tuning, default 1
            max_sentence: Maximum sentence character count for fine-tuning, default 200
            augment_config_path: Path to data augmentation configuration parameter file
            dataset_filters: List of dictionaries with 'name' and 'filter_fn' keys for dataset filtering.
                           Each dict should have:
                           - 'name': dataset name (partial match)
                           - 'filter_fn': lambda function that takes a row/example and returns True to keep it

            Example A params:
            silence_top_db: energy threshold for librosa.effects.split (smaller = more aggressive)
            silence_min_gap_ms: merge gaps shorter than this between voiced islands
            silence_pad_ms: pad each kept island to avoid cutting phones
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
        assert max_sentence <= 200, (
            f"max_sentence cannot be greater than 200, current value: {max_sentence}"
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

        # Example A config
        self.silence_top_db = silence_top_db
        self.silence_min_gap_ms = silence_min_gap_ms
        self.silence_pad_ms = silence_pad_ms
        self.dataset_filters = dataset_filters or []

        # Detect distributed training to avoid multiprocessing conflicts with CUDA
        # Check multiple environment variables used by different distributed training frameworks
        is_distributed = (
            os.environ.get("WORLD_SIZE", "1") != "1"  # PyTorch DDP / DeepSpeed
            or os.environ.get("LOCAL_RANK") is not None  # PyTorch DDP
            or os.environ.get("RANK") is not None  # General distributed training
            or os.environ.get("OMPI_COMM_WORLD_SIZE") is not None  # OpenMPI
            or os.environ.get("PMI_SIZE")
            is not None  # PMI (Process Management Interface)
        )

        # Disable multiprocessing in distributed training to prevent segmentation faults
        if is_distributed:
            self.num_proc = 0
            warnings.warn(
                "Distributed training detected. Disabling multiprocessing in dataset filtering "
                "to prevent CUDA/fork conflicts. This may slow down dataset loading but ensures stability.",
                UserWarning,
            )
        else:
            # Use multiprocessing for faster filtering in single-GPU mode
            self.num_proc = 4

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

        self.data_list: List[dict] = []
        self._load_data_list()

        self.augment_configs = None
        self.noises_path = None
        self.speed_rates = None
        if augment_config_path:
            with open(augment_config_path, "r", encoding="utf-8") as f:
                self.augment_configs = json.load(f)

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
            elif data_path.endswith(".csv"):
                # Load CSV file
                self._load_csv_data(data_path)
            else:
                # Get data list from JSON/JSONL
                # Find matching filter function for this dataset
                filter_fn = None
                dataset_name = os.path.basename(data_path)
                for filter_config in self.dataset_filters:
                    if filter_config["name"] in dataset_name:
                        filter_fn = filter_config["filter_fn"]
                        print(f"  Found filter for dataset '{filter_config['name']}'")
                        break

                with open(data_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                for line in tqdm(lines, desc=f"Reading data list from {data_path}"):
                    if isinstance(line, str):
                        line = json.loads(line)
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
                    if self.max_duration != -1 and line["duration"] > self.max_duration:
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

                    self.data_list.append(line)

    def _load_huggingface_dataset(self, data_path, dataset_subset=None):
        """Load data from a Hugging Face dataset folder."""
        print(
            f"Loading Hugging Face dataset from {data_path}"
            + (f" (subset: {dataset_subset})" if dataset_subset else "")
        )

        # Find matching filter function for this dataset
        filter_fn = None
        dataset_name = os.path.basename(data_path)
        for filter_config in self.dataset_filters:
            if filter_config["name"] in dataset_name:
                filter_fn = filter_config["filter_fn"]
                print(f"  Found filter for dataset '{filter_config['name']}'")
                break

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
                        if dataset_subset:
                            tsv_file = os.path.join(data_dir, f"{dataset_subset}.tsv")
                            if os.path.exists(tsv_file):
                                print(f"  Processing TSV file: {tsv_file}")
                                with open(tsv_file, "r", encoding="utf-8") as f:
                                    # Skip header if present
                                    next(f, None)  # Skip header line
                                    for line in f:  # Process line by line
                                        parts = line.strip().split("\t")
                                        if len(parts) >= 3:
                                            # Format: id, filename, transcription
                                            audio_filename = parts[1]
                                            transcription = parts[2]

                                            # Find the audio file in extracted directories
                                            audio_path = None
                                            for subdir in [
                                                "train",
                                                "test",
                                                "dev",
                                                "validation",
                                            ]:
                                                potential_path = os.path.join(
                                                    audio_dir, subdir, audio_filename
                                                )
                                                if os.path.exists(potential_path):
                                                    audio_path = potential_path
                                                    break

                                            if audio_path:
                                                yield {
                                                    "audio": {"path": audio_path},
                                                    "text": transcription,
                                                }

                    # Use the generator instead of a list
                    dataset = fleurs_generator()

                # Special handling for mozilla-foundation/common_voice_17_0 dataset
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

                    # Extract tar files if they exist
                    for tar_file in (
                        os.listdir(audio_dir) if os.path.exists(audio_dir) else []
                    ):
                        if tar_file.endswith(".tar"):
                            tar_path = os.path.join(audio_dir, tar_file)
                            # Extract to the audio directory itself
                            split_name = tar_file.replace(".tar", "")

                            # Check if already extracted
                            extract_marker = os.path.join(
                                audio_dir, f".{split_name}_extracted"
                            )
                            if not os.path.exists(extract_marker):
                                print(f"  Extracting {tar_file} to {audio_dir}")
                                with tarfile.open(tar_path, "r") as tar:
                                    # Extract files one by one to avoid memory issues
                                    for member in tar:
                                        tar.extract(member, audio_dir)
                                # Create marker file to indicate extraction is done
                                with open(extract_marker, "w") as f:
                                    f.write("extracted")

                    # Create a generator for Common Voice TSV files to avoid loading all in memory
                    def common_voice_generator():
                        if dataset_subset:
                            # Common Voice uses TSV files for metadata
                            tsv_file = os.path.join(
                                transcript_dir, f"{dataset_subset}.tsv"
                            )
                            if os.path.exists(tsv_file):
                                print(f"  Processing TSV file: {tsv_file}")
                                with open(tsv_file, "r", encoding="utf-8") as f:
                                    # Read header to get column indices
                                    header_line = next(f, None)
                                    if header_line:
                                        header = header_line.strip().split("\t")

                                        # Find column indices
                                        path_idx = (
                                            header.index("path")
                                            if "path" in header
                                            else 0
                                        )
                                        sentence_idx = (
                                            header.index("sentence")
                                            if "sentence" in header
                                            else 2
                                        )

                                        for line in f:  # Process remaining lines
                                            parts = line.strip().split("\t")
                                            if len(parts) > max(path_idx, sentence_idx):
                                                audio_filename = parts[path_idx]
                                                transcription = parts[sentence_idx]

                                                # Common Voice audio files are typically mp3
                                                if not audio_filename.endswith(".mp3"):
                                                    audio_filename = (
                                                        audio_filename + ".mp3"
                                                    )

                                                # Find the audio file
                                                audio_path = os.path.join(
                                                    audio_dir, audio_filename
                                                )

                                                if os.path.exists(audio_path):
                                                    yield {
                                                        "audio": {"path": audio_path},
                                                        "text": transcription,
                                                    }

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
                            streaming=False,  # Changed from True to False
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
                            streaming=False,  # Changed from True to False
                            download_mode="reuse_dataset_if_exists",
                            cache_dir=os.getenv("HF_DATASETS_CACHE", None),
                        )

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

            # Process the dataset entries with batch filtering for speed
            total_original = 0
            filter_kept = 0

            # Check if we have a filter and can batch process
            if filter_fn and not isinstance(dataset_iter, chain):
                # Batch processing for single dataset (not chained)
                if isinstance(dataset, DatasetDict):
                    # Process each split separately for batch filtering
                    for split_name in split_names:
                        split_data = dataset[split_name]
                        total_original += len(split_data)

                        # Apply filter in batch if dataset supports it
                        try:
                            # For uzbekvoice-filtered, use batch filtering
                            if "uzbekvoice" in data_path.lower():
                                # Filter using dataset's filter method (much faster)
                                blacklist_clients = [
                                    "56ac8e86-b8c9-4879-a342-0eeb94f686fc",
                                    "3d3fca02-6a07-41e2-9af4-60886ea60300",
                                    "231d3776-2dbe-4a42-a535-c67943427e3f",
                                    "e2716f95-70b5-4832-b903-eef2343591a4",
                                    "2a815774-e953-4031-931a-8a28052e5cf9",
                                    "d6fd3dc4-a55d-4a80-9bbf-b713325d05be",
                                    "10b29e87-bf01-4b16-bead-a044076f849b",
                                    "e3412d51-f079-4167-b3f9-311a976443ce",
                                ]
                                filtered_data = split_data.filter(
                                    lambda ex: (
                                        ex.get("reported_reasons") is None
                                        and ex.get("downvotes_count", 0) == 0
                                        and ex.get("reported_count", 0) == 0
                                        and ex.get("client_id") not in blacklist_clients
                                    ),
                                    batched=True,
                                    batch_size=1000,
                                    num_proc=self.num_proc,
                                    desc=f"Filtering {split_name}",
                                )
                                filter_kept += len(filtered_data)
                                # Process filtered data
                                for item in tqdm(
                                    filtered_data,
                                    desc=f"Processing filtered {split_name}",
                                ):
                                    self._process_item(item, data_entry={})
                            else:
                                # Generic batch filter
                                filtered_data = split_data.filter(
                                    filter_fn,
                                    batched=False,
                                    num_proc=self.num_proc,
                                    desc=f"Filtering {split_name}",
                                )
                                filter_kept += len(filtered_data)
                                for item in tqdm(
                                    filtered_data,
                                    desc=f"Processing filtered {split_name}",
                                ):
                                    self._process_item(item, data_entry={})
                        except:
                            # Fallback to item-by-item if batch fails
                            for item in tqdm(
                                split_data, desc=f"Processing {split_name}"
                            ):
                                if filter_fn(item):
                                    filter_kept += 1
                                    self._process_item(item, data_entry={})
                else:
                    # Single dataset
                    total_original = len(dataset)
                    if "uzbekvoice" in data_path.lower():
                        blacklist_clients = [
                            "56ac8e86-b8c9-4879-a342-0eeb94f686fc",
                            "3d3fca02-6a07-41e2-9af4-60886ea60300",
                            "231d3776-2dbe-4a42-a535-c67943427e3f",
                            "e2716f95-70b5-4832-b903-eef2343591a4",
                            "2a815774-e953-4031-931a-8a28052e5cf9",
                            "d6fd3dc4-a55d-4a80-9bbf-b713325d05be",
                            "10b29e87-bf01-4b16-bead-a044076f849b",
                            "e3412d51-f079-4167-b3f9-311a976443ce",
                        ]
                        filtered_data = dataset.filter(
                            lambda ex: (
                                ex.get("reported_reasons") is None
                                and ex.get("downvotes_count", 0) == 0
                                and ex.get("reported_count", 0) == 0
                                and ex.get("client_id") not in blacklist_clients
                            ),
                            batched=True,
                            batch_size=1000,
                            num_proc=self.num_proc,
                            desc="Filtering dataset",
                        )
                        filter_kept = len(filtered_data)
                        for item in tqdm(
                            filtered_data, desc="Processing filtered data"
                        ):
                            self._process_item(item, data_entry={})
                    else:
                        try:
                            filtered_data = dataset.filter(
                                filter_fn,
                                batched=False,
                                num_proc=self.num_proc,
                                desc="Filtering dataset",
                            )
                            filter_kept = len(filtered_data)
                            for item in tqdm(
                                filtered_data, desc="Processing filtered data"
                            ):
                                self._process_item(item, data_entry={})
                        except:
                            for item in tqdm(dataset, desc="Processing dataset"):
                                if filter_fn(item):
                                    filter_kept += 1
                                    self._process_item(item, data_entry={})
            else:
                # Original item-by-item processing for chained iterators or no filter
                batch_size = 1000  # Process in batches to manage memory
                for idx, item in enumerate(
                    tqdm(dataset_iter, desc=f"Processing HF dataset {data_path}")
                ):
                    # Count and apply dataset-level filter quickly
                    total_original += 1
                    if filter_fn and not filter_fn(item):
                        continue
                    if filter_fn:
                        filter_kept += 1
                    self._process_item(item, data_entry={})

                    # Periodically clear memory
                    if idx > 0 and idx % batch_size == 0:
                        gc.collect()

            # After iterating, report filter stats if available
            if filter_fn and total_original > 0:
                print(
                    f"  Filtered dataset: {total_original} -> {filter_kept} samples (kept {filter_kept / total_original * 100:.1f}%)"
                )

        except Exception as e:
            print(f"Error loading Hugging Face dataset from {data_path}: {e}")
            raise

    def _load_csv_data(self, data_path):
        """Load data from CSV file. Supports multiple CSV formats with proper header mapping."""
        # Find matching filter function for this dataset
        filter_fn = None
        dataset_name = os.path.basename(data_path)
        for filter_config in self.dataset_filters:
            if filter_config["name"] in dataset_name:
                filter_fn = filter_config["filter_fn"]
                print(f"  Found filter for dataset '{filter_config['name']}'")
                break

        with open(data_path, "r", encoding="utf-8") as f:
            # Try to detect CSV format by reading first few lines
            sample_lines = []
            for i, line in enumerate(f):
                sample_lines.append(line.strip())
                if i >= 2:  # Read first 3 lines to detect format
                    break

        # Reset file pointer
        with open(data_path, "r", encoding="utf-8") as f:
            # Detect delimiter and format
            has_header = False
            delimiter = ","

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
            if "|" in first_line and "," not in first_line:
                delimiter = "|"

            reader = csv.reader(f, delimiter=delimiter)

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
                    if delimiter == "|":
                        # LJSpeech format: filename|text
                        if "|" in row[0] and len(row) == 1:
                            filename, text = row[0].split("|", 1)
                        else:
                            filename, text = row[0], row[1] if len(row) > 1 else ""
                    else:
                        # Standard CSV format: filename,text or audio_path,transcription
                        filename, text = row[0], row[1]

                # Skip empty entries
                if not filename or not text:
                    continue

                # Create line dict in expected format
                line = {
                    "audio": {"path": filename},
                    "sentence": normalize_text(text.strip()),
                }

                # Try to get audio duration if file exists
                try:
                    if os.path.isfile(filename):
                        audio_path = filename
                    else:
                        # Try relative to CSV file directory
                        csv_dir = os.path.dirname(data_path)
                        audio_path = os.path.join(csv_dir, filename)

                    if os.path.isfile(audio_path):
                        line["audio"]["path"] = audio_path
                        # FAST: use soundfile.info (no full decode)
                        info = soundfile.info(audio_path)
                        duration = round(info.frames / float(info.samplerate), 2)
                        line["duration"] = duration
                    else:
                        # Skip if audio file not found
                        print(f"Warning: Audio file not found: {filename}")
                        continue

                except Exception as e:
                    print(f"Warning: Could not read audio file {filename}: {e}")
                    continue

                # Apply filtering criteria
                if "duration" in line:
                    if line["duration"] < self.min_duration:
                        continue
                    if self.max_duration != -1 and line["duration"] > self.max_duration:
                        continue

                # Check sentence length limits
                if (
                    len(line["sentence"]) < self.min_sentence
                    or len(line["sentence"]) > self.max_sentence
                ):
                    continue

                # Apply custom filter if available for CSV data
                if filter_fn and not filter_fn(line):
                    continue

                self.data_list.append(line)

    # Get audio data, sample rate, and text from data list
    def _get_list_data(self, idx):
        if self.data_list_path.endswith(".header"):
            data_list = self.dataset_reader.get_data(self.data_list[idx])
        else:
            data_list = self.data_list[idx]

        # Check if we have preloaded audio array from Hugging Face dataset
        if "audio_array" in data_list:
            # Use preloaded audio array from Hugging Face dataset
            sample = data_list["audio_array"]
            sample_rate = data_list.get("sampling_rate", 16000)

            # Ensure it's float32 and 2D for consistency
            if not isinstance(sample, np.ndarray):
                sample = np.array(sample, dtype=np.float32)
            else:
                sample = sample.astype(np.float32)

            # Make it 2D if it's 1D (add channel dimension)
            if sample.ndim == 1:
                sample = sample.reshape(-1, 1)
        else:
            # Original file-based loading
            # Split audio path and labels
            if isinstance(data_list.get("audio"), dict):
                audio_file = data_list["audio"]["path"]
            else:
                audio_file = data_list.get("wav", data_list.get("audio"))

            # --------- FAST IO + CHEAP MONO ----------
            if (
                isinstance(data_list.get("audio"), dict)
                and "start_time" in data_list["audio"].keys()
            ):
                start_time, end_time = (
                    data_list["audio"]["start_time"],
                    data_list["audio"]["end_time"],
                )
                # Split and read audio (ensure 2D)
                sample, sample_rate = self.slice_from_file(
                    audio_file, start=start_time, end=end_time
                )  # shape: (N, C)
            else:
                # Read as (frames, channels) without decoding twice
                sample, sample_rate = soundfile.read(
                    audio_file, dtype="float32", always_2d=True
                )  # shape: (N, C)

        # Handle transcript extraction with fallback to 'text' column
        if self.timestamps:
            transcript = data_list.get("sentences", data_list.get("text", ""))
        else:
            # Try 'sentence' first, then fall back to 'text'
            transcript = data_list.get("sentence", data_list.get("text", ""))

        # Normalize text to use uniform apostrophe characters
        transcript = normalize_text(transcript)

        language = data_list["language"] if "language" in data_list.keys() else None

        # Convert to mono channel cheaply
        # sample shape is (N, C). If mono requested and C>1, average channels.
        if self.mono:
            if sample.ndim == 2:
                if sample.shape[1] > 1:
                    sample = sample.mean(axis=1).astype(np.float32)
                else:
                    sample = sample[:, 0].astype(np.float32)
            else:
                sample = sample.astype(np.float32)
        else:
            # keep multi-channel by flattening last dim if needed
            if sample.ndim == 2 and sample.shape[1] == 1:
                sample = sample[:, 0].astype(np.float32)

        # ------------------------------
        # Example A: remove silence for non-timestamp training
        # ------------------------------
        if not self.timestamps:
            sample = remove_silence_librosa(
                sample,
                sample_rate,
                top_db=self.silence_top_db,
                min_silence_ms=self.silence_min_gap_ms,
                pad_ms=self.silence_pad_ms,
            )

        # Data augmentation (after silence removal)
        if self.augment_configs:
            sample, sample_rate = self.augment(sample, sample_rate)

        # Resample - only downsample to 16000 when original sample rate is higher than 16000
        if sample_rate > 16000:
            sample = self.resample(sample, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000
        elif self.sample_rate != sample_rate:
            sample = self.resample(
                sample, orig_sr=sample_rate, target_sr=self.sample_rate
            )
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

    def _process_item(self, item, data_entry):
        """Helper method to process a single dataset item."""
        try:
            # Handle audio data - HF datasets often have 'audio' column with dict
            if "audio" in item:
                audio_data = item["audio"]
                if isinstance(audio_data, dict):
                    # Standard HF audio format
                    if "path" in audio_data:
                        data_entry["wav"] = audio_data["path"]
                    elif "filename" in audio_data:
                        data_entry["wav"] = audio_data["filename"]

                    # Get sample rate if available
                    if "sampling_rate" in audio_data:
                        sr = audio_data["sampling_rate"]
                        # We might need to resample if it's not 16kHz

                    # Get audio array if available
                    if "array" in audio_data:
                        # Store the array for direct use
                        data_entry["audio_array"] = audio_data["array"]
                        if "sampling_rate" in audio_data:
                            data_entry["sampling_rate"] = audio_data["sampling_rate"]
                else:
                    # Audio might be a path string
                    data_entry["wav"] = str(audio_data)

            # Alternative audio column names
            elif "audio_path" in item:
                data_entry["wav"] = item["audio_path"]
            elif "file" in item:
                data_entry["wav"] = item["file"]
            elif "filename" in item:
                data_entry["wav"] = item["filename"]
            elif "path" in item:
                data_entry["wav"] = item["path"]

            # Handle transcription/text
            text = None
            for text_key in [
                "transcription",
                "text",
                "sentence",
                "transcript",
                "label",
            ]:
                if text_key in item:
                    text = item[text_key]
                    break

            if text:
                data_entry["sentence"] = normalize_text(text)

            # Get or compute duration if possible
            if "duration" in item:
                data_entry["duration"] = item["duration"]
            elif "audio_array" in data_entry and "sampling_rate" in data_entry:
                # Compute duration from array length
                data_entry["duration"] = (
                    len(data_entry["audio_array"]) / data_entry["sampling_rate"]
                )
            else:
                # We'll compute it later when loading the audio
                data_entry["duration"] = -1  # Flag to compute later

            # Skip if we don't have both audio and text
            if "wav" not in data_entry and "audio_array" not in data_entry:
                return
            if "sentence" not in data_entry:
                return

            # Apply duration filters if duration is known
            if data_entry["duration"] != -1:
                if data_entry["duration"] < self.min_duration:
                    return
                if (
                    self.max_duration != -1
                    and data_entry["duration"] > self.max_duration
                ):
                    return

            # Apply sentence length filters
            if len(data_entry["sentence"]) < self.min_sentence:
                return
            if len(data_entry["sentence"]) > self.max_sentence:
                return

            self.data_list.append(data_entry)

        except Exception as e:
            print(f"Error processing item: {e}")

    def __getitem__(self, idx):
        try:
            # Get audio data, sample rate, and text from data list
            sample, sample_rate, transcript, language = self._get_list_data(idx=idx)

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
                    # normalize to single example (drop possible batch dim)
                    if isinstance(feats, list):
                        feats = feats[0]
                    elif (
                        hasattr(feats, "shape")
                        and getattr(feats, "shape", [None])[0] == 1
                    ):
                        feats = feats[0]
                    data["input_features"] = feats
                else:
                    # -------- non-timestamps training: return features + RAW TEXT --------
                    feats = self.processor(
                        audio=sample, sampling_rate=self.sample_rate
                    ).input_features
                    # normalize to single example (drop possible batch dim)
                    if isinstance(feats, list):
                        feats = feats[0]
                    elif (
                        hasattr(feats, "shape")
                        and getattr(feats, "shape", [None])[0] == 1
                    ):
                        feats = feats[0]

                    data = {
                        "input_features": feats,  # (80, T) or torch tensor of same
                        "text": transcript,  # <-- collator will batch-tokenize
                    }
            else:
                # If there's no text, use <|nospeech|> token (kept as IDs; collator pads)
                data = self.processor(audio=sample, sampling_rate=self.sample_rate)
                data["labels"] = [self.startoftranscript, self.nospeech, self.endoftext]

            return data

        except Exception as e:
            print(
                f"Error reading data, index: {idx}, error message: {e}", file=sys.stderr
            )
            return self.__getitem__(random.randint(0, self.__len__() - 1))

    def __len__(self):
        return len(self.data_list)

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
        for config in self.augment_configs:
            if config["type"] == "speed" and random.random() < config["prob"]:
                if self.speed_rates is None:
                    min_speed_rate, max_speed_rate, num_rates = (
                        config["params"]["min_speed_rate"],
                        config["params"]["max_speed_rate"],
                        config["params"]["num_rates"],
                    )
                    self.speed_rates = np.linspace(
                        min_speed_rate, max_speed_rate, num_rates, endpoint=True
                    )
                rate = random.choice(self.speed_rates)
                sample = self.change_speed(sample, speed_rate=rate)
            if config["type"] == "shift" and random.random() < config["prob"]:
                min_shift_ms, max_shift_ms = (
                    config["params"]["min_shift_ms"],
                    config["params"]["max_shift_ms"],
                )
                shift_ms = random.randint(min_shift_ms, max_shift_ms)
                sample = self.shift(sample, sample_rate, shift_ms=shift_ms)
            if config["type"] == "volume" and random.random() < config["prob"]:
                min_gain_dBFS, max_gain_dBFS = (
                    config["params"]["min_gain_dBFS"],
                    config["params"]["max_gain_dBFS"],
                )
                gain = random.randint(min_gain_dBFS, max_gain_dBFS)
                sample = self.volume(sample, gain=gain)
            if config["type"] == "resample" and random.random() < config["prob"]:
                new_sample_rates = config["params"]["new_sample_rates"]
                new_sample_rate = np.random.choice(new_sample_rates)
                sample = self.resample(
                    sample, orig_sr=sample_rate, target_sr=new_sample_rate
                )
                sample_rate = new_sample_rate
            if config["type"] == "noise" and random.random() < config["prob"]:
                min_snr_dB, max_snr_dB = (
                    config["params"]["min_snr_dB"],
                    config["params"]["max_snr_dB"],
                )
                if self.noises_path is None:
                    self.noises_path = []
                    noise_dir = config["params"]["noise_dir"]
                    if os.path.exists(noise_dir):
                        for file in os.listdir(noise_dir):
                            self.noises_path.append(os.path.join(noise_dir, file))
                if self.noises_path:
                    noise_path = random.choice(self.noises_path)
                    snr_dB = random.randint(min_snr_dB, max_snr_dB)
                    sample = self.add_noise(
                        sample, sample_rate, noise_path=noise_path, snr_dB=snr_dB
                    )
            if config["type"] == "gaussian" and random.random() < config["prob"]:
                min_snr_dB, max_snr_dB = (
                    config["params"]["min_snr_dB"],
                    config["params"]["max_snr_dB"],
                )
                snr_dB = random.uniform(min_snr_dB, max_snr_dB)
                sample = self.add_gaussian_noise(sample, snr_dB=snr_dB)
        return sample, sample_rate

    # Change speech speed
    @staticmethod
    def change_speed(sample, speed_rate):
        if speed_rate == 1.0:
            return sample
        if speed_rate <= 0:
            raise ValueError("Speed rate should be greater than zero")
        old_length = sample.shape[0]
        new_length = int(old_length / speed_rate)
        old_indices = np.arange(old_length)
        new_indices = np.linspace(start=0, stop=old_length, num=new_length)
        sample = np.interp(new_indices, old_indices, sample).astype(np.float32)
        return sample

    # Audio shift
    @staticmethod
    def shift(sample, sample_rate, shift_ms):
        duration = sample.shape[0] / sample_rate
        if abs(shift_ms) / 1000.0 > duration:
            raise ValueError(
                "Absolute value of shift_ms should be less than audio duration"
            )
        shift_samples = int(shift_ms * sample_rate / 1000)
        if shift_samples > 0:
            sample[:-shift_samples] = sample[shift_samples:]
            sample[-shift_samples:] = 0
        elif shift_samples < 0:
            sample[-shift_samples:] = sample[:shift_samples]
            sample[:-shift_samples] = 0
        return sample

    # Change volume
    @staticmethod
    def volume(sample, gain):
        sample *= 10.0 ** (gain / 20.0)
        return sample

    # Audio resampling (fast polyphase with safe fallback)
    @staticmethod
    def resample(sample, orig_sr, target_sr):
        sample = np.asarray(sample, dtype=np.float32)
        if orig_sr == target_sr:
            return sample
        try:
            from math import gcd
            from scipy.signal import resample_poly

            g = gcd(int(orig_sr), int(target_sr))
            up = target_sr // g
            down = orig_sr // g
            # sample expected to be 1D; ensure it
            if sample.ndim > 1:
                sample = sample.reshape(-1)
            out = resample_poly(sample, up, down)
            return out.astype(np.float32, copy=False)
        except Exception:
            # Fallback to librosa if SciPy isn't available
            return librosa.resample(
                sample, orig_sr=orig_sr, target_sr=target_sr
            ).astype(np.float32, copy=False)

    # Add noise (fast: soundfile + cheap mono + our resampler)
    def add_noise(self, sample, sample_rate, noise_path, snr_dB, max_gain_db=300.0):
        # sample is 1D float32 mono
        noise, sr = soundfile.read(noise_path, dtype="float32", always_2d=True)
        # Cheap mono
        if noise.ndim == 2:
            if noise.shape[1] > 1:
                noise = noise.mean(axis=1).astype(np.float32)
            else:
                noise = noise[:, 0].astype(np.float32)
        else:
            noise = noise.astype(np.float32)

        # Resample noise if needed (use fast polyphase)
        if sr != sample_rate:
            noise = self.resample(noise, orig_sr=sr, target_sr=sample_rate)

        # Normalize audio volume to ensure noise is not too loud
        target_db = -20
        gain = min(max_gain_db, target_db - self.rms_db(sample))
        sample = (sample * (10.0 ** (gain / 20.0))).astype(np.float32)

        # Specify noise volume
        sample_rms_db, noise_rms_db = self.rms_db(sample), self.rms_db(noise)
        noise_gain_db = min(sample_rms_db - noise_rms_db - snr_dB, max_gain_db)
        noise = (noise * (10.0 ** (noise_gain_db / 20.0))).astype(np.float32)

        # Fix noise length
        if noise.shape[0] < sample.shape[0]:
            rep = int(np.ceil(sample.shape[0] / noise.shape[0]))
            noise = np.tile(noise, rep)[: sample.shape[0]]
        elif noise.shape[0] > sample.shape[0]:
            start_frame = random.randint(0, noise.shape[0] - sample.shape[0])
            noise = noise[start_frame : start_frame + sample.shape[0]]

        sample = (sample + noise).astype(np.float32)
        return sample

    def add_gaussian_noise(self, sample, snr_dB):
        """
        Add Gaussian white noise to audio sample.

        Args:
            sample: Audio signal as 1D numpy array
            snr_dB: Signal-to-noise ratio in decibels

        Returns:
            Noisy audio sample
        """
        # Calculate signal power and convert SNR from dB
        signal_power = np.mean(sample**2)
        snr_linear = 10 ** (snr_dB / 10)

        # Calculate noise power based on desired SNR
        noise_power = signal_power / snr_linear

        # Generate Gaussian noise with calculated power
        noise = np.random.normal(0, np.sqrt(noise_power), sample.shape)

        # Add noise to signal
        noisy_sample = (sample + noise).astype(np.float32)

        # Clip to prevent overflow (optional, but recommended for audio)
        noisy_sample = np.clip(noisy_sample, -1.0, 1.0)

        return noisy_sample

    @staticmethod
    def rms_db(sample):
        mean_square = np.mean(sample**2) + 1e-12  # numerical safety
        return 10 * np.log10(mean_square)
