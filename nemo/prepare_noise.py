#!/usr/bin/env python3
"""
Download and prepare noise WAVs + a NeMo-compatible noise manifest (JSONL) from multiple
Hugging Face datasets:

  - Nexdata/Scene_Noise_Data
  - Nexdata/Scene_Noise_Data_by_Voice_Recorder
  - HHoofs/car-noise

Outputs:
  1) Local WAV files under: <out_dir>/wav/
  2) One merged NeMo noise manifest (JSONL), one entry per line:
       {"audio_filepath": "/abs/path/to/file.wav", "duration": 3.42}

Install:
  pip install datasets soundfile

Example:
  python prepare_noise.py \
    --out_dir ./output/noise_hf \
    --manifest_path ./output/noise_hf/noise_manifest.json

Optional:
  --datasets Nexdata/Scene_Noise_Data Nexdata/Scene_Noise_Data_by_Voice_Recorder HHoofs/car-noise
  --split train
  --max_items_per_split 10000
  --audio_column audio
"""

import argparse
import gc
import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Dict, Optional

import soundfile as sf
from datasets import load_dataset

DEFAULT_DATASETS = [
    "Nexdata/Scene_Noise_Data",
    "Nexdata/Scene_Noise_Data_by_Voice_Recorder",
    "HHoofs/car-noise",
]


def mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def sanitize(s: str) -> str:
    # Safe filename component
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def is_wav(path: str) -> bool:
    return str(path).lower().endswith(".wav")


def duration_of_wav(path: Path) -> float:
    info = sf.info(str(path))
    return float(info.frames) / float(info.samplerate)


def export_audio_to_wav(audio_obj: Dict, dst: Path) -> float:
    """
    audio_obj is typically a datasets Audio feature output:
      {"path": "...", "array": np.ndarray, "sampling_rate": int}

    Strategy:
    1) If audio_obj["path"] exists and is already WAV -> copy it (fast, preserves original)
    2) Else write decoded samples to WAV with soundfile

    Returns: duration in seconds
    """
    src = audio_obj.get("path")
    if src and os.path.exists(src) and is_wav(src):
        shutil.copy2(src, dst)
        return duration_of_wav(dst)

    arr = audio_obj["array"]
    sr = int(audio_obj["sampling_rate"])
    sf.write(str(dst), arr, sr)
    duration = float(len(arr)) / float(sr)

    # Explicitly delete array to free memory
    del arr

    return duration


def pick_audio_column(sample_item: Dict, preferred: Optional[str] = None) -> str:
    """
    Determine which column contains HF Audio objects.

    Preference order:
      1) user-specified preferred column if present
      2) column named 'audio'
      3) first column that looks like an Audio dict (has array + sampling_rate)
    """
    if preferred and preferred in sample_item:
        return preferred
    if "audio" in sample_item:
        return "audio"

    for k, v in sample_item.items():
        if isinstance(v, dict) and "array" in v and "sampling_rate" in v:
            return k

    raise KeyError(
        f"Could not find an audio column. Available keys: {list(sample_item.keys())}. "
        f"Try --audio_column <name>."
    )


def load_dataset_with_retry(
    ds_name: str,
    streaming: bool = True,
    split: Optional[str] = None,
    max_retries: int = 5,
    initial_delay: float = 5.0,
    backoff_factor: float = 2.0,
):
    """
    Load a HuggingFace dataset with retry logic and exponential backoff.

    Handles transient errors like 504 Gateway Timeout.
    Uses streaming mode by default to avoid loading entire dataset into memory.
    """
    delay = initial_delay
    last_exception = None

    for attempt in range(1, max_retries + 1):
        try:
            return load_dataset(ds_name, streaming=streaming, split=split)
        except Exception as e:
            last_exception = e
            error_str = str(e)
            # Check for transient/retryable errors
            is_retryable = any(
                x in error_str
                for x in [
                    "504",
                    "502",
                    "503",
                    "timeout",
                    "Timeout",
                    "Connection",
                    "SSLError",
                ]
            )

            if not is_retryable or attempt == max_retries:
                print(f"  Failed to load {ds_name} after {attempt} attempt(s): {e}")
                raise

            print(
                f"  Attempt {attempt}/{max_retries} failed for {ds_name} "
                f"(retryable error). Retrying in {delay:.1f}s..."
            )
            time.sleep(delay)
            delay *= backoff_factor

    # Should not reach here, but just in case
    raise last_exception


def get_available_splits(ds_name: str, max_retries: int = 5) -> list:
    """
    Get available splits for a dataset without loading data.
    """
    from datasets import get_dataset_config_names, get_dataset_split_names

    delay = 5.0
    for attempt in range(1, max_retries + 1):
        try:
            # Try to get split names directly
            try:
                splits = get_dataset_split_names(ds_name)
                return splits
            except Exception:
                # Fallback: load with streaming to check keys
                ds = load_dataset(ds_name, streaming=True)
                if hasattr(ds, "keys"):
                    return list(ds.keys())
                return ["train"]
        except Exception as e:
            error_str = str(e)
            is_retryable = any(
                x in error_str
                for x in [
                    "504",
                    "502",
                    "503",
                    "timeout",
                    "Timeout",
                    "Connection",
                    "SSLError",
                ]
            )
            if not is_retryable or attempt == max_retries:
                raise
            print(f"  Retry {attempt}/{max_retries} getting splits for {ds_name}...")
            time.sleep(delay)
            delay *= 2.0

    return ["train"]


def process_dataset_streaming(
    ds_name: str,
    split: str,
    wav_dir: Path,
    manifest_file,
    audio_column: Optional[str],
    max_items: int,
    max_retries: int,
    gc_interval: int = 100,
) -> int:
    """
    Process a dataset split using streaming mode to minimize memory usage.
    Returns number of items written.
    """
    print(f"  Loading split '{split}' in streaming mode...")

    try:
        ds_stream = load_dataset_with_retry(
            ds_name,
            streaming=True,
            split=split,
            max_retries=max_retries,
        )
    except Exception as e:
        print(f"  ERROR: Could not load {ds_name}:{split} - {e}")
        return 0

    written = 0
    audio_col = None

    for idx, item in enumerate(ds_stream):
        if max_items and written >= max_items:
            break

        # Detect audio column from first item
        if audio_col is None:
            try:
                audio_col = pick_audio_column(item, audio_column)
                print(f"  Detected audio column: '{audio_col}'")
            except KeyError as e:
                print(f"  ERROR: {e}")
                return 0

        audio_obj = item.get(audio_col)
        if audio_obj is None:
            continue

        # Filename embeds dataset + split + index
        fname = f"{sanitize(ds_name)}_{sanitize(split)}_{idx:08d}.wav"
        wav_path = (wav_dir / fname).resolve()

        try:
            duration = export_audio_to_wav(audio_obj, wav_path)
        except Exception as e:
            print(f"    WARNING: failed item {idx} ({e}); skipped")
            continue

        manifest_file.write(
            json.dumps(
                {"audio_filepath": str(wav_path), "duration": duration},
                ensure_ascii=False,
            )
            + "\n"
        )
        manifest_file.flush()  # Flush after each write for safety

        written += 1

        # Periodic garbage collection to free memory
        if written % gc_interval == 0:
            gc.collect()
            print(f"    Processed {written} items...")

        # Clear references to free memory
        del audio_obj
        del item

    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out_dir",
        required=True,
        help="Base output directory; WAVs will be written under <out_dir>/wav/",
    )
    ap.add_argument(
        "--manifest_path",
        required=True,
        help="Output JSONL manifest file path (merged).",
    )
    ap.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="HF dataset identifiers to process (space-separated).",
    )
    ap.add_argument(
        "--split",
        default=None,
        help="Optional: process only one split name (e.g., 'train'). If omitted, process all splits found.",
    )
    ap.add_argument(
        "--max_items_per_split",
        type=int,
        default=0,
        help="Optional: limit number of items per split (0 = no limit).",
    )
    ap.add_argument(
        "--audio_column",
        default=None,
        help="Optional: specify the name of the dataset column containing audio.",
    )
    ap.add_argument(
        "--max_retries",
        type=int,
        default=5,
        help="Max retries for transient download failures (default: 5).",
    )
    ap.add_argument(
        "--gc_interval",
        type=int,
        default=100,
        help="Run garbage collection every N items (default: 100).",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    wav_dir = out_dir / "wav"
    manifest_path = Path(args.manifest_path).expanduser().resolve()

    mkdir(wav_dir)
    mkdir(manifest_path.parent)

    total_written = 0

    with open(manifest_path, "w", encoding="utf-8") as mf:
        for ds_name in args.datasets:
            print(f"\n{'=' * 60}")
            print(f"Processing dataset: {ds_name}")
            print(f"{'=' * 60}")

            # Get available splits
            try:
                available_splits = get_available_splits(ds_name, args.max_retries)
                print(f"  Available splits: {available_splits}")
            except Exception as e:
                print(f"  ERROR: Could not get splits for {ds_name}: {e}")
                print(f"  Skipping this dataset.")
                continue

            splits = [args.split] if args.split else available_splits

            for split in splits:
                if split not in available_splits:
                    print(
                        f"  WARNING: Split '{split}' not found. Available: {available_splits}"
                    )
                    continue

                written_split = process_dataset_streaming(
                    ds_name=ds_name,
                    split=split,
                    wav_dir=wav_dir,
                    manifest_file=mf,
                    audio_column=args.audio_column,
                    max_items=args.max_items_per_split,
                    max_retries=args.max_retries,
                    gc_interval=args.gc_interval,
                )

                print(f"  Split '{split}': exported {written_split} WAVs")
                total_written += written_split

                # Force garbage collection between splits
                gc.collect()

    print(f"\n{'=' * 60}")
    print("Completed.")
    print(f"{'=' * 60}")
    print(f"Datasets:       {args.datasets}")
    print(f"WAV directory:  {wav_dir}")
    print(f"Manifest file:  {manifest_path}")
    print(f"Total entries:  {total_written}")


if __name__ == "__main__":
    main()
