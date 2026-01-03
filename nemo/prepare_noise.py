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
import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional

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
    return float(len(arr)) / float(sr)


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


def iter_splits(ds) -> List[str]:
    # ds is a DatasetDict or a Dataset
    return list(ds.keys()) if hasattr(ds, "keys") else [""]


def load_dataset_with_retry(
    ds_name: str,
    max_retries: int = 5,
    initial_delay: float = 5.0,
    backoff_factor: float = 2.0,
):
    """
    Load a HuggingFace dataset with retry logic and exponential backoff.

    Handles transient errors like 504 Gateway Timeout.
    """
    delay = initial_delay
    last_exception = None

    for attempt in range(1, max_retries + 1):
        try:
            return load_dataset(ds_name)
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
    args = ap.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    wav_dir = out_dir / "wav"
    manifest_path = Path(args.manifest_path).expanduser().resolve()

    mkdir(wav_dir)
    mkdir(manifest_path.parent)

    total_written = 0

    with open(manifest_path, "w", encoding="utf-8") as mf:
        for ds_name in args.datasets:
            print(f"\nLoading dataset: {ds_name}")
            try:
                ds = load_dataset_with_retry(ds_name, max_retries=args.max_retries)
            except Exception as e:
                print(f"  ERROR: Skipping dataset {ds_name} due to: {e}")
                continue

            # Determine splits
            available_splits = list(ds.keys()) if hasattr(ds, "keys") else ["train"]
            splits = [args.split] if args.split else available_splits

            for split in splits:
                if hasattr(ds, "keys") and split not in ds:
                    raise ValueError(
                        f"Split '{split}' not found in {ds_name}. Available splits: {available_splits}"
                    )

                dsplit = ds[split] if hasattr(ds, "keys") else ds

                if len(dsplit) == 0:
                    print(f"  Split '{split}': 0 items (skipped)")
                    continue

                # Detect audio column using first item (already decoded by HF Audio feature)
                first_item = dsplit[0]
                audio_col = pick_audio_column(first_item, args.audio_column)

                written_split = 0
                limit = args.max_items_per_split

                for idx, item in enumerate(dsplit):
                    if limit and written_split >= limit:
                        break

                    audio_obj = item[audio_col]
                    # Filename embeds dataset + split + index
                    fname = f"{sanitize(ds_name)}_{sanitize(split)}_{idx:08d}.wav"
                    wav_path = (wav_dir / fname).resolve()

                    try:
                        duration = export_audio_to_wav(audio_obj, wav_path)
                    except Exception as e:
                        # Skip broken items but continue; log enough for troubleshooting
                        print(
                            f"    WARNING: failed item {idx} in {ds_name}:{split} ({e}); skipped"
                        )
                        continue

                    mf.write(
                        json.dumps(
                            {"audio_filepath": str(wav_path), "duration": duration},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                    written_split += 1
                    total_written += 1

                print(
                    f"  Split '{split}': exported {written_split} WAVs (audio column: '{audio_col}')"
                )

    print("\nCompleted.")
    print(f"Datasets:       {args.datasets}")
    print(f"WAV directory:  {wav_dir}")
    print(f"Manifest file:  {manifest_path}")
    print(f"Total entries:  {total_written}")


if __name__ == "__main__":
    main()
