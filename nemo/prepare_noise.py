#!/usr/bin/env python3
"""
Download and prepare noise WAVs + a NeMo-compatible noise manifest (JSONL) from multiple
Hugging Face datasets using direct file downloads (memory-efficient).

Datasets:
  - Nexdata/Scene_Noise_Data
  - Nexdata/Scene_Noise_Data_by_Voice_Recorder
  - HHoofs/car-noise

Outputs:
  1) Local WAV files under: <out_dir>/wav/
  2) One merged NeMo noise manifest (JSONL), one entry per line:
       {"audio_filepath": "/abs/path/to/file.wav", "duration": 3.42}

Install:
  pip install huggingface_hub soundfile

Example:
  python prepare_noise.py \
    --out_dir ./output/noise_hf \
    --manifest_path ./output/noise_hf/noise_manifest.json

Optional:
  --datasets Nexdata/Scene_Noise_Data Nexdata/Scene_Noise_Data_by_Voice_Recorder HHoofs/car-noise
  --max_files 10000
"""

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import List, Optional

import soundfile as sf
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError

DEFAULT_DATASETS = [
    "Nexdata/Scene_Noise_Data",
    "Nexdata/Scene_Noise_Data_by_Voice_Recorder",
    "HHoofs/car-noise",
]

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus"}


def mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def sanitize(s: str) -> str:
    """Safe filename component."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def is_audio_file(filename: str) -> bool:
    """Check if file has an audio extension."""
    return Path(filename).suffix.lower() in AUDIO_EXTENSIONS


def get_duration(path: Path) -> Optional[float]:
    """Get audio duration in seconds, returns None on failure."""
    try:
        info = sf.info(str(path))
        return float(info.frames) / float(info.samplerate)
    except Exception:
        return None


def convert_to_wav(src: Path, dst: Path) -> Optional[float]:
    """
    Convert audio file to WAV format if needed.
    Returns duration in seconds, or None on failure.
    """
    try:
        # Read audio data
        data, samplerate = sf.read(str(src))
        # Write as WAV
        sf.write(str(dst), data, samplerate)
        duration = float(len(data)) / float(samplerate)
        return duration
    except Exception as e:
        print(f"    WARNING: Could not process {src.name}: {e}")
        return None


def list_repo_audio_files(
    repo_id: str,
    max_retries: int = 5,
    initial_delay: float = 5.0,
) -> List[str]:
    """
    List all audio files in a HuggingFace dataset repository.
    Uses retry logic for transient errors.
    """
    api = HfApi()
    delay = initial_delay

    for attempt in range(1, max_retries + 1):
        try:
            files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
            audio_files = [f for f in files if is_audio_file(f)]
            return audio_files
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

            print(
                f"  Attempt {attempt}/{max_retries} failed. Retrying in {delay:.1f}s..."
            )
            time.sleep(delay)
            delay *= 2.0

    return []


def download_file_with_retry(
    repo_id: str,
    filename: str,
    local_dir: Path,
    max_retries: int = 5,
    initial_delay: float = 5.0,
) -> Optional[Path]:
    """
    Download a single file from HuggingFace with retry logic.
    Returns local path on success, None on failure.
    """
    delay = initial_delay

    for attempt in range(1, max_retries + 1):
        try:
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",
                local_dir=str(local_dir),
            )
            return Path(local_path)
        except HfHubHTTPError as e:
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
                print(f"    Failed to download {filename}: {e}")
                return None

            print(f"    Retry {attempt}/{max_retries} for {filename}...")
            time.sleep(delay)
            delay *= 2.0
        except Exception as e:
            print(f"    Failed to download {filename}: {e}")
            return None

    return None


def process_dataset(
    repo_id: str,
    wav_dir: Path,
    manifest_file,
    max_files: int = 0,
    max_retries: int = 5,
    cache_dir: Optional[Path] = None,
) -> int:
    """
    Process a single dataset: list files, download, convert to WAV, write manifest.
    Returns number of files successfully processed.
    """
    print(f"\n{'=' * 60}")
    print(f"Processing dataset: {repo_id}")
    print(f"{'=' * 60}")

    # List audio files in repository
    try:
        audio_files = list_repo_audio_files(repo_id, max_retries=max_retries)
        print(f"  Found {len(audio_files)} audio files")
    except Exception as e:
        print(f"  ERROR: Could not list files in {repo_id}: {e}")
        return 0

    if not audio_files:
        print(f"  No audio files found, skipping.")
        return 0

    # Limit files if requested
    if max_files > 0:
        audio_files = audio_files[:max_files]
        print(f"  Processing first {len(audio_files)} files (--max_files limit)")

    # Create a temp download directory
    download_dir = cache_dir or (wav_dir.parent / ".hf_cache" / sanitize(repo_id))
    mkdir(download_dir)

    written = 0
    dataset_prefix = sanitize(repo_id)

    for idx, remote_file in enumerate(audio_files):
        # Download file
        local_file = download_file_with_retry(
            repo_id=repo_id,
            filename=remote_file,
            local_dir=download_dir,
            max_retries=max_retries,
        )

        if local_file is None:
            continue

        # Determine output WAV path
        base_name = Path(remote_file).stem
        wav_name = f"{dataset_prefix}_{idx:08d}_{sanitize(base_name)}.wav"
        wav_path = wav_dir / wav_name

        # Convert to WAV (or copy if already WAV)
        if local_file.suffix.lower() == ".wav":
            # Already WAV - get duration and copy/link
            duration = get_duration(local_file)
            if duration is None:
                print(f"    WARNING: Could not read {local_file.name}, skipping")
                continue
            # Copy to output directory
            try:
                import shutil

                shutil.copy2(local_file, wav_path)
            except Exception as e:
                print(f"    WARNING: Could not copy {local_file.name}: {e}")
                continue
        else:
            # Convert to WAV
            duration = convert_to_wav(local_file, wav_path)
            if duration is None:
                continue

        # Write manifest entry
        manifest_file.write(
            json.dumps(
                {"audio_filepath": str(wav_path.resolve()), "duration": duration},
                ensure_ascii=False,
            )
            + "\n"
        )
        manifest_file.flush()

        written += 1

        # Progress indicator
        if written % 50 == 0:
            print(f"    Processed {written} files...")

        # Clean up downloaded file to save space
        try:
            local_file.unlink()
        except Exception:
            pass

    print(f"  Completed: {written} WAV files exported")
    return written


def main():
    ap = argparse.ArgumentParser(
        description="Download noise datasets from HuggingFace and create NeMo manifest"
    )
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
        "--max_files",
        type=int,
        default=0,
        help="Max files to download per dataset (0 = no limit).",
    )
    ap.add_argument(
        "--max_retries",
        type=int,
        default=5,
        help="Max retries for transient download failures (default: 5).",
    )
    ap.add_argument(
        "--cache_dir",
        default=None,
        help="Optional: directory for temporary downloads (cleaned up after processing).",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    wav_dir = out_dir / "wav"
    manifest_path = Path(args.manifest_path).expanduser().resolve()
    cache_dir = Path(args.cache_dir).expanduser().resolve() if args.cache_dir else None

    mkdir(wav_dir)
    mkdir(manifest_path.parent)
    if cache_dir:
        mkdir(cache_dir)

    total_written = 0

    with open(manifest_path, "w", encoding="utf-8") as mf:
        for ds_name in args.datasets:
            written = process_dataset(
                repo_id=ds_name,
                wav_dir=wav_dir,
                manifest_file=mf,
                max_files=args.max_files,
                max_retries=args.max_retries,
                cache_dir=cache_dir,
            )
            total_written += written

    print(f"\n{'=' * 60}")
    print("Completed.")
    print(f"{'=' * 60}")
    print(f"Datasets:       {args.datasets}")
    print(f"WAV directory:  {wav_dir}")
    print(f"Manifest file:  {manifest_path}")
    print(f"Total entries:  {total_written}")


if __name__ == "__main__":
    main()
