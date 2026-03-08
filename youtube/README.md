# YouTube -> Hugging Face Audio Dataset

This project downloads YouTube videos from `ids.txt`, fetches captions, cuts audio into VTT-aligned chunks, and saves a Hugging Face `audiofolder` dataset layout.

## What it does

1. Reads IDs/URLs from `ids.txt`.
2. Downloads Uzbek captions (manual first, auto fallback by default).
3. Downloads best audio stream and converts it to 16kHz mono WAV.
4. Uses each VTT cue block as one segment.
5. Cuts audio chunks with `ffmpeg`.
6. Writes a Hugging Face `audiofolder` layout (`audio/` + `metadata.csv`).

## Requirements

- Python 3.10+
- `ffmpeg` and `ffprobe` in PATH
- `yt-dlp` in PATH

## Setup

```bash
cd youtube
python3 -m venv .venv
source .venv/bin/activate
```

## Prepare `ids.txt`

`ids.txt` should have one YouTube video ID or full URL per line.

Example:

```text
dQw4w9WgXcQ
https://www.youtube.com/watch?v=aqz-KE-bpKQ
```

## Run

```bash
PYTHONPATH=src python3 -m youtube_hf_dataset \
  --ids-txt ids.txt \
  --work-dir data \
  --output-dir data/hf_audio_dataset
```

Useful flags:

- `--sub-langs "uz-orig,uz"` to override the default Uzbek subtitle selection
- `--prefer-auto-subs` to try auto captions first
- `--keep-intermediate` to keep raw downloads and full WAVs
- `--overwrite-output` to replace non-empty output directory

## Output

- `data/audio_chunks/`: segmented WAV files
- `data/segments.jsonl`: metadata for each chunk
- `data/hf_audio_dataset/audio/`: segmented WAV files
- `data/hf_audio_dataset/metadata.csv`: metadata for `audiofolder`

Load the resulting dataset:

```python
from datasets import load_dataset

ds = load_dataset("audiofolder", data_dir="data/hf_audio_dataset", split="train")
print(ds[0])
```
