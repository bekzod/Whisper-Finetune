#!/usr/bin/env python3
"""
transcribe_and_slice.py

Transcribe audio files in a folder using a NeMo ASR model with timestamps,
then slice each segment into a separate 16 kHz mono WAV file.

Output:
  - audio/  directory with numbered segment WAVs (16 kHz, mono)
  - transcription.jsonl with one line per segment:
      {"audio_filepath": "audio/000000000.wav", "duration": 4.896, "text": "..."}

Example:
  python transcribe_and_slice.py \
    --input-dir /path/to/audio/folder \
    --output-dir /path/to/output \
    --model nvidia/parakeet-tdt-0.6b-v3 \
    --batch-size 8

Install:
  pip install nemo_toolkit['asr'] soundfile librosa tqdm
"""

import argparse
import json
import logging
import os
from pathlib import Path

import nemo.collections.asr as nemo_asr
import soundfile as sf
import torch
from huggingface_hub import hf_hub_download
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a", ".aac"}


def find_audio_files(input_dir: str) -> list[str]:
    """Collect all audio files in the directory, sorted by name."""
    files = []
    for entry in sorted(os.listdir(input_dir)):
        if Path(entry).suffix.lower() in AUDIO_EXTENSIONS:
            files.append(os.path.join(input_dir, entry))
    return files


def transcribe_with_timestamps(model, audio_paths: list[str]) -> list[list[dict]]:
    """
    Transcribe audio files and return per-file list of segments
    with start, end, and text.
    """
    # Use transcribe with timestamps enabled
    outputs = model.transcribe(
        audio_paths,
        batch_size=len(audio_paths),
        timestamps=True,
    )

    all_segments = []
    for out in outputs:
        segments = []
        if hasattr(out, "timestep") and out.timestep is not None:
            # Handle NeMo Hypothesis with timestep info
            ts = out.timestep
            if hasattr(ts, "segments") and ts.segments:
                for seg in ts.segments:
                    segments.append(
                        {
                            "start": seg.start,
                            "end": seg.end,
                            "text": seg.text if hasattr(seg, "text") else str(seg),
                        }
                    )
            elif hasattr(ts, "words") and ts.words:
                # Fall back to word-level timestamps grouped into segments
                # Group words into segments by silence gaps
                segments = _words_to_segments(ts.words)
        elif hasattr(out, "segments") and out.segments:
            for seg in out.segments:
                segments.append(
                    {
                        "start": seg.start if hasattr(seg, "start") else seg["start"],
                        "end": seg.end if hasattr(seg, "end") else seg["end"],
                        "text": seg.text if hasattr(seg, "text") else seg["text"],
                    }
                )

        if not segments:
            # Fallback: entire file as one segment
            text = out.text if hasattr(out, "text") else str(out)
            segments.append(
                {
                    "start": None,
                    "end": None,
                    "text": text,
                }
            )

        all_segments.append(segments)
    return all_segments


def _words_to_segments(words, gap_threshold: float = 0.5) -> list[dict]:
    """Group word-level timestamps into segments based on silence gaps."""
    if not words:
        return []

    segments = []
    current_words = []
    current_start = None
    current_end = None

    for w in words:
        w_start = w.start if hasattr(w, "start") else w["start"]
        w_end = w.end if hasattr(w, "end") else w["end"]
        w_text = w.text if hasattr(w, "text") else w.get("text", str(w))

        if current_start is None:
            current_start = w_start
            current_end = w_end
            current_words.append(w_text)
        elif w_start - current_end > gap_threshold:
            segments.append(
                {
                    "start": current_start,
                    "end": current_end,
                    "text": " ".join(current_words).strip(),
                }
            )
            current_start = w_start
            current_end = w_end
            current_words = [w_text]
        else:
            current_end = w_end
            current_words.append(w_text)

    if current_words:
        segments.append(
            {
                "start": current_start,
                "end": current_end,
                "text": " ".join(current_words).strip(),
            }
        )

    return segments


def slice_and_save(
    source_path: str,
    start: float,
    end: float,
    output_path: str,
    target_sr: int = 16000,
) -> float:
    """
    Read a segment from source audio, resample to target_sr mono, save as WAV.
    Returns actual duration in seconds.
    """
    info = sf.info(source_path)
    sr = info.samplerate

    start_frame = int(start * sr)
    end_frame = int(end * sr)
    num_frames = end_frame - start_frame

    data, file_sr = sf.read(
        source_path, start=start_frame, frames=num_frames, dtype="float32"
    )

    # Convert to mono if needed
    if data.ndim > 1:
        data = data.mean(axis=1)

    # Resample if needed
    if file_sr != target_sr:
        import librosa

        data = librosa.resample(data, orig_sr=file_sr, target_sr=target_sr)

    sf.write(output_path, data, target_sr, subtype="PCM_16")
    return len(data) / target_sr


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio files and slice into segments"
    )
    parser.add_argument(
        "--input-dir", required=True, help="Folder containing audio files"
    )
    parser.add_argument(
        "--output-dir", required=True, help="Output folder for segments and manifest"
    )
    parser.add_argument(
        "--model",
        default="bekzod123/nemo_asr_2",
        help="NeMo/HF model name or local .nemo path",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    # Find audio files
    audio_files = find_audio_files(args.input_dir)
    if not audio_files:
        LOGGER.error("No audio files found in %s", args.input_dir)
        return

    LOGGER.info("Found %d audio files in %s", len(audio_files), args.input_dir)

    # Create output dirs
    audio_out_dir = os.path.join(args.output_dir, "audio")
    os.makedirs(audio_out_dir, exist_ok=True)

    manifest_path = os.path.join(args.output_dir, "transcription.jsonl")

    # Load model on CPU
    LOGGER.info("Loading model: %s", args.model)
    if args.model.endswith(".nemo") and os.path.exists(args.model):
        local_nemo_path = args.model
    else:
        local_nemo_path = hf_hub_download(
            repo_id=args.model, filename="nemo_asr_2-3.nemo", repo_type="model"
        )

    model = nemo_asr.models.ASRModel.restore_from(
        restore_path=local_nemo_path, map_location="cpu"
    )
    model.eval()
    LOGGER.info("Model ready on CPU")

    # Process in batches
    segment_counter = 0

    with open(manifest_path, "w", encoding="utf-8") as manifest_f:
        for i in tqdm(
            range(0, len(audio_files), args.batch_size),
            total=(len(audio_files) + args.batch_size - 1) // args.batch_size,
            desc="Transcribing",
        ):
            batch_paths = audio_files[i : i + args.batch_size]

            with torch.no_grad():
                batch_segments = transcribe_with_timestamps(model, batch_paths)

            for audio_path, segments in zip(batch_paths, batch_segments):
                for seg in segments:
                    text = seg["text"].strip()
                    if not text:
                        continue

                    out_filename = f"{segment_counter:09d}.wav"
                    out_path = os.path.join(audio_out_dir, out_filename)

                    if seg["start"] is not None and seg["end"] is not None:
                        duration = slice_and_save(
                            audio_path, seg["start"], seg["end"], out_path
                        )
                    else:
                        # No timestamps — copy entire file as 16kHz mono
                        info = sf.info(audio_path)
                        duration = slice_and_save(
                            audio_path, 0.0, info.duration, out_path
                        )

                    record = {
                        "audio_filepath": f"audio/{out_filename}",
                        "duration": round(duration, 3),
                        "text": text,
                    }
                    manifest_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    segment_counter += 1

            manifest_f.flush()

    LOGGER.info("Done. %d segments saved to %s", segment_counter, args.output_dir)
    LOGGER.info("Manifest: %s", manifest_path)
    print(f"\nTotal segments: {segment_counter}")
    print(f"Audio dir:     {audio_out_dir}")
    print(f"Manifest:      {manifest_path}")


if __name__ == "__main__":
    main()
