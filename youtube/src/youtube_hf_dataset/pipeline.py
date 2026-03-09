from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from youtube_hf_dataset.captions import parse_vtt_aligned_cues, pick_best_vtt
from youtube_hf_dataset.download import (
    convert_audio_to_wav16k,
    cut_audio_segment,
    download_audio,
    download_subtitle_candidates,
    probe_duration,
    resolve_video_id,
)
from youtube_hf_dataset.ids import read_ids, to_video_url
from youtube_hf_dataset.segment import cues_as_segments
from youtube_hf_dataset.utils import require_binaries


@dataclass
class BuildConfig:
    ids_txt: Path
    work_dir: Path
    output_dir: Path
    min_seconds: float = 0.4
    sub_langs: str = "uz-orig,uz,uz-Latn,uz-Cyrl"
    cookies: Path | None = None
    prefer_manual_subs: bool = True
    keep_intermediate: bool = False
    overwrite_output: bool = False


def build_hf_audio_dataset(config: BuildConfig) -> Path:
    require_binaries(["yt-dlp", "ffmpeg", "ffprobe"])

    ids = read_ids(config.ids_txt)
    if not ids:
        raise ValueError(f"No IDs found in {config.ids_txt}")

    downloads_dir = config.work_dir / "downloads"
    subtitles_dir = config.work_dir / "subtitles"
    full_wav_dir = config.work_dir / "wav_full"
    chunks_dir = config.output_dir / "audio"
    manifest_path = config.work_dir / "segments.jsonl"
    metadata_path = config.output_dir / "metadata.csv"

    if config.output_dir.exists() and any(config.output_dir.iterdir()):
        if not config.overwrite_output:
            raise RuntimeError(
                f"Output directory is not empty: {config.output_dir}. "
                "Use --overwrite-output to replace it."
            )
        shutil.rmtree(config.output_dir)

    for directory in (
        downloads_dir,
        subtitles_dir,
        full_wav_dir,
        chunks_dir,
        config.output_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    all_records: list[dict[str, str | float]] = []
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for raw_video in ids:
            video_url = to_video_url(raw_video)
            try:
                video_id = resolve_video_id(video_url, cookies=config.cookies)
            except Exception as exc:  # noqa: BLE001
                print(f"[skip] Could not resolve ID for {raw_video}: {exc}")
                continue

            print(f"[video] {video_id}")
            subtitle_files, subtitle_source = download_subtitle_candidates(
                video_url=video_url,
                output_dir=subtitles_dir,
                sub_langs=config.sub_langs,
                prefer_manual=config.prefer_manual_subs,
            )
            if not subtitle_files:
                print(f"  - no captions found for {video_id}; skipping")
                continue

            chosen_vtt, cues = pick_best_vtt(
                subtitle_files, parser=parse_vtt_aligned_cues
            )
            if chosen_vtt is None or not cues:
                print(
                    f"  - caption files exist but no usable cues for {video_id}; skipping"
                )
                continue

            segments = cues_as_segments(
                cues=cues,
                min_seconds=config.min_seconds,
            )
            if not segments:
                print(f"  - no segments produced for {video_id}; skipping")
                continue

            try:
                audio_file = download_audio(
                    video_url=video_url, output_dir=downloads_dir, video_id=video_id
                )
                full_wav = convert_audio_to_wav16k(
                    audio_file=audio_file, wav_dir=full_wav_dir
                )
                full_duration = probe_duration(full_wav)
            except Exception as exc:  # noqa: BLE001
                print(f"  - audio download/convert failed for {video_id}: {exc}")
                continue

            written = 0
            for index, segment in enumerate(segments):
                start = max(0.0, segment.start)
                end = min(full_duration, segment.end)
                if end - start < config.min_seconds:
                    continue

                chunk_path = chunks_dir / video_id / f"{video_id}_{index:05d}.wav"
                try:
                    cut_audio_segment(full_wav, chunk_path, start=start, end=end)
                    duration = probe_duration(chunk_path)
                except Exception as exc:  # noqa: BLE001
                    print(f"  - failed to cut segment {video_id}:{index}: {exc}")
                    continue

                if duration < config.min_seconds:
                    continue

                record = {
                    "id": f"{video_id}_{index:05d}",
                    "video_id": video_id,
                    "file_name": str(chunk_path.relative_to(config.output_dir)),
                    "text": segment.text,
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "duration": round(duration, 3),
                    "caption_source": subtitle_source,
                    "caption_file": chosen_vtt.name,
                    "youtube_url": video_url,
                }
                manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
                all_records.append(record)
                written += 1

            print(f"  - kept {written} chunks")

    if not all_records:
        raise RuntimeError(
            "No samples were created. Check IDs and captions availability."
        )

    fieldnames = [
        "id",
        "file_name",
        "text",
        "video_id",
        "start",
        "end",
        "duration",
        "caption_source",
        "caption_file",
        "youtube_url",
    ]
    with metadata_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)

    (config.output_dir / "README.txt").write_text(
        "Hugging Face audiofolder dataset layout.\n"
        "Use: load_dataset('audiofolder', data_dir='<this_dir>')\n",
        encoding="utf-8",
    )

    if not config.keep_intermediate:
        shutil.rmtree(downloads_dir, ignore_errors=True)
        shutil.rmtree(subtitles_dir, ignore_errors=True)
        shutil.rmtree(full_wav_dir, ignore_errors=True)

    print(
        f"[done] Saved audiofolder dataset with {len(all_records)} rows to {config.output_dir}"
    )
    print(f"[done] Segment manifest: {manifest_path}")
    return config.output_dir
