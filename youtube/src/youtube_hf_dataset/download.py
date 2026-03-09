from __future__ import annotations

import subprocess
from pathlib import Path

from youtube_hf_dataset.utils import run_command


def resolve_video_id(video_url: str, cookies: Path | None = None) -> str:
    cmd = [
        "yt-dlp",
        "--js-runtimes", "deno:/root/.deno/bin/deno",
        "--remote-components", "ejs:github",
    ]
    if cookies:
        cmd += ["--cookies", str(cookies)]
    cmd += [
        "--skip-download",
        "--no-playlist",
        "--print",
        "%(id)s",
        video_url,
    ]
    process = run_command(cmd)
    lines = [line.strip() for line in process.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"Could not resolve video ID for URL: {video_url}")
    return lines[-1]


def download_audio(
    video_url: str, output_dir: Path, video_id: str | None = None,
    cookies: Path | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "yt-dlp",
        "--js-runtimes", "deno:/root/.deno/bin/deno",
        "--remote-components", "ejs:github",
    ]
    if cookies:
        cmd += ["--cookies", str(cookies)]
    cmd += [
        "-f",
        "bestaudio/best",
        "--no-playlist",
        "-o",
        str(output_dir / "%(id)s.%(ext)s"),
        video_url,
    ]
    run_command(cmd)

    resolved_id = video_id or resolve_video_id(video_url)
    candidates = [
        path
        for path in output_dir.glob(f"{resolved_id}.*")
        if path.is_file() and path.suffix != ".part"
    ]
    if not candidates:
        raise FileNotFoundError(f"No audio download found for {video_url}")

    preferred_suffixes = (".m4a", ".webm", ".opus", ".mp3", ".aac", ".ogg")
    for suffix in preferred_suffixes:
        for candidate in candidates:
            if candidate.suffix.lower() == suffix:
                return candidate
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def download_subtitle_candidates(
    video_url: str,
    output_dir: Path,
    sub_langs: str,
    prefer_manual: bool,
    cookies: Path | None = None,
) -> tuple[list[Path], str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    video_id = resolve_video_id(video_url, cookies=cookies)

    def _clear_existing() -> None:
        for existing in output_dir.glob(f"{video_id}*.vtt"):
            existing.unlink()

    def _attempt(use_auto: bool) -> list[Path]:
        _clear_existing()
        caption_flag = "--write-auto-subs" if use_auto else "--write-subs"
        cmd = [
            "yt-dlp",
            "--js-runtimes", "deno:/root/.deno/bin/deno",
            "--remote-components", "ejs:github",
        ]
        if cookies:
            cmd += ["--cookies", str(cookies)]
        cmd += [
            "--skip-download",
            caption_flag,
            "--sub-langs",
            sub_langs,
            "--sub-format",
            "vtt",
            "--no-playlist",
            "-o",
            str(output_dir / "%(id)s.%(ext)s"),
            video_url,
        ]
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )
        files = [path for path in output_dir.glob(f"{video_id}*.vtt") if path.is_file()]
        files.sort(key=lambda p: p.name)
        if files:
            return files
        if process.returncode != 0:
            rendered = " ".join(process.args)
            raise RuntimeError(
                f"Command failed: {rendered}\nstdout:\n{process.stdout}\nstderr:\n{process.stderr}"
            )
        return files

    attempt_order = [False, True] if prefer_manual else [True, False]
    for use_auto in attempt_order:
        try:
            files = _attempt(use_auto=use_auto)
        except RuntimeError:
            files = []
        if files:
            source = "auto" if use_auto else "manual"
            return files, source
    return [], "none"


def convert_audio_to_wav16k(audio_file: Path, wav_dir: Path) -> Path:
    wav_dir.mkdir(parents=True, exist_ok=True)
    target = wav_dir / f"{audio_file.stem}.wav"
    run_command(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(audio_file),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-vn",
            "-loglevel",
            "error",
            str(target),
        ]
    )
    return target


def cut_audio_segment(
    source_wav: Path, output_wav: Path, start: float, end: float
) -> None:
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(source_wav),
            "-ss",
            f"{start:.3f}",
            "-to",
            f"{end:.3f}",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-vn",
            "-loglevel",
            "error",
            str(output_wav),
        ]
    )


def probe_duration(path: Path) -> float:
    process = run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
    )
    value = process.stdout.strip()
    if not value:
        raise RuntimeError(f"Could not read duration for: {path}")
    return float(value)
