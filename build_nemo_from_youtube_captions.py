import argparse
import html
import json
import re
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

import soundfile as sf
import webvtt

from nemo.utils import _APOSTROPHE_RE, _DASH_RE

_VTT_TIMESTAMP_LINE_RE = re.compile(
    r"^(?P<start>\d{2}:\d{2}(?::\d{2})?\.\d{3})\s*-->\s*(?P<end>\d{2}:\d{2}(?::\d{2})?\.\d{3})"
)
_VTT_WORD_TAG_RE = re.compile(
    r"<(?P<ts>\d{2}:\d{2}(?::\d{2})?\.\d{3})><c>(?P<word>[^<]*)</c>"
)


def run(cmd: List[str]) -> None:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}"
        )


def sanitize_id(s: str) -> str:
    s = s.strip()
    return re.sub(r"[^A-Za-z0-9_\-]", "_", s)


def read_ids(txt_path: Path) -> List[str]:
    ids = []
    for line in txt_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        ids.append(line)
    return ids


def to_youtube_url(video_id_or_url: str) -> str:
    if video_id_or_url.startswith("http://") or video_id_or_url.startswith("https://"):
        return video_id_or_url
    return f"https://www.youtube.com/watch?v={video_id_or_url}"


def download_audio(video: str, out_dir: Path) -> Path:
    """
    Download best audio only. Output file could be m4a/webm/etc.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    vid = sanitize_id(video)
    out_template = str(out_dir / f"{vid}.%(ext)s")

    cmd = [
        "yt-dlp",
        "-f",
        "bestaudio/best",
        "-o",
        out_template,
        "--no-playlist",
        to_youtube_url(video),
    ]
    run(cmd)

    candidates = list(out_dir.glob(f"{vid}.*"))
    if not candidates:
        raise FileNotFoundError(f"No audio output found for {video}")

    # Prefer typical audio containers
    for ext in (".m4a", ".webm", ".opus", ".mp3", ".aac", ".ogg", ".mka"):
        for c in candidates:
            if c.suffix.lower() == ext:
                return c
    return candidates[0]


def convert_to_wav_16k_mono(src: Path, wav_dir: Path) -> Path:
    wav_dir.mkdir(parents=True, exist_ok=True)
    dst = wav_dir / (src.stem + ".wav")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-vn",
        "-loglevel",
        "error",
        str(dst),
    ]
    run(cmd)
    return dst


def wav_duration(wav_path: Path) -> float:
    info = sf.info(str(wav_path))
    return float(info.frames) / float(info.samplerate)


def download_uzbek_vtt(
    video: str, subs_dir: Path, prefer_manual: bool
) -> Optional[Path]:
    """
    Attempts to download Uzbek subtitles as .vtt.
    We try a list of Uzbek language tags because YouTube sometimes uses variants.
    Returns path to the downloaded .vtt if found, else None.

    prefer_manual=True: try manual subs first; if not found, fallback to auto.
    """
    subs_dir.mkdir(parents=True, exist_ok=True)
    vid = sanitize_id(video)
    out_template = str(subs_dir / f"{vid}.%(ext)s")

    # Common Uzbek language tags seen on YouTube/yt-dlp
    # You can add/remove depending on what you see from `yt-dlp --list-subs`
    uz_langs = ["uz", "uz-Latn", "uz-Cyrl", "uzbek"]

    def attempt(auto: bool) -> Optional[Path]:
        cmd = [
            "yt-dlp",
            "--skip-download",
            "--write-subs" if not auto else "--write-auto-subs",
            "--sub-langs",
            ",".join(uz_langs),
            "--sub-format",
            "vtt",
            "-o",
            out_template,
            "--no-playlist",
            to_youtube_url(video),
        ]
        try:
            run(cmd)
        except Exception:
            return None

        # yt-dlp typically writes something like: <id>.<lang>.vtt
        vtts = list(subs_dir.glob(f"{vid}.*.vtt"))
        if not vtts:
            return None

        # If multiple, pick the shortest lang tag match preference order
        # (e.g., uz.vtt before uz-Latn.vtt)
        def rank(p: Path) -> int:
            name = p.name.lower()
            for i, lg in enumerate(uz_langs):
                if f".{lg.lower()}." in name:
                    return i
            return 999

        vtts.sort(key=rank)
        return vtts[0]

    if prefer_manual:
        v = attempt(auto=False)
        if v:
            return v
        return attempt(auto=True)
    else:
        v = attempt(auto=True)
        if v:
            return v
        return attempt(auto=False)


def clean_caption_text(text: str) -> str:
    # Remove VTT markup/HTML entities lightly; keep Uzbek characters intact.
    t = html.unescape(text)
    t = re.sub(r"<[^>]+>", "", t)  # tags
    t = t.replace("&nbsp;", " ")
    t = " ".join(part.strip() for part in t.split(">>") if part.strip())
    t = _APOSTROPHE_RE.sub("'", t)
    t = _DASH_RE.sub("-", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def parse_vtt_to_cues(vtt_path: Path) -> List[Tuple[float, float, str]]:
    cues = []
    v = webvtt.read(str(vtt_path))
    for c in v.captions:
        start = _parse_timestamp(c.start)
        end = _parse_timestamp(c.end)
        txt = clean_caption_text(c.text)
        if txt and end > start:
            cues.append((start, end, txt))
    return cues


def parse_vtt_to_word_segments(
    vtt_path: Path,
) -> List[Tuple[float, float, str]]:
    cues = _parse_vtt_raw_cues(vtt_path)
    has_word_tags = any(_VTT_WORD_TAG_RE.search(text or "") for _, _, text in cues)
    segments: List[Tuple[float, float, str]] = []
    last_end = 0.0

    for start, end, raw_text in cues:
        raw = html.unescape(raw_text or "")
        matches = list(_VTT_WORD_TAG_RE.finditer(raw))
        if matches:
            first_ts = _parse_timestamp(matches[0].group("ts"))
            prefix = clean_caption_text(raw[: matches[0].start()])
            prefix_words = _split_words(prefix)
            include_prefix = not segments or (start - last_end) > 1.0
            if include_prefix and prefix_words:
                segments.extend(
                    _spread_words(prefix_words, start, max(start, first_ts))
                )
                last_end = segments[-1][1]

            for idx, match in enumerate(matches):
                word_text = clean_caption_text(match.group("word"))
                tokens = _split_words(word_text)
                if not tokens:
                    continue
                word_start = _parse_timestamp(match.group("ts"))
                if idx + 1 < len(matches):
                    next_start = _parse_timestamp(matches[idx + 1].group("ts"))
                else:
                    next_start = end
                word_end = max(word_start, min(next_start, end))

                if len(tokens) == 1:
                    token_segments = [(word_start, word_end, tokens[0])]
                else:
                    token_segments = _spread_words(tokens, word_start, word_end)

                for seg_start, seg_end, token in token_segments:
                    if segments and seg_start < segments[-1][0] - 0.001:
                        continue
                    if (
                        segments
                        and seg_start <= segments[-1][1]
                        and token == segments[-1][2]
                    ):
                        continue
                    segments.append((seg_start, seg_end, token))
                    last_end = seg_end
            continue

        cleaned = clean_caption_text(raw)
        if not cleaned:
            continue
        words = _split_words(cleaned)
        if not words:
            continue
        if has_word_tags:
            continue
        segments.extend(_spread_words(words, start, end))
        last_end = segments[-1][1]

    segments.sort(key=lambda item: item[0])
    return segments


def _parse_timestamp(ts: str) -> float:
    # WebVTT timestamps: HH:MM:SS.mmm or MM:SS.mmm
    parts = ts.split(":")
    if len(parts) == 3:
        h = int(parts[0])
        m = int(parts[1])
        s = float(parts[2])
    elif len(parts) == 2:
        h = 0
        m = int(parts[0])
        s = float(parts[1])
    else:
        raise ValueError(f"Unrecognized timestamp: {ts}")
    return h * 3600 + m * 60 + s


def _parse_vtt_raw_cues(vtt_path: Path) -> List[Tuple[float, float, str]]:
    cues: List[Tuple[float, float, str]] = []
    lines = vtt_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
        if not line:
            idx += 1
            continue
        match = _VTT_TIMESTAMP_LINE_RE.match(line)
        if not match:
            idx += 1
            continue
        start = _parse_timestamp(match.group("start"))
        end = _parse_timestamp(match.group("end"))
        idx += 1
        text_lines: List[str] = []
        # Read all lines until we hit an empty line followed by another timestamp
        # or two consecutive empty lines (end of cue block)
        consecutive_empty = 0
        while idx < len(lines):
            current_line = lines[idx]
            if not current_line.strip():
                consecutive_empty += 1
                # Check if next non-empty line is a timestamp (new cue)
                if consecutive_empty >= 1:
                    peek_idx = idx + 1
                    while peek_idx < len(lines) and not lines[peek_idx].strip():
                        peek_idx += 1
                    if peek_idx >= len(lines) or _VTT_TIMESTAMP_LINE_RE.match(
                        lines[peek_idx].strip()
                    ):
                        break
                idx += 1
                continue
            consecutive_empty = 0
            text_lines.append(current_line)
            idx += 1
        cues.append((start, end, "\n".join(text_lines)))
    return cues


def _split_words(text: str) -> List[str]:
    return [token for token in text.split() if token]


def _spread_words(
    words: List[str], start: float, end: float
) -> List[Tuple[float, float, str]]:
    if not words:
        return []
    if end <= start:
        end = start + (0.05 * len(words))
    step = (end - start) / float(len(words))
    segments: List[Tuple[float, float, str]] = []
    for idx, word in enumerate(words):
        seg_start = start + (idx * step)
        seg_end = end if idx == len(words) - 1 else start + ((idx + 1) * step)
        segments.append((seg_start, seg_end, word))
    return segments


def sentences_for_slice(
    word_segments: List[Tuple[float, float, str]],
    slice_start: float,
    slice_end: float,
) -> List[dict]:
    sentences: List[dict] = []
    for seg_start, seg_end, token in word_segments:
        if seg_end <= slice_start:
            continue
        if seg_start >= slice_end:
            break
        rel_start = max(seg_start, slice_start) - slice_start
        rel_end = min(seg_end, slice_end) - slice_start
        if rel_end <= rel_start:
            continue
        sentences.append(
            {
                "start": round(rel_start, 3),
                "end": round(rel_end, 3),
                "text": token,
            }
        )
    return sentences


def pack_cues_into_slices(
    cues: List[Tuple[float, float, str]],
    min_sec: float = 0.5,
    max_sec: float = 20.0,
) -> List[Tuple[float, float, str]]:
    """
    Combine caption cues into slices with duration ideally in [min_sec, max_sec].
    """
    slices: List[Tuple[float, float, str]] = []
    cur_s = None
    cur_e = None
    parts: List[str] = []

    def flush():
        nonlocal cur_s, cur_e, parts
        if cur_s is None or cur_e is None:
            cur_s, cur_e, parts = None, None, []
            return
        txt = " ".join(p for p in parts if p).strip()
        if txt:
            slices.append((float(cur_s), float(cur_e), txt))
        cur_s, cur_e, parts = None, None, []

    for s, e, txt in cues:
        if cur_s is None:
            cur_s, cur_e = s, e
            parts = [txt]
            continue

        new_s = cur_s
        new_e = max(cur_e, e)
        if (new_e - new_s) <= max_sec:
            cur_e = new_e
            parts.append(txt)
        else:
            flush()
            cur_s, cur_e = s, e
            parts = [txt]

    flush()

    # Merge too-short slices if possible
    if not slices:
        return slices

    merged: List[Tuple[float, float, str]] = []
    i = 0
    while i < len(slices):
        s, e, txt = slices[i]
        dur = e - s
        if dur >= min_sec:
            merged.append((s, e, txt))
            i += 1
            continue

        # merge with next if fits
        if i + 1 < len(slices):
            s2, e2, txt2 = slices[i + 1]
            if (e2 - s) <= max_sec:
                merged.append((s, e2, (txt + " " + txt2).strip()))
                i += 2
                continue

        # merge with previous if fits
        if merged:
            ps, pe, ptxt = merged[-1]
            if (e - ps) <= max_sec:
                merged[-1] = (ps, e, (ptxt + " " + txt).strip())
                i += 1
                continue

        merged.append((s, e, txt))
        i += 1

    return merged


def slice_wav(full_wav: Path, out_wav: Path, start: float, end: float) -> None:
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(full_wav),
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
        str(out_wav),
    ]
    run(cmd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ids_txt", required=True, help="txt file with one YouTube ID or URL per line"
    )
    ap.add_argument("--out_dir", required=True, help="output dataset directory")
    ap.add_argument("--min_sec", type=float, default=0.5)
    ap.add_argument("--max_sec", type=float, default=20.0)
    ap.add_argument(
        "--prefer_manual",
        action="store_true",
        help="prefer manual subs; fallback to auto",
    )
    args = ap.parse_args()

    ids_txt = Path(args.ids_txt)
    out_dir = Path(args.out_dir)

    downloads_dir = out_dir / "downloads"
    wav_full_dir = out_dir / "wav_full"
    subs_dir = out_dir / "subs_vtt"
    slices_dir = out_dir / "wav_slices"
    manifest_path = out_dir / "manifest.jsonl"

    out_dir.mkdir(parents=True, exist_ok=True)

    videos = read_ids(ids_txt)
    if not videos:
        raise ValueError("No video IDs/URLs found in ids_txt.")

    with manifest_path.open("a", encoding="utf-8") as mf:
        for video in videos:
            vid = sanitize_id(video)
            print(f"\n=== Processing: {video} ===")

            # 1) download subtitles (Uzbek)
            vtt_path = download_uzbek_vtt(
                video, subs_dir, prefer_manual=args.prefer_manual
            )
            if not vtt_path:
                print("No Uzbek subtitles found (manual or auto). Skipping.")
                continue
            print(f"Subtitles: {vtt_path.name}")

            word_segments = parse_vtt_to_word_segments(vtt_path)
            if word_segments:
                slices = pack_cues_into_slices(
                    word_segments, min_sec=args.min_sec, max_sec=args.max_sec
                )
            else:
                cues = parse_vtt_to_cues(vtt_path)
                if not cues:
                    print("No usable cues in VTT. Skipping.")
                    continue
                slices = pack_cues_into_slices(
                    cues, min_sec=args.min_sec, max_sec=args.max_sec
                )
            if not slices:
                print("No slices produced after packing. Skipping.")
                continue

            # 2) download audio
            audio_file = download_audio(video, downloads_dir)
            print(f"Audio: {audio_file.name}")

            # 3) convert to wav 16k mono
            full_wav = convert_to_wav_16k_mono(audio_file, wav_full_dir)
            full_dur = wav_duration(full_wav)
            print(f"Full WAV: {full_wav.name} (duration={full_dur:.2f}s)")

            # 4) slice and write NeMo manifest
            wrote = 0
            for idx, (s, e, txt) in enumerate(slices):
                # clamp bounds
                s = max(0.0, s)
                e = min(full_dur, e)
                if e <= s:
                    continue

                slice_name = f"{vid}_{idx:05d}.wav"
                slice_path = slices_dir / vid / slice_name
                slice_wav(full_wav, slice_path, s, e)

                dur = wav_duration(slice_path)
                if dur < 0.1:
                    continue

                try:
                    rel_path = slice_path.resolve().relative_to(out_dir.resolve())
                except ValueError:
                    rel_path = slice_path

                if word_segments:
                    slice_end = min(e, s + dur)
                    sentences = sentences_for_slice(word_segments, s, slice_end)
                    if not sentences:
                        continue
                    txt = " ".join(seg["text"] for seg in sentences).strip()
                else:
                    sentences = [{"start": 0.0, "end": float(dur), "text": txt}]

                entry = {
                    "audio": str(rel_path),
                    "duration": float(dur),
                    "sentences": sentences,
                    "text": txt,
                }
                mf.write(json.dumps(entry, ensure_ascii=False) + "\n")
                wrote += 1

            print(f"Wrote {wrote} slices for {video}")

    print(f"\nDone. Manifest: {manifest_path.resolve()}")


if __name__ == "__main__":
    main()
