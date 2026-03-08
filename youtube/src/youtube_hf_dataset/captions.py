from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Callable

from youtube_hf_dataset.models import Cue

_TIMESTAMP_RE = re.compile(
    r"^(?P<start>\d{2}:\d{2}(?::\d{2})?\.\d{3})\s*-->\s*(?P<end>\d{2}:\d{2}(?::\d{2})?\.\d{3})"
)
_WORD_TAG_RE = re.compile(
    r"<(?P<ts>\d{2}:\d{2}(?::\d{2})?\.\d{3})><c>(?P<word>[^<]*)</c>"
)
_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"\s+")


def clean_caption_text(text: str) -> str:
    plain = html.unescape(text)
    plain = _TAG_RE.sub(" ", plain)
    plain = plain.replace("\n", " ")
    plain = _SPACE_RE.sub(" ", plain)
    return plain.strip()


def parse_vtt(path: Path) -> list[Cue]:
    cues: list[Cue] = []
    for start, end, raw_text in _parse_raw_vtt(path):
        text = _clean_cue_block(raw_text)
        if text and end > start:
            cues.append(Cue(start=start, end=end, text=text))
    return cues


def parse_vtt_aligned_cues(path: Path) -> list[Cue]:
    raw_cues = _parse_raw_vtt(path)
    cue_entries: list[tuple[Cue, str]] = []
    for start, end, raw_text in raw_cues:
        text = _clean_cue_block(raw_text)
        if end > start and text:
            cue_entries.append((Cue(start=start, end=end, text=text), raw_text))
    cue_windows = [cue for cue, _ in cue_entries]
    word_cues = parse_vtt_word_cues(path)
    if not word_cues:
        return cue_windows

    aligned: list[Cue] = []
    word_index = 0
    previous_lines: list[str] = []
    for cue, raw_text in cue_entries:
        while word_index < len(word_cues) and word_cues[word_index].start < cue.start:
            word_index += 1

        scan_index = word_index
        words: list[str] = []
        while scan_index < len(word_cues) and word_cues[scan_index].start < cue.end:
            words.append(word_cues[scan_index].text)
            scan_index += 1

        if words:
            text = " ".join(words).strip()
        else:
            current_lines = _clean_cue_block(raw_text).splitlines()
            trimmed_lines = _trim_seen_prefix_lines(current_lines, previous_lines)
            text = "\n".join(trimmed_lines).strip() or cue.text
        aligned.append(Cue(start=cue.start, end=cue.end, text=text))
        previous_lines = _clean_cue_block(raw_text).splitlines()
    return aligned


def parse_vtt_word_cues(path: Path) -> list[Cue]:
    raw_cues = _parse_raw_vtt(path)
    if any(_WORD_TAG_RE.search(text) for _, _, text in raw_cues):
        word_cues = _parse_word_level_cues(raw_cues)
        if word_cues:
            return word_cues

    cues: list[Cue] = []
    for start, end, raw_text in raw_cues:
        text = clean_caption_text(raw_text)
        if text and end > start:
            cues.append(Cue(start=start, end=end, text=text))
    return cues


def pick_best_vtt(
    candidates: list[Path], parser: Callable[[Path], list[Cue]] | None = None
) -> tuple[Path | None, list[Cue]]:
    parse = parser or parse_vtt
    best_path: Path | None = None
    best_cues: list[Cue] = []
    for candidate in sorted(candidates, key=lambda path: path.name):
        cues = parse(candidate)
        if len(cues) > len(best_cues):
            best_path = candidate
            best_cues = cues
    return best_path, best_cues


def _clean_cue_block(text: str) -> str:
    lines = [clean_caption_text(line) for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def _parse_raw_vtt(path: Path) -> list[tuple[float, float, str]]:
    cues: list[tuple[float, float, str]] = []
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
        match = _TIMESTAMP_RE.match(line)
        if not match:
            idx += 1
            continue

        start = _parse_timestamp(match.group("start"))
        end = _parse_timestamp(match.group("end"))
        idx += 1
        text_lines: list[str] = []
        while idx < len(lines):
            current = lines[idx]
            if not current.strip():
                peek_idx = idx + 1
                while peek_idx < len(lines) and not lines[peek_idx].strip():
                    peek_idx += 1
                if peek_idx >= len(lines) or _TIMESTAMP_RE.match(
                    lines[peek_idx].strip()
                ):
                    break
                idx += 1
                continue
            text_lines.append(current)
            idx += 1

        cues.append((start, end, "\n".join(text_lines)))
        idx += 1
    return cues


def _parse_word_level_cues(raw_cues: list[tuple[float, float, str]]) -> list[Cue]:
    cues: list[Cue] = []
    last_end = 0.0

    for start, end, raw_text in raw_cues:
        text = html.unescape(raw_text)
        matches = list(_WORD_TAG_RE.finditer(text))
        if not matches:
            continue

        first_word_start = _parse_timestamp(matches[0].group("ts"))
        prefix = clean_caption_text(text[: matches[0].start()])
        prefix_words = _split_words(prefix)
        if cues and (start - last_end) <= 1.0:
            prefix_words = _trim_seen_prefix(prefix_words, [cue.text for cue in cues])
        if prefix_words:
            prefix_cues = _spread_words(
                prefix_words, start, max(start, first_word_start)
            )
            cues.extend(prefix_cues)
            last_end = cues[-1].end

        for index, match in enumerate(matches):
            words = _split_words(clean_caption_text(match.group("word")))
            if not words:
                continue

            word_start = _parse_timestamp(match.group("ts"))
            if index + 1 < len(matches):
                next_start = _parse_timestamp(matches[index + 1].group("ts"))
            else:
                next_start = end
            word_end = max(word_start, min(next_start, end))

            for word_cue in _spread_words(words, word_start, word_end):
                if cues and word_cue.start < cues[-1].start - 0.001:
                    continue
                if (
                    cues
                    and word_cue.start <= cues[-1].end
                    and word_cue.text == cues[-1].text
                ):
                    continue
                cues.append(word_cue)
                last_end = word_cue.end

    return cues


def _spread_words(words: list[str], start: float, end: float) -> list[Cue]:
    if not words:
        return []
    if end <= start:
        end = start + (0.05 * len(words))

    step = (end - start) / len(words)
    cues: list[Cue] = []
    for index, word in enumerate(words):
        word_start = start + (index * step)
        word_end = end if index == len(words) - 1 else start + ((index + 1) * step)
        cues.append(Cue(start=word_start, end=word_end, text=word))
    return cues


def _split_words(text: str) -> list[str]:
    return [word for word in text.split() if word]


def _trim_seen_prefix(prefix_words: list[str], emitted_words: list[str]) -> list[str]:
    max_overlap = min(len(prefix_words), len(emitted_words))
    for overlap in range(max_overlap, 0, -1):
        if emitted_words[-overlap:] == prefix_words[:overlap]:
            return prefix_words[overlap:]
    return prefix_words


def _trim_seen_prefix_lines(
    current_lines: list[str], previous_lines: list[str]
) -> list[str]:
    max_overlap = min(len(current_lines), len(previous_lines))
    for overlap in range(max_overlap, 0, -1):
        if previous_lines[-overlap:] == current_lines[:overlap]:
            return current_lines[overlap:]
    return current_lines


def _parse_timestamp(timestamp: str) -> float:
    parts = timestamp.split(":")
    if len(parts) == 3:
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
    elif len(parts) == 2:
        hours = 0
        minutes = int(parts[0])
        seconds = float(parts[1])
    else:
        raise ValueError(f"Invalid VTT timestamp: {timestamp}")
    return hours * 3600 + minutes * 60 + seconds
