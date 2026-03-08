from __future__ import annotations

import math
from dataclasses import replace

from youtube_hf_dataset.models import Cue, Segment


def cues_as_segments(
    cues: list[Cue], max_seconds: float, min_seconds: float
) -> list[Segment]:
    segments: list[Segment] = []
    for cue in sorted(cues, key=lambda item: item.start):
        duration = cue.end - cue.start
        if duration < min_seconds or not cue.text.strip():
            continue
        if duration <= max_seconds:
            segments.append(Segment(start=cue.start, end=cue.end, text=cue.text))
            continue
        for split_cue in _split_long_cue(cue, max_seconds=max_seconds):
            if (
                split_cue.end - split_cue.start >= min_seconds
                and split_cue.text.strip()
            ):
                segments.append(
                    Segment(
                        start=split_cue.start,
                        end=split_cue.end,
                        text=split_cue.text,
                    )
                )
    return segments


def build_segments(
    cues: list[Cue], max_seconds: float, min_seconds: float
) -> list[Segment]:
    if max_seconds <= 0:
        raise ValueError("max_seconds must be > 0")
    normalized = _deduplicate_cues(cues)
    expanded: list[Cue] = []
    for cue in normalized:
        expanded.extend(_split_long_cue(cue, max_seconds=max_seconds))
    if not expanded:
        return []

    segments: list[Segment] = []
    current_start = expanded[0].start
    current_end = expanded[0].end
    text_parts = [expanded[0].text]

    for cue in expanded[1:]:
        proposed_end = max(current_end, cue.end)
        if proposed_end - current_start <= max_seconds:
            current_end = proposed_end
            text_parts.append(cue.text)
            continue

        segments.append(
            Segment(
                start=current_start, end=current_end, text=" ".join(text_parts).strip()
            )
        )
        current_start = cue.start
        current_end = cue.end
        text_parts = [cue.text]

    segments.append(
        Segment(start=current_start, end=current_end, text=" ".join(text_parts).strip())
    )
    merged = _merge_short_segments(
        segments, max_seconds=max_seconds, min_seconds=min_seconds
    )
    return [
        segment
        for segment in merged
        if segment.duration >= min_seconds and segment.text
    ]


def _deduplicate_cues(cues: list[Cue]) -> list[Cue]:
    deduped: list[Cue] = []
    for cue in sorted(cues, key=lambda item: item.start):
        if not deduped:
            deduped.append(cue)
            continue
        last = deduped[-1]
        overlap = cue.start <= last.end + 0.05
        same_text = cue.text == last.text
        if overlap and same_text:
            deduped[-1] = replace(last, end=max(last.end, cue.end))
            continue
        deduped.append(cue)
    return deduped


def _split_long_cue(cue: Cue, max_seconds: float) -> list[Cue]:
    if cue.end - cue.start <= max_seconds:
        return [cue]

    words = cue.text.split()
    if not words:
        return []

    chunks = max(1, math.ceil((cue.end - cue.start) / max_seconds))
    chunk_size = math.ceil(len(words) / chunks)
    duration = cue.end - cue.start

    split_cues: list[Cue] = []
    cursor = cue.start
    for index in range(chunks):
        begin = index * chunk_size
        end_index = min((index + 1) * chunk_size, len(words))
        if begin >= len(words):
            break
        text = " ".join(words[begin:end_index]).strip()
        word_fraction = (end_index - begin) / len(words)
        next_cursor = min(cue.end, cursor + (duration * word_fraction))
        if next_cursor <= cursor:
            next_cursor = min(cue.end, cursor + max_seconds)
        split_cues.append(Cue(start=cursor, end=next_cursor, text=text))
        cursor = next_cursor
    if split_cues and split_cues[-1].end < cue.end:
        last = split_cues[-1]
        split_cues[-1] = replace(last, end=cue.end)
    return split_cues


def _merge_short_segments(
    segments: list[Segment], max_seconds: float, min_seconds: float
) -> list[Segment]:
    if not segments:
        return []

    merged: list[Segment] = []
    index = 0
    while index < len(segments):
        segment = segments[index]
        if segment.duration >= min_seconds:
            merged.append(segment)
            index += 1
            continue

        merged_with_neighbor = False
        if merged and segment.end - merged[-1].start <= max_seconds:
            previous = merged[-1]
            merged[-1] = Segment(
                start=previous.start,
                end=segment.end,
                text=f"{previous.text} {segment.text}".strip(),
            )
            merged_with_neighbor = True

        if not merged_with_neighbor and index + 1 < len(segments):
            next_segment = segments[index + 1]
            if next_segment.end - segment.start <= max_seconds:
                merged.append(
                    Segment(
                        start=segment.start,
                        end=next_segment.end,
                        text=f"{segment.text} {next_segment.text}".strip(),
                    )
                )
                index += 2
                continue

        if not merged_with_neighbor:
            merged.append(segment)
        index += 1

    return merged
