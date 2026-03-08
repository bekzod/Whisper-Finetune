from __future__ import annotations

from youtube_hf_dataset.models import Cue, Segment


def cues_as_segments(cues: list[Cue], min_seconds: float) -> list[Segment]:
    segments: list[Segment] = []
    for cue in sorted(cues, key=lambda item: item.start):
        duration = cue.end - cue.start
        if duration < min_seconds or not cue.text.strip():
            continue
        segments.append(Segment(start=cue.start, end=cue.end, text=cue.text))
    return segments
