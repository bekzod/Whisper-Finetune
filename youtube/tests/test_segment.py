from youtube_hf_dataset.models import Cue
from youtube_hf_dataset.segment import build_segments, cues_as_segments


def test_build_segments_keeps_max_duration():
    cues = [
        Cue(start=0.0, end=8.0, text="a"),
        Cue(start=8.0, end=16.0, text="b"),
        Cue(start=16.0, end=24.0, text="c"),
        Cue(start=24.0, end=32.0, text="d"),
    ]

    segments = build_segments(cues, max_seconds=28.0, min_seconds=0.1)

    assert len(segments) == 2
    assert segments[0].start == 0.0
    assert segments[0].end == 24.0
    assert segments[1].start == 24.0
    assert segments[1].end == 32.0
    assert max(segment.duration for segment in segments) <= 28.0


def test_build_segments_splits_single_long_cue():
    cue = Cue(start=0.0, end=60.0, text=" ".join(str(i) for i in range(20)))

    segments = build_segments([cue], max_seconds=28.0, min_seconds=0.1)

    assert len(segments) >= 2
    assert max(segment.duration for segment in segments) <= 28.0
    assert "0" in segments[0].text


def test_cues_as_segments_keeps_each_cue_separate():
    cues = [
        Cue(start=0.0, end=2.0, text="first"),
        Cue(start=2.0, end=4.0, text="second"),
    ]

    segments = cues_as_segments(cues, max_seconds=28.0, min_seconds=0.1)

    assert [segment.text for segment in segments] == ["first", "second"]
    assert [segment.start for segment in segments] == [0.0, 2.0]
