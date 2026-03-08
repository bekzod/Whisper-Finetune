from youtube_hf_dataset.models import Cue
from youtube_hf_dataset.segment import cues_as_segments


def test_cues_as_segments_keeps_each_cue_separate():
    cues = [
        Cue(start=0.0, end=2.0, text="first"),
        Cue(start=2.0, end=4.0, text="second"),
    ]

    segments = cues_as_segments(cues, min_seconds=0.1)

    assert [segment.text for segment in segments] == ["first", "second"]
    assert [segment.start for segment in segments] == [0.0, 2.0]


def test_cues_as_segments_filters_tiny_cues():
    cues = [
        Cue(start=0.0, end=0.05, text="tiny"),
        Cue(start=0.1, end=1.0, text="kept"),
    ]

    segments = cues_as_segments(cues, min_seconds=0.1)

    assert [segment.text for segment in segments] == ["kept"]
