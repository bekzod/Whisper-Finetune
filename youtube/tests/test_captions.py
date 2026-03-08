from pathlib import Path

from youtube_hf_dataset.captions import (
    parse_vtt,
    parse_vtt_aligned_cues,
    parse_vtt_word_cues,
)


def test_parse_vtt_keeps_original_cue_blocks(tmp_path: Path):
    vtt_path = tmp_path / "sample.vtt"
    vtt_path.write_text(
        "\n".join(
            [
                "WEBVTT",
                "",
                "00:04:48.320 --> 00:04:50.469 align:start position:0%",
                "shu forta toshib berib turaman.",
                "Asosan<00:04:48.720><c> bozorni</c><00:04:49.080><c> ichiga</c><00:04:49.400><c> toshiymiz.</c><00:04:50.199><c> Undan</c>",
                "",
            ]
        ),
        encoding="utf-8",
    )

    cues = parse_vtt(vtt_path)

    assert len(cues) == 1
    assert cues[0].start == 288.32
    assert cues[0].end == 290.469
    assert (
        cues[0].text
        == "shu forta toshib berib turaman.\nAsosan bozorni ichiga toshiymiz. Undan"
    )


def test_parse_vtt_word_cues_deduplicates_word_timed_rolling_captions(tmp_path: Path):
    vtt_path = tmp_path / "rolling.vtt"
    vtt_path.write_text(
        "\n".join(
            [
                "WEBVTT",
                "",
                "00:00:00.280 --> 00:00:02.950 align:start position:0%",
                " ",
                "Assalomu<00:00:00.599><c> alaykum</c><00:00:00.919><c> azizlar.</c>",
                "",
                "00:00:02.950 --> 00:00:02.960 align:start position:0%",
                "Assalomu alaykum azizlar.",
                " ",
                "",
                "00:00:02.960 --> 00:00:05.309 align:start position:0%",
                "Assalomu alaykum azizlar.",
                "yakshanba.<00:00:04.080><c> Orta</c><00:00:04.359><c> qolayotgan</c>",
                "",
            ]
        ),
        encoding="utf-8",
    )

    cues = parse_vtt_word_cues(vtt_path)

    assert [cue.text for cue in cues] == [
        "Assalomu",
        "alaykum",
        "azizlar.",
        "yakshanba.",
        "Orta",
        "qolayotgan",
    ]


def test_parse_vtt_aligned_cues_uses_words_started_in_cue_window(tmp_path: Path):
    vtt_path = tmp_path / "aligned.vtt"
    vtt_path.write_text(
        "\n".join(
            [
                "WEBVTT",
                "",
                "00:00:00.280 --> 00:00:02.950 align:start position:0%",
                " ",
                "Assalomu<00:00:00.599><c> alaykum</c><00:00:00.919><c> azizlar.</c><00:00:01.760><c> Bugun</c><00:00:02.159><c> 8martt</c>",
                "",
                "00:00:02.960 --> 00:00:05.309 align:start position:0%",
                "Assalomu alaykum azizlar. Bugun 8martt",
                "yakshanba.<00:00:04.080><c> Orta</c><00:00:04.359><c> qolayotgan</c>",
                "",
            ]
        ),
        encoding="utf-8",
    )

    cues = parse_vtt_aligned_cues(vtt_path)

    assert [cue.text for cue in cues] == [
        "Assalomu alaykum azizlar. Bugun 8martt",
        "yakshanba. Orta qolayotgan",
    ]


def test_parse_vtt_aligned_cues_trims_carried_line_when_last_cue_has_no_word_tags(
    tmp_path: Path,
):
    vtt_path = tmp_path / "last-cue-no-tags.vtt"
    vtt_path.write_text(
        "\n".join(
            [
                "WEBVTT",
                "",
                "00:09:09.160 --> 00:09:10.790 align:start position:0%",
                "tugatishni bilmayotganga o'xshaydi.",
                "Xudda<00:09:09.480><c> Yevropa</c><00:09:09.839><c> sharqidagi</c><00:09:10.279><c> bir</c><00:09:10.399><c> davlatga</c>",
                "",
                "00:09:10.790 --> 00:09:10.800 align:start position:0%",
                "Xudda Yevropa sharqidagi bir davlatga",
                "",
                "00:09:10.800 --> 00:09:13.040 align:start position:0%",
                "Xudda Yevropa sharqidagi bir davlatga",
                "o'xshab",
                "",
            ]
        ),
        encoding="utf-8",
    )

    cues = parse_vtt_aligned_cues(vtt_path)

    assert cues[-1].text == "o'xshab"
