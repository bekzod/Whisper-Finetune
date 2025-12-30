#!/usr/bin/env python3
"""Shared text cleanup utilities for NeMo prep scripts."""

from __future__ import annotations

import re
from typing import Any

# Regex to match C or c NOT followed by h or H
_STANDALONE_C_RE = re.compile(r"[Cc](?![Hh])")

_APOSTROPHE_TRANSLATION = str.maketrans(
    {
        "\u2019": "'",
        "\u02bc": "'",
        "\u02bb": "'",
        "`": "'",
        "´": "'",
        "ʻ": "'",
        "ʼ": "'",
        "‛": "'",
    }
)
_ALLOWED_TEXT_RE = re.compile(r"[^a-zA-ZА-Яа-яЎўҚқҒғҲҳ0-9\s,.'-]+")
_MULTISPACE_RE = re.compile(r"\s+")
_UZBEK_CYRILLIC_TO_LATIN = {
    "А": "A",
    "а": "a",
    "Б": "B",
    "б": "b",
    "В": "V",
    "в": "v",
    "Г": "G",
    "г": "g",
    "Д": "D",
    "д": "d",
    "Е": "E",
    "е": "e",
    "Ё": "Yo",
    "ё": "yo",
    "Ж": "J",
    "ж": "j",
    "З": "Z",
    "з": "z",
    "И": "I",
    "и": "i",
    "Й": "Y",
    "й": "y",
    "К": "K",
    "к": "k",
    "Л": "L",
    "л": "l",
    "М": "M",
    "м": "m",
    "Н": "N",
    "н": "n",
    "О": "O",
    "о": "o",
    "П": "P",
    "п": "p",
    "Р": "R",
    "р": "r",
    "С": "S",
    "с": "s",
    "Т": "T",
    "т": "t",
    "У": "U",
    "у": "u",
    "Ф": "F",
    "ф": "f",
    "Х": "X",
    "х": "x",
    "Ц": "Ts",
    "ц": "ts",
    "Ч": "Ch",
    "ч": "ch",
    "Ш": "Sh",
    "ш": "sh",
    "Щ": "Sh",
    "щ": "sh",
    "Ъ": "'",
    "ъ": "'",
    "Ы": "I",
    "ы": "i",
    "Ь": "",
    "ь": "",
    "Э": "E",
    "э": "e",
    "Ю": "Yu",
    "ю": "yu",
    "Я": "Ya",
    "я": "ya",
    "Ў": "O'",
    "ў": "o'",
    "Қ": "Q",
    "қ": "q",
    "Ғ": "G'",
    "ғ": "g'",
    "Ҳ": "H",
    "ҳ": "h",
}
_UZBEK_CYRILLIC_CHARS = set(_UZBEK_CYRILLIC_TO_LATIN.keys())


def normalize_text(text: Any) -> str:
    """Normalize text to match training cleanup in utils/reader.py."""
    if text is None:
        return ""
    try:
        normalized = str(text)
    except Exception:
        return ""

    normalized = _transliterate_uzbek_cyrillic(normalized)
    normalized = normalized.translate(_APOSTROPHE_TRANSLATION)
    normalized = _ALLOWED_TEXT_RE.sub("", normalized)
    normalized = _MULTISPACE_RE.sub(" ", normalized).strip()
    return normalized


def _transliterate_uzbek_cyrillic(text: str) -> str:
    if not text:
        return text
    if not any(char in _UZBEK_CYRILLIC_CHARS for char in text):
        return text
    return "".join(_UZBEK_CYRILLIC_TO_LATIN.get(char, char) for char in text)


def contains_standalone_c(text: str) -> bool:
    """Check if text contains 'C' or 'c' not followed by 'h' or 'H'.

    Returns True if the text contains a standalone C/c (i.e., not part of Ch/ch).
    Used to filter out dataset items with invalid characters.
    """
    if not text:
        return False
    return bool(_STANDALONE_C_RE.search(text))


__all__ = ["normalize_text", "contains_standalone_c"]
