#!/usr/bin/env python3
"""Shared text cleanup utilities for NeMo prep scripts."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

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

# Common Uzbek misspellings: missing apostrophes, variant spellings, etc.
# Format: incorrect -> correct
_UZBEK_MISSPELLINGS = {
    # Missing apostrophes in common words
    "boladi": "bo'ladi",
    "bolish": "bo'lish",
    "bolgan": "bo'lgan",
    "bolib": "bo'lib",
    "bolsa": "bo'lsa",
    "bolsin": "bo'lsin",
    "bolar": "bo'lar",
    "bolmaydi": "bo'lmaydi",
    "bolmas": "bo'lmas",
    "bolmasin": "bo'lmasin",
    "oladi": "o'ladi",
    "olib": "o'lib",
    "olgan": "o'lgan",
    "olar": "o'lar",
    "ozim": "o'zim",
    "ozing": "o'zing",
    "ozi": "o'zi",
    "ozimiz": "o'zimiz",
    "ozingiz": "o'zingiz",
    "ozlari": "o'zlari",
    "qoladi": "qo'ladi",
    "qolib": "qo'lib",
    "qolgan": "qo'lgan",
    "qolar": "qo'lar",
    "qoydi": "qo'ydi",
    "qoyib": "qo'yib",
    "qoygan": "qo'ygan",
    "qoyar": "qo'yar",
    "korib": "ko'rib",
    "koradi": "ko'radi",
    "korgan": "ko'rgan",
    "korar": "ko'rar",
    "korish": "ko'rish",
    "kop": "ko'p",
    "kopchilik": "ko'pchilik",
    "koplab": "ko'plab",
    "kocha": "ko'cha",
    "kochada": "ko'chada",
    "togri": "to'g'ri",
    "togrisida": "to'g'risida",
    "tola": "to'la",
    "toliq": "to'liq",
    "toldirib": "to'ldirib",
    "gozal": "go'zal",
    "sorov": "so'rov",
    "sozlar": "so'zlar",
    "sozlash": "so'zlash",
    # Common g' misspellings
    "gapiradi": "g'apiradi",
    "gapirib": "g'apirib",
    "gapirish": "g'apirish",
    # Common word variants
    "kerak": "kerak",
    "keyingi": "keyingi",
}

# Build regex pattern for whole-word matching (case-insensitive)
_UZBEK_MISSPELLING_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in _UZBEK_MISSPELLINGS.keys()) + r")\b",
    re.IGNORECASE,
)
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


@dataclass
class MisspellingStats:
    """Track misspelling fix statistics."""

    total_fixes: int = 0
    fixes_by_word: Counter = field(default_factory=Counter)

    def record_fix(self, original: str, replacement: str) -> None:
        """Record a single fix."""
        self.total_fixes += 1
        self.fixes_by_word[f"{original.lower()} -> {replacement}"] += 1

    def merge(self, other: "MisspellingStats") -> None:
        """Merge another stats object into this one."""
        self.total_fixes += other.total_fixes
        self.fixes_by_word.update(other.fixes_by_word)

    def report(self) -> str:
        """Generate a human-readable report."""
        if self.total_fixes == 0:
            return "No misspellings fixed."
        lines = [f"Total misspellings fixed: {self.total_fixes}"]
        lines.append("Fixes by word:")
        for word_pair, count in self.fixes_by_word.most_common():
            lines.append(f"  {word_pair}: {count}")
        return "\n".join(lines)

    def reset(self) -> None:
        """Reset all statistics."""
        self.total_fixes = 0
        self.fixes_by_word.clear()


# Global stats tracker
_misspelling_stats = MisspellingStats()


def get_misspelling_stats() -> MisspellingStats:
    """Get the global misspelling statistics."""
    return _misspelling_stats


def reset_misspelling_stats() -> None:
    """Reset the global misspelling statistics."""
    _misspelling_stats.reset()


def _fix_uzbek_misspellings(text: str, stats: Optional[MisspellingStats] = None) -> str:
    """Fix common Uzbek misspellings, preserving original case."""
    if stats is None:
        stats = _misspelling_stats

    def replace_match(match: re.Match) -> str:
        word = match.group(0)
        replacement = _UZBEK_MISSPELLINGS.get(word.lower(), word)
        if replacement != word.lower():
            stats.record_fix(word, replacement)
        # Preserve original case
        if word.isupper():
            return replacement.upper()
        elif word[0].isupper():
            return replacement.capitalize()
        return replacement

    return _UZBEK_MISSPELLING_PATTERN.sub(replace_match, text)


def normalize_text(text: Any, stats: Optional[MisspellingStats] = None) -> str:
    """Normalize text to match training cleanup in utils/reader.py."""
    if text is None:
        return ""
    try:
        normalized = str(text)
    except Exception:
        return ""

    normalized = _transliterate_uzbek_cyrillic(normalized)
    normalized = normalized.translate(_APOSTROPHE_TRANSLATION)
    normalized = _fix_uzbek_misspellings(normalized, stats)
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


__all__ = [
    "normalize_text",
    "contains_standalone_c",
    "MisspellingStats",
    "get_misspelling_stats",
    "reset_misspelling_stats",
]
