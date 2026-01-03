#!/usr/bin/env python3
"""Shared text cleanup utilities for NeMo prep scripts."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, List, Optional, Set, Tuple

# Regex to match C or c NOT followed by h or H
_STANDALONE_C_RE = re.compile(r"[Cc](?![Hh])")

_APOSTROPHE_TRANSLATION = str.maketrans(
    {
        "\u2019": "'",  # RIGHT SINGLE QUOTATION MARK
        "\u02bc": "'",  # MODIFIER LETTER APOSTROPHE
        "\u02bb": "'",  # MODIFIER LETTER TURNED COMMA
        "`": "'",  # GRAVE ACCENT
        "´": "'",  # ACUTE ACCENT
        "‛": "'",  # SINGLE HIGH-REVERSED-9 QUOTATION MARK
    }
)
_ALLOWED_TEXT_RE = re.compile(r"[^a-zA-ZА-Яа-яЎўҚқҒғҲҳ0-9\s,.'-]+")

# Common Uzbek misspellings: missing apostrophes, variant spellings, etc.
# Format: incorrect -> correct
_UZBEK_MISSPELLINGS = {
    # Missing apostrophes in bo'lmoq (to be/become) forms
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
    # Missing apostrophes in o'z (self) forms
    "ozim": "o'zim",
    "ozing": "o'zing",
    "ozimiz": "o'zimiz",
    "ozingiz": "o'zingiz",
    "ozlari": "o'zlari",
    # Missing apostrophes in qo'ymoq (to put) forms
    "qoydi": "qo'ydi",
    "qoyib": "qo'yib",
    "qoygan": "qo'ygan",
    "qoyar": "qo'yar",
    "qoyish": "qo'yish",
    "qoyadi": "qo'yadi",
    # Missing apostrophes in ko'rmoq (to see) forms
    "korib": "ko'rib",
    "koradi": "ko'radi",
    "korgan": "ko'rgan",
    "korar": "ko'rar",
    "korish": "ko'rish",
    "kordi": "ko'rdi",
    "korsat": "ko'rsat",
    "korsatdi": "ko'rsatdi",
    "korsatish": "ko'rsatish",
    # Missing apostrophes in ko'p (many) forms
    "kop": "ko'p",
    "kopchilik": "ko'pchilik",
    "koplab": "ko'plab",
    "koprok": "ko'proq",
    # Missing apostrophes in ko'cha (street) forms
    "kocha": "ko'cha",
    "kochada": "ko'chada",
    "kochasi": "ko'chasi",
    # Missing apostrophes in to'g'ri (correct/straight)
    "togri": "to'g'ri",
    "togrisida": "to'g'risida",
    # Missing apostrophes in to'liq (complete) forms
    "toliq": "to'liq",
    "toldirib": "to'ldirib",
    "toldirish": "to'ldirish",
    # Missing apostrophes in go'zal (beautiful)
    "gozal": "go'zal",
    # Missing apostrophes in so'rov (request)
    "sorov": "so'rov",
    # Missing apostrophes in o'rganmoq (to learn) forms
    "organmoq": "o'rganmoq",
    "organdi": "o'rgandi",
    "organib": "o'rganib",
    "organish": "o'rganish",
    "organadi": "o'rganadi",
    # Missing apostrophes in o'ylamoq (to think) forms
    "oyladi": "o'yladi",
    "oylab": "o'ylab",
    "oylash": "o'ylash",
    "oylaydi": "o'ylaydi",
    # Missing apostrophes in o'tirmoq (to sit) forms
    "otirdi": "o'tirdi",
    "otirib": "o'tirib",
    "otirish": "o'tirish",
    "otiradi": "o'tiradi",
    # Missing apostrophes in o'ynamoq (to play) forms
    "oynadi": "o'ynadi",
    "oynab": "o'ynab",
    "oynash": "o'ynash",
    "oynaydi": "o'ynaydi",
    # Missing apostrophes in o'qimoq (to read/study) - careful: oqmoq (to flow) is different
    "oqituvchi": "o'qituvchi",
    "oquvchi": "o'quvchi",
    # Missing apostrophes in o'zgarmoq (to change) forms
    "ozgardi": "o'zgardi",
    "ozgarib": "o'zgarib",
    "ozgarish": "o'zgarish",
    "ozgaradi": "o'zgaradi",
    # Missing apostrophes in other o' words
    "ogil": "o'g'il",
    "orta": "o'rta",
    "ortada": "o'rtada",
    "ormon": "o'rmon",
    "osha": "o'sha",
    "oshanda": "o'shanda",
    "orin": "o'rin",
    "orinda": "o'rinda",
    "olik": "o'lik",
    "otkazdi": "o'tkazdi",
    "otkazish": "o'tkazish",
    # Missing apostrophes in g' words
    "galaba": "g'alaba",
    "goya": "g'oya",
    "goyat": "g'oyat",
    "garb": "g'arb",
    "garbiy": "g'arbiy",
    "gayrat": "g'ayrat",
    "gazab": "g'azab",
    "galati": "g'alati",
    "gamgin": "g'amgin",
    "gildirak": "g'ildirak",
    "gisht": "g'isht",
    "goyib": "g'oyib",
    "gam": "g'am",
    "gurur": "g'urur",
    # Soft h -> hard x corrections (common Uzbek pronunciation/spelling errors)
    # xabar (news) and related forms
    "habar": "xabar",
    "habari": "xabari",
    "habarlar": "xabarlar",
    "habarlari": "xabarlari",
    "habarnoma": "xabarnoma",
    "habarchi": "xabarchi",
    "habardon": "xabardon",
    # xafa (upset) and related forms
    "hafa": "xafa",
    "hafalik": "xafalik",
    # xarid (purchase) and related forms
    "harid": "xarid",
    "haridor": "xaridor",
    "haridorlar": "xaridorlar",
    # xizmat (service) and related forms
    "hizmat": "xizmat",
    "hizmati": "xizmati",
    "hizmatlar": "xizmatlar",
    "hizmatchi": "xizmatchi",
    # xalq (people/nation) and related forms
    "halq": "xalq",
    "halqi": "xalqi",
    "halqaro": "xalqaro",
    "halqlararo": "xalqlararo",
    # xotin (wife) and related forms
    "hotin": "xotin",
    "hotini": "xotini",
    "hotinlar": "xotinlar",
    # xona (room) and related forms
    "hona": "xona",
    "honada": "xonada",
    "honasi": "xonasi",
    "honalar": "xonalar",
    "honadon": "xonadon",
    # xotira (memory) and related forms
    "hotira": "xotira",
    "hotirasi": "xotirasi",
    "hotiralar": "xotiralar",
    "hotirlash": "xotirlash",
    "hotirladi": "xotirladi",
    # xavf (danger) and related forms
    "havf": "xavf",
    "havfli": "xavfli",
    "havfsiz": "xavfsiz",
    "havfsizlik": "xavfsizlik",
    # xato (mistake) and related forms
    "hato": "xato",
    "hatosi": "xatosi",
    "hatolar": "xatolar",
    "hatolik": "xatolik",
    # xush (pleasant) and related forms
    "hush": "xush",
    "hushnud": "xushnud",
    "hushmuomala": "xushmuomala",
    # xursand (happy) and related forms
    "hursand": "xursand",
    "hursandlik": "xursandlik",
    "hursandchilik": "xursandchilik",
    # xayr (goodbye) and related forms
    "hayr": "xayr",
    "hayrlashdi": "xayrlashdi",
    "hayriya": "xayriya",
    "hayriyachi": "xayriyachi",
    # xudo (god) and related forms
    "hudo": "xudo",
    "hudoga": "xudoga",
    "hudoning": "xudoning",
    "hudoyo": "xudoyo",
    # xulosa (conclusion) and related forms
    "hulosa": "xulosa",
    "hulosasi": "xulosasi",
    "hulosalar": "xulosalar",
    # xuddi (exactly)
    "huddi": "xuddi",
    # xo'ja (master/teacher) and related forms
    "hoja": "xo'ja",
    "hojalik": "xo'jalik",
    "hojali": "xo'jali",
    "hojayin": "xo'jayin",
    # xo'roz (rooster)
    "horoz": "xo'roz",
    # xom (raw) and related forms
    "hom": "xom",
    "homashyo": "xomashyo",
    # xor (choir/humiliated) and related forms
    "hor": "xor",
    "horlik": "xorlik",
    # xossa (property/characteristic)
    "hossa": "xossa",
    "hossalar": "xossalar",
    # xorazm (Khorezm region)
    "horazm": "xorazm",
    "horazmlik": "xorazmlik",
    # xat (letter) and related forms
    "hat": "xat",
    "hati": "xati",
    "hatlar": "xatlar",
    # xalol (honest/halal in Uzbek context)
    "halol": "xalol",
    "halollik": "xalollik",
}

# Build regex pattern for whole-word matching (case-insensitive)
_UZBEK_MISSPELLING_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in _UZBEK_MISSPELLINGS.keys()) + r")\b",
    re.IGNORECASE,
)
_MULTISPACE_RE = re.compile(r"\s+")
# Pattern to match spaces before punctuation (e.g., "qurildi ." => "qurildi.")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([.,])")
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

# Word tokenization pattern for frequency analysis
_WORD_TOKENIZE_RE = re.compile(r"[a-zA-Z']+")


@dataclass
class MisspellingStats:
    """Track misspelling fix statistics."""

    total_fixes: int = 0
    fixes_by_word: Counter = field(default_factory=Counter)
    fixes_by_text: Counter = field(default_factory=Counter)

    def record_fix(self, original: str, replacement: str) -> None:
        """Record a single fix."""
        self.total_fixes += 1
        self.fixes_by_word[f"{original.lower()} -> {replacement}"] += 1

    def record_text_fix(self, original: str, dataset_label: str) -> None:
        """Record the full text that contained a fix and its dataset label."""
        self.fixes_by_text[(dataset_label, original)] += 1

    def merge(self, other: "MisspellingStats") -> None:
        """Merge another stats object into this one."""
        self.total_fixes += other.total_fixes
        self.fixes_by_word.update(other.fixes_by_word)
        self.fixes_by_text.update(other.fixes_by_text)

    def report(self) -> str:
        """Generate a human-readable report."""
        if self.total_fixes == 0:
            return "No misspellings fixed."
        lines = [f"Total misspellings fixed: {self.total_fixes}"]
        lines.append("Fixes by word:")
        for word_pair, count in self.fixes_by_word.most_common():
            lines.append(f"  {word_pair}: {count}")
        if self.fixes_by_text:
            lines.append("Fixes by text (dataset -> original):")
            for (dataset_label, original), count in self.fixes_by_text.most_common():
                lines.append(f"  [{dataset_label}] {original}: {count}")
        return "\n".join(lines)

    def reset(self) -> None:
        """Reset all statistics."""
        self.total_fixes = 0
        self.fixes_by_word.clear()
        self.fixes_by_text.clear()


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
        lower_word = word.lower()
        replacement: str = _UZBEK_MISSPELLINGS.get(lower_word, lower_word)
        if replacement != lower_word:
            stats.record_fix(word, replacement)
        # Preserve original case
        if word.isupper():
            return replacement.upper()
        elif word[0].isupper():
            return replacement.capitalize()
        return replacement

    return _UZBEK_MISSPELLING_PATTERN.sub(replace_match, text)


def normalize_text(
    text: Any,
    stats: Optional[MisspellingStats] = None,
    dataset_label: Optional[str] = None,
) -> str:
    """Normalize text to match training cleanup in utils/reader.py."""
    if text is None:
        return ""
    try:
        normalized = str(text)
    except Exception:
        return ""

    normalized = _transliterate_uzbek_cyrillic(normalized)
    normalized = normalized.translate(_APOSTROPHE_TRANSLATION)
    before_fix = normalized
    normalized = _fix_uzbek_misspellings(normalized, stats)
    if stats is not None and dataset_label and normalized != before_fix:
        stats.record_text_fix(before_fix, dataset_label)
    normalized = _ALLOWED_TEXT_RE.sub("", normalized)
    normalized = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", normalized)
    normalized = _MULTISPACE_RE.sub(" ", normalized).strip()
    if normalized.startswith("-"):
        normalized = normalized[1:].lstrip()
    if normalized.endswith("-"):
        normalized = normalized[:-1].rstrip()
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


# =============================================================================
# Frequency-Based Typo Detection
# =============================================================================


class WordFrequencyCollector:
    """Collects word frequencies across the entire dataset.

    This class is used in a first pass over the dataset to build
    a frequency distribution of all words, which is then used by
    FrequencyBasedTypoDetector to identify potential typos.
    """

    def __init__(self) -> None:
        self._word_counts: Counter = Counter()
        self._total_words: int = 0

    def add_text(self, text: str) -> None:
        """Add text to the frequency collection.

        Args:
            text: Text to tokenize and count. Should be normalized first.
        """
        if not text:
            return
        words = _WORD_TOKENIZE_RE.findall(text.lower())
        self._word_counts.update(words)
        self._total_words += len(words)

    def add_texts(self, texts: List[str]) -> None:
        """Add multiple texts to the frequency collection."""
        for text in texts:
            self.add_text(text)

    @property
    def word_counts(self) -> Counter:
        """Get the word frequency counter."""
        return self._word_counts

    @property
    def total_words(self) -> int:
        """Get total number of words processed."""
        return self._total_words

    @property
    def vocabulary_size(self) -> int:
        """Get the number of unique words."""
        return len(self._word_counts)

    def get_frequency(self, word: str) -> int:
        """Get the frequency of a specific word."""
        return self._word_counts.get(word.lower(), 0)

    def get_relative_frequency(self, word: str) -> float:
        """Get the relative frequency of a word (count / total)."""
        if self._total_words == 0:
            return 0.0
        return self._word_counts.get(word.lower(), 0) / self._total_words

    def most_common(self, n: Optional[int] = None) -> List[Tuple[str, int]]:
        """Get the n most common words."""
        return self._word_counts.most_common(n)

    def reset(self) -> None:
        """Reset all collected frequencies."""
        self._word_counts.clear()
        self._total_words = 0

    def merge(self, other: "WordFrequencyCollector") -> None:
        """Merge another collector into this one."""
        self._word_counts.update(other._word_counts)
        self._total_words += other._total_words


def _edit_distance(s1: str, s2: str) -> int:
    """Compute the Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return _edit_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # j+1 instead of j since previous_row and current_row are one character longer
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def _apostrophe_variants(word: str) -> List[str]:
    """Generate variants of a word with apostrophes inserted.

    For Uzbek, common positions are after o, g for o', g'.
    """
    variants = []
    # Try inserting apostrophe after each character
    for i in range(1, len(word)):
        variant = word[:i] + "'" + word[i:]
        variants.append(variant)
    return variants


def _remove_apostrophes(word: str) -> str:
    """Remove all apostrophes from a word."""
    return word.replace("'", "")


@dataclass
class TypoCandidate:
    """Represents a potential typo and its suggested correction."""

    typo: str
    correction: str
    typo_frequency: int
    correction_frequency: int
    confidence: float  # 0.0 to 1.0, higher = more confident it's a typo
    reason: str  # Description of why this is considered a typo


@dataclass
class TypoDetectionStats:
    """Statistics about typo detection and corrections."""

    total_typos_detected: int = 0
    total_corrections_applied: int = 0
    typos_by_word: Counter = field(default_factory=Counter)

    def record_detection(self, typo: str, correction: str) -> None:
        """Record a detected typo."""
        self.total_typos_detected += 1
        self.typos_by_word[f"{typo} -> {correction}"] += 1

    def record_correction(self) -> None:
        """Record an applied correction."""
        self.total_corrections_applied += 1

    def reset(self) -> None:
        """Reset all statistics."""
        self.total_typos_detected = 0
        self.total_corrections_applied = 0
        self.typos_by_word.clear()

    def report(self) -> str:
        """Generate a human-readable report."""
        if self.total_typos_detected == 0:
            return "No frequency-based typos detected."
        lines = [
            f"Total typos detected: {self.total_typos_detected}",
            f"Total corrections applied: {self.total_corrections_applied}",
            "Typos by word (top 50):",
        ]
        for word_pair, count in self.typos_by_word.most_common(50):
            lines.append(f"  {word_pair}: {count}")
        return "\n".join(lines)


class FrequencyBasedTypoDetector:
    """Detects potential typos based on word frequency analysis.

    A word is considered a potential typo if:
    1. It has low frequency in the corpus
    2. A similar word (by edit distance or apostrophe insertion) has much higher frequency

    This is particularly useful for Uzbek where apostrophes (o', g') are often
    mistakenly omitted.
    """

    def __init__(
        self,
        frequency_collector: WordFrequencyCollector,
        min_frequency_ratio: float = 50.0,
        max_edit_distance: int = 2,
        min_correction_frequency: int = 500,
        min_typo_length: int = 3,
        confidence_threshold: float = 0.7,
    ) -> None:
        """Initialize the typo detector.

        Args:
            frequency_collector: A WordFrequencyCollector with frequencies from the dataset
            min_frequency_ratio: Minimum ratio of correction_freq / typo_freq to consider a typo
            max_edit_distance: Maximum edit distance to consider for corrections
            min_correction_frequency: Minimum frequency of the correction word
            min_typo_length: Minimum length of word to consider as potential typo
            confidence_threshold: Minimum confidence score to include a typo
        """
        self._collector = frequency_collector
        self._min_frequency_ratio = min_frequency_ratio
        self._max_edit_distance = max_edit_distance
        self._min_correction_frequency = min_correction_frequency
        self._min_typo_length = min_typo_length
        self._confidence_threshold = confidence_threshold

        # Cache for detected typos: typo -> correction
        self._typo_corrections: Dict[str, str] = {}
        self._analyzed = False
        self._stats = TypoDetectionStats()

        # High-frequency words cache for faster lookup
        self._high_freq_words: Set[str] = set()

    @property
    def stats(self) -> TypoDetectionStats:
        """Get typo detection statistics."""
        return self._stats

    def analyze(self) -> List[TypoCandidate]:
        """Analyze the frequency data to detect potential typos.

        Returns:
            List of TypoCandidate objects representing detected typos
        """
        candidates = []
        word_counts = self._collector.word_counts

        # Build set of high-frequency words for faster lookup
        self._high_freq_words = {
            word
            for word, count in word_counts.items()
            if count >= self._min_correction_frequency
        }

        # Also index by apostrophe-stripped version
        stripped_to_original: Dict[str, List[str]] = {}
        for word in self._high_freq_words:
            stripped = _remove_apostrophes(word)
            if stripped not in stripped_to_original:
                stripped_to_original[stripped] = []
            stripped_to_original[stripped].append(word)

        # Check each word for potential typos
        for word, count in word_counts.items():
            if len(word) < self._min_typo_length:
                continue

            # Skip if this word is already high frequency
            if word in self._high_freq_words:
                continue

            # Check 1: Word without apostrophe might be typo for word with apostrophe
            stripped = _remove_apostrophes(word)
            if stripped == word:  # Word has no apostrophes
                # Look for high-freq words that match when apostrophes are removed
                if stripped in stripped_to_original:
                    for potential_correction in stripped_to_original[stripped]:
                        if potential_correction == word:
                            continue
                        correction_freq = word_counts[potential_correction]
                        if correction_freq < self._min_correction_frequency:
                            continue
                        if (
                            count > 0
                            and correction_freq / count >= self._min_frequency_ratio
                        ):
                            confidence = self._calculate_confidence(
                                count,
                                correction_freq,
                                edit_dist=len(potential_correction) - len(word),
                            )
                            if confidence >= self._confidence_threshold:
                                candidate = TypoCandidate(
                                    typo=word,
                                    correction=potential_correction,
                                    typo_frequency=count,
                                    correction_frequency=correction_freq,
                                    confidence=confidence,
                                    reason="missing apostrophe",
                                )
                                candidates.append(candidate)
                                self._typo_corrections[word] = potential_correction

            # Check 2: Edit distance to high-frequency words
            best_match = self._find_best_edit_distance_match(word, count)
            if best_match and word not in self._typo_corrections:
                candidates.append(best_match)
                self._typo_corrections[word] = best_match.correction

        self._analyzed = True
        return candidates

    def _find_best_edit_distance_match(
        self, word: str, word_freq: int
    ) -> Optional[TypoCandidate]:
        """Find the best high-frequency word within edit distance."""
        best_candidate = None
        best_confidence = 0.0

        for high_freq_word in self._high_freq_words:
            # Quick length check to avoid expensive edit distance calculation
            len_diff = abs(len(high_freq_word) - len(word))
            if len_diff > self._max_edit_distance:
                continue

            edit_dist = _edit_distance(word, high_freq_word)
            if edit_dist > self._max_edit_distance or edit_dist == 0:
                continue

            correction_freq = self._collector.get_frequency(high_freq_word)
            if (
                word_freq > 0
                and correction_freq / word_freq >= self._min_frequency_ratio
            ):
                confidence = self._calculate_confidence(
                    word_freq, correction_freq, edit_dist
                )
                if (
                    confidence > best_confidence
                    and confidence >= self._confidence_threshold
                ):
                    best_confidence = confidence
                    best_candidate = TypoCandidate(
                        typo=word,
                        correction=high_freq_word,
                        typo_frequency=word_freq,
                        correction_frequency=correction_freq,
                        confidence=confidence,
                        reason=f"edit distance {edit_dist}",
                    )

        return best_candidate

    def _calculate_confidence(
        self, typo_freq: int, correction_freq: int, edit_dist: int
    ) -> float:
        """Calculate confidence score for a typo detection.

        Confidence is based on:
        - Frequency ratio (higher = more confident)
        - Edit distance (lower = more confident)
        """
        if typo_freq == 0:
            freq_ratio = correction_freq
        else:
            freq_ratio = correction_freq / typo_freq

        # Normalize frequency ratio to 0-1 range (cap at 1000x ratio)
        freq_score = min(freq_ratio / 1000.0, 1.0)

        # Edit distance score (1 = best, decreases with distance)
        edit_score = 1.0 / (1.0 + edit_dist)

        # Combined score
        return (freq_score * 0.7) + (edit_score * 0.3)

    def get_correction(self, word: str) -> Optional[str]:
        """Get the correction for a word if it's detected as a typo.

        Args:
            word: The word to check (case-insensitive)

        Returns:
            The correction if the word is a typo, None otherwise
        """
        if not self._analyzed:
            self.analyze()
        return self._typo_corrections.get(word.lower())

    def correct_text(self, text: str, record_stats: bool = True) -> str:
        """Apply typo corrections to text.

        Args:
            text: Text to correct
            record_stats: Whether to record statistics about corrections

        Returns:
            Text with typo corrections applied
        """
        if not self._analyzed:
            self.analyze()

        if not self._typo_corrections:
            return text

        words = _WORD_TOKENIZE_RE.findall(text)
        result = text

        for word in words:
            lower_word = word.lower()
            correction_lookup = self._typo_corrections.get(lower_word)
            if correction_lookup is not None:
                correction: str = correction_lookup
                # Preserve case
                if word.isupper():
                    correction = correction.upper()
                elif word[0].isupper():
                    correction = correction.capitalize()

                # Replace whole word only
                pattern = r"\b" + re.escape(word) + r"\b"
                result = re.sub(pattern, correction, result)

                if record_stats:
                    self._stats.record_detection(lower_word, correction_lookup)
                    self._stats.record_correction()

        return result

    def get_typo_report(self) -> str:
        """Generate a report of all detected typos."""
        if not self._analyzed:
            self.analyze()

        if not self._typo_corrections:
            return "No frequency-based typos detected."

        lines = [
            f"Frequency-Based Typo Detection Report",
            f"=====================================",
            f"Total unique typos detected: {len(self._typo_corrections)}",
            f"",
            f"Detected typos (sorted by typo frequency):",
        ]

        # Sort by typo frequency
        sorted_typos = sorted(
            self._typo_corrections.items(),
            key=lambda x: self._collector.get_frequency(x[0]),
            reverse=True,
        )

        for typo, correction in sorted_typos[:100]:  # Top 100
            typo_freq = self._collector.get_frequency(typo)
            correction_freq = self._collector.get_frequency(correction)
            lines.append(
                f"  '{typo}' ({typo_freq}x) -> '{correction}' ({correction_freq}x)"
            )

        if len(sorted_typos) > 100:
            lines.append(f"  ... and {len(sorted_typos) - 100} more")

        return "\n".join(lines)

    def reset(self) -> None:
        """Reset the detector state."""
        self._typo_corrections.clear()
        self._high_freq_words.clear()
        self._analyzed = False
        self._stats.reset()


# Global instances for easy access
_global_frequency_collector: Optional[WordFrequencyCollector] = None
_global_typo_detector: Optional[FrequencyBasedTypoDetector] = None


def get_frequency_collector() -> WordFrequencyCollector:
    """Get or create the global frequency collector."""
    global _global_frequency_collector
    if _global_frequency_collector is None:
        _global_frequency_collector = WordFrequencyCollector()
    return _global_frequency_collector


def reset_frequency_collector() -> None:
    """Reset the global frequency collector."""
    global _global_frequency_collector
    if _global_frequency_collector is not None:
        _global_frequency_collector.reset()
    _global_frequency_collector = None


def get_typo_detector(
    min_frequency_ratio: float = 50.0,
    max_edit_distance: int = 2,
    min_correction_frequency: int = 500,
    min_typo_length: int = 3,
    confidence_threshold: float = 0.7,
) -> FrequencyBasedTypoDetector:
    """Get or create the global typo detector.

    If the detector doesn't exist, creates one using the global frequency collector.
    """
    global _global_typo_detector
    if _global_typo_detector is None:
        collector = get_frequency_collector()
        _global_typo_detector = FrequencyBasedTypoDetector(
            collector,
            min_frequency_ratio=min_frequency_ratio,
            max_edit_distance=max_edit_distance,
            min_correction_frequency=min_correction_frequency,
            min_typo_length=min_typo_length,
            confidence_threshold=confidence_threshold,
        )
    return _global_typo_detector


def reset_typo_detector() -> None:
    """Reset the global typo detector."""
    global _global_typo_detector
    if _global_typo_detector is not None:
        _global_typo_detector.reset()
    _global_typo_detector = None


__all__ = [
    "normalize_text",
    "contains_standalone_c",
    "MisspellingStats",
    "get_misspelling_stats",
    "reset_misspelling_stats",
    # Frequency-based typo detection
    "WordFrequencyCollector",
    "FrequencyBasedTypoDetector",
    "TypoCandidate",
    "TypoDetectionStats",
    "get_frequency_collector",
    "reset_frequency_collector",
    "get_typo_detector",
    "reset_typo_detector",
]
