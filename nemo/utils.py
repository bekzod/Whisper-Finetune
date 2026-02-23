#!/usr/bin/env python3
"""Shared text cleanup utilities for NeMo prep scripts."""

from __future__ import annotations

import logging
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from hunspell import HunSpell

_LOGGER = logging.getLogger(__name__)

# Regex to match C or c NOT followed by h or H
_STANDALONE_C_RE = re.compile(r"[Cc](?![Hh])")

_APOSTROPHE_TRANSLATION = str.maketrans(
    {
        "\u2019": "'",  # RIGHT SINGLE QUOTATION MARK
        "\u2018": "'",  # LEFT SINGLE QUOTATION MARK
        "\u02bc": "'",  # MODIFIER LETTER APOSTROPHE
        "\u02bb": "'",  # MODIFIER LETTER TURNED COMMA
        "\u02b9": "'",  # MODIFIER LETTER PRIME
        "\u02c8": "'",  # MODIFIER LETTER VERTICAL LINE
        "`": "'",  # GRAVE ACCENT
        "´": "'",  # ACUTE ACCENT
        "‛": "'",  # SINGLE HIGH-REVERSED-9 QUOTATION MARK
        "′": "'",  # PRIME
        "ʻ": "'",  # MODIFIER LETTER TURNED COMMA (alternative)
        "ʼ": "'",  # MODIFIER LETTER APOSTROPHE (alternative)
        "'": "'",  # FULLWIDTH APOSTROPHE
        "ˈ": "'",  # MODIFIER LETTER VERTICAL LINE (IPA)
        "ʹ": "'",  # MODIFIER LETTER PRIME
        "\u0027": "'",  # APOSTROPHE (standard, for completeness)
        "\u055a": "'",  # ARMENIAN APOSTROPHE
        "\ua78c": "'",  # LATIN SMALL LETTER SALTILLO
    }
)
_ALLOWED_TEXT_RE = re.compile(r"[^a-zA-ZА-Яа-яЎўҚқҒғҲҳ0-9\s,.'\-?]+")

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
    "bolmoq": "bo'lmoq",
    "boldi": "bo'ldi",
    "bolasiz": "bo'lasiz",
    "bolaman": "bo'laman",
    "bolishga": "bo'lishga",
    "bolishini": "bo'lishini",
    "bolmasa": "bo'lmasa",
    "bolmasdan": "bo'lmasdan",
    "bolardi": "bo'lardi",
    "bolayotgan": "bo'layotgan",
    # Missing apostrophes in o'z (self) forms
    "ozim": "o'zim",
    "ozing": "o'zing",
    "ozimiz": "o'zimiz",
    "ozingiz": "o'zingiz",
    "ozlari": "o'zlari",
    "oziga": "o'ziga",
    "ozini": "o'zini",
    "ozidan": "o'zidan",
    "ozining": "o'zining",
    "ozicha": "o'zicha",
    "ozaro": "o'zaro",
    "ozbek": "o'zbek",
    "ozbekiston": "o'zbekiston",
    "ozbekcha": "o'zbekcha",
    # Missing apostrophes in qo'ymoq (to put) forms
    "qoydi": "qo'ydi",
    "qoyib": "qo'yib",
    "qoygan": "qo'ygan",
    "qoyar": "qo'yar",
    "qoyish": "qo'yish",
    "qoyadi": "qo'yadi",
    "qoymoq": "qo'ymoq",
    "qoyildi": "qo'yildi",
    "qoyilgan": "qo'yilgan",
    "qoysa": "qo'ysa",
    "qoyaman": "qo'yaman",
    "qoyasiz": "qo'yasiz",
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
    "kormoq": "ko'rmoq",
    "kordim": "ko'rdim",
    "kording": "ko'rding",
    "kordik": "ko'rdik",
    "korishga": "ko'rishga",
    "korinish": "ko'rinish",
    "korinadi": "ko'rinadi",
    "korsatmoq": "ko'rsatmoq",
    "korsatildi": "ko'rsatildi",
    "korsatilgan": "ko'rsatilgan",
    "korgazma": "ko'rgazma",
    # Missing apostrophes in ko'p (many) forms
    "kop": "ko'p",
    "kopchilik": "ko'pchilik",
    "koplab": "ko'plab",
    "koprok": "ko'proq",
    "kopdan": "ko'pdan",
    "kopga": "ko'pga",
    "kopgina": "ko'pgina",
    "kopaydi": "ko'paydi",
    "kopayish": "ko'payish",
    "kopaytirish": "ko'paytirish",
    "kopincha": "ko'pincha",
    # Missing apostrophes in ko'cha (street) forms
    "kocha": "ko'cha",
    "kochada": "ko'chada",
    "kochasi": "ko'chasi",
    "kochalar": "ko'chalar",
    "kochalarda": "ko'chalarda",
    "kochaning": "ko'chaning",
    # Missing apostrophes in ko'z (eye) forms
    "koz": "ko'z",
    "kozim": "ko'zim",
    "kozi": "ko'zi",
    "kozlar": "ko'zlar",
    "kozlari": "ko'zlari",
    "kozga": "ko'zga",
    "kozdan": "ko'zdan",
    "kozoynak": "ko'zoynak",
    # Missing apostrophes in ko'nmoq (to agree/get used to) forms
    "kondi": "ko'ndi",
    "konib": "ko'nib",
    "konish": "ko'nish",
    "konadi": "ko'nadi",
    "kongil": "ko'ngil",
    "kongilsiz": "ko'ngilsiz",
    "kongilga": "ko'ngilga",
    # Missing apostrophes in ko'chirmoq (to copy/move) forms
    "kochirmoq": "ko'chirmoq",
    "kochirdi": "ko'chirdi",
    "kochirish": "ko'chirish",
    "kochirib": "ko'chirib",
    "kochirgan": "ko'chirgan",
    # Missing apostrophes in to'g'ri (correct/straight)
    "togri": "to'g'ri",
    "togrisida": "to'g'risida",
    "togridan": "to'g'ridan",
    "togrisi": "to'g'risi",
    "togrilab": "to'g'rilab",
    "togrilash": "to'g'rilash",
    # Missing apostrophes in to'liq (complete) forms
    "toliq": "to'liq",
    "toldirib": "to'ldirib",
    "toldirish": "to'ldirish",
    "toldirdi": "to'ldirdi",
    "tolgan": "to'lgan",
    "toladi": "to'ladi",
    "tolov": "to'lov",
    "tolovlar": "to'lovlar",
    "tolash": "to'lash",
    # Missing apostrophes in to'xta (stop) forms
    "toxtamoq": "to'xtamoq",
    "toxtadi": "to'xtadi",
    "toxtab": "to'xtab",
    "toxtash": "to'xtash",
    "toxtatmoq": "to'xtatmoq",
    "toxtatdi": "to'xtatdi",
    "toxtatish": "to'xtatish",
    # Missing apostrophes in to'pla (collect) forms
    "toplamoq": "to'plamoq",
    "topladi": "to'pladi",
    "toplab": "to'plab",
    "toplash": "to'plash",
    "toplam": "to'plam",
    "toplangan": "to'plangan",
    # Missing apostrophes in to'y (wedding) forms
    "toy": "to'y",
    "toyda": "to'yda",
    "toyi": "to'yi",
    "toylar": "to'ylar",
    "toying": "to'ying",
    # Missing apostrophes in go'zal (beautiful)
    "gozal": "go'zal",
    "gozallik": "go'zallik",
    "gozalligi": "go'zalligi",
    # Missing apostrophes in go'sht (meat) forms
    "gosht": "go'sht",
    "goshtli": "go'shtli",
    "goshtxona": "go'shtxona",
    # Missing apostrophes in go'yo (as if) forms
    "goyo": "go'yo",
    "goyoki": "go'yoki",
    # Missing apostrophes in so'z (word) forms
    "sozlar": "so'zlar",
    "sozlari": "so'zlari",
    "sozlash": "so'zlash",
    "sozlashdi": "so'zlashdi",
    "sozlashmoq": "so'zlashmoq",
    "sozlashib": "so'zlashib",
    "sozni": "so'zni",
    "sozning": "so'zning",
    "sozga": "so'zga",
    "sozsiz": "so'zsiz",
    "sozma": "so'zma",
    # Missing apostrophes in so'rov (request)
    "sorov": "so'rov",
    "sorovnoma": "so'rovnoma",
    "sorovlar": "so'rovlar",
    # Missing apostrophes in so'ng (after/end) forms
    "song": "so'ng",
    "songgi": "so'nggi",
    "songra": "so'ngra",
    # Missing apostrophes in o'rganmoq (to learn) forms
    "organmoq": "o'rganmoq",
    "organdi": "o'rgandi",
    "organib": "o'rganib",
    "organish": "o'rganish",
    "organadi": "o'rganadi",
    "organdim": "o'rgandim",
    "organaman": "o'rganaman",
    "organildi": "o'rganildi",
    "organilgan": "o'rganilgan",
    # Missing apostrophes in o'ylamoq (to think) forms
    "oyladi": "o'yladi",
    "oylab": "o'ylab",
    "oylash": "o'ylash",
    "oylaydi": "o'ylaydi",
    "oylamoq": "o'ylamoq",
    "oyladim": "o'yladim",
    "oylagan": "o'ylagan",
    "oylayman": "o'ylayman",
    # Missing apostrophes in o'tirmoq (to sit) forms
    "otirdi": "o'tirdi",
    "otirib": "o'tirib",
    "otirish": "o'tirish",
    "otiradi": "o'tiradi",
    "otirmoq": "o'tirmoq",
    "otirdim": "o'tirdim",
    "otirgan": "o'tirgan",
    "otiraman": "o'tiraman",
    # Missing apostrophes in o'ynamoq (to play) forms
    "oynadi": "o'ynadi",
    "oynab": "o'ynab",
    "oynash": "o'ynash",
    "oynaydi": "o'ynaydi",
    "oynamoq": "o'ynamoq",
    "oynadim": "o'ynadim",
    "oynagan": "o'ynagan",
    "oyin": "o'yin",
    "oyinlar": "o'yinlar",
    "oyinchi": "o'yinchi",
    # Missing apostrophes in o'qimoq (to read/study) - careful: oqmoq (to flow) is different
    "oqituvchi": "o'qituvchi",
    "oquvchi": "o'quvchi",
    "oqimoq": "o'qimoq",
    "oqidi": "o'qidi",
    "oqish": "o'qish",
    "oqiydi": "o'qiydi",
    "oqidim": "o'qidim",
    "oqigan": "o'qigan",
    "oqiyman": "o'qiyman",
    "oqitish": "o'qitish",
    "oqitmoq": "o'qitmoq",
    # Missing apostrophes in o'zgarmoq (to change) forms
    "ozgardi": "o'zgardi",
    "ozgarib": "o'zgarib",
    "ozgarish": "o'zgarish",
    "ozgaradi": "o'zgaradi",
    "ozgarmoq": "o'zgarmoq",
    "ozgardim": "o'zgardim",
    "ozgargan": "o'zgargan",
    "ozgartirmoq": "o'zgartirmoq",
    "ozgartirish": "o'zgartirish",
    "ozgartirdi": "o'zgartirdi",
    # Missing apostrophes in o'tmoq (to pass) forms
    "otmoq": "o'tmoq",
    "otadi": "o'tadi",
    "otgan": "o'tgan",
    "otmish": "o'tmish",
    "otgan": "o'tgan",
    # Missing apostrophes in other o' words
    "ogil": "o'g'il",
    "ogil": "o'g'il",
    "oglon": "o'g'lon",
    "ogli": "o'g'li",
    "ogillar": "o'g'illar",
    "orta": "o'rta",
    "ortada": "o'rtada",
    "ortadagi": "o'rtadagi",
    "ortacha": "o'rtacha",
    "ortasida": "o'rtasida",
    "ortasidan": "o'rtasidan",
    "ormon": "o'rmon",
    "ormonda": "o'rmonda",
    "ormonlar": "o'rmonlar",
    "osha": "o'sha",
    "oshanda": "o'shanda",
    "oshanday": "o'shanday",
    "orin": "o'rin",
    "orinda": "o'rinda",
    "orinli": "o'rinli",
    "orinsiz": "o'rinsiz",
    "orinbos": "o'rinbos",
    "olik": "o'lik",
    "oldir": "o'ldir",
    "otkazdi": "o'tkazdi",
    "otkazish": "o'tkazish",
    "otkazmoq": "o'tkazmoq",
    "otkazilib": "o'tkazilib",
    "otkazildi": "o'tkazildi",
    "otkazilgan": "o'tkazilgan",
    "olcham": "o'lcham",
    "olchov": "o'lchov",
    "olchovlar": "o'lchovlar",
    "osim": "o'sim",
    "osimlik": "o'simlik",
    "osimliklar": "o'simliklar",
    "oqi": "o'qi",
    "oqlar": "o'qlar",
    "otov": "o'tov",
    "oroq": "o'roq",
    # Missing apostrophes in g' words
    "galaba": "g'alaba",
    "galabali": "g'alabali",
    "galabaga": "g'alabaga",
    "goya": "g'oya",
    "goyalar": "g'oyalar",
    "goyasi": "g'oyasi",
    "goyat": "g'oyat",
    "garb": "g'arb",
    "garbiy": "g'arbiy",
    "garbda": "g'arbda",
    "gayrat": "g'ayrat",
    "gayratli": "g'ayratli",
    "gazab": "g'azab",
    "gazabli": "g'azabli",
    "gazablandi": "g'azablandi",
    "galati": "g'alati",
    "galatiroq": "g'alatiroq",
    "gamgin": "g'amgin",
    "gamginlik": "g'amginlik",
    "gildirak": "g'ildirak",
    "gildiraklar": "g'ildiraklar",
    "gisht": "g'isht",
    "gishtlar": "g'ishtlar",
    "gishtli": "g'ishtli",
    "goyib": "g'oyib",
    "gam": "g'am",
    "gamxo'rlik": "g'amxo'rlik",
    "gamli": "g'amli",
    "gurur": "g'urur",
    "gururli": "g'ururli",
    "gururlanmoq": "g'ururlanmoq",
    "govur": "g'ovur",
    "goz": "g'oz",
    "gozlar": "g'ozlar",
    "galla": "g'alla",
    "gallalar": "g'allalar",
    "galla": "g'alla",
    "govak": "g'ovak",
    "goldir": "g'oldir",
    "golib": "g'olib",
    "goyilik": "g'oyilik",
    "gijim": "g'ijim",
    "gijimlab": "g'ijimlab",
    "gijimlash": "g'ijimlash",
    "girrom": "g'irrom",
    "girt": "g'irt",
    # Soft h -> hard x corrections (common Uzbek pronunciation/spelling errors)
    # xabar (news) and related forms
    "habar": "xabar",
    "habari": "xabari",
    "habarlar": "xabarlar",
    "habarlari": "xabarlari",
    "habarnoma": "xabarnoma",
    "habarchi": "xabarchi",
    "habardon": "xabardon",
    "habarsiz": "xabarsiz",
    "habardor": "xabardor",
    # xafa (upset) and related forms
    "hafa": "xafa",
    "hafalik": "xafalik",
    "hafagarchilik": "xafagarchilik",
    # xarid (purchase) and related forms
    "harid": "xarid",
    "haridor": "xaridor",
    "haridorlar": "xaridorlar",
    "haridorlik": "xaridorlik",
    # xizmat (service) and related forms
    "hizmat": "xizmat",
    "hizmati": "xizmati",
    "hizmatlar": "xizmatlar",
    "hizmatchi": "xizmatchi",
    "hizmatchilar": "xizmatchilar",
    "hizmatkor": "xizmatkor",
    # xalq (people/nation) and related forms
    "halq": "xalq",
    "halqi": "xalqi",
    "halqaro": "xalqaro",
    "halqlararo": "xalqlararo",
    "halqlar": "xalqlar",
    "halqning": "xalqning",
    "halqparvar": "xalqparvar",
    # xotin (wife) and related forms
    "hotin": "xotin",
    "hotini": "xotini",
    "hotinlar": "xotinlar",
    "hotinlari": "xotinlari",
    # xona (room) and related forms
    "hona": "xona",
    "honada": "xonada",
    "honasi": "xonasi",
    "honalar": "xonalar",
    "honadon": "xonadon",
    "honalardan": "xonalardan",
    "honaning": "xonaning",
    # xotira (memory) and related forms
    "hotira": "xotira",
    "hotirasi": "xotirasi",
    "hotiralar": "xotiralar",
    "hotirlash": "xotirlash",
    "hotirladi": "xotirladi",
    "hotirlamoq": "xotirlamoq",
    "hotirjam": "xotirjam",
    # xavf (danger) and related forms
    "havf": "xavf",
    "havfli": "xavfli",
    "havfsiz": "xavfsiz",
    "havfsizlik": "xavfsizlik",
    "havflilik": "xavflilik",
    # xato (mistake) and related forms
    "hato": "xato",
    "hatosi": "xatosi",
    "hatolar": "xatolar",
    "hatolik": "xatolik",
    "hatosiz": "xatosiz",
    # xush (pleasant) and related forms
    "hush": "xush",
    "hushnud": "xushnud",
    "hushmuomala": "xushmuomala",
    "hushbo'y": "xushbo'y",
    "hushhol": "xushhol",
    "hushxabar": "xushxabar",
    "hushhavo": "xushhavo",
    # xursand (happy) and related forms
    "hursand": "xursand",
    "hursandlik": "xursandlik",
    "hursandchilik": "xursandchilik",
    # xayr (goodbye) and related forms
    "hayr": "xayr",
    "hayrlashdi": "xayrlashdi",
    "hayriya": "xayriya",
    "hayriyachi": "xayriyachi",
    "hayrlashmoq": "xayrlashmoq",
    "hayrlashish": "xayrlashish",
    "hayrixoh": "xayrixoh",
    # xudo (god) and related forms
    "hudo": "xudo",
    "hudoga": "xudoga",
    "hudoning": "xudoning",
    "hudoyo": "xudoyo",
    "hudojo": "xudojo",
    # xulosa (conclusion) and related forms
    "hulosa": "xulosa",
    "hulosasi": "xulosasi",
    "hulosalar": "xulosalar",
    "hulosalash": "xulosalash",
    # xuddi (exactly)
    "huddi": "xuddi",
    # xo'ja (master/teacher) and related forms
    "hoja": "xo'ja",
    "hojalik": "xo'jalik",
    "hojali": "xo'jali",
    "hojayin": "xo'jayin",
    "hojayinlar": "xo'jayinlar",
    # xo'roz (rooster)
    "horoz": "xo'roz",
    "horozlar": "xo'rozlar",
    # xom (raw) and related forms
    "hom": "xom",
    "homashyo": "xomashyo",
    "homaki": "xomaki",
    # xor (choir/humiliated) and related forms
    "hor": "xor",
    "horlik": "xorlik",
    "horlangan": "xorlangan",
    # xossa (property/characteristic)
    "hossa": "xossa",
    "hossalar": "xossalar",
    "hossali": "xossali",
    # xorazm (Khorezm region)
    "horazm": "xorazm",
    "horazmlik": "xorazmlik",
    "horazmda": "xorazmda",
    # xat (letter) and related forms
    "hat": "xat",
    "hati": "xati",
    "hatlar": "xatlar",
    "hatni": "xatni",
    "hatning": "xatning",
    # xalol (honest/halal in Uzbek context)
    "halol": "xalol",
    "halollik": "xalollik",
    "haloldan": "xaloldan",
    # xayol (thought/imagination) forms
    "hayol": "xayol",
    "hayollar": "xayollar",
    "hayoliy": "xayoliy",
    "hayolot": "xayolot",
    "hayolparast": "xayolparast",
    # xazon (autumn leaves) forms
    "hazon": "xazon",
    "hazonlar": "xazonlar",
    "hazonrez": "xazonrez",
    # xurmo (date palm/fruit)
    "hurmo": "xurmo",
    "hurmolar": "xurmolar",
    # xurofot (superstition)
    "hurofot": "xurofot",
    "hurofotlar": "xurofotlar",
    "hurofotchi": "xurofotchi",
    # xushtor (lover/fan)
    "hushtor": "xushtor",
    "hushtorlar": "xushtorlar",
    # xo'p (okay)
    "hop": "xo'p",
    # Additional common words
    "harakat": "xarakat",
    "harakatlar": "xarakatlar",
    "harakatlanmoq": "xarakatlanmoq",
    "ham": "xam",  # Note: 'ham' (also) is different from 'xam' (bent) - use with caution
}

# Build regex pattern for whole-word matching (case-insensitive)
_UZBEK_MISSPELLING_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in _UZBEK_MISSPELLINGS.keys()) + r")\b",
    re.IGNORECASE,
)
_MULTISPACE_RE = re.compile(r"\s+")
# Pattern to match spaces before punctuation (e.g., "qurildi ." => "qurildi.")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([.,])")
# Pattern to match spaces between digits (e.g., "600 000" => "600000")
_SPACED_NUMBER_RE = re.compile(r"(\d)\s+(?=\d)")
# Pattern to collapse comma-separated thousands (e.g., "43,000" => "43000").
_COMMA_THOUSANDS_RE = re.compile(r"\b(\d{1,3})((?:,\d{3})+)\b")
# Pattern to match standalone integer tokens for number normalization.
_NUMBER_TOKEN_RE = re.compile(r"\b\d+\b")
# Pattern to match decimal numbers (e.g., 3.14 or 3,14).
_DECIMAL_NUMBER_RE = re.compile(r"\b(\d+)[.,](\d+)\b")
# Pattern to match Uzbek-style phone chunks like "90 426 53 14".
_UZBEK_PHONE_GROUP_RE = re.compile(
    r"(?<!\d)(\d{2})[\s-]+(\d{3})[\s-]+(\d{2})[\s-]+(\d{2})(?!\d)"
)
# Pattern to match common Uzbek abbreviations and variants.
_UZBEK_YEAR_ABBREV_RE = re.compile(r"\b(\d+)\s*(?:-\s*)?y\.", re.IGNORECASE)
_UZBEK_NUMBER_KG_RE = re.compile(r"\b(\d+)\s*(?:-\s*)?(kg)\b", re.IGNORECASE)
_UZBEK_NUMBER_SUM_RE = re.compile(r"\b(\d+)\s*(?:-\s*)?(sum|so'm)\b", re.IGNORECASE)
_UZBEK_STANDALONE_KG_RE = re.compile(r"(?<![a-zA-Z])(kg)(?![a-zA-Z])", re.IGNORECASE)
_UZBEK_STANDALONE_SUM_RE = re.compile(
    r"(?<![a-zA-Z])(sum|so'm)(?![a-zA-Z])", re.IGNORECASE
)
# Pattern to match number+suffix tokens (e.g., 5-ta, 10yil, 2-sinf).
_NUMBER_WITH_SUFFIX_RE = re.compile(
    r"\b(\d+)([-']?)([a-zA-ZА-Яа-яЎўҚқҒғҲҳ][a-zA-ZА-Яа-яЎўҚқҒғҲҳ']*)\b"
)
_UZBEK_MONTHS = (
    "yanvar",
    "fevral",
    "mart",
    "aprel",
    "may",
    "iyun",
    "iyul",
    "avgust",
    "sentyabr",
    "oktyabr",
    "noyabr",
    "dekabr",
)
_UZBEK_DAY_MONTH_RE = re.compile(
    r"\b(0?[1-9]|[12][0-9]|3[01])(?:\s*[-/.]\s*|\s+)("
    + "|".join(_UZBEK_MONTHS)
    + r")\b",
    re.IGNORECASE,
)
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
_COMMON_HUNSPELL_DICT_DIRS = (
    "/usr/share/hunspell",
    "/usr/share/myspell",
    "/usr/share/myspell/dicts",
    "/opt/homebrew/share/hunspell",
    "/Library/Spelling",
)
_UZBEK_HUNSPELL_BASENAMES = ("uz_UZ", "uz_Latn_UZ", "uz")
_UZBEK_NUMBER_UNITS = (
    "nol",
    "bir",
    "ikki",
    "uch",
    "to'rt",
    "besh",
    "olti",
    "yetti",
    "sakkiz",
    "to'qqiz",
)
_UZBEK_NUMBER_TENS = {
    10: "o'n",
    20: "yigirma",
    30: "o'ttiz",
    40: "qirq",
    50: "ellik",
    60: "oltmish",
    70: "yetmish",
    80: "sakson",
    90: "to'qson",
}
_UZBEK_NUMBER_SCALES = (
    "",
    "ming",
    "million",
    "milliard",
    "trillion",
    "kvadrillion",
    "kvintillion",
)
_VALID_DECIMAL_MODES = {"fractional", "digit"}
_ATTACHED_NUMERIC_SUFFIXES = {
    "ta",
    "tacha",
    "tadan",
    "tasi",
    "talik",
    "chi",
    "nchi",
    "inchi",
    "ga",
    "gacha",
    "da",
    "dan",
    "ni",
    "ning",
    "na",
    "lab",
    "lik",
    "lar",
    "lari",
    "larga",
    "larda",
    "lardan",
}
# Suffixes that, when hyphenated with a number (e.g. "109-bet"),
# indicate ordinal form: "bir yuz to'qqizinchi bet".
# Compound forms like "yilgacha", "yilda" are matched via prefix check.
_ORDINAL_TRIGGER_SUFFIXES = {
    "bet",
    "mayda",
    "bob",
    "band",
    "bosqich",
    "guruh",
    "kurs",
    "sinf",
    "qism",
    "qator",
    "sahifa",
    "son",
    "joy",
    "o'rin",
    "daraja",
    "pog'ona",
    "tur",
    "davr",
    "qadam",
    "xona",
    "etaj",
    "yil",
}


def _is_ordinal_trigger(suffix_lower: str) -> bool:
    """Check if a suffix (or compound suffix) triggers ordinal conversion."""
    if suffix_lower in _ORDINAL_TRIGGER_SUFFIXES:
        return True
    return any(
        suffix_lower.startswith(stem) and len(suffix_lower) > len(stem)
        for stem in _ORDINAL_TRIGGER_SUFFIXES
    )


_UZBEK_DAY_ORDINALS = {
    1: "birinchi",
    2: "ikkinchi",
    3: "uchinchi",
    4: "to'rtinchi",
    5: "beshinchi",
    6: "oltinchi",
    7: "yettinchi",
    8: "sakkizinchi",
    9: "to'qqizinchi",
    10: "o'ninchi",
    11: "o'n birinchi",
    12: "o'n ikkinchi",
    13: "o'n uchinchi",
    14: "o'n to'rtinchi",
    15: "o'n beshinchi",
    16: "o'n oltinchi",
    17: "o'n yettinchi",
    18: "o'n sakkizinchi",
    19: "o'n to'qqizinchi",
    20: "yigirmanchi",
    21: "yigirma birinchi",
    22: "yigirma ikkinchi",
    23: "yigirma uchinchi",
    24: "yigirma to'rtinchi",
    25: "yigirma beshinchi",
    26: "yigirma oltinchi",
    27: "yigirma yettinchi",
    28: "yigirma sakkizinchi",
    29: "yigirma to'qqizinchi",
    30: "o'ttizinchi",
    31: "o'ttiz birinchi",
}


def _preserve_word_case(original: str, replacement: str) -> str:
    """Return replacement with casing aligned to original."""
    if original.isupper():
        return replacement.upper()
    if original and original[0].isupper():
        return replacement.capitalize()
    return replacement


def _uzbek_under_thousand(value: int) -> str:
    """Convert an integer from 0..999 to spoken Uzbek."""
    if value == 0:
        return _UZBEK_NUMBER_UNITS[0]

    parts: List[str] = []
    hundreds = value // 100
    remainder = value % 100

    if hundreds:
        parts.append(_UZBEK_NUMBER_UNITS[hundreds])
        parts.append("yuz")

    if remainder >= 20:
        tens = (remainder // 10) * 10
        parts.append(_UZBEK_NUMBER_TENS[tens])
        remainder %= 10
        if remainder:
            parts.append(_UZBEK_NUMBER_UNITS[remainder])
    elif remainder >= 10:
        parts.append(_UZBEK_NUMBER_TENS[10])
        remainder -= 10
        if remainder:
            parts.append(_UZBEK_NUMBER_UNITS[remainder])
    elif remainder > 0:
        parts.append(_UZBEK_NUMBER_UNITS[remainder])

    return " ".join(parts)


def _number_to_spoken_uzbek(value: int) -> str:
    """Convert an integer to spoken Uzbek words."""
    if value == 0:
        return _UZBEK_NUMBER_UNITS[0]

    is_negative = value < 0
    number = abs(value)
    scale_idx = 0
    parts: List[str] = []

    while number > 0:
        number, chunk = divmod(number, 1000)
        if chunk:
            chunk_words = _uzbek_under_thousand(chunk)
            scale = _UZBEK_NUMBER_SCALES[scale_idx]
            parts.append(f"{chunk_words} {scale}".strip())
        scale_idx += 1

    result = " ".join(reversed(parts))
    if is_negative:
        return f"minus {result}"
    return result


_UZBEK_VOWELS = set("aeiouaoʻ'")


def _number_to_ordinal_uzbek(value: int) -> str:
    """Convert an integer to its spoken Uzbek ordinal form (e.g. 5 -> 'beshinchi')."""
    spoken = _number_to_spoken_uzbek(value)
    last_char = spoken[-1].lower()
    if last_char in _UZBEK_VOWELS:
        return spoken + "nchi"
    return spoken + "inchi"


def _digits_to_spoken_uzbek(digits: str) -> str:
    """Convert a digit sequence to spoken Uzbek digit-by-digit."""
    return " ".join(_UZBEK_NUMBER_UNITS[int(digit)] for digit in digits)


def _resolve_decimal_mode(decimal_mode: Optional[str]) -> str:
    """Resolve decimal normalization mode from arg/env with validation."""
    mode = decimal_mode or os.environ.get("UZBEK_DECIMAL_MODE", "fractional")
    normalized_mode = mode.strip().lower()
    if normalized_mode in _VALID_DECIMAL_MODES:
        return normalized_mode
    _LOGGER.warning(
        "Invalid decimal mode '%s'. Falling back to 'fractional'.",
        mode,
    )
    return "fractional"


def _normalize_decimals_to_spoken_uzbek(
    text: str,
    stats: Optional[MisspellingStats] = None,
    decimal_mode: str = "fractional",
) -> str:
    """Replace decimal numbers with spoken Uzbek text."""
    if not text:
        return text
    if stats is None:
        stats = _misspelling_stats

    def replace_match(match: re.Match) -> str:
        raw_decimal = match.group(0)
        integer_part = int(match.group(1))
        fractional_part_raw = match.group(2)

        integer_spoken = _number_to_spoken_uzbek(integer_part)
        if decimal_mode == "digit":
            fractional_spoken = _digits_to_spoken_uzbek(fractional_part_raw)
            spoken = f"{integer_spoken} nuqta {fractional_spoken}"
        else:
            fractional_part = int(fractional_part_raw)
            fractional_spoken = _number_to_spoken_uzbek(fractional_part)
            spoken = f"{integer_spoken} butun {fractional_spoken}"

        if spoken != raw_decimal:
            stats.record_fix(raw_decimal, spoken)
        return spoken

    return _DECIMAL_NUMBER_RE.sub(replace_match, text)


def _normalize_number_suffixes_to_spoken_uzbek(
    text: str, stats: Optional[MisspellingStats] = None
) -> str:
    """Expand numeric part in number+suffix tokens while keeping suffix form."""
    if not text:
        return text
    if stats is None:
        stats = _misspelling_stats

    def replace_match(match: re.Match) -> str:
        raw_token = match.group(0)
        number_part = match.group(1)
        separator = match.group(2)
        suffix = match.group(3)

        try:
            numeric_value = int(number_part)
        except ValueError:
            return raw_token

        spoken_number = _number_to_spoken_uzbek(numeric_value)
        suffix_lower = suffix.lower()

        if separator and _is_ordinal_trigger(suffix_lower):
            ordinal = _number_to_ordinal_uzbek(numeric_value)
            replacement = f"{ordinal} {suffix}"
        elif separator:
            replacement = f"{spoken_number}{separator}{suffix}"
        elif suffix_lower in _ATTACHED_NUMERIC_SUFFIXES:
            replacement = f"{spoken_number}{suffix}"
        else:
            replacement = f"{spoken_number} {suffix}"

        if replacement != raw_token:
            stats.record_fix(raw_token, replacement)
        return replacement

    return _NUMBER_WITH_SUFFIX_RE.sub(replace_match, text)


def _normalize_phone_numbers_to_spoken_uzbek(
    text: str, stats: Optional[MisspellingStats] = None
) -> str:
    """Normalize grouped phone-number chunks to spoken Uzbek with commas."""
    if not text:
        return text
    if stats is None:
        stats = _misspelling_stats

    def replace_match(match: re.Match) -> str:
        raw_phone = match.group(0)
        groups = [match.group(1), match.group(2), match.group(3), match.group(4)]
        spoken_groups = [_number_to_spoken_uzbek(int(group)) for group in groups]
        replacement = ", ".join(spoken_groups)
        if replacement != raw_phone:
            stats.record_fix(raw_phone, replacement)
        return replacement

    return _UZBEK_PHONE_GROUP_RE.sub(replace_match, text)


def _normalize_numbers_to_spoken_uzbek(
    text: str, stats: Optional[MisspellingStats] = None
) -> str:
    """Replace integer tokens with spoken Uzbek and record normalization fixes."""
    if not text:
        return text
    if stats is None:
        stats = _misspelling_stats

    def replace_match(match: re.Match) -> str:
        raw_number = match.group(0)
        try:
            numeric_value = int(raw_number)
        except ValueError:
            return raw_number
        spoken = _number_to_spoken_uzbek(numeric_value)
        if spoken != raw_number:
            stats.record_fix(raw_number, spoken)
        return spoken

    return _NUMBER_TOKEN_RE.sub(replace_match, text)


def _normalize_uzbek_dates_to_spoken(
    text: str, stats: Optional[MisspellingStats] = None
) -> str:
    """Convert day+month date expressions to spoken Uzbek ordinal forms."""
    if not text:
        return text
    if stats is None:
        stats = _misspelling_stats

    def replace_match(match: re.Match) -> str:
        raw_date = match.group(0)
        day = int(match.group(1))
        month = match.group(2).lower()
        ordinal_day = _UZBEK_DAY_ORDINALS.get(
            day, f"{_number_to_spoken_uzbek(day)}inchi"
        )
        spoken_date = f"{ordinal_day} {month}"
        if spoken_date != raw_date:
            stats.record_fix(raw_date, spoken_date)
        return spoken_date

    return _UZBEK_DAY_MONTH_RE.sub(replace_match, text)


def _normalize_uzbek_abbreviations(
    text: str, stats: Optional[MisspellingStats] = None
) -> str:
    """Expand selected Uzbek abbreviations and unify common variants."""
    if not text:
        return text
    if stats is None:
        stats = _misspelling_stats

    def replace_year(match: re.Match) -> str:
        raw = match.group(0)
        replacement = f"{match.group(1)} yil"
        if replacement != raw:
            stats.record_fix(raw, replacement)
        return replacement

    normalized = _UZBEK_YEAR_ABBREV_RE.sub(replace_year, text)

    def replace_number_kg(match: re.Match) -> str:
        raw = match.group(0)
        number = match.group(1)
        unit = _preserve_word_case(match.group(2), "kilogram")
        replacement = f"{number} {unit}"
        if replacement != raw:
            stats.record_fix(raw, replacement)
        return replacement

    normalized = _UZBEK_NUMBER_KG_RE.sub(replace_number_kg, normalized)

    def replace_number_sum(match: re.Match) -> str:
        raw = match.group(0)
        number = match.group(1)
        currency = _preserve_word_case(match.group(2), "so'm")
        replacement = f"{number} {currency}"
        if replacement != raw:
            stats.record_fix(raw, replacement)
        return replacement

    normalized = _UZBEK_NUMBER_SUM_RE.sub(replace_number_sum, normalized)

    def replace_standalone_kg(match: re.Match) -> str:
        raw = match.group(0)
        replacement = _preserve_word_case(raw, "kilogram")
        if replacement != raw:
            stats.record_fix(raw, replacement)
        return replacement

    normalized = _UZBEK_STANDALONE_KG_RE.sub(replace_standalone_kg, normalized)

    def replace_standalone_sum(match: re.Match) -> str:
        raw = match.group(0)
        replacement = _preserve_word_case(raw, "so'm")
        if replacement != raw:
            stats.record_fix(raw, replacement)
        return replacement

    return _UZBEK_STANDALONE_SUM_RE.sub(replace_standalone_sum, normalized)


@dataclass
class NormalizedWordStats:
    """Track all normalized words for post-run inspection."""

    total_texts: int = 0
    total_words: int = 0
    words: Counter = field(default_factory=Counter)
    texts_by_dataset: Counter = field(default_factory=Counter)

    def record_text(self, normalized_text: str, dataset_label: Optional[str]) -> None:
        """Record every normalized word from a text."""
        self.total_texts += 1
        words = _WORD_TOKENIZE_RE.findall(normalized_text.lower())
        self.total_words += len(words)
        self.words.update(words)
        if dataset_label:
            self.texts_by_dataset[dataset_label] += 1

    def report(self, top_k: int = 100) -> str:
        """Generate a readable summary of normalized-word statistics."""
        if self.total_texts == 0:
            return "No normalized texts recorded."
        lines = [
            f"Total normalized texts: {self.total_texts}",
            f"Total normalized words: {self.total_words}",
            f"Unique normalized words: {len(self.words)}",
            f"Top normalized words:",
        ]
        for word, count in self.words.most_common(top_k):
            lines.append(f"  {word}: {count}")
        return "\n".join(lines)

    def reset(self) -> None:
        """Reset tracked normalized-word statistics."""
        self.total_texts = 0
        self.total_words = 0
        self.words.clear()
        self.texts_by_dataset.clear()


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


# Global stats trackers
_normalized_word_stats = NormalizedWordStats()
_misspelling_stats = MisspellingStats()


def get_normalized_word_stats() -> NormalizedWordStats:
    """Get the global normalized-word statistics."""
    return _normalized_word_stats


def reset_normalized_word_stats() -> None:
    """Reset the global normalized-word statistics."""
    _normalized_word_stats.reset()


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
        return _preserve_word_case(word, replacement)

    return _UZBEK_MISSPELLING_PATTERN.sub(replace_match, text)


def _candidate_hunspell_paths() -> List[Tuple[str, str]]:
    """Build candidate dictionary (dic, aff) path pairs for Uzbek Hunspell."""
    env_dic = os.environ.get("UZBEK_HUNSPELL_DIC")
    env_aff = os.environ.get("UZBEK_HUNSPELL_AFF")
    if env_dic and env_aff:
        return [(env_dic, env_aff)]

    dict_dirs: List[str] = []
    env_dict_dir = os.environ.get("HUNSPELL_DICT_DIR")
    if env_dict_dir:
        dict_dirs.append(env_dict_dir)
    dict_dirs.extend(_COMMON_HUNSPELL_DICT_DIRS)

    pairs: List[Tuple[str, str]] = []
    for dict_dir in dict_dirs:
        for basename in _UZBEK_HUNSPELL_BASENAMES:
            dic_path = os.path.join(dict_dir, f"{basename}.dic")
            aff_path = os.path.join(dict_dir, f"{basename}.aff")
            if os.path.exists(dic_path) and os.path.exists(aff_path):
                pairs.append((dic_path, aff_path))
    return pairs


class HunspellUzbekCorrector:
    """Hunspell-based Uzbek spelling correction with strict edits."""

    def __init__(self, max_edit_distance: int = 1) -> None:
        self._max_edit_distance = max_edit_distance
        self._hunspell: Optional[Any] = None
        self._suggestion_cache: Dict[str, Optional[str]] = {}
        self._initialize_hunspell()

    @property
    def is_available(self) -> bool:
        return self._hunspell is not None

    def _initialize_hunspell(self) -> None:
        for dic_path, aff_path in _candidate_hunspell_paths():
            try:
                self._hunspell = HunSpell(dic_path, aff_path)
                _LOGGER.info(
                    "Loaded Uzbek Hunspell dictionary: dic=%s aff=%s",
                    dic_path,
                    aff_path,
                )
                return
            except Exception as exc:
                _LOGGER.warning(
                    "Failed to initialize Hunspell with dic=%s aff=%s: %s",
                    dic_path,
                    aff_path,
                    exc,
                )

        _LOGGER.warning(
            "Hunspell is installed but no Uzbek dictionary was found. "
            "Set UZBEK_HUNSPELL_DIC and UZBEK_HUNSPELL_AFF to enable correction."
        )

    def _pick_single_edit_suggestion(self, word: str) -> Optional[str]:
        lower_word = word.lower()
        if lower_word in self._suggestion_cache:
            return self._suggestion_cache[lower_word]
        if self._hunspell is None:
            self._suggestion_cache[lower_word] = None
            return None

        try:
            if self._hunspell.spell(lower_word):
                self._suggestion_cache[lower_word] = None
                return None
            suggestions = self._hunspell.suggest(lower_word)
        except Exception as exc:
            _LOGGER.warning("Hunspell suggestion lookup failed for '%s': %s", word, exc)
            self._suggestion_cache[lower_word] = None
            return None

        for suggestion in suggestions:
            normalized_suggestion = suggestion.strip().lower()
            if not normalized_suggestion or normalized_suggestion == lower_word:
                continue
            if not _WORD_TOKENIZE_RE.fullmatch(normalized_suggestion):
                continue
            if (
                _edit_distance(lower_word, normalized_suggestion)
                == self._max_edit_distance
            ):
                self._suggestion_cache[lower_word] = normalized_suggestion
                return normalized_suggestion

        self._suggestion_cache[lower_word] = None
        return None

    def correct_text(self, text: str, stats: Optional[MisspellingStats] = None) -> str:
        """Correct words only when Hunspell suggests a 1-edit-distance replacement."""
        if not text or self._hunspell is None:
            return text

        def replace_match(match: re.Match) -> str:
            word = match.group(0)
            suggestion = self._pick_single_edit_suggestion(word)
            if suggestion is None:
                return word
            corrected = _preserve_word_case(word, suggestion)
            if stats is not None:
                stats.record_fix(word, suggestion)
            return corrected

        return _WORD_TOKENIZE_RE.sub(replace_match, text)

    def reset(self) -> None:
        """Reset cache while keeping loaded dictionary."""
        self._suggestion_cache.clear()


_global_hunspell_uzbek_corrector: Optional[HunspellUzbekCorrector] = None


def get_hunspell_uzbek_corrector(max_edit_distance: int = 1) -> HunspellUzbekCorrector:
    """Get or create the global Hunspell Uzbek corrector."""
    global _global_hunspell_uzbek_corrector
    if _global_hunspell_uzbek_corrector is None:
        _global_hunspell_uzbek_corrector = HunspellUzbekCorrector(
            max_edit_distance=max_edit_distance
        )
    return _global_hunspell_uzbek_corrector


def reset_hunspell_uzbek_corrector() -> None:
    """Reset the global Hunspell Uzbek corrector."""
    global _global_hunspell_uzbek_corrector
    if _global_hunspell_uzbek_corrector is not None:
        _global_hunspell_uzbek_corrector.reset()
    _global_hunspell_uzbek_corrector = None


def normalize_text(
    text: Any,
    stats: Optional[MisspellingStats] = None,
    dataset_label: Optional[str] = None,
    decimal_mode: Optional[str] = None,
) -> str:
    """Normalize text to match training cleanup in utils/reader.py."""
    if text is None:
        return ""
    try:
        normalized = str(text)
    except Exception:
        return ""
    resolved_decimal_mode = _resolve_decimal_mode(decimal_mode)

    normalized = _transliterate_uzbek_cyrillic(normalized)
    normalized = normalized.translate(_APOSTROPHE_TRANSLATION)
    normalized = _normalize_uzbek_abbreviations(normalized, stats=stats)
    before_fix = normalized
    normalized = _fix_uzbek_misspellings(normalized, stats)
    hunspell_corrector = get_hunspell_uzbek_corrector(max_edit_distance=1)
    normalized = hunspell_corrector.correct_text(normalized, stats=stats)
    if stats is not None and dataset_label and normalized != before_fix:
        stats.record_text_fix(before_fix, dataset_label)
    normalized = _ALLOWED_TEXT_RE.sub("", normalized)
    normalized = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", normalized)
    normalized = _normalize_phone_numbers_to_spoken_uzbek(normalized, stats=stats)
    normalized = _SPACED_NUMBER_RE.sub(r"\1", normalized)
    normalized = _COMMA_THOUSANDS_RE.sub(
        lambda m: m.group(1) + m.group(2).replace(",", ""), normalized
    )
    normalized = _normalize_uzbek_dates_to_spoken(normalized, stats=stats)
    normalized = _normalize_decimals_to_spoken_uzbek(
        normalized, stats=stats, decimal_mode=resolved_decimal_mode
    )
    normalized = _normalize_number_suffixes_to_spoken_uzbek(normalized, stats=stats)
    normalized = _normalize_numbers_to_spoken_uzbek(normalized, stats=stats)
    normalized = _MULTISPACE_RE.sub(" ", normalized).strip()
    if normalized.startswith("-"):
        normalized = normalized[1:].lstrip()
    if normalized.endswith("-"):
        normalized = normalized[:-1].rstrip()
    _normalized_word_stats.record_text(normalized, dataset_label)
    words = _WORD_TOKENIZE_RE.findall(normalized.lower())
    if words:
        _LOGGER.debug(
            "Normalized words dataset=%s words=%s",
            dataset_label or "unknown",
            " ".join(words),
        )
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
    fixes_by_text: Counter = field(default_factory=Counter)

    def record_detection(self, typo: str, correction: str) -> None:
        """Record a detected typo."""
        self.total_typos_detected += 1
        self.typos_by_word[f"{typo} -> {correction}"] += 1

    def record_correction(self) -> None:
        """Record an applied correction."""
        self.total_corrections_applied += 1

    def record_text_fix(self, original: str, dataset_label: str) -> None:
        """Record full text that was corrected along with dataset label."""
        self.fixes_by_text[(dataset_label, original)] += 1

    def reset(self) -> None:
        """Reset all statistics."""
        self.total_typos_detected = 0
        self.total_corrections_applied = 0
        self.typos_by_word.clear()
        self.fixes_by_text.clear()

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
        if self.fixes_by_text:
            lines.append("Fixes by text (dataset -> original):")
            for (dataset_label, original), count in self.fixes_by_text.most_common():
                lines.append(f"  [{dataset_label}] {original}: {count}")
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
        max_edit_distance: int = 3,
        min_correction_frequency: int = 450,
        min_typo_length: int = 3,
        confidence_threshold: float = 0.6,
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

    def analyze(self, verbose: bool = True) -> List[TypoCandidate]:
        """Analyze the frequency data to detect potential typos.

        Args:
            verbose: Whether to print progress information

        Returns:
            List of TypoCandidate objects representing detected typos
        """
        candidates = []
        word_counts = self._collector.word_counts

        if verbose:
            print(f"  Building high-frequency word index...")
            sys.stdout.flush()

        # Build set of high-frequency words for faster lookup
        self._high_freq_words = {
            word
            for word, count in word_counts.items()
            if count >= self._min_correction_frequency
        }

        if verbose:
            print(f"  Found {len(self._high_freq_words)} high-frequency words")
            sys.stdout.flush()

        # Also index by apostrophe-stripped version
        stripped_to_original: Dict[str, List[str]] = {}
        for word in self._high_freq_words:
            stripped = _remove_apostrophes(word)
            if stripped not in stripped_to_original:
                stripped_to_original[stripped] = []
            stripped_to_original[stripped].append(word)

        # Build length-based buckets for high-frequency words to speed up edit distance search
        # Only compare words whose lengths differ by at most max_edit_distance
        high_freq_by_length: Dict[int, List[str]] = defaultdict(list)
        for word in self._high_freq_words:
            high_freq_by_length[len(word)].append(word)

        if verbose:
            print(f"  Built length-based index for edit distance search")
            sys.stdout.flush()

        # Filter words that need checking (not high-frequency, meets length requirement)
        words_to_check = [
            (word, count)
            for word, count in word_counts.items()
            if len(word) >= self._min_typo_length and word not in self._high_freq_words
        ]
        total_words = len(words_to_check)

        if verbose:
            print(f"  Checking {total_words} candidate words for typos...")
            sys.stdout.flush()

        # Check each word for potential typos
        for idx, (word, count) in enumerate(words_to_check):
            # Progress reporting every 10000 words
            if verbose and idx > 0 and idx % 10000 == 0:
                print(
                    f"    Progress: {idx}/{total_words} words checked ({100 * idx // total_words}%), found {len(candidates)} typos so far"
                )
                sys.stdout.flush()

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

            # Check 2: Edit distance to high-frequency words (using length buckets)
            if word not in self._typo_corrections:
                best_match = self._find_best_edit_distance_match(
                    word, count, high_freq_by_length
                )
                if best_match:
                    candidates.append(best_match)
                    self._typo_corrections[word] = best_match.correction

        if verbose:
            print(f"  Analysis complete: found {len(candidates)} potential typos")
            sys.stdout.flush()

        self._analyzed = True
        return candidates

    def _find_best_edit_distance_match(
        self,
        word: str,
        word_freq: int,
        high_freq_by_length: Dict[int, List[str]],
    ) -> Optional[TypoCandidate]:
        """Find the best high-frequency word within edit distance.

        Uses length-based bucketing to only compare words of similar length,
        dramatically reducing the number of edit distance calculations.
        """
        best_candidate = None
        best_confidence = 0.0
        word_len = len(word)

        # Only check words within max_edit_distance of our word's length
        for length_offset in range(
            -self._max_edit_distance, self._max_edit_distance + 1
        ):
            target_len = word_len + length_offset
            if target_len < self._min_typo_length:
                continue

            for high_freq_word in high_freq_by_length.get(target_len, []):
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
    "NormalizedWordStats",
    "get_normalized_word_stats",
    "reset_normalized_word_stats",
    "MisspellingStats",
    "get_misspelling_stats",
    "reset_misspelling_stats",
    "HunspellUzbekCorrector",
    "get_hunspell_uzbek_corrector",
    "reset_hunspell_uzbek_corrector",
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
