# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# USAGE: python process_asr_text_tokenizer.py --manifest=<path to train manifest files, seperated by commas> \
#         --data_root="<output directory>" \
#         --vocab_size=<number of tokens in vocabulary> \
#         --tokenizer=<"spe" or "wpe"> \
#         --log
# where <manifest> can be: train_clean_100, train_clean_360, train_other_500
# You can also put more than one data_set comma-separated:
# --manifest="train_clean_100,train_clean_360,train_other_500"
# or
#       python process_asr_text_tokenizer.py --data_file=<path to train text file> \
#         --data_root="<output directory>" \
#         --vocab_size=<number of tokens in vocabulary> \
#         --tokenizer=<"bpe" or "wpe"> \
#         --log
# where <manifest> can be: train_clean_100, train_clean_360, train_other_500
# You can also put more than one data_set comma-separated:
# --manifest="train_clean_100,train_clean_360,train_other_500"
#
# Args:
#   --manifest or --data_file: If your text data lies inside of an ASR manifest file,
#       then use the --manifest path. If instead the text data is inside a file with separate lines
#       corresponding to different text lines, then use --data_file.
#       In either case, you can add commas to concatenate different manifests or different data files.
#
#   --data_root: The output directory (whose subdirectories will be created if not present) where
#       the tokenizers will be placed.
#
#   --vocab_size: The size of the tokenizer vocabulary. Larger vocabularies can accommodate almost entire,
#       words but the decoder size of any model will grow proportionally.
#
#   --tokenizer: Can be either spe or wpe . spe refers to the Google sentencepiece library tokenizer.
#       wpe refers to the HuggingFace BERT Word Piece tokenizer.
#
#   --no_lower_case: When this flag is passed, it will force the tokenizer to create seperate tokens for
#       upper and lower case characters. By default, the script will turn all the text to lower case
#       before tokenization (and if upper case characters are passed during training/inference, the
#       tokenizer will emit a token equivalent to Out-Of-Vocabulary). Used primarily for the
#       English language.
#
#    --spe_type: The sentencepiece library has a few implementations of the tokenization technique, and
#       spe_type refers to these implementations. Currently supported types are unigram, bpe, char, word.
#       Defaults to bpe.
#
#   --spe_character_coverage: The sentencepiece library considers how much of the original vocabulary it
#       should cover in its "base set" of tokens (akin to the lower and upper case characters of the
#       English language). For almost all languages with small base token sets (<1000 tokens), this
#       should be kept at its default of 1.0. For languages with larger vocabularies (say Japanese,
#       Mandarin, Korean etc), the suggested value is 0.9995.
#
#   --spe_user_defined_symbols: The sentencepiece library allows you to add your own tokens to the base set.
#      This flag allows you to pass a space separated list of tokens that you want to add to the base set.
#      These tokens remain in the decoded text and are encoded automatically when present in the input text.
#
#   --spe_control_symbols: The sentencepiece library allows you to add your own tokens to the base set.
#      This flag allows you to pass a space separated list of tokens that you want to add to the base set.
#      These tokens get removed at decode time and are not encoded from the text - can only be added to the
#      input programatically.
#
#   --spe_byte_fallback: If <unk>, fallback to a byte sequence of the characters.
#
#   --spe_split_digits: If true, digits are split into individual tokens.
#
#   --spe_sample_size: If the dataset is too large, consider using a sampled dataset indicated by a
#       positive integer. By default, any negative value (default = -1) will use the entire dataset.
#
#   --spe_train_extremely_large_corpus: When training a sentencepiece tokenizer on very large amounts of text,
#       sometimes the tokenizer will run out of memory or wont be able to process so much data on RAM.
#       At some point you might receive the following error - "Input corpus too large, try with
#       train_extremely_large_corpus=true". If your machine has large amounts of RAM, it might still be possible
#       to build the tokenizer using the above flag. Will silently fail if it runs out of RAM.
#
#   --spe_max_sentencepiece_length: Limits the maximum length that any any SentencePiece subword can be.
#       Using this will change the subword tokens generated.
#
#   --spe_pad: Adds <pad> as special token.
#
#   --spe_bos: Adds <s> as Begining-of-Sentence special token.
#
#   --spe_eos: Adds </s> as End-of-Sentence special token.
#
#   --log: Whether the script should display log messages


import argparse
import json
import logging
import os
import re
import time
from collections import Counter
from typing import Dict, List, Optional

import tokenizers
from nemo.collections.common.tokenizers.sentencepiece_tokenizer import create_spt_model
from nemo.utils.data_utils import DataStoreObject

from utils import normalize_text

try:
    import orjson
except ImportError:
    orjson = None

parser = argparse.ArgumentParser(description="Create tokenizer")
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument(
    "--manifest", default=None, type=str, help="Comma separated list of manifest files"
)
group.add_argument(
    "--data_file", default=None, help="data file from which to create tokenizer model"
)
parser.add_argument(
    "--data_root", required=True, default=None, type=str, help="Output directory"
)
parser.add_argument("--vocab_size", default=1024, type=int, help="Vocabulary size")
parser.add_argument(
    "--tokenizer",
    default="wpe",
    choices=["spe", "wpe"],
    help="Type of tokenization to perform",
)
parser.add_argument(
    "--spe_type",
    default="bpe",
    choices=["bpe", "unigram", "char", "word"],
    help="Type of the SentencePiece model. Can be `bpe`, `unigram`, `char` or `word`."
    "Used only if --tokenizer == `spe`",
)
parser.add_argument(
    "--spe_character_coverage",
    type=float,
    default=1.0,
    help="Character coverage percentage for SentencePiece tokenization. For languages "
    "with large vocabulary, should be close to 0.9995, otherwise kept as 1.0",
)
parser.add_argument(
    "--spe_bos", action="store_true", help="Add <s> token to SentencePiece Tokenizer."
)
parser.add_argument(
    "--spe_eos", action="store_true", help="Add </s> token to SentencePiece Tokenizer."
)
parser.add_argument(
    "--spe_pad", action="store_true", help="Add <pad> token to SentencePiece Tokenizer."
)
parser.add_argument(
    "--spe_user_defined_symbols",
    default=None,
    type=str,
    nargs="+",
    help="User defined symbols for SentencePiece",
)
parser.add_argument(
    "--spe_control_symbols",
    default=None,
    type=str,
    nargs="+",
    help="Control symbols for SentencePiece",
)
parser.add_argument(
    "--spe_split_digits", action="store_true", help="Split digits into separate tokens."
)
parser.add_argument(
    "--spe_remove_extra_whitespaces",
    action="store_true",
    help="Remove leading, trailing, and duplicate internal whitespace.",
)

parser.add_argument(
    "--spe_sample_size",
    type=int,
    default=-1,
    help="Samples the dataset by `sample_size` if positive integer, otherwise uses whole dataset",
)
parser.add_argument("--spe_train_extremely_large_corpus", action="store_true", help="")
parser.add_argument(
    "--spe_max_sentencepiece_length",
    type=int,
    default=-1,
    help="Limit the maximum number of tokens in each SentencePiece subword. "
    "Must be a positive integer > 0. By default places no limit on subword length.",
)
parser.add_argument(
    "--spe_no_split_by_unicode_script",
    dest="spe_split_by_unicode_script",
    action="store_false",
    help="Don't use Unicode script to split sentence pieces.",
)
parser.add_argument(
    "--spe_byte_fallback",
    dest="spe_byte_fallback",
    action="store_true",
    help="If <unk>, fallback to a byte sequence of the characters.",
)
parser.add_argument("--no_lower_case", dest="lower_case", action="store_false")
parser.add_argument(
    "--normalize_text_corpus",
    action="store_true",
    help="Normalize manifest text before tokenizer training to reduce noisy variants.",
)
parser.add_argument(
    "--max_line_occurrence",
    type=int,
    default=3,
    help="Maximum times the same cleaned line can appear in corpus (0 = no limit).",
)
parser.add_argument(
    "--min_chars",
    type=int,
    default=1,
    help="Drop cleaned lines shorter than this length when building text corpus.",
)
parser.add_argument(
    "--max_chars",
    type=int,
    default=0,
    help="Drop cleaned lines longer than this length when building text corpus (0 = no limit).",
)
parser.add_argument(
    "--force_rebuild_text_corpus",
    action="store_true",
    help="Rebuild <data_root>/text_corpus/document.txt even if it already exists.",
)
parser.add_argument(
    "--progress_log_interval",
    type=int,
    default=200000,
    help="Log progress every N processed lines while building text corpus (0 disables periodic progress logs).",
)
parser.add_argument(
    "--write_buffer_lines",
    type=int,
    default=10000,
    help="Number of cleaned lines to buffer before writing to disk.",
)
parser.add_argument("--log", action="store_true")
parser.set_defaults(log=False, lower_case=True, spe_train_extremely_large_corpus=False)
args = parser.parse_args()

WHITESPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.])")
REPEATED_PUNCT_RE = re.compile(r"[,.]{2,}")


def __json_loads(raw: str):
    if orjson is not None:
        return orjson.loads(raw)
    return json.loads(raw)


def __keep_last_punct(match: re.Match) -> str:
    return match.group(0)[-1]


def __clean_line(text: str, normalize: bool) -> str:
    cleaned = normalize_text(text) if normalize else str(text).strip()
    if not cleaned:
        return ""
    if (" " in cleaned) and ("," in cleaned or "." in cleaned):
        cleaned = WHITESPACE_BEFORE_PUNCT_RE.sub(r"\1", cleaned)
    if ("," in cleaned or "." in cleaned) and REPEATED_PUNCT_RE.search(cleaned):
        cleaned = REPEATED_PUNCT_RE.sub(__keep_last_punct, cleaned)
    return cleaned.strip()


def __build_document_from_manifests(
    data_root: str,
    manifests: str,
    normalize_text_corpus: bool,
    max_line_occurrence: int,
    min_chars: int,
    max_chars: int,
    force_rebuild_text_corpus: bool,
    progress_log_interval: int,
    write_buffer_lines: int,
):
    if "," in manifests:
        manifests = manifests.split(",")
    else:
        manifests = [manifests]

    document_dir = os.path.join(data_root, "text_corpus")
    os.makedirs(document_dir, exist_ok=True)

    document_path = os.path.join(document_dir, "document.txt")

    if os.path.exists(document_path):
        if not force_rebuild_text_corpus:
            logging.info("Corpus already exists at path : %s", document_path)
            return document_path
        logging.info("Rebuilding corpus at path : %s", document_path)

    min_len = max(min_chars, 0)
    max_len = max_chars if max_chars > 0 else 0
    dedupe_enabled = max_line_occurrence > 0
    line_counts: Optional[Counter] = Counter() if dedupe_enabled else None
    write_buffer_lines = max(1, write_buffer_lines)
    stats: Dict[str, int] = {
        "total": 0,
        "written": 0,
        "empty": 0,
        "too_short": 0,
        "too_long": 0,
        "overrepresented": 0,
    }
    buffer: List[str] = []
    total_start_time = time.perf_counter()

    with open(
        document_path, "w", encoding="utf-8", buffering=1024 * 1024
    ) as out_writer:
        for manifest in manifests:
            source_start = time.perf_counter()
            source_seen = 0
            source_written = 0
            with open(
                DataStoreObject(manifest).get(), "r", encoding="utf-8"
            ) as in_reader:
                for line in in_reader:
                    stats["total"] += 1
                    source_seen += 1
                    item = __json_loads(line)
                    text = item.get("text", item.get("normalized_text", ""))
                    cleaned = __clean_line(text, normalize_text_corpus)

                    if not cleaned:
                        stats["empty"] += 1
                        continue

                    if min_len and len(cleaned) < min_len:
                        stats["too_short"] += 1
                        continue

                    if max_len and len(cleaned) > max_len:
                        stats["too_long"] += 1
                        continue

                    if dedupe_enabled:
                        current_count = line_counts[cleaned]
                        if current_count >= max_line_occurrence:
                            stats["overrepresented"] += 1
                            continue
                        line_counts[cleaned] = current_count + 1

                    buffer.append(cleaned + "\n")
                    if len(buffer) >= write_buffer_lines:
                        out_writer.writelines(buffer)
                        buffer.clear()
                    stats["written"] += 1
                    source_written += 1

                    if (
                        progress_log_interval > 0
                        and stats["total"] % progress_log_interval == 0
                    ):
                        elapsed = max(time.perf_counter() - total_start_time, 1e-6)
                        logging.info(
                            "Progress: processed=%d written=%d dropped=%d rate=%.1f lines/s",
                            stats["total"],
                            stats["written"],
                            stats["total"] - stats["written"],
                            stats["total"] / elapsed,
                        )

            if buffer:
                out_writer.writelines(buffer)
                buffer.clear()

            logging.info(
                "Finished extracting manifest: %s (seen=%d, written=%d, elapsed=%.2fs)",
                manifest,
                source_seen,
                source_written,
                time.perf_counter() - source_start,
            )

        total_elapsed = max(time.perf_counter() - total_start_time, 1e-6)
        unique_kept = len(line_counts) if line_counts is not None else 0

        logging.info(
            "Finished extracting manifests. Total lines: %d, written: %d, dropped empty: %d, "
            "dropped too short: %d, dropped too long: %d, dropped duplicate cap: %d, unique kept: %d, "
            "elapsed: %.2fs, throughput: %.1f lines/s",
            stats["total"],
            stats["written"],
            stats["empty"],
            stats["too_short"],
            stats["too_long"],
            stats["overrepresented"],
            unique_kept,
            total_elapsed,
            stats["total"] / total_elapsed,
        )
    return document_path


def __build_document_from_data_files(
    data_root: str,
    data_files: str,
    normalize_text_corpus: bool,
    max_line_occurrence: int,
    min_chars: int,
    max_chars: int,
    force_rebuild_text_corpus: bool,
    progress_log_interval: int,
    write_buffer_lines: int,
):
    if "," in data_files:
        data_files = data_files.split(",")
    else:
        data_files = [data_files]

    document_dir = os.path.join(data_root, "text_corpus")
    os.makedirs(document_dir, exist_ok=True)

    document_path = os.path.join(document_dir, "document.txt")

    if os.path.exists(document_path):
        if not force_rebuild_text_corpus:
            logging.info("Corpus already exists at path : %s", document_path)
            return document_path
        logging.info("Rebuilding corpus at path : %s", document_path)

    min_len = max(min_chars, 0)
    max_len = max_chars if max_chars > 0 else 0
    dedupe_enabled = max_line_occurrence > 0
    line_counts: Optional[Counter] = Counter() if dedupe_enabled else None
    write_buffer_lines = max(1, write_buffer_lines)
    stats: Dict[str, int] = {
        "total": 0,
        "written": 0,
        "empty": 0,
        "too_short": 0,
        "too_long": 0,
        "overrepresented": 0,
    }
    buffer: List[str] = []
    total_start_time = time.perf_counter()

    with open(
        document_path, "w", encoding="utf-8", buffering=1024 * 1024
    ) as out_writer:
        for data_file in data_files:
            source_start = time.perf_counter()
            source_seen = 0
            source_written = 0
            with open(
                DataStoreObject(data_file).get(), "r", encoding="utf-8"
            ) as in_reader:
                for line in in_reader:
                    stats["total"] += 1
                    source_seen += 1
                    cleaned = __clean_line(line.rstrip("\n"), normalize_text_corpus)

                    if not cleaned:
                        stats["empty"] += 1
                        continue

                    if min_len and len(cleaned) < min_len:
                        stats["too_short"] += 1
                        continue

                    if max_len and len(cleaned) > max_len:
                        stats["too_long"] += 1
                        continue

                    if dedupe_enabled:
                        current_count = line_counts[cleaned]
                        if current_count >= max_line_occurrence:
                            stats["overrepresented"] += 1
                            continue
                        line_counts[cleaned] = current_count + 1

                    buffer.append(cleaned + "\n")
                    if len(buffer) >= write_buffer_lines:
                        out_writer.writelines(buffer)
                        buffer.clear()
                    stats["written"] += 1
                    source_written += 1

                    if (
                        progress_log_interval > 0
                        and stats["total"] % progress_log_interval == 0
                    ):
                        elapsed = max(time.perf_counter() - total_start_time, 1e-6)
                        logging.info(
                            "Progress: processed=%d written=%d dropped=%d rate=%.1f lines/s",
                            stats["total"],
                            stats["written"],
                            stats["total"] - stats["written"],
                            stats["total"] / elapsed,
                        )

            if buffer:
                out_writer.writelines(buffer)
                buffer.clear()

            logging.info(
                "Finished extracting data file: %s (seen=%d, written=%d, elapsed=%.2fs)",
                data_file,
                source_seen,
                source_written,
                time.perf_counter() - source_start,
            )

        total_elapsed = max(time.perf_counter() - total_start_time, 1e-6)
        unique_kept = len(line_counts) if line_counts is not None else 0

        logging.info(
            "Finished extracting text files. Total lines: %d, written: %d, dropped empty: %d, "
            "dropped too short: %d, dropped too long: %d, dropped duplicate cap: %d, unique kept: %d, "
            "elapsed: %.2fs, throughput: %.1f lines/s",
            stats["total"],
            stats["written"],
            stats["empty"],
            stats["too_short"],
            stats["too_long"],
            stats["overrepresented"],
            unique_kept,
            total_elapsed,
            stats["total"] / total_elapsed,
        )
    return document_path


def __process_data(
    text_path: str,
    dst_folder: str,
    vocab_size: int,
    tokenizer_type: str,
    spe_type: str,
    spe_character_coverage: float,
    spe_train_extremely_large_corpus: bool,
    spe_sample_size: int,
    spe_max_sentencepiece_length: int,
    spe_split_by_unicode_script: bool,
    spe_bos: bool,
    spe_eos: bool,
    spe_pad: bool,
    spe_control_symbols: Optional[List[str]],
    spe_user_defined_symbols: Optional[List[str]],
    spe_byte_fallback: bool,
    spe_split_digits: bool,
    spe_remove_extra_whitespaces: bool,
    lower_case: bool,
):
    """
    Converts flac to wav and build manifests's json
    Args:
        text_path: source with text lines
        dst_folder: where wav files will be stored
        vocab_size: vocabular size used in encoding the text
        tokenizer_type: type of tokenization to perform - wpe or spe
        spe_type: type of tokenization model used for spe.
        spe_character_coverage: float value between 0 and 1 (as a percentage). For languages with a vast charset,
            can be < 1.0, but for all other languages, it should be set as 1.0
        spe_sample_size: int, default of -1. If positive integer is used, samples the dataset
            by given sample size.
        spe_train_extremely_large_corpus: bool. If dataset is too large, and user has sufficient RAM,
            this flag can be set to try to trained the tokenizer. Will silently fail if it runs out of RAM.
        spe_max_sentencepiece_length: Limits the maximum length of the SentencePiece subword that can be constructed.
            By default, no limit is placed.
        spe_bos: Bool flag, whether to add <s> to SentencePiece tokenizer vocabulary.
        spe_eos: Bool flag, whether to add </s> to SentencePiece tokenizer vocabulary.
        spe_pad: Bool flag, whether to add <pad> to SentencePiece tokenizer vocabulary.
        spe_control_symbols: control symbols to add to tokenizer, as defined by sentencepiece.
            These tokens get removed at decode time and are not encoded from the text - can only be added to the input programatically.
        spe_user_defined_symbols: user symbols to add to tokenizer, as defined by sentencepiece.
            These tokens remain in the decoded text and are encoded automatically when present in the input text.
        spe_byte_fallback: If <unk>, fallback to a byte sequence of the character.
        spe_split_digits: If true, digits are split into individual tokens.
        spe_remove_extra_whitespaces: If true, removes leading, trailing, and duplicate internal whitespace.
        lower_case: whether to tokenize with lower case character set only (for english)

    Returns:
    """
    if tokenizer_type == "spe":
        # Prepare directory of tokenizer
        if spe_max_sentencepiece_length > 0:
            tokenizer_dir = os.path.join(
                dst_folder, "tokenizer_{}_{}_v{}_max_{}"
            ).format(tokenizer_type, spe_type, vocab_size, spe_max_sentencepiece_length)
        else:
            tokenizer_dir = os.path.join(dst_folder, "tokenizer_{}_{}_v{}").format(
                tokenizer_type, spe_type, vocab_size
            )

        if spe_pad:
            tokenizer_dir = f"{tokenizer_dir}_pad"
        if spe_bos:
            tokenizer_dir = f"{tokenizer_dir}_bos"
        if spe_eos:
            tokenizer_dir = f"{tokenizer_dir}_eos"

        if not os.path.exists(tokenizer_dir):
            os.makedirs(tokenizer_dir)

        if os.path.exists(os.path.join(tokenizer_dir, "tokenizer.model")):
            logging.warning("Model file already exists, overriding old model file !")
            os.remove(os.path.join(tokenizer_dir, "tokenizer.model"))

        # Build tokenizer
        tokenizer_path, vocab_path = create_spt_model(
            data_file=text_path,
            vocab_size=vocab_size,
            sample_size=spe_sample_size,
            do_lower_case=lower_case,
            output_dir=tokenizer_dir,
            tokenizer_type=spe_type,
            character_coverage=spe_character_coverage,
            train_extremely_large_corpus=spe_train_extremely_large_corpus,
            max_sentencepiece_length=spe_max_sentencepiece_length,
            split_by_unicode_script=spe_split_by_unicode_script,
            bos=spe_bos,
            eos=spe_eos,
            pad=spe_pad,
            control_symbols=spe_control_symbols,
            user_defined_symbols=spe_user_defined_symbols,
            byte_fallback=spe_byte_fallback,
            split_digits=spe_split_digits,
            remove_extra_whitespaces=spe_remove_extra_whitespaces,
        )

    else:
        tokenizer_dir = os.path.join(dst_folder, "tokenizer_{}_v{}").format(
            tokenizer_type, vocab_size
        )

        if not os.path.exists(tokenizer_dir):
            os.makedirs(tokenizer_dir)

        tokenizer = tokenizers.BertWordPieceTokenizer(lowercase=lower_case)

        tokenizer.train(text_path, vocab_size=vocab_size)
        tokenizer.save_model(tokenizer_dir)

    return tokenizer_dir


def main():
    start_time = time.perf_counter()
    data_root = args.data_root
    manifests = args.manifest
    data_file = args.data_file
    vocab_size = args.vocab_size
    tokenizer = args.tokenizer
    spe_type = args.spe_type
    spe_character_coverage = args.spe_character_coverage
    spe_sample_size = args.spe_sample_size
    spe_train_extremely_large_corpus = args.spe_train_extremely_large_corpus
    spe_max_sentencepiece_length = args.spe_max_sentencepiece_length
    spe_split_by_unicode_script = args.spe_split_by_unicode_script
    spe_bos, spe_eos, spe_pad = args.spe_bos, args.spe_eos, args.spe_pad
    spe_control_symbols = args.spe_control_symbols
    spe_user_defined_symbols = args.spe_user_defined_symbols
    spe_byte_fallback = args.spe_byte_fallback
    spe_split_digits = args.spe_split_digits
    spe_remove_extra_whitespaces = args.spe_remove_extra_whitespaces
    lower_case = args.lower_case
    normalize_text_corpus = args.normalize_text_corpus
    max_line_occurrence = args.max_line_occurrence
    min_chars = args.min_chars
    max_chars = args.max_chars
    force_rebuild_text_corpus = args.force_rebuild_text_corpus
    progress_log_interval = args.progress_log_interval
    write_buffer_lines = args.write_buffer_lines

    if not os.path.exists(data_root):
        os.makedirs(data_root)

    if args.log:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    logging.info(
        "Starting tokenizer preparation (tokenizer=%s, vocab=%d, normalize=%s, dedupe_cap=%d)",
        tokenizer,
        vocab_size,
        normalize_text_corpus,
        max_line_occurrence,
    )

    if manifests:
        text_corpus_path = __build_document_from_manifests(
            data_root=data_root,
            manifests=manifests,
            normalize_text_corpus=normalize_text_corpus,
            max_line_occurrence=max_line_occurrence,
            min_chars=min_chars,
            max_chars=max_chars,
            force_rebuild_text_corpus=force_rebuild_text_corpus,
            progress_log_interval=progress_log_interval,
            write_buffer_lines=write_buffer_lines,
        )
    else:
        text_corpus_path = __build_document_from_data_files(
            data_root=data_root,
            data_files=data_file,
            normalize_text_corpus=normalize_text_corpus,
            max_line_occurrence=max_line_occurrence,
            min_chars=min_chars,
            max_chars=max_chars,
            force_rebuild_text_corpus=force_rebuild_text_corpus,
            progress_log_interval=progress_log_interval,
            write_buffer_lines=write_buffer_lines,
        )
    tokenizer_path = __process_data(
        text_corpus_path,
        data_root,
        vocab_size,
        tokenizer,
        spe_type,
        lower_case=lower_case,
        spe_character_coverage=spe_character_coverage,
        spe_sample_size=spe_sample_size,
        spe_train_extremely_large_corpus=spe_train_extremely_large_corpus,
        spe_max_sentencepiece_length=spe_max_sentencepiece_length,
        spe_split_by_unicode_script=spe_split_by_unicode_script,
        spe_bos=spe_bos,
        spe_eos=spe_eos,
        spe_pad=spe_pad,
        spe_control_symbols=spe_control_symbols,
        spe_user_defined_symbols=spe_user_defined_symbols,
        spe_byte_fallback=spe_byte_fallback,
        spe_split_digits=spe_split_digits,
        spe_remove_extra_whitespaces=spe_remove_extra_whitespaces,
    )

    print("Serialized tokenizer at location :", tokenizer_path)
    logging.info("Done! Total elapsed: %.2fs", time.perf_counter() - start_time)


if __name__ == "__main__":
    main()
