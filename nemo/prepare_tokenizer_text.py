#!/usr/bin/env python3
"""Clean text lines for tokenizer training using the same normalization as dataset prep."""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _load_cleanup_utils() -> Any:
    utils_path = Path(__file__).with_name("utils.py")
    spec = importlib.util.spec_from_file_location("nemo_cleanup_utils", utils_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load cleanup utils from {utils_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_cleanup_utils = _load_cleanup_utils()
normalize_text = _cleanup_utils.normalize_text


def _expand_input_paths(raw_inputs: Sequence[str]) -> List[Path]:
    inputs: List[Path] = []
    for entry in raw_inputs:
        for part in entry.split(","):
            cleaned = part.strip()
            if cleaned:
                inputs.append(Path(cleaned).expanduser())
    return inputs


def _resolve_output_path(
    output: Optional[str],
    data_root: Optional[str],
    input_paths: Sequence[Path],
) -> Path:
    if output:
        return Path(output).expanduser()
    if data_root:
        return Path(data_root).expanduser() / "text_corpus" / "document.txt"
    if len(input_paths) == 1:
        return input_paths[0].with_suffix(".cleaned.txt")
    return Path("cleaned_corpus.txt")


def _iter_lines(paths: Sequence[Path]) -> Iterable[str]:
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                yield line.rstrip("\n")


def _filter_and_clean_lines(
    lines: Iterable[str],
    min_chars: int,
    max_chars: int,
) -> Tuple[Iterable[str], Dict[str, int]]:
    counts: Dict[str, int] = {
        "total": 0,
        "written": 0,
        "empty": 0,
        "too_short": 0,
        "too_long": 0,
    }
    min_len = max(min_chars, 0)
    max_len = max_chars if max_chars > 0 else 0

    def generator() -> Iterable[str]:
        for line in lines:
            counts["total"] += 1
            cleaned = normalize_text(line)
            cleaned = re.sub(r"\s+([,.])", r"\1", cleaned)
            cleaned = re.sub(r"[,.]{2,}", lambda match: match.group(0)[-1], cleaned)
            if not cleaned:
                counts["empty"] += 1
                continue
            if min_len and len(cleaned) < min_len:
                counts["too_short"] += 1
                continue
            if max_len and len(cleaned) > max_len:
                counts["too_long"] += 1
                continue
            counts["written"] += 1
            yield cleaned

    return generator(), counts


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for cleaning a line-based text corpus."""
    parser = argparse.ArgumentParser(
        description=(
            "Clean a line-based text corpus using the same normalization as NeMo dataset prep."
        )
    )
    parser.add_argument(
        "--input_files",
        nargs="+",
        required=True,
        help="One or more text files (one sample per line). Comma-separated lists supported.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for cleaned text. Defaults to <data_root>/text_corpus/document.txt "
        "when --data_root is set, otherwise <input>.cleaned.txt.",
    )
    parser.add_argument(
        "--data_root",
        default=None,
        help="Optional output root; writes cleaned text to <data_root>/text_corpus/document.txt.",
    )
    parser.add_argument(
        "--min_chars",
        type=int,
        default=1,
        help="Drop cleaned lines shorter than this length (default: 1).",
    )
    parser.add_argument(
        "--max_chars",
        type=int,
        default=0,
        help="Drop cleaned lines longer than this length (0 = no limit).",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for cleaning and writing the normalized text corpus."""
    args = parse_args()
    input_paths = _expand_input_paths(args.input_files)
    if not input_paths:
        print("No input files provided.", file=sys.stderr)
        sys.exit(2)

    missing = [path for path in input_paths if not path.exists()]
    if missing:
        missing_list = ", ".join(str(path) for path in missing)
        print(f"Missing input files: {missing_list}", file=sys.stderr)
        sys.exit(2)

    output_path = _resolve_output_path(args.output, args.data_root, input_paths)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = _iter_lines(input_paths)
    cleaned_lines, counts = _filter_and_clean_lines(
        lines, min_chars=args.min_chars, max_chars=args.max_chars
    )

    with output_path.open("w", encoding="utf-8") as handle:
        for cleaned in cleaned_lines:
            handle.write(cleaned + "\n")

    print(
        "Cleaned text written to: {}\n"
        "Total lines: {total}\n"
        "Written: {written}\n"
        "Dropped empty: {empty}\n"
        "Dropped too short: {too_short}\n"
        "Dropped too long: {too_long}".format(output_path, **counts)
    )


if __name__ == "__main__":
    main()
