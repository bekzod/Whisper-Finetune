"""
TSV parsing utilities for Whisper fine-tuning datasets.

This module provides functions to parse TSV files with proper handling of:
- UTF-8 BOM removal
- Header detection and normalization
- Column mapping and validation
- Robust CSV parsing with quoting support
"""

import csv
import sys
from itertools import chain, zip_longest
from typing import Any, Dict, Iterator, List, Optional, Sequence, TextIO, Tuple

try:
    csv.field_size_limit(sys.maxsize)
except (OverflowError, AttributeError):
    # Fall back to a large but safe limit if sys.maxsize is not accepted
    csv.field_size_limit(2**31 - 1)


def _strip_bom(value: str) -> str:
    """Remove a UTF-8 BOM prefix if present."""
    if value and value[0] == "\ufeff":
        return value[1:]
    return value


def _iter_tsv_rows(
    file_obj: TextIO,
    *,
    delimiter: str = "\t",
    strip_cells: bool = True,
    skip_comments: bool = True,
) -> Iterator[List[str]]:
    """
    Yield TSV rows using the csv module to honor quoting and embedded delimiters.
    """
    reader = csv.reader(
        file_obj,
        delimiter=delimiter,
        quoting=csv.QUOTE_MINIMAL,
        skipinitialspace=False,
    )
    for raw_row in reader:
        if not raw_row:
            continue
        if strip_cells:
            row = []
            for cell in raw_row:
                cell = _strip_bom(cell.strip()) if isinstance(cell, str) else cell
                row.append(cell)
        else:
            row = [
                _strip_bom(cell) if isinstance(cell, str) else cell for cell in raw_row
            ]
        if skip_comments:
            for cell in row:
                if not cell:
                    continue
                if isinstance(cell, str) and cell.startswith("#"):
                    row = []
                break
            if not row:
                continue
        yield row


def _prepare_tsv_header(raw_header: Sequence[str]) -> List[str]:
    """
    Normalize TSV header names to lowercase unique keys for robust lookups.
    Empty headers are replaced with positional placeholders.
    """
    normalized: List[str] = []
    seen: Dict[str, int] = {}
    for idx, cell in enumerate(raw_header):
        if isinstance(cell, str):
            header_name = _strip_bom(cell).strip().lower()
        elif cell is None:
            header_name = ""
        else:
            header_name = str(cell).strip().lower()

        if not header_name:
            header_name = f"column_{idx}"

        base = header_name
        count = seen.get(base, 0)
        if count > 0:
            header_name = f"{base}_{count + 1}"
        seen[base] = count + 1
        normalized.append(header_name)

    return normalized


def _iter_tsv_dict_rows(
    file_obj: TextIO,
    *,
    delimiter: str = "\t",
    strip_cells: bool = True,
    skip_comments: bool = True,
) -> Tuple[List[str], Dict[str, str], Iterator[Tuple[int, Dict[str, str]]]]:
    """
    Yield TSV rows as dictionaries keyed by normalized lowercase header names.

    Returns:
        header_keys: normalized lowercase header names.
        header_lookup: mapping of normalized names to their original header string.
        row_iter: iterator yielding (line_number, row_dict) pairs.
    """
    row_iter = _iter_tsv_rows(
        file_obj,
        delimiter=delimiter,
        strip_cells=strip_cells,
        skip_comments=skip_comments,
    )
    header_row = next(row_iter, None)
    if not header_row:
        return [], {}, iter(())

    original_header: List[str] = []
    normalized_header_lower: List[str] = []
    for idx, cell in enumerate(header_row):
        if isinstance(cell, str):
            cleaned = _strip_bom(cell).strip()
        elif cell is None:
            cleaned = ""
        else:
            cleaned = str(cell).strip()
        original_header.append(cleaned)
        normalized_header_lower.append(cleaned.lower())

    header_hint_substrings = (
        "path",
        "audio",
        "filename",
        "file",
        "text",
        "sentence",
        "transcript",
        "transcription",
        "normalized",
        "raw",
        "duration",
        "language",
        "id",
    )
    header_present = any(
        any(hint in cell for hint in header_hint_substrings) and cell
        for cell in normalized_header_lower
    )

    if header_present:
        header_keys = _prepare_tsv_header(original_header)
        header_lookup = {
            header_keys[idx]: (original_header[idx] or header_keys[idx])
            for idx in range(len(header_keys))
        }
        data_iter: Iterator[List[str]] = row_iter
        first_data_line_number = 2
    else:
        header_keys = [f"column_{idx}" for idx in range(len(original_header))]
        header_lookup = {key: key for key in header_keys}
        data_iter = chain([original_header], row_iter)
        first_data_line_number = 1
        source_desc = getattr(file_obj, "name", "<tsv>")
        print(f"  Info: No header detected in {source_desc}; using positional columns.")

    def generator() -> Iterator[Tuple[int, Dict[str, str]]]:
        extras_warning_emitted = False
        for line_number, row in enumerate(data_iter, start=first_data_line_number):
            if not row:
                continue
            if len(row) > len(header_keys) and not extras_warning_emitted:
                extras_warning_emitted = True
                ignored = len(row) - len(header_keys)
                print(
                    f"  Warning: detected {ignored} extra columns starting at line {line_number}; ignoring surplus values."
                )

            row_dict: Dict[str, str] = {}
            for key, value in zip_longest(header_keys, row, fillvalue=""):
                if key is None:
                    continue
                if value is None:
                    cell_value = ""
                elif isinstance(value, str):
                    cell_value = value
                else:
                    cell_value = str(value)
                row_dict[key] = cell_value
            yield line_number, row_dict

    return header_keys, header_lookup, generator()


def _detect_column_index(
    header: Sequence[str], candidates: Sequence[str], default_idx: Optional[int]
) -> Optional[int]:
    """
    Locate the index of the first matching candidate column in a normalized header.
    """
    normalized = {col.strip().lower(): idx for idx, col in enumerate(header) if col}
    for candidate in candidates:
        idx = normalized.get(candidate)
        if idx is not None:
            return idx
    return default_idx


def iter_tsv_dict_rows(
    file_obj: TextIO,
    *,
    delimiter: str = "\t",
    strip_cells: bool = True,
    skip_comments: bool = True,
) -> Tuple[List[str], Dict[str, str], Iterator[Tuple[int, Dict[str, str]]]]:
    """
    Public interface for iterating over TSV rows as dictionaries.

    Args:
        file_obj: Open file object to read from
        delimiter: Column delimiter (default: tab)
        strip_cells: Whether to strip whitespace from cells
        skip_comments: Whether to skip rows starting with '#'

    Returns:
        Tuple of (header_keys, header_lookup, row_iterator)
        - header_keys: List of normalized column names
        - header_lookup: Mapping from normalized to original header names
        - row_iterator: Iterator yielding (line_number, row_dict) tuples
    """
    return _iter_tsv_dict_rows(
        file_obj,
        delimiter=delimiter,
        strip_cells=strip_cells,
        skip_comments=skip_comments,
    )


def detect_column_index(
    header: Sequence[str], candidates: Sequence[str], default_idx: Optional[int] = None
) -> Optional[int]:
    """
    Public interface for detecting column indices by candidate names.

    Args:
        header: List of header column names
        candidates: List of candidate column names to search for
        default_idx: Default index to return if no match found

    Returns:
        Index of first matching column, or default_idx if none found
    """
    return _detect_column_index(header, candidates, default_idx)
