#!/usr/bin/env python3
"""
Dataset Validation Utility for Whisper Fine-tuning

This script helps identify potential issues with dataset loading, particularly
TSV/CSV parsing problems that can lead to overly long labels exceeding token limits.

Usage:
    python utils/validate_dataset.py --data_path path/to/dataset.csv
    python utils/validate_dataset.py --data_path path/to/datasets.json --model openai/whisper-small
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import soundfile
from transformers import WhisperProcessor
from tqdm import tqdm


def validate_csv_file(file_path, processor=None, max_examples=10):
    """Validate CSV/TSV file for common parsing issues."""
    print(f"\n🔍 Validating CSV file: {file_path}")

    issues_found = []
    examples_logged = 0

    with open(file_path, "r", encoding="utf-8") as f:
        # Detect delimiter
        sample = f.read(1024)
        f.seek(0)

        delimiter = ","
        if "|" in sample and sample.count("|") > sample.count(","):
            delimiter = "|"
            print(f"   Detected pipe delimiter (LJSpeech format)")
        elif "\t" in sample and sample.count("\t") > sample.count(","):
            delimiter = "\t"
            print(f"   Detected tab delimiter (TSV format)")
        else:
            print(f"   Using comma delimiter (CSV format)")

        reader = csv.reader(f, delimiter=delimiter)

        # Check for header
        first_row = next(reader, None)
        if not first_row:
            return ["Empty file"]

        has_header = any(
            keyword in first_row[0].lower()
            for keyword in ["filename", "path", "audio", "text", "transcript"]
        )

        if has_header:
            print(f"   Header detected: {first_row}")
            data_start_row = 2
        else:
            print(f"   No header, first row: {first_row}")
            data_start_row = 1
            # Process first row as data
            f.seek(0)
            reader = csv.reader(f, delimiter=delimiter)

        row_count = 0
        for row_idx, row in enumerate(reader, start=data_start_row):
            if has_header and row_idx == 2:
                # Skip header row
                continue

            row_count += 1

            if len(row) < 2:
                issues_found.append(f"Row {row_idx}: insufficient columns ({len(row)})")
                continue

            # Extract filename and text based on format
            if delimiter == "|" and len(row) == 1 and "|" in row[0]:
                filename, text = row[0].split("|", 1)
            else:
                filename, text = row[0], row[1]

            # Check for parsing issues
            text_len = len(text)
            char_issues = []

            if text_len > 2000:
                char_issues.append(f"very long ({text_len} chars)")

            if "\n" in text:
                char_issues.append(f"{text.count('\n')} newlines")

            if "\t" in text and delimiter != "\t":
                char_issues.append(f"{text.count('\t')} unexpected tabs")

            if text.count(delimiter) > 5 and delimiter in text:
                char_issues.append(f"contains {text.count(delimiter)} delimiter chars")

            # Check token length if processor available
            token_issues = []
            if processor:
                try:
                    tokens = processor.tokenizer.encode(text, add_special_tokens=False)
                    token_len = len(tokens)
                    if token_len > 448:
                        token_issues.append(f"exceeds max tokens ({token_len} > 448)")
                    elif token_len > 350:
                        token_issues.append(f"approaching token limit ({token_len})")
                except Exception as e:
                    token_issues.append(f"tokenization error: {e}")

            # Report issues
            if char_issues or token_issues:
                issue_desc = f"Row {row_idx}: {', '.join(char_issues + token_issues)}"
                issues_found.append(issue_desc)

                if examples_logged < max_examples:
                    print(f"   ⚠️  {issue_desc}")
                    print(f"       Filename: {filename}")
                    print(
                        f"       Text preview: '{text[:150]}{'...' if len(text) > 150 else ''}'"
                    )
                    examples_logged += 1

            # Stop after reasonable sample for large files
            if row_count > 1000:
                break

    print(f"   Processed {row_count} rows")
    print(f"   Found {len(issues_found)} potential issues")

    return issues_found


def validate_json_manifest(file_path, processor=None, max_examples=10):
    """Validate JSON manifest file."""
    print(f"\n🔍 Validating JSON manifest: {file_path}")

    issues_found = []
    examples_logged = 0

    with open(file_path, "r", encoding="utf-8") as f:
        if file_path.endswith(".jsonl"):
            # JSONL format
            for line_idx, line in enumerate(f, 1):
                if not line.strip():
                    continue

                try:
                    data = json.loads(line.strip())
                except json.JSONDecodeError as e:
                    issues_found.append(f"Line {line_idx}: JSON decode error - {e}")
                    continue

                issues = validate_manifest_entry(data, line_idx, processor)
                issues_found.extend(issues)

                if issues and examples_logged < max_examples:
                    print(f"   ⚠️  Line {line_idx} issues: {', '.join(issues)}")
                    examples_logged += 1

                if line_idx > 1000:  # Sample limit
                    break
        else:
            # Regular JSON format
            try:
                data = json.load(f)
                if isinstance(data, list):
                    for idx, entry in enumerate(data):
                        issues = validate_manifest_entry(entry, idx + 1, processor)
                        issues_found.extend(issues)

                        if issues and examples_logged < max_examples:
                            print(f"   ⚠️  Entry {idx + 1} issues: {', '.join(issues)}")
                            examples_logged += 1
                else:
                    issues_found.append("JSON file is not a list of entries")
            except json.JSONDecodeError as e:
                issues_found.append(f"JSON decode error: {e}")

    print(f"   Found {len(issues_found)} potential issues")
    return issues_found


def validate_manifest_entry(entry, entry_id, processor=None):
    """Validate a single manifest entry."""
    issues = []

    # Check required fields
    if "audio" not in entry and "audio_path" not in entry:
        issues.append("missing audio path")

    # Get text content
    text = entry.get("sentence") or entry.get("text") or entry.get("transcription", "")

    if not text:
        issues.append("missing text content")
        return issues

    # Check text length
    text_len = len(text)
    if text_len > 2000:
        issues.append(f"very long text ({text_len} chars)")

    # Check for parsing artifacts
    if "\n" in text:
        issues.append(f"contains {text.count('\n')} newlines")

    if "\t" in text:
        issues.append(f"contains {text.count('\t')} tabs")

    # Check token length
    if processor:
        try:
            tokens = processor.tokenizer.encode(text, add_special_tokens=False)
            token_len = len(tokens)
            if token_len > 448:
                issues.append(f"exceeds token limit ({token_len} > 448)")
            elif token_len > 350:
                issues.append(f"approaching token limit ({token_len})")
        except Exception as e:
            issues.append(f"tokenization error: {e}")

    return issues


def validate_audio_files(data_paths, max_check=50):
    """Validate that audio files exist and are readable."""
    print(
        f"\n🔍 Validating audio file accessibility (checking up to {max_check} files)"
    )

    audio_paths = []
    issues_found = []

    for path in data_paths:
        if path.endswith(".csv") or path.endswith(".tsv"):
            # Extract audio paths from CSV
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.reader(f, delimiter="|" if path.endswith(".tsv") else ",")
                for row_idx, row in enumerate(reader):
                    if len(row) >= 2:
                        audio_path = row[0]
                        # Try both absolute and relative to CSV directory
                        if not os.path.isfile(audio_path):
                            audio_path = os.path.join(os.path.dirname(path), audio_path)
                        audio_paths.append(audio_path)

                        if len(audio_paths) >= max_check:
                            break

    print(f"   Checking {len(audio_paths)} audio files...")
    accessible_count = 0

    for audio_path in tqdm(audio_paths[:max_check]):
        try:
            if os.path.isfile(audio_path):
                # Try to read audio file info
                info = soundfile.info(audio_path)
                accessible_count += 1
            else:
                issues_found.append(f"File not found: {audio_path}")
        except Exception as e:
            issues_found.append(f"Cannot read {audio_path}: {e}")

    print(f"   {accessible_count}/{len(audio_paths[:max_check])} files accessible")
    print(f"   Found {len(issues_found)} audio file issues")

    return issues_found


def main():
    parser = argparse.ArgumentParser(
        description="Validate datasets for Whisper fine-tuning"
    )
    parser.add_argument(
        "--data_path", required=True, help="Path to dataset file or manifest"
    )
    parser.add_argument(
        "--model", default="openai/whisper-small", help="Whisper model for tokenization"
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=10,
        help="Max examples to show per issue type",
    )
    parser.add_argument(
        "--check_audio",
        action="store_true",
        help="Also validate audio file accessibility",
    )
    parser.add_argument(
        "--max_audio_check", type=int, default=50, help="Max audio files to check"
    )

    args = parser.parse_args()

    # Load processor for tokenization
    processor = None
    try:
        print(f"Loading Whisper processor: {args.model}")
        processor = WhisperProcessor.from_pretrained(args.model)
    except Exception as e:
        print(
            f"Warning: Could not load processor ({e}). Token length validation disabled."
        )

    # Validate the dataset
    all_issues = []

    if args.data_path.endswith(".json"):
        # Check if it's a datasets.json config file
        with open(args.data_path, "r") as f:
            config = json.load(f)

        if "datasets" in config:
            print("📋 Detected datasets.json config file")
            # Validate each dataset in the config
            for dataset_info in config["datasets"]:
                dataset_path = dataset_info.get("path") or dataset_info.get("name")
                if dataset_path and os.path.isfile(dataset_path):
                    if dataset_path.endswith((".csv", ".tsv")):
                        issues = validate_csv_file(
                            dataset_path, processor, args.max_examples
                        )
                        all_issues.extend(issues)
                    elif dataset_path.endswith((".json", ".jsonl")):
                        issues = validate_json_manifest(
                            dataset_path, processor, args.max_examples
                        )
                        all_issues.extend(issues)
        else:
            # Regular JSON manifest
            issues = validate_json_manifest(
                args.data_path, processor, args.max_examples
            )
            all_issues.extend(issues)

    elif args.data_path.endswith(".jsonl"):
        issues = validate_json_manifest(args.data_path, processor, args.max_examples)
        all_issues.extend(issues)

    elif args.data_path.endswith((".csv", ".tsv")):
        issues = validate_csv_file(args.data_path, processor, args.max_examples)
        all_issues.extend(issues)

    else:
        print(f"❌ Unsupported file format: {args.data_path}")
        sys.exit(1)

    # Check audio files if requested
    if args.check_audio:
        audio_issues = validate_audio_files([args.data_path], args.max_audio_check)
        all_issues.extend(audio_issues)

    # Summary
    print(f"\n📊 VALIDATION SUMMARY")
    print(f"   Total issues found: {len(all_issues)}")

    if all_issues:
        print(f"\n❌ Issues that may cause training problems:")
        for issue in all_issues[:20]:  # Show first 20 issues
            print(f"   • {issue}")

        if len(all_issues) > 20:
            print(f"   ... and {len(all_issues) - 20} more issues")

        print(f"\n💡 Common fixes:")
        print(f"   • Check TSV/CSV delimiter (comma, pipe, or tab)")
        print(f"   • Verify text column doesn't contain multi-line content")
        print(f"   • Ensure text transcriptions are reasonable length")
        print(f"   • Consider splitting very long transcriptions")

        sys.exit(1)
    else:
        print(f"✅ No major issues detected!")
        sys.exit(0)


if __name__ == "__main__":
    main()
