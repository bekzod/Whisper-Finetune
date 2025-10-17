#!/usr/bin/env python
"""Pre-download HuggingFace datasets defined in the training/eval config."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterable as IterableCollection
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from datasets import load_dataset
from datasets.utils.logging import set_verbosity_info

DatasetEntry = Dict[str, object]
SplitKey = Tuple[str, Optional[str], Optional[str], str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Warm the HuggingFace datasets cache for configured splits."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/datasets.json"),
        help="Path to the dataset configuration JSON.",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        default=["train", "eval"],
        help="Top-level groups in the config to pre-download (default: train eval).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional HuggingFace datasets cache directory.",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Optional HuggingFace token for gated datasets.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be downloaded without performing the calls.",
    )
    return parser.parse_args()


def load_dataset_config(path: Path) -> Dict[str, Sequence[DatasetEntry]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_selected_entries(
    config: Dict[str, Sequence[DatasetEntry]],
    groups: Sequence[str],
) -> IterableCollection[Tuple[str, DatasetEntry]]:
    for group in groups:
        for entry in config.get(group, []):
            if isinstance(entry, dict):
                yield group, entry


def coerce_splits(raw_value: object) -> List[str]:
    if raw_value is None:
        return ["train"]
    if isinstance(raw_value, str):
        return [raw_value]
    if isinstance(raw_value, IterableCollection):
        return [str(split) for split in raw_value]
    raise TypeError(f"Unsupported split definition: {raw_value!r}")


def parse_repo_spec(name: str) -> Tuple[str, Optional[str], Optional[str]]:
    normalized = name.strip()
    if normalized.startswith("hf://"):
        normalized = normalized[5:]
    subset = None
    revision = None
    before_hash = normalized
    if "#" in normalized:
        before_hash, subset = normalized.split("#", 1)
    if "@" in before_hash:
        repo, revision = before_hash.split("@", 1)
    else:
        repo = before_hash
    return repo, revision, subset


def is_local_reference(name: str) -> bool:
    potential_path = Path(name)
    return potential_path.exists() or os.path.isabs(name)


def extract_optional_kwargs(entry: DatasetEntry) -> Dict[str, object]:
    optional_keys = ("data_dir", "data_files", "trust_remote_code")
    return {key: entry[key] for key in optional_keys if key in entry}


def warm_split(
    group: str,
    repo: str,
    subset: Optional[str],
    revision: Optional[str],
    split: str,
    cache_dir: Optional[Path],
    hf_token: Optional[str],
    extra_kwargs: Dict[str, object],
    dry_run: bool,
) -> bool:
    display_subset = subset or "default"
    print(f"[{group}] caching {repo} ({display_subset}) split={split}")
    if dry_run:
        return True

    load_kwargs: Dict[str, object] = {"split": split}
    if revision:
        load_kwargs["revision"] = revision
    if cache_dir:
        load_kwargs["cache_dir"] = str(cache_dir)
    if hf_token:
        load_kwargs["use_auth_token"] = hf_token
    load_kwargs.update(extra_kwargs)

    try:
        if subset:
            dataset = load_dataset(repo, name=subset, **load_kwargs)
        else:
            dataset = load_dataset(repo, **load_kwargs)
        try:
            _ = len(dataset)
        except (TypeError, NotImplementedError):
            pass
        return True
    except Exception as exc:  # pylint: disable=broad-except
        print(f"  ⚠️ failed to download {repo} ({display_subset}) split={split}: {exc}")
        return False


def main() -> None:
    args = parse_args()
    set_verbosity_info()

    config = load_dataset_config(args.config)
    seen: Set[SplitKey] = set()
    requested = 0
    completed = 0

    for group, entry in iter_selected_entries(config, args.groups):
        name = entry.get("name")
        if not name:
            print(f"[{group}] skipping entry without a 'name' field: {entry}")
            continue

        name_str = str(name)
        if is_local_reference(name_str):
            print(f"[{group}] skipping local dataset reference '{name_str}'")
            continue

        repo, revision, inline_subset = parse_repo_spec(name_str)
        subset = str(entry["subset"]) if entry.get("subset") else inline_subset
        splits = coerce_splits(entry.get("splits"))
        extra_kwargs = extract_optional_kwargs(entry)

        for split in splits:
            key: SplitKey = (repo, subset, revision, split)
            if key in seen:
                continue
            seen.add(key)
            requested += 1
            if warm_split(
                group,
                repo,
                subset,
                revision,
                split,
                args.cache_dir,
                args.hf_token,
                extra_kwargs,
                args.dry_run,
            ):
                completed += 1

    status = "planned" if args.dry_run else "completed"
    print(f"{status.capitalize()} {completed}/{requested} dataset split downloads.")


if __name__ == "__main__":
    main()
