#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pre-download HuggingFace datasets to avoid rate limits during training.
This script downloads datasets to local cache so that training can proceed
without hitting API rate limits.
"""

import argparse
import os
import sys
import time
from datasets import load_dataset
from tqdm import tqdm


def rate_limited_download(func, *args, **kwargs):
    """
    Execute a function with exponential backoff retry for rate limiting.
    """
    max_retries = 5
    base_delay = 60  # Start with 1 minute delay

    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_msg = str(e).lower()
            if "rate limit" in error_msg or "quota" in error_msg or "2500" in error_msg:
                if attempt < max_retries - 1:
                    delay = base_delay * (2**attempt)  # Exponential backoff
                    print(
                        f"Rate limit hit. Waiting {delay} seconds before retry {attempt + 1}/{max_retries}..."
                    )
                    time.sleep(delay)
                    continue
                else:
                    print(
                        "Max retries reached for rate limiting. Please upgrade your HF organization or wait."
                    )
                    raise
            else:
                # Re-raise non-rate-limit errors immediately
                raise


def download_dataset(repo_id, subset=None, split=None, revision=None, cache_dir=None):
    """
    Download a dataset with rate limiting protection.
    """
    print(f"\n📥 Downloading dataset: {repo_id}")
    if subset:
        print(f"   Subset: {subset}")
    if split:
        print(f"   Split: {split}")
    if revision:
        print(f"   Revision: {revision}")

    try:
        # First try with the specified subset
        try:
            dataset = rate_limited_download(
                load_dataset,
                repo_id,
                name=subset,
                split=split,
                revision=revision,
                cache_dir=cache_dir,
                download_mode="reuse_dataset_if_exists",
            )
        except Exception as subset_error:
            if "not found" in str(subset_error).lower() and subset:
                print(f"   Subset '{subset}' not found, trying with default config...")
                # Try without subset (use default config)
                dataset = rate_limited_download(
                    load_dataset,
                    repo_id,
                    split=split,
                    revision=revision,
                    cache_dir=cache_dir,
                    download_mode="reuse_dataset_if_exists",
                )
            else:
                raise subset_error

        # Get dataset size info
        if hasattr(dataset, "__len__"):
            print(f"✅ Successfully downloaded {len(dataset)} examples")
        else:
            print(f"✅ Successfully downloaded dataset")

        return True

    except Exception as e:
        print(f"❌ Failed to download {repo_id}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Pre-download HuggingFace datasets to avoid rate limits during training"
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Dataset repository IDs to download (e.g., 'mozilla-foundation/common_voice_17_0')",
    )

    parser.add_argument(
        "--subsets",
        nargs="*",
        help="Dataset subsets/configs to download (e.g., 'uz' for Uzbek). If not specified, downloads all available.",
    )

    parser.add_argument(
        "--splits",
        nargs="*",
        default=["train", "validation", "test"],
        help="Splits to download (default: train, validation, test)",
    )

    parser.add_argument(
        "--revision",
        help="Specific revision/branch to download (e.g., 'refs/convert/parquet')",
    )

    parser.add_argument(
        "--cache-dir",
        help="Custom cache directory for datasets (default: uses HuggingFace default)",
    )

    parser.add_argument(
        "--common-voice-uz",
        action="store_true",
        help="Quick option to download Uzbek Common Voice 17.0 dataset",
    )

    args = parser.parse_args()

    # Set up cache directory
    cache_dir = args.cache_dir or os.getenv("HF_DATASETS_CACHE")
    if cache_dir:
        print(f"Using cache directory: {cache_dir}")

    # Quick option for Common Voice Uzbek
    if args.common_voice_uz:
        print("🚀 Downloading Uzbek Common Voice 17.0 dataset...")
        datasets_to_download = [
            {
                "repo_id": "mozilla-foundation/common_voice_17_0",
                "subset": "uz",
                "splits": ["train", "validation", "test"],
                "revision": None,  # Remove specific revision
            }
        ]
    else:
        if not args.datasets:
            print("❌ Error: Please specify --datasets or use --common-voice-uz")
            sys.exit(1)

        datasets_to_download = []
        for repo_id in args.datasets:
            datasets_to_download.append(
                {
                    "repo_id": repo_id,
                    "subset": args.subsets[0] if args.subsets else None,
                    "splits": args.splits,
                    "revision": args.revision,
                }
            )

    # Download datasets
    total_datasets = len(datasets_to_download)
    successful_downloads = 0

    print(f"🎯 Planning to download {total_datasets} dataset(s)")
    print("=" * 50)

    for i, dataset_config in enumerate(datasets_to_download, 1):
        repo_id = dataset_config["repo_id"]
        subset = dataset_config["subset"]
        splits = dataset_config["splits"]
        revision = dataset_config["revision"]

        print(f"\n[{i}/{total_datasets}] Processing {repo_id}")

        dataset_success = True

        # If splits are specified, download each split separately
        if splits:
            for split in splits:
                try:
                    success = download_dataset(
                        repo_id=repo_id,
                        subset=subset,
                        split=split,
                        revision=revision,
                        cache_dir=cache_dir,
                    )
                    if not success:
                        dataset_success = False
                except Exception as e:
                    print(f"❌ Error downloading {repo_id}:{split} - {e}")
                    dataset_success = False
        else:
            # Download entire dataset
            try:
                success = download_dataset(
                    repo_id=repo_id,
                    subset=subset,
                    revision=revision,
                    cache_dir=cache_dir,
                )
                if not success:
                    dataset_success = False
            except Exception as e:
                print(f"❌ Error downloading {repo_id} - {e}")
                dataset_success = False

        if dataset_success:
            successful_downloads += 1

    # Summary
    print("\n" + "=" * 50)
    print(f"📊 Download Summary:")
    print(f"   Total datasets: {total_datasets}")
    print(f"   Successful: {successful_downloads}")
    print(f"   Failed: {total_datasets - successful_downloads}")

    if successful_downloads == total_datasets:
        print("🎉 All datasets downloaded successfully!")
        print("\n💡 Tips:")
        print("   - Datasets are now cached locally and won't trigger API rate limits")
        print("   - You can now run your training script without rate limit issues")
        print("   - Consider setting HF_DATASETS_OFFLINE=1 to force offline mode")
    else:
        print("⚠️  Some datasets failed to download. Check the errors above.")
        print("   You may need to:")
        print("   - Wait for rate limits to reset")
        print("   - Upgrade your HuggingFace organization plan")
        print("   - Check dataset names and availability")
        print(
            "   - Some datasets may use 'default' config instead of language-specific subsets"
        )


if __name__ == "__main__":
    main()
