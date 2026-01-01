#!/usr/bin/env python3
"""Diagnostic script to debug HuggingFace dataset loading issues."""

import sys
from pathlib import Path

# Allow running from nemo directory
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main():
    print("=" * 60)
    print("Dataset Loading Diagnostic Script")
    print("=" * 60)

    # Import dependencies
    try:
        import numpy as np
        from datasets import Audio, load_dataset
    except ImportError as e:
        print(f"ERROR: Missing dependency: {e}")
        print("Please install: pip install datasets numpy")
        return 1

    dataset_name = "IbratDO/first_dataset"
    split = "train"

    print(f"\n1. Loading dataset: {dataset_name} (split={split})")
    print("-" * 40)

    try:
        ds = load_dataset(dataset_name, split=split)
        print(f"   ✓ Dataset loaded successfully")
        print(f"   Length: {len(ds)}")
    except Exception as e:
        print(f"   ✗ Failed to load dataset: {e}")
        return 1

    print(f"\n2. Dataset Features")
    print("-" * 40)
    print(f"   Features: {ds.features}")
    print(f"   Column names: {ds.column_names}")

    for col_name in ds.column_names:
        feature = ds.features.get(col_name)
        print(f"   - {col_name}: {type(feature).__name__} = {feature}")

    print(f"\n3. First 3 raw items (BEFORE any casting)")
    print("-" * 40)
    for i in range(min(3, len(ds))):
        print(f"\n   Item {i}:")
        item = ds[i]
        for k, v in item.items():
            v_type = type(v).__name__
            if v is None:
                print(f"      {k}: None")
            elif isinstance(v, dict):
                dict_keys = list(v.keys())
                has_array = "array" in v and v["array"] is not None
                if has_array:
                    arr = np.asarray(v["array"])
                    array_info = f"shape={arr.shape}, dtype={arr.dtype}"
                else:
                    array_info = "NO ARRAY"
                sr = v.get("sampling_rate")
                path = v.get("path")
                print(f"      {k}: dict")
                print(f"         keys: {dict_keys}")
                print(f"         array: {array_info}")
                print(f"         sampling_rate: {sr}")
                print(f"         path: {path}")
            elif isinstance(v, str):
                preview = v[:80] + "..." if len(v) > 80 else v
                print(f"      {k}: str = '{preview}'")
            elif isinstance(v, np.ndarray):
                print(f"      {k}: ndarray shape={v.shape}, dtype={v.dtype}")
            elif isinstance(v, (bytes, bytearray)):
                print(f"      {k}: bytes len={len(v)}")
            else:
                print(f"      {k}: {v_type} = {str(v)[:80]}")

    # Check if audio column needs casting
    print(f"\n4. Checking Audio column status")
    print("-" * 40)

    audio_col = None
    for col in ds.column_names:
        feature = ds.features.get(col)
        feature_str = str(feature)
        if (
            isinstance(feature, Audio)
            or "Audio" in feature_str
            or "audio" in feature_str.lower()
        ):
            audio_col = col
            print(f"   Found audio column: '{col}'")
            print(f"   Feature type: {type(feature).__name__}")
            print(f"   Feature: {feature}")
            break

    if audio_col is None:
        print("   ✗ No audio column detected!")
        print("   Available columns:", ds.column_names)

    # Try casting audio column
    print(f"\n5. Casting audio column to ensure decoding")
    print("-" * 40)

    if audio_col:
        try:
            ds_casted = ds.cast_column(audio_col, Audio(sampling_rate=16000))
            print(f"   ✓ Cast successful")

            print(f"\n6. First 3 items AFTER casting")
            print("-" * 40)
            for i in range(min(3, len(ds_casted))):
                print(f"\n   Item {i}:")
                item = ds_casted[i]
                for k, v in item.items():
                    v_type = type(v).__name__
                    if v is None:
                        print(f"      {k}: None")
                    elif isinstance(v, dict):
                        dict_keys = list(v.keys())
                        has_array = "array" in v and v["array"] is not None
                        if has_array:
                            arr = np.asarray(v["array"])
                            array_info = f"shape={arr.shape}, dtype={arr.dtype}, min={arr.min():.4f}, max={arr.max():.4f}"
                        else:
                            array_info = "NO ARRAY"
                        sr = v.get("sampling_rate")
                        path = v.get("path")
                        print(f"      {k}: dict")
                        print(f"         has_array: {has_array}")
                        print(f"         array_info: {array_info}")
                        print(f"         sampling_rate: {sr}")
                        print(f"         path: {path}")
                    elif isinstance(v, str):
                        preview = v[:80] + "..." if len(v) > 80 else v
                        print(f"      {k}: str = '{preview}'")
                    elif isinstance(v, np.ndarray):
                        print(f"      {k}: ndarray shape={v.shape}, dtype={v.dtype}")
                    else:
                        print(f"      {k}: {v_type} = {str(v)[:80]}")

        except Exception as e:
            print(f"   ✗ Cast failed: {e}")
            import traceback

            traceback.print_exc()
    else:
        # Try to find any column that might contain audio
        print("\n   Trying to find audio data in all columns...")
        for col in ds.column_names:
            print(f"\n   Checking column '{col}':")
            sample_val = ds[0][col]
            print(f"      Type: {type(sample_val).__name__}")
            if isinstance(sample_val, dict):
                print(f"      Dict keys: {list(sample_val.keys())}")
            elif isinstance(sample_val, str):
                print(f"      Value: {sample_val[:100]}")

    print("\n" + "=" * 60)
    print("Diagnostic complete")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
