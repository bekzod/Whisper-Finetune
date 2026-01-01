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

    def inspect_object(obj, name="object"):
        """Thoroughly inspect an object to understand its interface."""
        print(f"         --- Inspecting {name} ---")
        print(f"         Type: {type(obj)}")
        print(f"         Module: {type(obj).__module__}")

        # Get all attributes and methods
        attrs = [a for a in dir(obj) if not a.startswith("_")]
        print(f"         Public attributes/methods: {attrs}")

        # Check for common audio-related attributes
        for attr in [
            "array",
            "waveform",
            "samples",
            "data",
            "audio",
            "sampling_rate",
            "sample_rate",
            "sr",
            "path",
            "file",
        ]:
            if hasattr(obj, attr):
                try:
                    val = getattr(obj, attr)
                    val_type = type(val).__name__
                    if isinstance(val, np.ndarray):
                        print(f"         .{attr}: ndarray shape={val.shape}")
                    elif val is None:
                        print(f"         .{attr}: None")
                    else:
                        print(f"         .{attr}: {val_type} = {str(val)[:50]}")
                except Exception as e:
                    print(f"         .{attr}: ERROR accessing - {e}")

        # Check if callable
        print(f"         Callable: {callable(obj)}")

        # Check for __getitem__
        if hasattr(obj, "__getitem__"):
            print(f"         Has __getitem__: True")
            try:
                item = obj[0]
                print(f"         obj[0] = {type(item).__name__}")
            except Exception as e:
                print(f"         obj[0] ERROR: {e}")

        # Check for iter
        if hasattr(obj, "__iter__"):
            print(f"         Has __iter__: True")

        # Try to get the actual decoding method
        for method_name in [
            "decode",
            "read",
            "load",
            "get",
            "to_dict",
            "to_numpy",
            "__call__",
        ]:
            if hasattr(obj, method_name):
                method = getattr(obj, method_name)
                print(f"         Has .{method_name}(): {callable(method)}")

    def try_decode_audio(v, verbose=False):
        """Try to decode an AudioDecoder or similar object."""
        if verbose:
            inspect_object(v, "AudioDecoder")

        # Method 1: Check if it's an AudioDecoder (callable)
        if callable(v):
            try:
                decoded = v()
                if verbose:
                    print(f"         v() returned: {type(decoded).__name__}")
                if isinstance(decoded, dict):
                    return decoded
                if isinstance(decoded, np.ndarray):
                    return {"array": decoded, "sampling_rate": None}
            except Exception as e:
                if verbose:
                    print(f"         v() failed: {e}")

        # Method 2: Check for array/sampling_rate attributes
        if hasattr(v, "array") and hasattr(v, "sampling_rate"):
            try:
                return {
                    "array": v.array,
                    "sampling_rate": v.sampling_rate,
                    "path": getattr(v, "path", None),
                }
            except Exception as e:
                if verbose:
                    print(f"         Attribute access failed: {e}")

        # Method 3: Try get_all_samples() for torchcodec AudioDecoder
        if hasattr(v, "get_all_samples"):
            try:
                samples = v.get_all_samples()
                if verbose:
                    print(
                        f"         v.get_all_samples() returned: {type(samples).__name__}"
                    )
                if samples is not None:
                    # torchcodec returns a tensor, convert to numpy
                    if hasattr(samples, "numpy"):
                        arr = samples.numpy()
                    elif hasattr(samples, "data"):
                        arr = np.asarray(samples.data)
                    else:
                        arr = np.asarray(samples)
                    if verbose:
                        print(f"         Array shape: {arr.shape}")
                    # Get sample rate from metadata if available
                    sr = None
                    if hasattr(v, "metadata"):
                        meta = v.metadata
                        if verbose:
                            print(f"         metadata: {meta}")
                        if hasattr(meta, "sample_rate"):
                            sr = meta.sample_rate
                        elif isinstance(meta, dict):
                            sr = meta.get("sample_rate") or meta.get("sampling_rate")
                    return {"array": arr, "sampling_rate": sr}
            except Exception as e:
                if verbose:
                    print(f"         v.get_all_samples() failed: {e}")
                    import traceback

                    traceback.print_exc()

        # Method 4: Try decode() method
        if hasattr(v, "decode"):
            try:
                decoded = v.decode()
                if verbose:
                    print(f"         v.decode() returned: {type(decoded).__name__}")
                if isinstance(decoded, dict):
                    return decoded
            except Exception as e:
                if verbose:
                    print(f"         v.decode() failed: {e}")

        # Method 5: Try __getitem__ with slice
        if hasattr(v, "__getitem__"):
            try:
                decoded = v[:]
                if verbose:
                    print(f"         v[:] returned: {type(decoded).__name__}")
                if isinstance(decoded, dict):
                    return decoded
                if isinstance(decoded, np.ndarray):
                    return {"array": decoded, "sampling_rate": None}
            except Exception as e:
                if verbose:
                    print(f"         v[:] failed: {e}")

        # Method 6: Try to_dict
        if hasattr(v, "to_dict"):
            try:
                decoded = v.to_dict()
                if verbose:
                    print(f"         v.to_dict() returned: {type(decoded).__name__}")
                if isinstance(decoded, dict):
                    return decoded
            except Exception as e:
                if verbose:
                    print(f"         v.to_dict() failed: {e}")

        return None

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
            elif "AudioDecoder" in v_type or "Decoder" in v_type:
                print(f"      {k}: {v_type} (lazy decoder)")
                # Try to decode it with verbose output for first item
                decoded = try_decode_audio(v, verbose=(i == 0))
                if decoded:
                    arr = decoded.get("array")
                    sr = decoded.get("sampling_rate")
                    if arr is not None:
                        arr = np.asarray(arr)
                        print(
                            f"         ✓ Decoded: shape={arr.shape}, dtype={arr.dtype}, sr={sr}"
                        )
                    else:
                        print(
                            f"         ✗ Decoded but no array, keys={list(decoded.keys())}"
                        )
                else:
                    print(f"         ✗ Failed to decode")
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
                    elif "AudioDecoder" in v_type or "Decoder" in v_type:
                        print(
                            f"      {k}: {v_type} (lazy decoder) - STILL NOT DECODED!"
                        )
                        # Try to decode it
                        decoded = try_decode_audio(v)
                        if decoded:
                            arr = decoded.get("array")
                            sr = decoded.get("sampling_rate")
                            if arr is not None:
                                arr = np.asarray(arr)
                                print(
                                    f"         ✓ Manual decode: shape={arr.shape}, dtype={arr.dtype}, sr={sr}"
                                )
                            else:
                                print(f"         ✗ Manual decode but no array")
                        else:
                            print(f"         ✗ Manual decode failed")
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
