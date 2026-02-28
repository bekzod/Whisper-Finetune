#!/usr/bin/env python3
import argparse
import contextlib
import hashlib
import json
from pathlib import Path
from typing import Any

AUDIO_PATH_KEYS = ("audio_filepath", "audio_file", "file", "filepath", "path")
FILE_HASH_ALGO = "sha256"
FILE_READ_CHUNK_SIZE = 1024 * 1024


def _extract_audio_ref(row: dict[str, Any]) -> str | None:
    for key in AUDIO_PATH_KEYS:
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _normalize_duration(value: Any) -> float | None:
    try:
        return round(float(value), 6)
    except (TypeError, ValueError):
        return None


def _resolve_audio_path(
    audio_ref: str, manifest_path: Path, audio_root: Path | None
) -> Path:
    raw_path = Path(audio_ref).expanduser()
    if raw_path.is_absolute():
        return raw_path

    candidates: list[Path] = []
    if audio_root is not None:
        candidates.append(audio_root / raw_path)
    candidates.extend((Path.cwd() / raw_path, manifest_path.parent / raw_path))

    for candidate in candidates:
        if candidate.exists():
            return candidate

    if audio_root is not None:
        return audio_root / raw_path
    return Path.cwd() / raw_path


def _hash_audio_path(path: Path) -> str:
    hasher = hashlib.new(FILE_HASH_ALGO)
    with path.open("rb") as f:
        while True:
            chunk = f.read(FILE_READ_CHUNK_SIZE)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def dedup_jsonl(
    input_path: str,
    output_path: str,
    log_every: int = 100_000,
    removed_files_txt: str | None = None,
    audio_root: str | None = None,
) -> None:
    manifest_path = Path(input_path).expanduser().resolve()
    audio_root_path = (
        Path(audio_root).expanduser().resolve() if audio_root is not None else None
    )

    # Keep the first row per (duration, text). If we see the same pair again, we hash files
    # and drop rows that match an already-seen hash in that pair.
    groups: dict[tuple[float, str], dict[str, Any]] = {}
    hash_cache: dict[Path, str | None] = {}
    processed = 0
    kept = 0
    dropped = 0
    hash_failures = 0

    with contextlib.ExitStack() as stack:
        fin = stack.enter_context(open(input_path, "r", encoding="utf-8"))
        fout = stack.enter_context(open(output_path, "w", encoding="utf-8"))
        removed_fout = (
            stack.enter_context(open(removed_files_txt, "w", encoding="utf-8"))
            if removed_files_txt
            else None
        )

        for line in fin:
            processed += 1
            line = line.strip()
            if line:
                row = json.loads(line)
                text = row.get("text")
                duration = _normalize_duration(row.get("duration"))
                audio_ref = _extract_audio_ref(row)
                dedup_key = (
                    (duration, text)
                    if duration is not None and isinstance(text, str)
                    else None
                )

                if dedup_key is None:
                    fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                    kept += 1
                else:
                    state = groups.get(dedup_key)
                    if state is None:
                        groups[dedup_key] = {
                            "first_audio_ref": audio_ref,
                            "hashes": None,
                        }
                        fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                        kept += 1
                    else:
                        hashes = state.get("hashes")
                        if hashes is None:
                            hashes = set()
                            first_audio_ref = state.get("first_audio_ref")
                            if isinstance(first_audio_ref, str) and first_audio_ref:
                                first_path = _resolve_audio_path(
                                    first_audio_ref,
                                    manifest_path=manifest_path,
                                    audio_root=audio_root_path,
                                )
                                if first_path in hash_cache:
                                    first_hash = hash_cache[first_path]
                                else:
                                    try:
                                        first_hash = _hash_audio_path(first_path)
                                    except OSError:
                                        first_hash = None
                                        hash_failures += 1
                                    hash_cache[first_path] = first_hash
                                if first_hash is not None:
                                    hashes.add(first_hash)
                            state["hashes"] = hashes
                            state["first_audio_ref"] = None

                        current_hash: str | None = None
                        if isinstance(audio_ref, str) and audio_ref:
                            current_path = _resolve_audio_path(
                                audio_ref,
                                manifest_path=manifest_path,
                                audio_root=audio_root_path,
                            )
                            if current_path in hash_cache:
                                current_hash = hash_cache[current_path]
                            else:
                                try:
                                    current_hash = _hash_audio_path(current_path)
                                except OSError:
                                    current_hash = None
                                    hash_failures += 1
                                hash_cache[current_path] = current_hash

                        if current_hash is not None and current_hash in hashes:
                            dropped += 1
                            if removed_fout is not None:
                                removed_ref = (
                                    audio_ref
                                    or row.get("audio_filepath")
                                    or row.get("audio_file")
                                    or row.get("file")
                                    or row.get("filepath")
                                    or row.get("path")
                                )
                                removed_fout.write(f"{removed_ref}\n")
                        else:
                            if current_hash is not None:
                                hashes.add(current_hash)
                            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                            kept += 1

            if log_every > 0 and processed % log_every == 0:
                print(
                    f"Processed: {processed:,} | Kept: {kept:,} | Removed duplicates: {dropped:,}"
                )

    print(
        f"Done. Processed: {processed:,} | Kept: {kept:,} | Removed duplicates: {dropped:,}"
    )
    if hash_failures > 0:
        print(
            f"Warning: failed to hash {hash_failures:,} file(s); those rows were kept."
        )
    if removed_files_txt:
        print(f"Removed-file log written to: {removed_files_txt}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("input_jsonl")
    p.add_argument("output_jsonl")
    p.add_argument(
        "--log-every",
        type=int,
        default=100_000,
        help="Print progress every N input lines (set <=0 to disable).",
    )
    p.add_argument(
        "--removed-files-txt",
        default=None,
        help="Optional path to write one file reference per removed duplicate row.",
    )
    p.add_argument(
        "--audio-root",
        default=None,
        help="Optional root directory for relative audio paths in the JSONL.",
    )
    args = p.parse_args()
    dedup_jsonl(
        args.input_jsonl,
        args.output_jsonl,
        log_every=args.log_every,
        removed_files_txt=args.removed_files_txt,
        audio_root=args.audio_root,
    )
