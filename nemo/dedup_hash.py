#!/usr/bin/env python3
import argparse
import contextlib
import hashlib
import json
import sqlite3
import struct
import tempfile
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Final

AUDIO_PATH_KEYS = ("audio_filepath", "audio_file", "file", "filepath", "path")
FILE_HASH_ALGO = "sha256"
FILE_READ_CHUNK_SIZE = 1024 * 1024
GROUP_KEY_HASH_BYTES = 16
DEFAULT_HASH_CACHE_SIZE = 200_000
DEFAULT_COMMIT_EVERY = 50_000
_CACHE_MISS: Final[object] = object()


class AudioHashCache:
    def __init__(self, max_entries: int) -> None:
        self.max_entries = max(0, int(max_entries))
        self._data: OrderedDict[str, bytes | None] = OrderedDict()

    def get(self, key: str) -> bytes | None | object:
        if self.max_entries <= 0:
            return _CACHE_MISS
        value = self._data.pop(key, _CACHE_MISS)
        if value is not _CACHE_MISS:
            self._data[key] = value
        return value

    def put(self, key: str, value: bytes | None) -> None:
        if self.max_entries <= 0:
            return
        self._data[key] = value
        self._data.move_to_end(key)
        if len(self._data) > self.max_entries:
            self._data.popitem(last=False)


def _extract_audio_ref(row: dict[str, Any]) -> str | None:
    for key in AUDIO_PATH_KEYS:
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _normalize_duration_micros(value: Any) -> int | None:
    try:
        return int(round(float(value) * 1_000_000))
    except (TypeError, ValueError):
        return None


def _make_group_key(duration_micros: int, text: str) -> bytes:
    hasher = hashlib.blake2b(digest_size=GROUP_KEY_HASH_BYTES)
    hasher.update(struct.pack("<q", duration_micros))
    hasher.update(text.encode("utf-8", "ignore"))
    return hasher.digest()


def _resolve_audio_path(
    audio_ref: str, manifest_path: Path, audio_root: Path | None
) -> Path:
    raw_path = Path(audio_ref).expanduser()
    if raw_path.is_absolute():
        return raw_path

    if audio_root is not None:
        return audio_root / raw_path
    return manifest_path.parent / raw_path


SAMPLE_CHUNK = 64 * 1024  # 64 KiB from head + tail


def _hash_audio_path(path: Path) -> bytes:
    stat = path.stat()
    size = stat.st_size
    hasher = hashlib.blake2b(digest_size=32)
    hasher.update(struct.pack("<q", size))
    with path.open("rb") as f:
        head = f.read(SAMPLE_CHUNK)
        hasher.update(head)
        if size > SAMPLE_CHUNK * 2:
            f.seek(-SAMPLE_CHUNK, 2)
            hasher.update(f.read(SAMPLE_CHUNK))
        elif size > len(head):
            hasher.update(f.read())
    return hasher.digest()


def _init_state_db(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA cache_size=-200000")
    conn.execute("PRAGMA mmap_size=268435456")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS group_state (
            group_key BLOB PRIMARY KEY,
            first_audio_ref TEXT,
            initialized INTEGER NOT NULL DEFAULT 0
        ) WITHOUT ROWID
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS group_hash (
            group_key BLOB NOT NULL,
            audio_hash BLOB NOT NULL,
            PRIMARY KEY (group_key, audio_hash)
        ) WITHOUT ROWID
        """
    )
    conn.commit()


def dedup_jsonl(
    input_path: str,
    output_path: str,
    log_every: int = 100_000,
    removed_files_txt: str | None = None,
    audio_root: str | None = None,
    state_db: str | None = None,
    hash_cache_size: int = DEFAULT_HASH_CACHE_SIZE,
    commit_every: int = DEFAULT_COMMIT_EVERY,
    reverse: bool = False,
) -> None:
    manifest_path = Path(input_path).expanduser().resolve()
    audio_root_path = (
        Path(audio_root).expanduser().resolve() if audio_root is not None else None
    )
    hash_cache = AudioHashCache(hash_cache_size)
    processed = 0
    kept = 0
    dropped = 0
    hash_failures = 0
    collisions = 0
    cache_hits = 0
    cache_misses = 0
    t_start = time.monotonic()
    t_last_log = t_start

    temp_db_path: Path | None = None
    if state_db is None:
        with tempfile.NamedTemporaryFile(
            prefix="dedup_hash_", suffix=".sqlite3", delete=False
        ) as tmp:
            temp_db_path = Path(tmp.name)
        state_db_path = temp_db_path
    else:
        state_db_path = Path(state_db).expanduser().resolve()
        state_db_path.parent.mkdir(parents=True, exist_ok=True)

    with contextlib.ExitStack() as stack:
        if reverse:
            print("Reverse mode: reading all lines into memory...")
            with open(input_path, "r", encoding="utf-8") as f:
                all_lines = f.readlines()
            all_lines.reverse()
            print(f"Loaded {len(all_lines):,} lines (reversed). Starting dedup...")
            line_iter = iter(all_lines)
        else:
            fin = stack.enter_context(open(input_path, "r", encoding="utf-8"))
            line_iter = iter(fin)

        out_lines: list[str] | None = [] if reverse else None
        fout = (
            None
            if reverse
            else stack.enter_context(open(output_path, "w", encoding="utf-8"))
        )
        removed_fout = (
            stack.enter_context(open(removed_files_txt, "w", encoding="utf-8"))
            if removed_files_txt
            else None
        )
        conn = stack.enter_context(sqlite3.connect(state_db_path))
        _init_state_db(conn)
        conn.execute("BEGIN")

        for line in line_iter:
            processed += 1
            line = line.strip()
            if line:
                row = json.loads(line)
                text = row.get("text")
                duration_micros = _normalize_duration_micros(row.get("duration"))
                audio_ref = _extract_audio_ref(row)
                dedup_key = None
                if duration_micros is not None and isinstance(text, str):
                    dedup_key = _make_group_key(duration_micros, text)

                if dedup_key is None:
                    _out_line = json.dumps(row, ensure_ascii=False) + "\n"
                    if out_lines is not None:
                        out_lines.append(_out_line)
                    else:
                        fout.write(_out_line)
                    kept += 1
                else:
                    state_row = conn.execute(
                        "SELECT initialized, first_audio_ref FROM group_state WHERE group_key = ?",
                        (dedup_key,),
                    ).fetchone()
                    if state_row is None:
                        conn.execute(
                            "INSERT INTO group_state (group_key, first_audio_ref, initialized) VALUES (?, ?, 0)",
                            (dedup_key, audio_ref),
                        )
                        _out_line = json.dumps(row, ensure_ascii=False) + "\n"
                    if out_lines is not None:
                        out_lines.append(_out_line)
                    else:
                        fout.write(_out_line)
                        kept += 1
                    else:
                        collisions += 1
                        initialized = bool(state_row[0])
                        first_audio_ref = state_row[1] if not initialized else None

                        if not initialized:
                            if isinstance(first_audio_ref, str) and first_audio_ref:
                                first_path = _resolve_audio_path(
                                    first_audio_ref,
                                    manifest_path=manifest_path,
                                    audio_root=audio_root_path,
                                )
                                first_cache_key = str(first_path)
                                cached_first_hash = hash_cache.get(first_cache_key)
                                if cached_first_hash is _CACHE_MISS:
                                    cache_misses += 1
                                    try:
                                        first_hash = _hash_audio_path(first_path)
                                    except OSError:
                                        first_hash = None
                                        hash_failures += 1
                                    hash_cache.put(first_cache_key, first_hash)
                                else:
                                    cache_hits += 1
                                    first_hash = cached_first_hash

                                if first_hash is not None:
                                    conn.execute(
                                        "INSERT OR IGNORE INTO group_hash (group_key, audio_hash) VALUES (?, ?)",
                                        (dedup_key, first_hash),
                                    )
                            conn.execute(
                                "UPDATE group_state SET initialized = 1, first_audio_ref = NULL WHERE group_key = ?",
                                (dedup_key,),
                            )

                        current_hash: bytes | None = None
                        if isinstance(audio_ref, str) and audio_ref:
                            current_path = _resolve_audio_path(
                                audio_ref,
                                manifest_path=manifest_path,
                                audio_root=audio_root_path,
                            )
                            current_cache_key = str(current_path)
                            cached_current_hash = hash_cache.get(current_cache_key)
                            if cached_current_hash is _CACHE_MISS:
                                cache_misses += 1
                                try:
                                    current_hash = _hash_audio_path(current_path)
                                except OSError:
                                    current_hash = None
                                    hash_failures += 1
                                hash_cache.put(current_cache_key, current_hash)
                            else:
                                cache_hits += 1
                                current_hash = cached_current_hash

                        if current_hash is not None:
                            exists = conn.execute(
                                "SELECT 1 FROM group_hash WHERE group_key = ? AND audio_hash = ? LIMIT 1",
                                (dedup_key, current_hash),
                            ).fetchone()
                            if exists is not None:
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
                                    removed_fout.write(f"{removed_ref or ''}\n")
                                continue
                            conn.execute(
                                "INSERT INTO group_hash (group_key, audio_hash) VALUES (?, ?)",
                                (dedup_key, current_hash),
                            )
                        # If we can't hash current audio, we keep the row.
                        _out_line = json.dumps(row, ensure_ascii=False) + "\n"
                    if out_lines is not None:
                        out_lines.append(_out_line)
                    else:
                        fout.write(_out_line)
                        kept += 1

            if log_every > 0 and processed % log_every == 0:
                t_now = time.monotonic()
                elapsed = t_now - t_start
                interval = t_now - t_last_log
                rate = log_every / interval if interval > 0 else 0
                t_last_log = t_now
                print(
                    f"Processed: {processed:,} | Kept: {kept:,} | Removed: {dropped:,} | "
                    f"Collisions: {collisions:,} | Cache hit/miss: {cache_hits:,}/{cache_misses:,} | "
                    f"Hash errors: {hash_failures:,} | "
                    f"Rate: {rate:,.0f} rows/s | Elapsed: {elapsed:.1f}s"
                )
            if commit_every > 0 and processed % commit_every == 0:
                conn.commit()
                conn.execute("BEGIN")

        conn.commit()

    if reverse and out_lines is not None:
        print(f"Writing {len(out_lines):,} kept lines in original order...")
        out_lines.reverse()
        with open(output_path, "w", encoding="utf-8") as fout_rev:
            fout_rev.writelines(out_lines)
        del out_lines

    total_time = time.monotonic() - t_start
    print(
        f"Done. Processed: {processed:,} | Kept: {kept:,} | Removed: {dropped:,} | "
        f"Collisions: {collisions:,} | Cache hit/miss: {cache_hits:,}/{cache_misses:,} | "
        f"Total time: {total_time:.1f}s ({processed / total_time:,.0f} rows/s)"
    )
    if hash_failures > 0:
        print(
            f"Warning: failed to hash {hash_failures:,} file(s); those rows were kept."
        )
    if removed_files_txt:
        print(f"Removed-file log written to: {removed_files_txt}")
    if temp_db_path is not None:
        try:
            temp_db_path.unlink()
        except OSError:
            pass


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
    p.add_argument(
        "--state-db",
        default=None,
        help="Optional SQLite path for dedupe state (default: temporary file).",
    )
    p.add_argument(
        "--hash-cache-size",
        type=int,
        default=DEFAULT_HASH_CACHE_SIZE,
        help="In-memory LRU size for audio file hashes (set <=0 to disable).",
    )
    p.add_argument(
        "--commit-every",
        type=int,
        default=DEFAULT_COMMIT_EVERY,
        help="Commit SQLite state every N rows (set <=0 to commit only at the end).",
    )
    p.add_argument(
        "--reverse",
        action="store_true",
        help="Process lines from bottom to top (keep last occurrence, remove earlier duplicates).",
    )
    args = p.parse_args()
    dedup_jsonl(
        args.input_jsonl,
        args.output_jsonl,
        log_every=args.log_every,
        removed_files_txt=args.removed_files_txt,
        audio_root=args.audio_root,
        state_db=args.state_db,
        hash_cache_size=args.hash_cache_size,
        commit_every=args.commit_every,
        reverse=args.reverse,
    )
