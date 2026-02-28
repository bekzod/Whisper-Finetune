#!/usr/bin/env python3
import argparse
import contextlib
import json


def dedup_jsonl(
    input_path: str,
    output_path: str,
    log_every: int = 100_000,
    removed_files_txt: str | None = None,
) -> None:
    seen = set()
    processed = 0
    kept = 0
    dropped = 0

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
                key = (row.get("filehash"), row.get("text"))

                if key in seen:
                    dropped += 1
                    if removed_fout is not None:
                        # Keep one useful identifier per removed row for quick cleanup.
                        removed_ref = (
                            row.get("audio_filepath")
                            or row.get("audio_file")
                            or row.get("file")
                            or row.get("filepath")
                            or row.get("path")
                            or row.get("filehash")
                        )
                        removed_fout.write(f"{removed_ref}\n")
                else:
                    seen.add(key)
                    fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                    kept += 1

            if log_every > 0 and processed % log_every == 0:
                print(
                    f"Processed: {processed:,} | Kept: {kept:,} | Removed duplicates: {dropped:,}"
                )

    print(
        f"Done. Processed: {processed:,} | Kept: {kept:,} | Removed duplicates: {dropped:,}"
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
    args = p.parse_args()
    dedup_jsonl(
        args.input_jsonl,
        args.output_jsonl,
        log_every=args.log_every,
        removed_files_txt=args.removed_files_txt,
    )
