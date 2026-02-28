#!/usr/bin/env python3
import argparse
import json


def dedup_jsonl(input_path: str, output_path: str) -> None:
    seen = set()
    kept = 0
    dropped = 0

    with (
        open(input_path, "r", encoding="utf-8") as fin,
        open(output_path, "w", encoding="utf-8") as fout,
    ):
        for line in fin:
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)
            key = (row.get("filehash"), row.get("text"))

            if key in seen:
                dropped += 1
                continue

            seen.add(key)
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1

    print(f"Kept: {kept}, Removed duplicates: {dropped}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("input_jsonl")
    p.add_argument("output_jsonl")
    args = p.parse_args()
    dedup_jsonl(args.input_jsonl, args.output_jsonl)
