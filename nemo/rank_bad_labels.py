#!/usr/bin/env python3
"""
rank_bad_labels.py

Rank likely-bad ASR labels by retranscribing audio with a NeMo / Parakeet model
and comparing the hypothesis to the existing transcript.

Input:
  JSONL manifest with at least:
    {"audio_filepath": "/abs/or/rel/path.wav", "text": "reference transcript"}
  Optional:
    {"duration": 12.34, "id": "utt_001"}

Output:
  - ranked CSV
  - ranked JSONL
  sorted by suspicion_score descending

Example:
  python rank_bad_labels.py \
    --manifest train_manifest.jsonl \
    --model nvidia/parakeet-tdt-0.6b-v3 \
    --batch-size 8 \
    --output-prefix ranked_train \
    --device cuda

Install:
  pip install nemo_toolkit['asr'] jiwer soundfile librosa pandas tqdm

Notes:
  - Streams results to disk incrementally — only one batch + a top-K heap
    are held in memory at a time.
  - Supports --resume to continue from where a previous run left off.
  - This version uses robust, model-agnostic signals:
      * WER / CER between existing text and model hypothesis
      * normalized edit distances
      * text-length mismatch
      * speaking-rate mismatch (chars/sec)
  - It does NOT auto-replace labels.
  - Review top suspicious items manually.
"""

import argparse
import heapq
import json
import logging
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import pandas as pd
import torch
import torch.multiprocessing as mp
from tqdm import tqdm

try:
    import soundfile as sf
except ImportError:
    sf = None

# NeMo
import nemo.collections.asr as nemo_asr
from jiwer import cer, wer

_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
LOGGER = logging.getLogger(__name__)

TOP_K = 500


def normalize_text(
    text: str,
    lowercase: bool = True,
    remove_punctuation: bool = False,
    collapse_whitespace: bool = True,
) -> str:
    if text is None:
        return ""
    text = str(text).strip()
    if lowercase:
        text = text.lower()
    if remove_punctuation:
        text = _PUNCT_RE.sub(" ", text)
    if collapse_whitespace:
        text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    if b == 0:
        return default
    return a / b


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def get_duration_seconds(audio_path: str) -> float:
    if sf is None:
        return float("nan")
    info = sf.info(audio_path)
    return float(info.frames) / float(info.samplerate)


def count_manifest_lines(path: str) -> int:
    """Quick first pass to count non-empty lines for progress bars."""
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def iter_manifest(path: str) -> Iterator[Dict[str, Any]]:
    """Yield one row at a time from a JSONL manifest."""
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Bad JSON on line {lineno}: {e}") from e

            if "audio_filepath" not in item:
                raise ValueError(f"Line {lineno} missing 'audio_filepath'")
            if "text" not in item:
                raise ValueError(f"Line {lineno} missing 'text'")

            yield item


def batchify_iter(it: Iterator[Any], batch_size: int) -> Iterator[List[Any]]:
    """Collect items from an iterator into batches."""
    batch = []
    for item in it:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def transcribe_batch(model, audio_paths: List[str]) -> List[str]:
    """
    Works with NeMo ASR models whose transcribe() may return:
      - list[str]
      - list[Hypothesis/text-like objects]
    """
    outputs = model.transcribe(audio_paths, batch_size=len(audio_paths))
    hyps = []

    for out in outputs:
        if isinstance(out, str):
            hyps.append(out)
        elif hasattr(out, "text"):
            hyps.append(out.text)
        else:
            hyps.append(str(out))

    return hyps


def compute_row_scores(
    ref_raw: str,
    hyp_raw: str,
    duration: float,
    lowercase: bool,
    remove_punctuation: bool,
) -> Dict[str, float]:
    ref = normalize_text(
        ref_raw, lowercase=lowercase, remove_punctuation=remove_punctuation
    )
    hyp = normalize_text(
        hyp_raw, lowercase=lowercase, remove_punctuation=remove_punctuation
    )

    ref_words = ref.split()
    hyp_words = hyp.split()

    ref_chars = len(ref.replace(" ", ""))
    hyp_chars = len(hyp.replace(" ", ""))

    # Distance signals
    row_wer = wer(ref, hyp) if ref.strip() else (0.0 if not hyp.strip() else 1.0)
    row_cer = cer(ref, hyp) if ref.strip() else (0.0 if not hyp.strip() else 1.0)

    # Length mismatch
    word_count_ratio = safe_div(len(hyp_words), max(1, len(ref_words)), default=1.0)
    char_count_ratio = safe_div(hyp_chars, max(1, ref_chars), default=1.0)

    # Symmetric mismatch around 1.0
    word_len_mismatch = abs(math.log(max(word_count_ratio, 1e-6)))
    char_len_mismatch = abs(math.log(max(char_count_ratio, 1e-6)))

    # Speaking-rate mismatch: chars/sec
    if duration and not math.isnan(duration) and duration > 0:
        ref_cps = ref_chars / duration
        hyp_cps = hyp_chars / duration
        cps_ratio = safe_div(hyp_cps, max(ref_cps, 1e-6), default=1.0)
        cps_mismatch = abs(math.log(max(cps_ratio, 1e-6)))
    else:
        ref_cps = float("nan")
        hyp_cps = float("nan")
        cps_mismatch = 0.0

    # Heuristic score: emphasize transcript disagreement first.
    wer_s = clamp01(row_wer)
    cer_s = clamp01(row_cer)
    word_len_s = clamp01(word_len_mismatch / 1.0)
    char_len_s = clamp01(char_len_mismatch / 1.0)
    cps_s = clamp01(cps_mismatch / 0.7)

    suspicion = (
        0.45 * wer_s
        + 0.30 * cer_s
        + 0.15 * char_len_s
        + 0.05 * word_len_s
        + 0.05 * cps_s
    )

    if suspicion >= 0.85:
        bucket = "very_high"
    elif suspicion >= 0.65:
        bucket = "high"
    elif suspicion >= 0.40:
        bucket = "medium"
    else:
        bucket = "low"

    return {
        "ref_norm": ref,
        "hyp_norm": hyp,
        "wer": row_wer,
        "cer": row_cer,
        "ref_words": len(ref_words),
        "hyp_words": len(hyp_words),
        "ref_chars": ref_chars,
        "hyp_chars": hyp_chars,
        "word_count_ratio": word_count_ratio,
        "char_count_ratio": char_count_ratio,
        "ref_chars_per_sec": ref_cps,
        "hyp_chars_per_sec": hyp_cps,
        "suspicion_score": suspicion,
        "bucket": bucket,
    }


def estimate_max_workers(model_size_gb: float = 1.5, headroom_gb: float = 2.0) -> int:
    """Estimate how many model replicas fit in free VRAM."""
    if not torch.cuda.is_available():
        return 1
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    free_gb = free_bytes / (1024**3)
    usable = max(0, free_gb - headroom_gb)
    n = max(1, int(usable // model_size_gb))
    return n


def _collect_done_audio_paths(worker_paths: List[str]) -> set:
    """Read all existing worker files and collect audio_filepath values already processed."""
    done = set()
    for wpath in worker_paths:
        if os.path.exists(wpath):
            with open(wpath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        done.add(rec["audio_filepath"])
                    except (json.JSONDecodeError, KeyError):
                        continue
    return done


def _worker_transcribe(
    worker_id: int,
    model_name: str,
    device: str,
    chunk: List[Dict[str, Any]],
    batch_size: int,
    manifest_dir: str,
    resolve_relative: bool,
    lowercase: bool,
    remove_punctuation: bool,
    output_path: str,
    done_audio_paths: set,
):
    """Worker process: loads its own model replica, transcribes its chunk, writes results."""
    import nemo.collections.asr as nemo_asr  # re-import in subprocess

    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s | worker-{worker_id} | %(levelname)s | %(message)s",
    )
    log = logging.getLogger(f"worker-{worker_id}")

    # Resolve paths first so we can filter already-done rows
    for row in chunk:
        prepare_row(row, manifest_dir, resolve_relative)

    if done_audio_paths:
        before = len(chunk)
        chunk = [r for r in chunk if r["audio_filepath"] not in done_audio_paths]
        skipped = before - len(chunk)
        if skipped:
            log.info("Resuming: skipping %d already-processed rows", skipped)

    if not chunk:
        log.info("Nothing to do, all rows already processed")
        return

    log.info("Loading model: %s", model_name)
    if model_name.endswith(".nemo") and os.path.exists(model_name):
        model = nemo_asr.models.ASRModel.restore_from(model_name)
    else:
        model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)

    if device == "cuda":
        model = model.cuda()
    model.eval()
    log.info("Model ready, processing %d rows", len(chunk))

    with open(output_path, "a", encoding="utf-8") as out_f:
        for i in range(0, len(chunk), batch_size):
            batch_rows = chunk[i : i + batch_size]

            audio_paths = [r["audio_filepath"] for r in batch_rows]
            hyps = transcribe_batch(model, audio_paths)

            for row, hyp in zip(batch_rows, hyps):
                metrics = compute_row_scores(
                    ref_raw=row["text"],
                    hyp_raw=hyp,
                    duration=row.get("duration", float("nan")),
                    lowercase=lowercase,
                    remove_punctuation=remove_punctuation,
                )
                out = dict(row)
                out["model_hypothesis"] = hyp
                out.update(metrics)
                out_f.write(json.dumps(out, ensure_ascii=False) + "\n")

            out_f.flush()

    log.info("Done: wrote %d rows to %s", len(chunk), output_path)


def prepare_row(row: Dict[str, Any], manifest_dir: str, resolve_relative: bool) -> None:
    """Resolve paths and fill missing durations in-place."""
    if resolve_relative and not os.path.isabs(row["audio_filepath"]):
        row["audio_filepath"] = str(Path(manifest_dir) / row["audio_filepath"])

    if "duration" not in row or row["duration"] in (None, ""):
        try:
            row["duration"] = get_duration_seconds(row["audio_filepath"])
        except Exception:
            row["duration"] = float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="Input JSONL manifest")
    parser.add_argument(
        "--model",
        default="nvidia/parakeet-tdt-0.6b-v3",
        help="NeMo/HF model name or local .nemo path",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-prefix", default="ranked_manifest")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--lowercase", action="store_true", default=False)
    parser.add_argument("--remove-punctuation", action="store_true", default=False)
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    parser.add_argument(
        "--resolve-relative-to-manifest", action="store_true", default=False
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume from a previous interrupted run",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel model replicas (0 = auto-detect from free VRAM)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    manifest_dir = str(Path(args.manifest).resolve().parent)
    unsorted_path = f"{args.output_prefix}._unsorted.jsonl"

    # Count total rows for progress bar
    LOGGER.info("Counting manifest rows: %s", args.manifest)
    total_rows = count_manifest_lines(args.manifest)
    LOGGER.info("Total rows: %d", total_rows)

    # Determine number of workers
    num_workers = args.num_workers
    if num_workers <= 0:
        if args.device == "cuda" and torch.cuda.is_available():
            num_workers = estimate_max_workers(model_size_gb=1.5, headroom_gb=2.0)
        else:
            num_workers = 1
    LOGGER.info("Using %d parallel worker(s)", num_workers)

    # Resume support: count already-processed lines
    skip_rows = 0
    if args.resume and os.path.exists(unsorted_path):
        with open(unsorted_path, "r", encoding="utf-8") as f:
            for _ in f:
                skip_rows += 1
        LOGGER.info("Resuming: skipping %d already-processed rows", skip_rows)

    # Read all remaining rows into memory for chunking
    manifest_iter = iter_manifest(args.manifest)
    if skip_rows > 0:
        for _ in zip(range(skip_rows), manifest_iter):
            pass
    remaining_rows = list(manifest_iter)
    remaining = len(remaining_rows)
    LOGGER.info("Rows to process: %d (skipped %d)", remaining, skip_rows)

    if num_workers > 1 and remaining > 0:
        # --- Multi-worker parallel path ---
        mp.set_start_method("spawn", force=True)

        # Split rows into roughly equal chunks
        chunk_size = math.ceil(remaining / num_workers)
        chunks = [
            remaining_rows[i : i + chunk_size] for i in range(0, remaining, chunk_size)
        ]
        actual_workers = len(chunks)

        # Stable worker output files (survive interrupts for resume)
        worker_paths = [
            f"{args.output_prefix}._worker_{i}.jsonl" for i in range(actual_workers)
        ]

        # On resume, collect already-processed audio paths from worker files
        done_audio_paths: set = set()
        if args.resume:
            done_audio_paths = _collect_done_audio_paths(worker_paths)
            if done_audio_paths:
                LOGGER.info(
                    "Resume: found %d already-processed rows across worker files",
                    len(done_audio_paths),
                )

        LOGGER.info(
            "Spawning %d workers (chunks of ~%d rows each)",
            actual_workers,
            chunk_size,
        )

        processes = []
        for i, (chunk, wpath) in enumerate(zip(chunks, worker_paths)):
            p = mp.Process(
                target=_worker_transcribe,
                args=(
                    i,
                    args.model,
                    args.device,
                    chunk,
                    args.batch_size,
                    manifest_dir,
                    args.resolve_relative_to_manifest,
                    args.lowercase,
                    args.remove_punctuation,
                    wpath,
                    done_audio_paths,
                ),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        # Check for failures
        for i, p in enumerate(processes):
            if p.exitcode != 0:
                LOGGER.error("Worker %d exited with code %d", i, p.exitcode)

        # Merge worker outputs into unsorted file
        with open(unsorted_path, "w", encoding="utf-8") as out_f:
            for wpath in worker_paths:
                if os.path.exists(wpath):
                    with open(wpath, "r", encoding="utf-8") as wf:
                        for line in wf:
                            out_f.write(line)

    elif remaining > 0:
        # --- Single-worker path (original logic) ---
        LOGGER.info("Loading model: %s", args.model)
        if args.model.endswith(".nemo") and os.path.exists(args.model):
            model = nemo_asr.models.ASRModel.restore_from(args.model)
        else:
            model = nemo_asr.models.ASRModel.from_pretrained(model_name=args.model)

        if args.device == "cuda":
            model = model.cuda()
        model.eval()
        LOGGER.info("Model ready on device: %s", args.device)

        total_batches = math.ceil(remaining / max(1, args.batch_size))
        open_mode = "a" if (args.resume and skip_rows > 0) else "w"

        with open(unsorted_path, open_mode, encoding="utf-8") as out_f:
            for i in tqdm(
                range(0, remaining, args.batch_size),
                total=total_batches,
                desc="Transcribing & scoring",
            ):
                batch_rows = remaining_rows[i : i + args.batch_size]

                for row in batch_rows:
                    prepare_row(row, manifest_dir, args.resolve_relative_to_manifest)

                audio_paths = [r["audio_filepath"] for r in batch_rows]
                hyps = transcribe_batch(model, audio_paths)

                for row, hyp in zip(batch_rows, hyps):
                    metrics = compute_row_scores(
                        ref_raw=row["text"],
                        hyp_raw=hyp,
                        duration=row.get("duration", float("nan")),
                        lowercase=args.lowercase,
                        remove_punctuation=args.remove_punctuation,
                    )
                    out = dict(row)
                    out["model_hypothesis"] = hyp
                    out.update(metrics)
                    out_f.write(json.dumps(out, ensure_ascii=False) + "\n")

                out_f.flush()

    LOGGER.info("Transcription pass complete: %d rows processed", remaining + skip_rows)

    # Build bucket_counts and top_k_heap from the full unsorted file
    top_k_heap: List[Tuple[float, int, Dict]] = []
    bucket_counts: Counter = Counter()
    rows_processed = 0
    if os.path.exists(unsorted_path):
        with open(unsorted_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                rec = json.loads(line)
                bucket_counts[rec["bucket"]] += 1
                score = rec["suspicion_score"]
                entry = (score, idx, rec)
                if len(top_k_heap) < TOP_K:
                    heapq.heappush(top_k_heap, entry)
                elif score > top_k_heap[0][0]:
                    heapq.heapreplace(top_k_heap, entry)
                rows_processed += 1

    # Sort and write final outputs
    LOGGER.info("Reading unsorted results for final sort")
    df = pd.read_json(unsorted_path, lines=True)
    df = df.sort_values(["suspicion_score", "wer", "cer"], ascending=False).reset_index(
        drop=True
    )

    csv_path = f"{args.output_prefix}.csv"
    jsonl_path = f"{args.output_prefix}.jsonl"
    top500_path = f"{args.output_prefix}.top500.csv"

    LOGGER.info("Writing outputs with prefix: %s", args.output_prefix)
    df.to_csv(csv_path, index=False)

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in df.to_dict(orient="records"):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Top 500 from heap (already in memory, no need to read from disk)
    top_k_sorted = sorted(top_k_heap, key=lambda x: x[0], reverse=True)
    top_k_rows = [entry[2] for entry in top_k_sorted]
    pd.DataFrame(top_k_rows).to_csv(top500_path, index=False)

    LOGGER.info("Saved: %s", csv_path)
    LOGGER.info("Saved: %s", jsonl_path)
    LOGGER.info("Saved: %s", top500_path)

    # Clean up temp files
    try:
        os.remove(unsorted_path)
    except OSError:
        pass
    # Remove worker shard files from multi-worker runs
    import glob as _glob

    for wf in _glob.glob(f"{args.output_prefix}._worker_*.jsonl"):
        try:
            os.remove(wf)
        except OSError:
            pass

    # Summary
    print()
    print("Bucket counts:")
    for bucket in ["very_high", "high", "medium", "low"]:
        print(f"  {bucket:>10s}  {bucket_counts.get(bucket, 0)}")

    print()
    print("Top 10 suspicious rows:")
    cols = [
        "audio_filepath",
        "duration",
        "suspicion_score",
        "wer",
        "cer",
        "text",
        "model_hypothesis",
    ]
    top10 = [entry[2] for entry in top_k_sorted[:10]]
    print(pd.DataFrame(top10)[cols].to_string(index=False))


if __name__ == "__main__":
    main()
